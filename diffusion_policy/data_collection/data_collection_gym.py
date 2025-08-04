from scripts.diffusion_env import DiffusionEnv
import time
import numpy as np
import os
import pickle
from scipy.spatial.transform import Rotation as R


class PickPlacePolicy:
    def __init__(self, steps_per_phase=100):
        self.state_id = 0
        self.step_in_phase = 0
        self.steps_per_phase = steps_per_phase
        self.gripper = 1.0  # Open initially

        self.trajectory = []  # Will be list of [position, quaternion, gripper]

    def plan_trajectory(self, obs):
        cube = np.array(obs["cube"])
        target = np.array(obs["target"])

        down_quat = R.from_euler("xyz", [np.pi, 0, 0]).as_quat()

        # Define key poses
        waypoints = [
            (cube + [0, 0, 0.5], 1.0),  # above cube
            (cube + [0, 0, 0.15], 1.0),  # at cube (open)
            (cube + [0, 0, 0.15], 0.0),  # grasp (close)
            (cube + [0, 0, 0.3], 0.0),  # lift
            (target + [0, 0, 0.4], 0.0),  # above target
            (target + [0, 0, 0.3], 0.0),  # at target (closed)
            (target + [0, 0, 0.3], 1.0),  # release
            (target + [0, 0, 0.3], 1.0),  # hover
        ]

        self.trajectory = []
        for i in range(len(waypoints) - 1):
            start_pos, start_grip = waypoints[i]
            end_pos, end_grip = waypoints[i + 1]
            for s in range(self.steps_per_phase):
                alpha = s / self.steps_per_phase
                pos = (1 - alpha) * start_pos + alpha * end_pos
                grip = (1 - alpha) * start_grip + alpha * end_grip
                self.trajectory.append((pos, down_quat, grip))

    def __call__(self, obs):
        if not self.trajectory:
            self.plan_trajectory(obs)

        if self.state_id >= len(self.trajectory):
            # Hold last pose
            pos, quat, grip = self.trajectory[-1]
        else:
            pos, quat, grip = self.trajectory[self.state_id]
            self.state_id += 1

        return list(pos) + list(quat) + [grip]


CONFIG = "../config/global_config.yaml"
num_trajectories = 300
save_dir = "trajectories"
os.makedirs(save_dir, exist_ok=True)

env = DiffusionEnv(config=CONFIG, sim=True)
policy = PickPlacePolicy(steps_per_phase=75)

for traj_id in range(num_trajectories):
    observation, info = env.reset()
    policy = PickPlacePolicy(steps_per_phase=75)
    time.sleep(2.0)  # Allow environment to reset
    trajectory = []

    while True:
        action = policy(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        # Flatten obs/state and next_obs/state
        state = (
            list(observation["cube"])
            + list(observation["target"])
            + list(observation["gripper"])
            + list(observation["franka_ee"][0])
        )

        next_state = (
            list(next_observation["cube"])
            + list(next_observation["target"])
            + list(next_observation["gripper"])
            + list(next_observation["franka_ee"][0])
        )

        row = {"state": state, "action": list(action), "next_state": next_state}
        trajectory.append(row)

        if terminated:
            print(
                f"SUCCESS: Trajectory {traj_id} ended. Saving to pickle, steps:{info}"
            )
            # Save as pickle
            pkl_path = os.path.join(save_dir, f"trajectory_{traj_id:03d}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(trajectory, f)
            break
        elif truncated:
            print(f"FAILED: Trajectory {traj_id} truncated, steps:{info}")
            break
        else:
            observation = next_observation

        time.sleep(0.1)
