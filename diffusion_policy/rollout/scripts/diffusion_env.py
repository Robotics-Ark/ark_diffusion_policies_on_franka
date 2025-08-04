from ark.env.ark_env import ArkEnv
from arktypes import joint_group_command_t, joint_state_t, rigid_body_state_t
from arktypes.utils import pack, unpack
from arktypes import task_space_command_t, pose_t
import numpy as np


class DiffusionEnv(ArkEnv):

    def __init__(self, config=None, sim=True):
        environment_name = "diffusion_env"

        action_channels = {
            "franka/cartesian_command/sim": task_space_command_t,
        }

        observation_channels = {
            "franka/ee_state/sim": pose_t,
            "franka/joint_states/sim": joint_state_t,
            "cube/ground_truth/sim": rigid_body_state_t,
            "target/ground_truth/sim": rigid_body_state_t,
        }

        self.steps = 0
        self.max_steps = 500

        super().__init__(
            environment_name=environment_name,
            action_channels=action_channels,
            observation_channels=observation_channels,
            global_config=config,
            sim=sim,
        )

    def action_packing(self, action):
        """
        Packs the action into a joint_group_command_t format.

        List of:
        [EE X, EE_Y, EE_Z, EE_Quaternion_X, EE_Quaternion_Y, EE_Quaternion_Z, EE_Quaternion_W, Gripper]
        """

        xyz_command = np.array(action[:3])
        quaternion_command = np.array(action[3:7])
        gripper_command = action[7]

        franka_cartesian_command = pack.task_space_command(
            "all", xyz_command, quaternion_command, gripper_command
        )
        return {"franka/cartesian_command/sim": franka_cartesian_command}

    def observation_unpacking(self, observation_dict):
        """
        Unpacks the observation from the environment.

        Returns a dictionary with keys
        """
        cube_state = observation_dict["cube/ground_truth/sim"]
        target_state = observation_dict["target/ground_truth/sim"]
        joint_state = observation_dict["franka/joint_states/sim"]
        ee_state = observation_dict["franka/ee_state/sim"]

        _, cube_position, _, _, _ = unpack.rigid_body_state(cube_state)
        _, target_position, _, _, _ = unpack.rigid_body_state(target_state)
        _, _, franka_joint_position, _, _ = unpack.joint_state(joint_state)
        franka_ee_position, franka_ee_orientation = unpack.pose(ee_state)

        gripper_position = franka_joint_position[
            -2
        ]  # Assuming last two joints are gripper

        return {
            "cube": cube_position,
            "target": target_position,
            "gripper": [gripper_position],
            "franka_ee": (franka_ee_position, franka_ee_orientation),
        }

    def reset_objects(self):
        self.steps = 0
        self.reset_component("cube")
        self.reset_component("target")
        self.reset_component("franka")

    def reward(self, state, action, next_state):
        return 0.0

    def step(self, action):
        self.steps += 1
        return super().step(action)

    def terminated_truncated_info(self, state, action, next_state):
        cube_pos = np.array(state["cube"])
        target_pos = np.array(state["target"])

        # Terminate if cube is within 5 cm of target
        distance = np.linalg.norm(cube_pos - target_pos)
        terminated = distance < 0.1

        if terminated:
            print("Cube is close enough to the target. Terminating episode.")

        if self.steps >= self.max_steps:
            print("Max steps reached. Terminating episode.")
            truncated = True
        else:
            truncated = False

        return terminated, truncated, self.steps
