import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import pickle
import csv

from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path

from diffusion_network import *
import matplotlib.pyplot as plt
from scripts.diffusion_env import DiffusionEnv

CONFIG = "../config/global_config.yaml"

model_path = "../training/models/ema_noise_pred_net_epoch_210.pth"

pred_horizon   = 16  # p
obs_horizon    = 8   # o
action_horizon = 8   # a
subsample: int = 2

batch_size = 1

obs_dim = 10
action_dim = 8
num_diffusion_iters = 100

# load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
state_dict = torch.load(model_path, map_location=device)

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75
)

# for this demo, we use DDPMScheduler with 100 diffusion iterations

noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

ema_noise_pred_net = noise_pred_net
ema_noise_pred_net.load_state_dict(state_dict)

# add a check to ensure the model is in eval mode
ema_noise_pred_net.eval()


def inference(nobs, noise_scheduler: DDPMScheduler, num_diffusion_iters: int = 100):
    nobs = nobs.to(device, dtype=torch.float32)
    B = 1
    with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            # print(pred_horizon, action_dim)
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            # send to cpu
            naction = noisy_action.to('cpu')
            obs_cond = obs_cond.to('cpu')

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                # naction
            
                noise_pred = ema_noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

    naction = naction.detach().to('cpu').numpy()
    # (B, pred_horizon, action_dim)
    naction = naction[0]
    return naction


env = DiffusionEnv(config=CONFIG, sim=True)
observation, info = env.reset()



SUCCESS = 0
for traj_id in range(50):
    print(f"Starting trajectory {traj_id}") 
    observation, info = env.reset()

    obs = list(observation["cube"]) + \
                list(observation["target"]) + \
                list(observation["gripper"]) + \
                list(observation["franka_ee"][0]) 

    print(obs)
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    time.sleep(2.0)  # Allow environment to reset
    
    B = 1

    while True:
        obs_seq = np.stack(obs_deque)

        action = inference(
            nobs=torch.tensor(obs_seq, dtype=torch.float32),
            noise_scheduler=noise_scheduler,
            num_diffusion_iters=num_diffusion_iters
        )

        # print(f"Action shape: {action.shape}")


        start = obs_horizon +1
        end = start + action_horizon
        action = action[start:end,:]

        for i in range(len(action)):
            next_observation, reward, terminated, truncated, info = env.step(action[i])

            obs = list(observation["cube"]) + \
                list(observation["target"]) + \
                list(observation["gripper"]) + \
                list(observation["franka_ee"][0])
            
            obs_deque.append(obs)
            # print("EXECUTED i")
            time.sleep(0.1)

        if terminated:
            print(f"SUCCESS: Trajectory {traj_id} ended")
            SUCCESS += 1
            break
            
        elif truncated:
            print(f"FAILED: Trajectory {traj_id} truncated")
            break
        else:
            observation = next_observation

        time.sleep(0.1)

print(f"Total successful trajectories: {SUCCESS}")