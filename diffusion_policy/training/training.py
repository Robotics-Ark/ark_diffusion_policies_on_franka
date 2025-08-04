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
# env import
import gym
from gym import spaces
import os

from diffusion_data_loader import DiffusionPolicyDataset
from diffusion_network import *

trajectory_dir = "../data_collection/trajectories"

# horizons (same convention as Push‑T example)
pred_horizon   = 16  # p
obs_horizon    = 8   # o
action_horizon = 8   # a
subsample: int = 2

num_epochs = 20000

obs_dim = 10
action_dim = 8
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

dataset = DiffusionPolicyDataset(
    dataset_path   = trajectory_dir,
    pred_horizon   = pred_horizon,
    obs_horizon    = obs_horizon,
    action_horizon = action_horizon,
    subsample      = subsample,
)

dataloader = DataLoader(
    dataset,
    batch_size          = 256,
    num_workers         = 1,
    shuffle             = True,
    pin_memory          = True,
    persistent_workers  = True,
)

# quick sanity‑check
batch = next(iter(dataloader))
print("batch['obs'].shape   :", batch["obs"].shape)      # expected (batch_size, obs_horizon, obs_dim)
print("batch['action'].shape:", batch["action"].shape)   # expected (batch_size, pred_horizon, action_dim)

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# example inputs
noised_action = torch.randn((1, pred_horizon, action_dim))
obs = torch.zeros((1, obs_horizon, obs_dim))
diffusion_iter = torch.zeros((1,))

# the noise prediction network
# takes noisy action, diffusion iteration and observation as input
# predicts the noise added to action
noise = noise_pred_net(
    sample=noised_action,
    timestep=diffusion_iter,
    global_cond=obs.flatten(start_dim=1))

# illustration of removing noise
# the actual noise removal is performed by NoiseScheduler
# and is dependent on the diffusion noise schedule
denoised_action = noised_action - noise

# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
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

# device transfer
device = torch.device('cuda')
_ = noise_pred_net.to(device)

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-3, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=50,
    num_training_steps=len(dataloader) * num_epochs
)

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nobs = nbatch['obs'].to(device)
                naction = nbatch['action'].to(device)
                B = nobs.shape[0]

                # observation as FiLM conditioning
                # (B, obs_horizon, obs_dim)
                obs_cond = nobs[:,:obs_horizon,:]
                # (B, obs_horizon * obs_dim)
                obs_cond = obs_cond.flatten(start_dim=1)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)

                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(noise_pred_net.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                print(f"Epoch {epoch_idx}, loss = {loss_cpu}")
                tepoch.set_postfix(loss=loss_cpu)

        # Weights of the EMA model
        # is used for inference
        if epoch_idx % 10 == 0:
            ema_noise_pred_net = noise_pred_net
            # save EMA model weights every 10 epochs
            torch.save(
                ema_noise_pred_net.state_dict(),
                f"ema_noise_pred_net_epoch_{epoch_idx}.pth"
            )
            
        tglobal.set_postfix(loss=np.mean(epoch_loss))
        # save the model
        torch.save(
            noise_pred_net.state_dict(),
            f"noise_pred_net_epoch_{epoch_idx}.pth"
        )
        