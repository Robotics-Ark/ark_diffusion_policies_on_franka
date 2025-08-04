# Diffusion Policies in Ark

A minimal, end‑to‑end pipeline for learning diffusion‑based manipulation policies on the Franka Emika Panda robot in simulation

Repository Layout

```
ark_diffusion_policies_on_franka/
├── diffusion_policy/
│   ├── config/           # YAML & JSON hyper‑parameter files
│   ├── data_collection/ # Nodes for recording demonstrations
│   ├── robots/          # simulation abstractions
│   ├── rollout/         # Inference scripts to deploy a trained policy
│   └── training/        # Training loops & utilities
├── gen.py               # Utility code‑generation script
├── .gitignore
├── LICENSE
└── README.md            # You are here
```

TL;DR – Run everything under data_collection/ to collect data, everything under training/ to train the model, and everything under rollout/ to deploy the trained policy.

## Data Collection

All launch files & nodes live in diffusion_policy/data_collection/. They stream sensor observations & joint commands, saving them as trajectory JSON/NPZ files.

## Training

The training loop consumes the trajectories recorded above and writes checkpoints to training/output/…

## Rollout / Deployment

Deploy a trained policy in simulation using the scripts in diffusion_policy/rollout/.
