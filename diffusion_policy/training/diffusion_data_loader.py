import os
import glob
import pickle
from pathlib import Path
from typing import List, Tuple, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class DiffusionPolicyDataset(Dataset):
    """
    A window‑based dataset for trajectories saved as pickled lists of dicts:
        {"state": [...], "action": [...], "next_state": [...]} (next_state is optional)

    Parameters
    ----------
    dataset_path : str | Path
        Folder that contains *.pkl trajectory files.
    pred_horizon : int
        How many *future* actions the model should predict (p in the ASCII sketch).
    obs_horizon : int
        How many most‑recent observations are fed to the model (o in the sketch).
    action_horizon : int
        How many actions that have *already happened* are provided as context (a in the sketch).
    subsample : int | Sequence[int], default 1
        Gap (in raw environment timesteps) between consecutive elements *within a single window*.
        A single integer applies the same stride to both observations and prediction targets.
        Passing a 1‑element tuple or list is also allowed for Hydra/OMEGACONF compatibility.
    in_memory : bool, default True
        Load whole trajectories into RAM (faster, but uses more memory).
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_subsample(value: Union[int, Sequence[int]]) -> int:
        """Ensure *value* is a positive integer; gracefully handle 1‑element sequences."""
        if isinstance(value, int):
            subsample_int = value
        elif isinstance(value, Sequence):
            if len(value) != 1:
                raise ValueError(
                    "subsample expects a single integer (or 1‑element sequence), "
                    f"got sequence of length {len(value)}: {value}"
                )
            subsample_int = int(value[0])
        else:
            raise TypeError(
                "subsample must be int or sequence of int; "
                f"got type {type(value).__name__}"
            )
        if subsample_int < 1:
            raise ValueError("subsample must be >= 1, got {subsample_int}")
        return subsample_int

    def __init__(
        self,
        dataset_path: str | Path,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        subsample: Union[int, Sequence[int]] = 1,
        in_memory: bool = True,
    ):
        super().__init__()
        self.subsample = self._coerce_subsample(subsample)

        self.dataset_path = Path(dataset_path)
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.in_memory = in_memory

        # ------------------------------------------------------------------
        # 1. Gather all pickle files
        # ------------------------------------------------------------------
        self.trajectories: List[List[dict]] = []
        self.file_index: List[str] = sorted(glob.glob(str(self.dataset_path / "*.pkl")))
        if not self.file_index:
            raise FileNotFoundError(f"No *.pkl files in {self.dataset_path}")

        # ------------------------------------------------------------------
        # 2. Load (or prepare to lazy‑load) trajectories
        # ------------------------------------------------------------------
        if self.in_memory:
            for file in self.file_index:
                with open(file, "rb") as f:
                    self.trajectories.append(pickle.load(f))  # type: ignore[arg-type]
        else:
            # store paths only; actual loading happens in __getitem__
            self.trajectories = [None] * len(self.file_index)  # placeholder

        # ------------------------------------------------------------------
        # 3. Build a global index mapping (traj_id, start_step) → sample
        # ------------------------------------------------------------------
        # A valid *raw* window must span the following number of env steps:
        #     span = (obs_horizon + action_horizon + pred_horizon - 1) * subsample + 1
        span = (
            self.obs_horizon + self.action_horizon + self.pred_horizon - 1
        ) * self.subsample + 1
        self.sample_index: List[Tuple[int, int]] = []
        for tid, traj in enumerate(
            self._get_traj(i) for i in range(len(self.file_index))
        ):
            if len(traj) < span:
                continue  # trajectory too short for even one window
            last_valid_start = len(traj) - span
            self.sample_index.extend([(tid, s) for s in range(last_valid_start + 1)])

        # ------------------------------------------------------------------
        # 4. Compute min‑max stats for normalisation / scaling (optional)
        # ------------------------------------------------------------------
        states: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        for traj in (self._get_traj(i) for i in range(len(self.file_index))):
            states.extend([row["state"] for row in traj])
            actions.extend([row["action"] for row in traj])
        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.float32)

        self.stats = {
            "state": {
                "min": torch.from_numpy(states_np.min(axis=0)),
                "max": torch.from_numpy(states_np.max(axis=0)),
            },
            "action": {
                "min": torch.from_numpy(actions_np.min(axis=0)),
                "max": torch.from_numpy(actions_np.max(axis=0)),
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_traj(self, idx: int) -> List[dict]:
        """Return trajectory `idx` (load from disk if necessary)."""
        if self.in_memory:
            return self.trajectories[idx]  # already a list
        if self.trajectories[idx] is None:  # lazy‑load
            with open(self.file_index[idx], "rb") as f:
                self.trajectories[idx] = pickle.load(f)
        return self.trajectories[idx]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj_id, start = self.sample_index[idx]
        traj = self._get_traj(traj_id)
        step = self.subsample

        # Build index lists -------------------------------------------------
        obs_indices = [start + i * step for i in range(self.obs_horizon)]
        pred_indices = [
            start + (self.obs_horizon + i) * step for i in range(self.pred_horizon)
        ]

        obs_seq = torch.tensor(
            [traj[t]["state"] for t in obs_indices], dtype=torch.float32
        )
        prediction_actions = torch.tensor(
            [traj[t]["action"] for t in pred_indices], dtype=torch.float32
        )

        return {
            "obs": obs_seq,  # (obs_horizon, state_dim)
            "action": prediction_actions,  # (pred_horizon, action_dim)
        }
