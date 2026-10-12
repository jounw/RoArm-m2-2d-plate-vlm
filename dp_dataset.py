"""Dataset for the eye-in-hand descend+grab Diffusion Policy.

Loads the HDF5 episodes written by dp_collect_data.py (each has
`cam_eih` [T,H,W,3] uint8, `pose` [T,4] float32 (x,y,z,t), `grip_closed`
[T] bool) and turns them into fixed-length (obs_horizon, action_horizon)
training windows, following the standard Diffusion Policy formulation
(Chi et al., 2023): at each timestep the model conditions on the last
`obs_horizon` observations and predicts the next `action_horizon` actions.

Actions are the raw (x, y, z, t) pose commands -- the same 4 numbers
pick_place.py already sends to RoArm.move_to()/servo_to(), so a trained
policy's output can replace track_and_descend's move_to() calls directly
with no extra conversion step.

Normalization: per-dimension min/max computed over the whole training set,
mapped to [-1, 1] (standard for diffusion policies, whose output head
trains best against a bounded range). Stats are saved alongside the model
checkpoint by dp_train.py and must be reused at inference time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # allow dp_dataset.py's pure-numpy pieces to be used/tested without torch installed
    torch = None
    Dataset = object


@dataclass
class Stats:
    action_min: np.ndarray
    action_max: np.ndarray

    def normalize(self, action: np.ndarray) -> np.ndarray:
        span = np.maximum(self.action_max - self.action_min, 1e-6)
        return ((action - self.action_min) / span) * 2.0 - 1.0

    def unnormalize(self, action: np.ndarray) -> np.ndarray:
        span = np.maximum(self.action_max - self.action_min, 1e-6)
        return (action + 1.0) / 2.0 * span + self.action_min

    def to_json(self) -> dict:
        return {"action_min": self.action_min.tolist(), "action_max": self.action_max.tolist()}

    @staticmethod
    def from_json(data: dict) -> "Stats":
        return Stats(action_min=np.array(data["action_min"], dtype=np.float32), action_max=np.array(data["action_max"], dtype=np.float32))

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_json()), encoding="utf-8")

    @staticmethod
    def load(path: str) -> "Stats":
        return Stats.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def compute_stats(episode_paths: list[Path]) -> Stats:
    all_poses = []
    for p in episode_paths:
        with h5py.File(p, "r") as f:
            all_poses.append(np.asarray(f["pose"], dtype=np.float32))
    poses = np.concatenate(all_poses, axis=0)  # (N, 4)
    return Stats(action_min=poses.min(axis=0), action_max=poses.max(axis=0))


def list_episodes(dataset_dir: str, object_name: str | None = None) -> list[Path]:
    d = Path(dataset_dir)
    pattern = f"{object_name}_ep*.hdf5" if object_name else "*.hdf5"
    return sorted(d.glob(pattern))


class EihEpisodeDataset(Dataset):
    """One training sample = one (obs_horizon, action_horizon) window from
    one episode. Images are returned as float32 CHW in [0, 1]; actions are
    normalized to [-1, 1] via `stats`.
    """

    def __init__(
        self,
        episode_paths: list[Path],
        stats: Stats,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        pad_before: bool = True,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required to use EihEpisodeDataset (pip install torch)")
        self.stats = stats
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pad_before = pad_before
        self._episodes: list[dict] = []
        self._index: list[tuple[int, int]] = []  # (episode_idx, start_t)

        for ep_idx, p in enumerate(episode_paths):
            with h5py.File(p, "r") as f:
                frames = np.asarray(f["cam_eih"], dtype=np.uint8)  # (T,H,W,3)
                poses = np.asarray(f["pose"], dtype=np.float32)  # (T,4)
            T = frames.shape[0]
            self._episodes.append({"frames": frames, "poses": poses, "T": T})
            start_min = 0 if pad_before else obs_horizon - 1
            for t in range(start_min, T):
                self._index.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self._index)

    def _get_obs_window(self, ep: dict, t: int) -> np.ndarray:
        """Returns obs_horizon frames ending at t (inclusive), padding with
        the first available frame if t - obs_horizon + 1 < 0."""
        T = ep["T"]
        idxs = [max(0, t - k) for k in reversed(range(self.obs_horizon))]
        frames = ep["frames"][idxs]  # (obs_horizon, H, W, 3)
        return frames

    def _get_action_window(self, ep: dict, t: int) -> np.ndarray:
        """Returns action_horizon actions starting at t, padding with the
        last available action if the episode ends early."""
        T = ep["T"]
        idxs = [min(T - 1, t + k) for k in range(self.action_horizon)]
        return ep["poses"][idxs]  # (action_horizon, 4)

    def __getitem__(self, i: int):
        ep_idx, t = self._index[i]
        ep = self._episodes[ep_idx]
        obs_frames = self._get_obs_window(ep, t)  # (obs_horizon, H, W, 3) uint8
        actions = self._get_action_window(ep, t)  # (action_horizon, 4)

        obs_frames = obs_frames.astype(np.float32) / 255.0
        obs_frames = np.transpose(obs_frames, (0, 3, 1, 2))  # (obs_horizon, 3, H, W)
        actions_norm = self.stats.normalize(actions)

        return {
            "obs": torch.from_numpy(obs_frames),  # (obs_horizon, 3, H, W)
            "action": torch.from_numpy(actions_norm.astype(np.float32)),  # (action_horizon, 4)
        }
