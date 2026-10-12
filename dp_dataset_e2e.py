"""Dataset for the end-to-end (start-pose-to-grab) Diffusion Policy.

Same windowing scheme as dp_dataset.py (obs_horizon history -> action_horizon
prediction, actions = raw (x,y,z,t) pose commands, [-1,1] normalization via
Stats), but each observation now carries BOTH camera views -- fixed
overview + eye-in-hand -- since the policy has to do its own rough approach,
which needs the wide view the eye-in-hand camera alone can't provide.

Reuses dp_dataset.Stats as-is (it only operates on the 4-D action/pose
vectors, camera-agnostic).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dp_dataset import Stats, compute_stats  # noqa: F401 (re-exported for convenience)

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


def list_episodes(dataset_dir: str, object_name: str | None = None) -> list[Path]:
    d = Path(dataset_dir)
    pattern = f"{object_name}_ep*.hdf5" if object_name else "*.hdf5"
    return sorted(d.glob(pattern))


class E2eEpisodeDataset(Dataset):
    """One training sample = one (obs_horizon, action_horizon) window from
    one episode, with obs containing both camera streams stacked on a new
    leading "camera" axis: (obs_horizon, n_cameras=2, 3, H, W)."""

    def __init__(
        self,
        episode_paths: list[Path],
        stats: Stats,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        pad_before: bool = True,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required to use E2eEpisodeDataset (pip install torch)")
        self.stats = stats
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pad_before = pad_before
        self._episodes: list[dict] = []
        self._index: list[tuple[int, int]] = []

        for ep_idx, p in enumerate(episode_paths):
            with h5py.File(p, "r") as f:
                frames_fixed = np.asarray(f["cam_fixed"], dtype=np.uint8)  # (T,H,W,3)
                frames_eih = np.asarray(f["cam_eih"], dtype=np.uint8)  # (T,H,W,3)
                poses = np.asarray(f["pose"], dtype=np.float32)  # (T,4)
            T = frames_fixed.shape[0]
            self._episodes.append({"frames_fixed": frames_fixed, "frames_eih": frames_eih, "poses": poses, "T": T})
            start_min = 0 if pad_before else obs_horizon - 1
            for t in range(start_min, T):
                self._index.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self._index)

    def _get_obs_window(self, ep: dict, t: int) -> np.ndarray:
        idxs = [max(0, t - k) for k in reversed(range(self.obs_horizon))]
        fixed = ep["frames_fixed"][idxs]  # (obs_horizon, H, W, 3)
        eih = ep["frames_eih"][idxs]  # (obs_horizon, H, W, 3)
        stacked = np.stack([fixed, eih], axis=1)  # (obs_horizon, 2, H, W, 3)
        return stacked

    def _get_action_window(self, ep: dict, t: int) -> np.ndarray:
        T = ep["T"]
        idxs = [min(T - 1, t + k) for k in range(self.action_horizon)]
        return ep["poses"][idxs]

    def __getitem__(self, i: int):
        ep_idx, t = self._index[i]
        ep = self._episodes[ep_idx]
        obs_frames = self._get_obs_window(ep, t)  # (obs_horizon, 2, H, W, 3) uint8
        actions = self._get_action_window(ep, t)  # (action_horizon, 4)

        obs_frames = obs_frames.astype(np.float32) / 255.0
        obs_frames = np.transpose(obs_frames, (0, 1, 4, 2, 3))  # (obs_horizon, 2, 3, H, W)
        actions_norm = self.stats.normalize(actions)

        return {
            "obs": torch.from_numpy(obs_frames),  # (obs_horizon, 2, 3, H, W)
            "action": torch.from_numpy(actions_norm.astype(np.float32)),  # (action_horizon, 4)
        }
