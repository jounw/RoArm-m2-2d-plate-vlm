"""Inference wrapper for the trained end-to-end (start-pose-to-grab)
Diffusion Policy. Same receding-horizon idea as dp_policy_runner.py, but
reads BOTH cameras each look and feeds a stacked (obs_horizon, 2, 3, H, W)
observation to DiffusionPolicyE2E.

Run (imported, not usually run standalone):
  from dp_policy_runner_e2e import DiffusionPolicyE2ERunner
  runner = DiffusionPolicyE2ERunner("dp_e2e_checkpoints/dp_e2e_latest.pt", "dp_e2e_checkpoints/stats.json")
  ok = runner.run(arm, fixed_cap, eih_cap)
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np
import torch

from dp_dataset import Stats
from dp_model_e2e import DiffusionPolicyE2E
from roarm_driver import Pose, RoArm


class DiffusionPolicyE2ERunner:
    def __init__(
        self,
        checkpoint_path: str,
        stats_path: str,
        img_w: int = 224,
        img_h: int = 224,
        exec_horizon: int = 6,
        num_inference_steps: int = 16,
        grip_close_threshold: float = 2.5,
        max_steps: int = 150,
        move_speed: float = 0.3,
        device: str | None = None,
    ) -> None:
        self.img_w = img_w
        self.img_h = img_h
        self.exec_horizon = exec_horizon
        self.num_inference_steps = num_inference_steps
        self.grip_close_threshold = grip_close_threshold
        self.max_steps = max_steps
        self.move_speed = move_speed

        self.device = torch.device(device) if device else self._auto_device()
        self.stats = Stats.load(stats_path)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.obs_horizon = ckpt["obs_horizon"]
        self.action_horizon = ckpt["action_horizon"]
        self.n_cameras = ckpt.get("n_cameras", 2)
        self.model = DiffusionPolicyE2E(
            action_horizon=ckpt["action_horizon"],
            obs_horizon=ckpt["obs_horizon"],
            n_cameras=self.n_cameras,
            num_diffusion_steps=ckpt["num_diffusion_steps"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @staticmethod
    def _auto_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _frame_to_tensor(self, frame_bgr: np.ndarray) -> torch.Tensor:
        if frame_bgr.shape[1] != self.img_w or frame_bgr.shape[0] != self.img_h:
            frame_bgr = cv2.resize(frame_bgr, (self.img_w, self.img_h))
        x = frame_bgr.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # (3, H, W)
        return torch.from_numpy(x)

    @torch.no_grad()
    def _predict_actions(self, obs_pairs: deque[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """obs_pairs: deque of (frame_fixed, frame_eih) BGR pairs, oldest
        first. Returns the unnormalized (action_horizon, 4) pose sequence."""
        pairs = list(obs_pairs)
        while len(pairs) < self.obs_horizon:
            pairs.insert(0, pairs[0])
        # (obs_horizon, 2, 3, H, W)
        stacked = torch.stack(
            [torch.stack([self._frame_to_tensor(fixed), self._frame_to_tensor(eih)], dim=0) for fixed, eih in pairs],
            dim=0,
        )
        obs = stacked.unsqueeze(0).to(self.device)  # (1, obs_horizon, 2, 3, H, W)
        actions_norm = self.model.sample(obs, num_inference_steps=self.num_inference_steps)
        actions_norm = actions_norm[0].cpu().numpy()
        return self.stats.unnormalize(actions_norm)

    def run(self, arm: RoArm, fixed_cap, eih_cap, show_window: str | None = None) -> bool:
        """Runs the receding-horizon control loop, reading both cameras
        each look, until the policy closes the gripper or --max-steps is
        hit. Returns True on a (predicted) successful grab, False on
        timeout. This covers the FULL trajectory -- caller should just have
        the arm at the start pose with the gripper open before calling."""
        obs_pairs: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.obs_horizon)
        step = 0

        while step < self.max_steps:
            ok1, frame_fixed = fixed_cap.read()
            ok2, frame_eih = eih_cap.read()
            if not (ok1 and ok2):
                time.sleep(0.05)
                continue
            obs_pairs.append((frame_fixed, frame_eih))

            actions = self._predict_actions(obs_pairs)  # (action_horizon, 4)
            for a in actions[: self.exec_horizon]:
                x, y, z, t = (float(v) for v in a)
                arm.move_to(Pose(x=x, y=y, z=z, t=t), speed=self.move_speed)
                if show_window:
                    ok3, live = eih_cap.read()
                    if ok3:
                        disp = live.copy()
                        cv2.putText(disp, f"[DP-E2E] step {step} x={x:.0f} y={y:.0f} z={z:.0f} t={t:.2f}",
                                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
                        cv2.imshow(show_window, disp)
                        cv2.waitKey(1)
                step += 1
                if t >= self.grip_close_threshold:
                    print(f"[DP-E2E] gripper close predicted at step {step} (t={t:.2f} >= {self.grip_close_threshold})")
                    arm.set_gripper(t)
                    return True
                if step >= self.max_steps:
                    break

        print(f"[DP-E2E] max_steps ({self.max_steps}) reached without a grab decision")
        return False
