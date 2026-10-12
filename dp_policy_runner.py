"""Inference wrapper for the trained eye-in-hand descend+grab Diffusion
Policy. `DiffusionPolicyRunner.run(arm, cap)` plays the same ROLE as
pick_place.track_and_descend(arm, cap, target, floor_z, args) -- take the
arm from the hand-off pose to a completed grab -- but as a bound method on
a loaded model instead of a free function (it needs no `target`/`floor_z`,
the policy only looks at pixels). pick_place_fixed_to_eih.py's --policy dp
flag switches the stage-2 call between the two.

Receding-horizon execution (standard for Diffusion Policy): each "look",
the model predicts an action_horizon-length sequence of (x,y,z,t) poses
from the current obs_horizon frames, but only the first --exec-horizon of
them are actually sent to the robot before looking again -- this bounds
how far the plan can drift from what the camera is currently seeing.

Stops when either:
  - the predicted/executed gripper command crosses --grip-close-threshold
    (the policy decided to grab), or
  - --max-steps is hit without that happening (treated as failure)

Run (imported, not usually run standalone):
  from dp_policy_runner import DiffusionPolicyRunner
  runner = DiffusionPolicyRunner("dp_checkpoints/dp_latest.pt", "dp_checkpoints/stats.json")
  ok = runner.run(arm, eih_cap)
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np
import torch

from dp_dataset import Stats
from dp_model import DiffusionPolicy
from roarm_driver import Pose, RoArm


class DiffusionPolicyRunner:
    def __init__(
        self,
        checkpoint_path: str,
        stats_path: str,
        img_w: int = 224,
        img_h: int = 224,
        exec_horizon: int = 4,
        num_inference_steps: int = 16,
        grip_close_threshold: float = 2.5,
        max_steps: int = 60,
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
        self.model = DiffusionPolicy(
            action_horizon=ckpt["action_horizon"],
            obs_horizon=ckpt["obs_horizon"],
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
    def _predict_actions(self, obs_frames: deque[np.ndarray]) -> np.ndarray:
        """obs_frames: deque of the last obs_horizon BGR frames (oldest
        first). Returns the unnormalized (action_horizon, 4) pose sequence."""
        frames = list(obs_frames)
        while len(frames) < self.obs_horizon:
            frames.insert(0, frames[0])  # pad with the earliest frame
        tensors = torch.stack([self._frame_to_tensor(f) for f in frames], dim=0)  # (obs_horizon, 3, H, W)
        obs = tensors.unsqueeze(0).to(self.device)  # (1, obs_horizon, 3, H, W)
        actions_norm = self.model.sample(obs, num_inference_steps=self.num_inference_steps)  # (1, action_horizon, 4)
        actions_norm = actions_norm[0].cpu().numpy()
        return self.stats.unnormalize(actions_norm)  # (action_horizon, 4)

    def run(self, arm: RoArm, cap, show_window: str | None = "RoArm Eye-in-Hand") -> bool:
        """Runs the receding-horizon control loop until the policy closes
        the gripper or --max-steps is hit. Returns True on a (predicted)
        successful grab, False on timeout. Caller is expected to have
        already positioned the arm at the hand-off pose (e.g. the
        look-down pose in pick_place_fixed_to_eih.py) before calling this."""
        obs_frames: deque[np.ndarray] = deque(maxlen=self.obs_horizon)
        step = 0

        while step < self.max_steps:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            obs_frames.append(frame)

            actions = self._predict_actions(obs_frames)  # (action_horizon, 4)
            for a in actions[: self.exec_horizon]:
                x, y, z, t = (float(v) for v in a)
                arm.move_to(Pose(x=x, y=y, z=z, t=t), speed=self.move_speed)
                if show_window:
                    ok2, live = cap.read()
                    if ok2:
                        disp = live.copy()
                        cv2.putText(disp, f"[DP] step {step} x={x:.0f} y={y:.0f} z={z:.0f} t={t:.2f}",
                                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
                        cv2.imshow(show_window, disp)
                        cv2.waitKey(1)
                step += 1
                if t >= self.grip_close_threshold:
                    print(f"[DP] gripper close predicted at step {step} (t={t:.2f} >= {self.grip_close_threshold})")
                    arm.set_gripper(t)
                    return True
                if step >= self.max_steps:
                    break

        print(f"[DP] max_steps ({self.max_steps}) reached without a grab decision")
        return False
