"""Diffusion Policy model for the end-to-end (start-pose-to-grab) task.

Same architecture as dp_model.DiffusionPolicy (small CNN vision encoder +
FiLM-conditioned 1D temporal U-Net + DDPM), except the observation now has
TWO camera views per timestep -- fixed overview + eye-in-hand -- run through
one SHARED VisionEncoder (same weights for both cameras; the model has to
generalize to either viewpoint anyway, and sharing keeps the parameter
count/data requirement down) and concatenated per timestep before flattening
across obs_horizon, same as dp_model.py does for a single camera.

Action dim = 4 (x, y, z, t), identical to dp_model.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dp_model import ACTION_DIM, ConditionalResBlock1D, SinusoidalPosEmb, VisionEncoder


class DiffusionPolicyE2E(nn.Module):
    def __init__(
        self,
        action_horizon: int = 16,
        obs_horizon: int = 2,
        n_cameras: int = 2,
        action_dim: int = ACTION_DIM,
        vision_feat_dim: int = 128,
        hidden_channels: int = 64,
        n_res_blocks: int = 3,
        num_diffusion_steps: int = 100,
    ) -> None:
        super().__init__()
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.n_cameras = n_cameras
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps

        self.vision_encoder = VisionEncoder(out_dim=vision_feat_dim)  # shared across both cameras
        cond_dim = vision_feat_dim * n_cameras * obs_horizon + hidden_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels), nn.Mish(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.in_proj = nn.Conv1d(action_dim, hidden_channels, 1)
        self.blocks = nn.ModuleList([ConditionalResBlock1D(hidden_channels, cond_dim) for _ in range(n_res_blocks)])
        self.out_proj = nn.Conv1d(hidden_channels, action_dim, 1)

        betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_horizon, n_cameras, 3, H, W) -> (B, vision_feat_dim * n_cameras * obs_horizon)
        B, T, C = obs.shape[:3]
        flat = obs.reshape(B * T * C, *obs.shape[3:])
        feat = self.vision_encoder(flat)  # (B*T*C, vision_feat_dim)
        return feat.reshape(B, -1)

    def predict_noise(self, noisy_action: torch.Tensor, timestep: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep)
        cond = torch.cat([obs_cond, t_emb], dim=-1)
        x = self.in_proj(noisy_action.transpose(1, 2))
        for block in self.blocks:
            x = block(x, cond)
        return self.out_proj(x).transpose(1, 2)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B = action.shape[0]
        device = action.device
        obs_cond = self.encode_obs(obs)

        timestep = torch.randint(0, self.num_diffusion_steps, (B,), device=device).long()
        noise = torch.randn_like(action)
        sqrt_alpha_cumprod = self.alphas_cumprod[timestep].sqrt().view(B, 1, 1)
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[timestep]).sqrt().view(B, 1, 1)
        noisy_action = sqrt_alpha_cumprod * action + sqrt_one_minus_alpha_cumprod * noise

        pred_noise = self.predict_noise(noisy_action, timestep, obs_cond)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, num_inference_steps: int | None = None) -> torch.Tensor:
        """Deterministic DDIM-style sampling (eta=0), see dp_model.py."""
        device = obs.device
        B = obs.shape[0]
        obs_cond = self.encode_obs(obs)

        steps = num_inference_steps or self.num_diffusion_steps
        step_indices = torch.linspace(self.num_diffusion_steps - 1, 0, steps, device=device).long()

        x = torch.randn(B, self.action_horizon, self.action_dim, device=device)
        for i, t in enumerate(step_indices):
            t_batch = t.expand(B)
            pred_noise = self.predict_noise(x, t_batch, obs_cond)
            alpha_cumprod_t = self.alphas_cumprod[t]

            x0_pred = (x - (1 - alpha_cumprod_t).sqrt() * pred_noise) / alpha_cumprod_t.sqrt().clamp(min=1e-6)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if i < len(step_indices) - 1:
                alpha_cumprod_prev = self.alphas_cumprod[step_indices[i + 1]]
                dir_xt = (1 - alpha_cumprod_prev).sqrt() * pred_noise
                x = alpha_cumprod_prev.sqrt() * x0_pred + dir_xt
            else:
                x = x0_pred
        return x
