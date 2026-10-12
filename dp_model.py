"""Diffusion Policy model for the eye-in-hand descend+grab stage.

A compact version of Chi et al. 2023's "Diffusion Policy": a small CNN
vision encoder turns each observed frame into a feature vector, the
obs_horizon of features is flattened into one conditioning vector, and a
1D temporal U-Net predicts the noise added to an `action_horizon`-length
action sequence at a given diffusion timestep (DDPM). Sampling denoises
from pure noise back to a predicted action sequence conditioned on the
current observation.

Deliberately self-contained (no `diffusers`/`robomimic` dependency) so it
only needs plain PyTorch. Sizes are kept small (this stage's action space
is 4-D: x, y, z, gripper) since the training set will realistically be a
few dozen to a couple hundred short demonstrations, not thousands.

Action dim = 4 (x, y, z, t) -- see dp_dataset.py / dp_collect_data.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_DIM = 4


class VisionEncoder(nn.Module):
    """Small CNN -> global-average-pool -> linear. Not a pretrained backbone
    on purpose (keeps this runnable/trainable on a laptop CPU/MPS for a
    small dataset); swap in a frozen ResNet18 later if more data exists."""

    def __init__(self, out_dim: int = 128, in_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        f = self.net(x)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)  # (B, 128)
        return self.proj(f)  # (B, out_dim)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalResBlock1D(nn.Module):
    """1D conv residual block conditioned on a global feature vector
    (diffusion timestep embedding + observation embedding), via FiLM."""

    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.cond_proj = nn.Linear(cond_dim, channels * 2)  # FiLM scale+shift

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L), cond: (B, cond_dim)
        h = F.mish(self.norm1(self.conv1(x)))
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.mish(self.norm2(self.conv2(h)))
        return x + h


class DiffusionPolicy(nn.Module):
    """Predicts noise added to an action_horizon-length action sequence,
    conditioned on obs_horizon observation frames and the diffusion
    timestep. Call `sample()` for inference (DDPM ancestral sampling)."""

    def __init__(
        self,
        action_horizon: int = 8,
        obs_horizon: int = 2,
        action_dim: int = ACTION_DIM,
        vision_feat_dim: int = 128,
        hidden_channels: int = 64,
        n_res_blocks: int = 3,
        num_diffusion_steps: int = 100,
    ) -> None:
        super().__init__()
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps

        self.vision_encoder = VisionEncoder(out_dim=vision_feat_dim)
        cond_dim = vision_feat_dim * obs_horizon + hidden_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels), nn.Mish(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.in_proj = nn.Conv1d(action_dim, hidden_channels, 1)
        self.blocks = nn.ModuleList([ConditionalResBlock1D(hidden_channels, cond_dim) for _ in range(n_res_blocks)])
        self.out_proj = nn.Conv1d(hidden_channels, action_dim, 1)

        # DDPM noise schedule (linear beta), precomputed as buffers.
        betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_horizon, 3, H, W) -> (B, vision_feat_dim * obs_horizon)
        B, T = obs.shape[:2]
        flat = obs.reshape(B * T, *obs.shape[2:])
        feat = self.vision_encoder(flat)  # (B*T, vision_feat_dim)
        return feat.reshape(B, -1)

    def predict_noise(self, noisy_action: torch.Tensor, timestep: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        # noisy_action: (B, action_horizon, action_dim)
        t_emb = self.time_mlp(timestep)  # (B, hidden_channels)
        cond = torch.cat([obs_cond, t_emb], dim=-1)  # (B, cond_dim)
        x = self.in_proj(noisy_action.transpose(1, 2))  # (B, hidden, action_horizon)
        for block in self.blocks:
            x = block(x, cond)
        return self.out_proj(x).transpose(1, 2)  # (B, action_horizon, action_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Training step: sample a random diffusion timestep per batch item,
        add noise to the ground-truth action, predict it back, return the
        MSE loss (standard DDPM epsilon-prediction objective)."""
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
        """Inference: denoise from Gaussian noise to an action sequence,
        conditioned on `obs` (B, obs_horizon, 3, H, W). Returns actions in
        the same normalized [-1, 1] space training used -- unnormalize with
        the Stats saved by dp_train.py before sending to the robot.

        Deterministic DDIM-style sampling (eta=0, no injected noise) over
        the epsilon predictor trained with the DDPM objective in forward():
        fewer steps than num_diffusion_steps run just as well and it's
        reproducible run to run, both useful for a real-time control loop."""
        device = obs.device
        B = obs.shape[0]
        obs_cond = self.encode_obs(obs)

        steps = num_inference_steps or self.num_diffusion_steps
        step_indices = torch.linspace(self.num_diffusion_steps - 1, 0, steps, device=device).long()

        x = torch.randn(B, self.action_horizon, self.action_dim, device=device)
        for i, t in enumerate(step_indices):
            t_batch = t.expand(B)
            pred_noise = self.predict_noise(x, t_batch, obs_cond)
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            x0_pred = (x - (1 - alpha_cumprod_t).sqrt() * pred_noise) / alpha_cumprod_t.sqrt().clamp(min=1e-6)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if i < len(step_indices) - 1:
                alpha_cumprod_prev = self.alphas_cumprod[step_indices[i + 1]]
                dir_xt = (1 - alpha_cumprod_prev).sqrt() * pred_noise
                x = alpha_cumprod_prev.sqrt() * x0_pred + dir_xt
            else:
                x = x0_pred
        return x
