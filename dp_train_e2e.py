"""Trains the end-to-end (start-pose-to-grab) DiffusionPolicyE2E on episodes
collected by dp_collect_data_e2e.py. Mirrors dp_train.py but for the
dual-camera dataset/model.

You need real demonstration data first -- these are FULL trajectories
(rough approach through grab), so expect them to need more demos than the
short descend-only clips dp_train.py trains on to reach similar
reliability; the policy now also has to learn the coarse search/approach
behavior that used to be handled by the fixed camera's homography.

Run:
  python3 dp_train_e2e.py --dataset-dir dp_e2e_dataset --object red_block --epochs 300 --out-dir dp_e2e_checkpoints
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dp_dataset import Stats, compute_stats
from dp_dataset_e2e import E2eEpisodeDataset, list_episodes
from dp_model_e2e import ACTION_DIM, DiffusionPolicyE2E


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the end-to-end (start-pose-to-grab) Diffusion Policy")
    p.add_argument("--dataset-dir", default="dp_e2e_dataset")
    p.add_argument("--object", default=None)
    p.add_argument("--out-dir", default="dp_e2e_checkpoints")
    p.add_argument("--obs-horizon", type=int, default=2)
    p.add_argument("--action-horizon", type=int, default=16, help="longer than dp_train.py's default: full trajectories are longer than a short descend clip")
    p.add_argument("--batch-size", type=int, default=8, help="smaller default than dp_train.py: two camera streams per sample doubles memory use")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-diffusion-steps", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=20)
    return p


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(out_dir: Path, model: DiffusionPolicyE2E, stats: Stats, args: argparse.Namespace, epoch: int, val_loss: float | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"dp_e2e_epoch{epoch:04d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "action_horizon": args.action_horizon,
            "obs_horizon": args.obs_horizon,
            "n_cameras": 2,
            "action_dim": ACTION_DIM,
            "num_diffusion_steps": args.num_diffusion_steps,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        ckpt_path,
    )
    stats.save(str(out_dir / "stats.json"))
    latest_path = out_dir / "dp_e2e_latest.pt"
    torch.save(torch.load(ckpt_path, map_location="cpu"), latest_path)
    return ckpt_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    episode_paths = list_episodes(args.dataset_dir, args.object)
    if not episode_paths:
        print(f"no episodes found in {args.dataset_dir} (object={args.object!r}) -- run dp_collect_data_e2e.py first")
        return 1
    print(f"found {len(episode_paths)} episodes")

    stats = compute_stats(episode_paths)
    print(f"action stats: min={stats.action_min.tolist()} max={stats.action_max.tolist()}")

    full_dataset = E2eEpisodeDataset(episode_paths, stats, obs_horizon=args.obs_horizon, action_horizon=args.action_horizon)
    n_val = max(1, int(len(full_dataset) * args.val_fraction)) if len(full_dataset) > 10 else 0
    n_train = len(full_dataset) - n_val
    if n_val > 0:
        train_set, val_set = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_set, val_set = full_dataset, None
    print(f"train samples: {n_train}, val samples: {n_val}")

    device = pick_device(args.device)
    print(f"device: {device}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=len(train_set) > args.batch_size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False) if val_set is not None else None

    model = DiffusionPolicyE2E(
        action_horizon=args.action_horizon,
        obs_horizon=args.obs_horizon,
        n_cameras=2,
        num_diffusion_steps=args.num_diffusion_steps,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_losses = []
        for batch in train_loader:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            loss = model(obs, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / max(1, len(train_losses))

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    obs = batch["obs"].to(device)
                    action = batch["action"].to(device)
                    val_losses.append(model(obs, action).item())
            val_loss = sum(val_losses) / max(1, len(val_losses))

        dt = time.time() - t0
        msg = f"epoch {epoch:04d}/{args.epochs} train_loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f" val_loss={val_loss:.4f}"
        msg += f" ({dt:.1f}s)"
        print(msg)

        should_save = (epoch % args.save_every == 0) or (epoch == args.epochs)
        is_best = val_loss is not None and val_loss < best_val
        if is_best:
            best_val = val_loss
        if should_save or is_best:
            ckpt_path = save_checkpoint(out_dir, model, stats, args, epoch, val_loss)
            tag = " (best)" if is_best else ""
            print(f"  saved: {ckpt_path.name}{tag}")

    print(f"\ndone. latest checkpoint + stats.json in {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
