"""
Train a reward predictor for GPC-RANK.

The reward predictor scores a single image: lower = closer to task completion.
At inference time, GPC-RANK uses it to rank world-model-predicted future frames
and selects the action candidate with the lowest (best) score.

Architecture: ResNet18 + MLP -> scalar  (same as GPC paper)

Training labels (time-to-go cost):
  - Successful episodes: label = (T - t) / T   (1.0 at start -> 0.0 at end)
  - Failed episodes:     label = 1.0 for all frames (never reaches goal)

This gives a smooth regression target where argmin(score) picks the frame
that looks most like task completion.

Input: LeRobot format dataset. Supported robot_type values select the
camera video key (see CAMERA_KEY below): SimplerEnv google/widowx
(from collect_finetune_dreamdojo_data.py), or agilex real-robot episodes.

Usage (8-GPU DDP, SimplerEnv Google):
    torchrun --nproc_per_node=8 train_reward_predictor.py \
        --dataset_dir data/simpler_env_dreamdojo \
        --robot_type google \
        --save_dir checkpoints/reward_predictor_google \
        --pretrained_backbone \
        --num_epochs 100

Usage (Agilex real-robot):
    torchrun --nproc_per_node=8 train_reward_predictor.py \
        --dataset_dir ../openpi/data/pnp_cup_0415 \
        --robot_type agilex \
        --save_dir checkpoints/reward_predictor_agilex_pnp_cup \
        --pretrained_backbone \
        --num_epochs 100
"""

import warnings
warnings.filterwarnings("ignore", message=".*video decoding and encoding.*deprecated.*")

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.models as models
import torchvision.transforms.v2 as v2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Camera video key per robot type
CAMERA_KEY = {
    "google": "observation.images.image",
    "widowx": "observation.images.image_0",
    "agilex": "observation.images.cam_high",
}


def setup_distributed():
    """Initialize DDP if launched via torchrun, otherwise single-GPU."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return device, rank, local_rank, world_size
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


class RewardPredictorDataset(Dataset):
    """
    Loads (image, cost_label) pairs from a LeRobot-format SimplerEnv dataset.

    Label scheme (time-to-go cost):
      - Successful episode of length T: frame t gets label (T - t) / T
      - Failed episode: all frames get label 1.0

    Args:
        treat_all_success: If True, treat ALL episodes as successful regardless
            of the `success` column.  Useful for human demonstration datasets
            (e.g. fractal20220817) where the success label is missing or wrong.
    """

    def __init__(self, dataset_dir: str, robot_type: str, img_size: int = 96,
                 treat_all_success: bool = False):
        self.img_size = img_size
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(img_size),
            v2.ToDtype(torch.float32, scale=True),  # -> [0, 1]
        ])

        camera_key = CAMERA_KEY[robot_type]
        dataset_path = Path(dataset_dir)

        # Discover all episodes
        self.samples = []  # list of (video_path, frame_idx, label)

        data_root = dataset_path / "data"
        video_root = dataset_path / "videos"

        n_success, n_fail = 0, 0
        for chunk_dir in sorted(data_root.glob("chunk-*")):
            chunk_name = chunk_dir.name
            for pq_path in sorted(chunk_dir.glob("episode_*.parquet")):
                ep_name = pq_path.stem
                vid_path = video_root / chunk_name / camera_key / f"{ep_name}.mp4"

                if not vid_path.exists():
                    continue

                df = pd.read_parquet(pq_path)
                n = len(df)
                if n == 0:
                    continue

                # Determine success
                if treat_all_success:
                    success = True
                elif "success" in df.columns:
                    success = bool(df["success"].iloc[-1])
                else:
                    success = True

                if success:
                    n_success += 1
                else:
                    n_fail += 1

                # Compute per-frame cost labels
                for t in range(n):
                    if success:
                        label = (n - 1 - t) / max(n - 1, 1)  # 1.0 -> 0.0
                    else:
                        label = 1.0
                    self.samples.append((str(vid_path), t, label))

        print(f"RewardPredictorDataset: {len(self.samples)} samples from {dataset_dir}")
        print(f"  Episodes: {n_success} success, {n_fail} fail "
              f"(treat_all_success={treat_all_success})")

        # Pre-group by video for efficient loading
        self._video_cache_path = None
        self._video_cache_frames = None

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        """Load a single frame from video, with simple caching."""
        if self._video_cache_path != video_path:
            import torchvision.io
            video, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
            self._video_cache_path = video_path
            self._video_cache_frames = video  # (T, H, W, 3) uint8 tensor

        idx = min(frame_idx, len(self._video_cache_frames) - 1)
        return self._video_cache_frames[idx].numpy()

    def __getitem__(self, idx):
        video_path, frame_idx, label = self.samples[idx]
        frame = self._load_frame(video_path, frame_idx)  # (H, W, 3) uint8
        img = self.transform(frame)  # (3, img_size, img_size) float [0, 1]
        return img, torch.tensor(label, dtype=torch.float32)


def _decode_episode_for_reward(args_tuple):
    """Worker: decode one episode's video, batch-resize, compute labels.

    Returns (frames, labels, success) or None.
      frames: (T, 3, img_size, img_size) float32 [0, 1]
      labels: (T,) float32
    """
    pq_path, vid_path, treat_all_success, img_size = args_tuple

    import torchvision.io  # import in worker to avoid fork issues

    if not treat_all_success:
        df = pd.read_parquet(pq_path)
        n_pq = len(df)
        if n_pq == 0:
            return None
        success = bool(df["success"].iloc[-1]) if "success" in df.columns else True
    else:
        success = True
        n_pq = None

    video, _, _ = torchvision.io.read_video(vid_path, pts_unit="sec")
    n_vid = len(video)
    if n_vid == 0:
        return None

    min_n = min(n_pq, n_vid) if n_pq is not None else n_vid

    # Batch resize all frames at once (much faster than per-frame transforms)
    frames = video[:min_n].permute(0, 3, 1, 2).float()
    frames = F.interpolate(frames, size=(img_size, img_size),
                           mode='bilinear', align_corners=False)
    frames.div_(255.0)

    if success:
        t = torch.arange(min_n, dtype=torch.float32)
        labels = (min_n - 1 - t) / max(min_n - 1, 1)
    else:
        labels = torch.ones(min_n, dtype=torch.float32)

    # Return numpy arrays to avoid PyTorch shared-memory (mmap) IPC which
    # exhausts /dev/shm when many workers return large tensors in parallel.
    return (frames.numpy(), labels.numpy(), success)


#  Faster dataset: pre-extract all frames to memory
class RewardPredictorDatasetPreloaded(Dataset):
    """
    Same as RewardPredictorDataset but loads ALL frames into RAM at init.
    Much faster training at the cost of memory.

    Uses multiprocessing to decode videos in parallel for ~Nx speedup.
    """

    def __init__(self, dataset_dir: str, robot_type: str, img_size: int = 96,
                 treat_all_success: bool = False, num_load_workers: int = 0):
        self.img_size = img_size

        camera_key = CAMERA_KEY[robot_type]
        dataset_path = Path(dataset_dir)

        data_root = dataset_path / "data"
        video_root = dataset_path / "videos"

        episodes = []
        for chunk_dir in sorted(data_root.glob("chunk-*")):
            chunk_name = chunk_dir.name
            for pq_path in sorted(chunk_dir.glob("episode_*.parquet")):
                ep_name = pq_path.stem
                vid_path = video_root / chunk_name / camera_key / f"{ep_name}.mp4"
                if vid_path.exists():
                    episodes.append((str(pq_path), str(vid_path)))

        work_args = [(pq, vid, treat_all_success, img_size)
                     for pq, vid in episodes]

        all_frames = []  # list of (T_i, 3, H, W) tensors
        all_labels = []  # list of (T_i,) tensors
        n_success, n_fail = 0, 0

        if num_load_workers > 0:
            print(f"Loading {len(episodes)} episodes with "
                  f"{num_load_workers} parallel workers...")
            with mp.Pool(num_load_workers) as pool:
                for result in tqdm(
                    pool.imap(_decode_episode_for_reward, work_args,
                              chunksize=16),
                    total=len(work_args), desc="Loading episodes",
                ):
                    if result is None:
                        continue
                    frames, labels, success = result
                    all_frames.append(torch.from_numpy(frames))
                    all_labels.append(torch.from_numpy(labels))
                    if success:
                        n_success += 1
                    else:
                        n_fail += 1
        else:
            # Sequential fallback (original behaviour, but with batch resize)
            print(f"Loading {len(episodes)} episodes sequentially...")
            for args in tqdm(work_args, desc="Loading episodes"):
                result = _decode_episode_for_reward(args)
                if result is None:
                    continue
                frames, labels, success = result
                all_frames.append(torch.from_numpy(frames))
                all_labels.append(torch.from_numpy(labels))
                if success:
                    n_success += 1
                else:
                    n_fail += 1

        # Concatenate into flat tensors for fast indexing
        self.images = torch.cat(all_frames, dim=0)   # (N, 3, H, W)
        self.labels = torch.cat(all_labels, dim=0)    # (N,)

        print(f"Loaded {len(self.images)} frames "
              f"({n_success} success + {n_fail} fail episodes)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


#  Model (same architecture as GPC paper)
class RewardPredictor(nn.Module):
    """Scalar task-completion cost from a single image. Lower = closer to goal."""

    def __init__(self, output_dim: int = 1, pretrained_backbone: bool = False):
        super().__init__()
        self.resnet18 = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        )
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        """x: (B, 3, H, W) float [0, 1] -> (B, output_dim)"""
        return self.mlp(self.resnet18(x))


def train(args):
    device, rank, local_rank, world_size = setup_distributed()
    is_main = (rank == 0)

    if is_main:
        print(f"Device: {device}, World size: {world_size}")
    os.makedirs(args.save_dir, exist_ok=True)

    # wandb (only on rank 0)
    use_wandb = args.wandb and WANDB_AVAILABLE and is_main
    if args.wandb and not WANDB_AVAILABLE and is_main:
        print("WARNING: --wandb flag set but wandb not installed. Skipping.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # Dataset
    if args.preload:
        dataset = RewardPredictorDatasetPreloaded(
            args.dataset_dir, args.robot_type, args.img_size,
            treat_all_success=args.treat_all_success,
            num_load_workers=args.num_load_workers)
    else:
        dataset = RewardPredictorDataset(
            args.dataset_dir, args.robot_type, args.img_size,
            treat_all_success=args.treat_all_success)

    # Train/val split
    n = len(dataset)
    n_val = max(1, int(n * 0.1))
    n_train = n - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # DDP samplers
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    if is_main:
        print(f"Train: {n_train} samples, Val: {n_val} samples")
        print(f"Per-GPU batch size: {args.batch_size}, "
              f"Effective batch size: {args.batch_size * world_size}")

    model = RewardPredictor(
        output_dim=1,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    if is_main:
        print(f"Pretrained backbone: {args.pretrained_backbone}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs)

    best_val_loss = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_losses = []
        for imgs, labels in train_loader:
            imgs = imgs.to(device)       # (B, 3, H, W)
            labels = labels.to(device)   # (B,)

            preds = model(imgs).squeeze(-1)  # (B,)
            loss = F.mse_loss(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds = model(imgs).squeeze(-1)
                val_losses.append(F.mse_loss(preds, labels).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        # Reduce val_loss across ranks for accurate reporting
        if world_size > 1:
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()

        if is_main:
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "best_val_loss": best_val_loss,
                }, step=epoch)

            if epoch % args.log_every == 0 or epoch == 1:
                print(f"Epoch {epoch:4d}/{args.num_epochs}  "
                      f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
                      f"lr={scheduler.get_last_lr()[0]:.1e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state = model.module.state_dict() if world_size > 1 else model.state_dict()
                path = os.path.join(args.save_dir, "reward_predictor_best.pth")
                torch.save(state, path)

            if epoch % args.save_every == 0 or epoch == args.num_epochs:
                state = model.module.state_dict() if world_size > 1 else model.state_dict()
                path = os.path.join(args.save_dir, f"reward_predictor_epoch_{epoch}.pth")
                torch.save(state, path)

    if is_main:
        print(f"\nTraining complete. Best val_loss: {best_val_loss:.5f}")
        print(f"Checkpoints saved to {args.save_dir}/")

    if use_wandb:
        wandb.finish()

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(
        description="Train GPC-RANK reward predictor for SimplerEnv")

    # Data
    parser.add_argument("--dataset_dir", required=True,
                        help="LeRobot dataset dir (from collect_finetune_dreamdojo_data.py "
                             "or fractal20220817_data_dreamdojo / lerobot)")
    parser.add_argument("--robot_type", required=True, choices=["google", "widowx", "agilex"])
    parser.add_argument("--img_size", type=int, default=96,
                        help="Image resolution (must match world model, default: 96)")
    parser.add_argument("--preload", action="store_true",
                        help="Load all frames into RAM (faster training, more memory)")
    parser.add_argument("--treat_all_success", action="store_true",
                        help="Treat ALL episodes as successful (for human demo datasets "
                             "like fractal20220817 where success labels are missing/wrong)")
    parser.add_argument("--num_load_workers", type=int, default=0,
                        help="Number of parallel workers for video decoding during "
                             "preload (0 = sequential). Recommended: 8-16.")

    # Training
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_backbone", action="store_true",
                        help="Use ImageNet-pretrained ResNet18 backbone")

    # Output
    parser.add_argument("--save_dir", required=True,
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=5)

    # wandb
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="gpc-reward-predictor",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (auto-generated if not set)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
