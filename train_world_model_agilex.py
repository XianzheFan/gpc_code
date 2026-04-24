"""
Train GPC world model on Agilex dataset (action_dim=14).

Two-phase training following the GPC paper:
  Phase 1: Single-step warmup (pred_horizon=5, short sequences)
  Phase 2: Multi-step finetuning (pred_horizon=16, longer sequences)

Input: zarr dataset created by convert_agilex_to_zarr.py

Usage:
  # Phase 1
  python train_world_model_agilex.py \
      --config configs/train_agilex_phase_one.yml

  # Phase 2 (requires Phase 1 checkpoint)
  python train_world_model_agilex.py \
      --config configs/train_agilex_phase_two.yml
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from functools import partial as functools_partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
import zarr
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
from tqdm.auto import tqdm


###############################################################################
#  Data Loading
###############################################################################

def create_sample_indices(episode_ends, sequence_length, pad_before=0, pad_after=0):
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        for idx in range(-pad_before, episode_length - sequence_length + pad_after + 1):
            buf_start = max(idx, 0) + start_idx
            buf_end = min(idx + sequence_length, episode_length) + start_idx
            s_start = buf_start - (idx + start_idx)
            s_end = sequence_length - ((idx + sequence_length + start_idx) - buf_end)
            indices.append([buf_start, buf_end, s_start, s_end])
    return np.array(indices)


def sample_sequence(train_data, sequence_length, buf_start, buf_end, s_start, s_end):
    result = {}
    for key, arr in train_data.items():
        sample = arr[buf_start:buf_end]
        if s_start > 0 or s_end < sequence_length:
            data = np.zeros((sequence_length,) + arr.shape[1:], dtype=arr.dtype)
            if s_start > 0:
                data[:s_start] = sample[0]
            if s_end < sequence_length:
                data[s_end:] = sample[-1]
            data[s_start:s_end] = sample
        else:
            data = sample
        result[key] = data
    return result


def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
    return ndata * 2 - 1


class AgilexWorldModelDataset(torch.utils.data.Dataset):
    """
    Dataset for Agilex world model training.
    Reads from zarr (output of convert_agilex_to_zarr.py).

    Returns batches with keys:
      'image':  (pred_horizon, 3, 96, 96) float32 in [0, 1]
      'action': (pred_horizon, 14)        float32 in [-1, 1]
    """

    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon,
                 resize_scale=96, stats=None):
        root = zarr.open(dataset_path, mode='r')
        episode_ends = root['meta']['episode_ends'][:]
        num_frames = episode_ends[-1]

        self.train_images = root['data']['img'][:num_frames]      # (N, H, W, 3) uint8
        train_actions = root['data']['action'][:num_frames]        # (N, 14)

        # Compute or use provided action stats
        if stats is None:
            self.stats = {
                'action': {
                    'min': train_actions.min(axis=0),
                    'max': train_actions.max(axis=0),
                }
            }
        else:
            self.stats = stats

        self.normalized_actions = normalize_data(
            train_actions, self.stats['action']
        ).astype(np.float32)

        self.indices = create_sample_indices(
            episode_ends, pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )
        self.pred_horizon = pred_horizon
        self.action_dim = self.normalized_actions.shape[1]
        self.resize_scale = resize_scale
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(resize_scale),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buf_start, buf_end, s_start, s_end = self.indices[idx]

        # Sample actions
        act_sample = self.normalized_actions[buf_start:buf_end]
        if s_start > 0 or s_end < self.pred_horizon:
            act = np.zeros((self.pred_horizon, self.action_dim), dtype=np.float32)
            if s_start > 0:
                act[:s_start] = act_sample[0]
            if s_end < self.pred_horizon:
                act[s_end:] = act_sample[-1]
            act[s_start:s_end] = act_sample
        else:
            act = act_sample

        # Sample images
        img_sample = self.train_images[buf_start:buf_end]
        if s_start > 0 or s_end < self.pred_horizon:
            imgs_raw = np.zeros((self.pred_horizon,) + img_sample.shape[1:],
                                dtype=img_sample.dtype)
            if s_start > 0:
                imgs_raw[:s_start] = img_sample[0]
            if s_end < self.pred_horizon:
                imgs_raw[s_end:] = img_sample[-1]
            imgs_raw[s_start:s_end] = img_sample
        else:
            imgs_raw = img_sample

        # Transform images: (T, 3, H, W) float [0, 1]
        imgs = np.stack([
            self.transform(frame).numpy() for frame in imgs_raw
        ])

        return {'image': imgs, 'action': act}


###############################################################################
#  World Model Building Blocks (same as gpc_rank_agilex_infer.py)
###############################################################################

GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8
Conv1x1 = functools_partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = functools_partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)


class WMGroupNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(max(1, c // GN_GROUP_SIZE), c, eps=GN_EPS)
    def forward(self, x): return self.norm(x)

class AdaGroupNorm(nn.Module):
    def __init__(self, c, cc):
        super().__init__()
        self.in_channels, self.num_groups = c, max(1, c // GN_GROUP_SIZE)
        self.linear = nn.Linear(cc, c * 2)
    def forward(self, x, cond):
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        s, sh = self.linear(cond)[:,:,None,None].chunk(2, dim=1)
        return x * (1 + s) + sh

class SelfAttention2d(nn.Module):
    def __init__(self, c, hd=ATTN_HEAD_DIM):
        super().__init__()
        self.n_head = max(1, c // hd)
        self.norm = WMGroupNorm(c)
        self.qkv = Conv1x1(c, c * 3); self.out = Conv1x1(c, c)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, x):
        n,c,h,w = x.shape
        qkv = self.qkv(self.norm(x)).view(n, self.n_head*3, c//self.n_head, h*w).transpose(2,3).contiguous()
        q,k,v = qkv.chunk(3, dim=1)
        y = (F.softmax(q @ k.transpose(-2,-1) / math.sqrt(k.size(-1)), -1) @ v).transpose(2,3).reshape(n,c,h,w)
        return x + self.out(y)

class FourierFeatures(nn.Module):
    def __init__(self, cc):
        super().__init__()
        self.register_buffer("weight", torch.randn(1, cc // 2))
    def forward(self, x):
        f = 2 * math.pi * x.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], -1)

class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c,c,3,2,1); nn.init.orthogonal_(self.conv.weight)
    def forward(self, x): return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = Conv3x3(c,c)
    def forward(self, x): return self.conv(F.interpolate(x, scale_factor=2., mode='nearest'))

class ResBlock(nn.Module):
    def __init__(self, ci, co, cc, attn):
        super().__init__()
        self.proj = Conv1x1(ci,co) if ci!=co else nn.Identity()
        self.n1 = AdaGroupNorm(ci,cc); self.c1 = Conv3x3(ci,co)
        self.n2 = AdaGroupNorm(co,cc); self.c2 = Conv3x3(co,co)
        self.attn = SelfAttention2d(co) if attn else nn.Identity()
        nn.init.zeros_(self.c2.weight)
    def forward(self, x, cond):
        r = self.proj(x)
        return self.attn(self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x,cond))),cond))) + r)

class ResBlocks(nn.Module):
    def __init__(self, li, lo, cc, attn):
        super().__init__()
        self.resblocks = nn.ModuleList([ResBlock(i,o,cc,attn) for i,o in zip(li,lo)])
    def forward(self, x, cond, to_cat=None):
        outs = []
        for i, rb in enumerate(self.resblocks):
            if to_cat is not None: x = torch.cat((x, to_cat[i]), 1)
            x = rb(x, cond); outs.append(x)
        return x, outs

class UNet(nn.Module):
    def __init__(self, cc, depths, chs, attns):
        super().__init__()
        self._nd = len(chs)-1
        db, ub = [], []
        for i,n in enumerate(depths):
            c1,c2 = chs[max(0,i-1)], chs[i]
            db.append(ResBlocks([c1]+[c2]*(n-1),[c2]*n,cc,attns[i]))
            ub.append(ResBlocks([2*c2]*n+[c1+c2],[c2]*n+[c1],cc,attns[i]))
        self.db = nn.ModuleList(db); self.ub = nn.ModuleList(reversed(ub))
        self.mid = ResBlocks([chs[-1]]*2,[chs[-1]]*2,cc,True)
        self.ds = nn.ModuleList([nn.Identity()]+[Downsample(c) for c in chs[:-1]])
        self.us = nn.ModuleList([nn.Identity()]+[Upsample(c) for c in reversed(chs[:-1])])
    def forward(self, x, cond):
        *_,h,w = x.size(); n=self._nd
        x = F.pad(x,(0,math.ceil(w/2**n)*2**n-w,0,math.ceil(h/2**n)*2**n-h))
        do=[]
        for b,d in zip(self.db,self.ds): xd=d(x); x,bo=b(xd,cond); do.append((xd,*bo))
        x,_=self.mid(x,cond)
        for b,u,s in zip(self.ub,self.us,reversed(do)): x,_=b(u(x),cond,s[::-1])
        return x[...,:h,:w]


class InnerModel(nn.Module):
    def __init__(self, img_ch, n_cond, cond_ch, depths, channels, attn_depths, action_dim):
        super().__init__()
        self.noise_emb = FourierFeatures(cond_ch)
        self.noise_cond_emb = FourierFeatures(cond_ch)
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, cond_ch // n_cond), nn.ReLU(), nn.Flatten())
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_ch, cond_ch), nn.SiLU(), nn.Linear(cond_ch, cond_ch))
        self.conv_in = Conv3x3((n_cond + 1) * img_ch, channels[0])
        self.unet = UNet(cond_ch, depths, channels, attn_depths)
        self.norm_out = WMGroupNorm(channels[0])
        self.conv_out = Conv3x3(channels[0], img_ch)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy, cn, cnc, obs, act):
        ae = self.act_emb(act) if act is not None else 0
        cond = self.cond_proj(self.noise_emb(cn) + self.noise_cond_emb(cnc) + ae)
        x = self.conv_in(torch.cat((obs, noisy), 1))
        x = self.unet(x, cond)
        return self.conv_out(F.silu(self.norm_out(x)))


###############################################################################
#  Denoiser (training version with loss computation)
###############################################################################

def _add_dims(t, n):
    return t.reshape(t.shape + (1,) * (n - t.ndim))


class Denoiser(nn.Module):
    def __init__(self, img_ch=3, n_cond=4, cond_ch=256,
                 depths=None, channels=None, attn_depths=None,
                 action_dim=14, sigma_data=0.5, sigma_offset_noise=0.1,
                 noise_previous_obs=True):
        super().__init__()
        if depths is None: depths = [2,2,2,2]
        if channels is None: channels = [96,96,96,96]
        if attn_depths is None: attn_depths = [0,0,1,1]

        self.sigma_data = sigma_data
        self.sigma_offset_noise = sigma_offset_noise
        self.noise_previous_obs = noise_previous_obs
        self.n_cond = n_cond
        self.inner_model = InnerModel(img_ch, n_cond, cond_ch,
                                       depths, channels, attn_depths, action_dim)
        self.sample_sigma = None

    @property
    def _device(self):
        return self.inner_model.noise_emb.weight.device

    def setup_sigma_sampling(self, loc=-1.2, scale=1.2, smin=2e-3, smax=20):
        def _s(n, dev):
            return (torch.randn(n, device=dev) * scale + loc).exp().clip(smin, smax)
        self.sample_sigma = _s

    def _apply_noise(self, x, sigma):
        b, c = x.shape[:2]
        offset = self.sigma_offset_noise * torch.randn(b, c, 1, 1, device=self._device)
        return x + offset + torch.randn_like(x) * _add_dims(sigma, x.ndim)

    def _conditioners(self, sigma, sigma_cond=None):
        sd, so = self.sigma_data, self.sigma_offset_noise
        sigma = (sigma**2 + so**2).sqrt()
        c_in = 1 / (sigma**2 + sd**2).sqrt()
        c_skip = sd**2 / (sigma**2 + sd**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        c_noise_cond = sigma_cond.log() / 4 if sigma_cond is not None else torch.zeros_like(c_noise)
        return tuple(_add_dims(c, n) for c, n in zip(
            (c_in, c_out, c_skip, c_noise, c_noise_cond), (4,4,4,1,1)))

    @torch.no_grad()
    def denoise(self, noisy, sigma, sigma_cond, obs, act):
        c_in, c_out, c_skip, c_noise, c_noise_cond = self._conditioners(sigma, sigma_cond)
        mo = self.inner_model(noisy * c_in, c_noise, c_noise_cond, obs / self.sigma_data, act)
        d = c_skip * noisy + c_out * mo
        return d.clamp(-1,1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)

    def forward(self, batch, dev):
        """Training forward: compute denoising loss over sequence."""
        obs = batch['image'].to(dev)    # (B, T, C, H, W)
        act = batch['action'].to(dev)   # (B, T, action_dim)
        b, t, c, h, w = obs.size()
        n = self.n_cond
        seq_len = t - n  # number of frames to predict

        all_obs = obs.clone()
        # Convert images from [0,1] to [-1,1]
        all_obs = all_obs * 2 - 1

        loss = 0
        for i in range(seq_len):
            prev_obs = all_obs[:, i:n+i].reshape(b, n*c, h, w)
            prev_act = act[:, i:n+i]
            target = all_obs[:, n+i]

            sigma_cond = None
            if self.noise_previous_obs:
                sigma_cond = self.sample_sigma(b, dev)
                prev_obs = self._apply_noise(prev_obs, sigma_cond)

            sigma = self.sample_sigma(b, dev)
            noisy = self._apply_noise(target, sigma)

            c_in, c_out, c_skip, c_noise, c_noise_cond = self._conditioners(sigma, sigma_cond)
            mo = self.inner_model(noisy * c_in, c_noise, c_noise_cond,
                                  prev_obs / self.sigma_data, prev_act)

            # MSE loss on denoising target
            gt = (target - c_skip * noisy) / c_out
            loss += F.mse_loss(mo, gt)

            # Autoregressive: use denoised prediction for next step
            with torch.no_grad():
                d = c_skip * noisy + c_out * mo
                d = d.clamp(-1,1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
                all_obs[:, n+i] = d

        loss /= max(seq_len, 1)
        return loss, {"loss_denoising": loss.item()}


###############################################################################
#  Training Loop
###############################################################################

def save_checkpoint(model, save_dir):
    """Save checkpoint. Unwraps DDP if needed."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "denoiser.pth")
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state_dict, path)
    print(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override dataset_path in config")
    parser.add_argument("--models_save_dir", type=str, default=None, help="Override models_save_dir in config")
    parser.add_argument("--phase_one_checkpoint", type=str, default=None, help="Override phase_one_checkpoint in config")
    parser.add_argument("--run_name", type=str, default=None, help="Override run_name in config")
    args = parser.parse_args()

    # DDP setup
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dataset_path is not None:
        cfg["dataset_path"] = args.dataset_path
    if args.models_save_dir is not None:
        cfg["models_save_dir"] = args.models_save_dir
    if args.phase_one_checkpoint is not None:
        cfg["phase_one_checkpoint"] = args.phase_one_checkpoint
    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    # Optional wandb (rank 0 only)
    if cfg.get("wandb", False) and rank == 0:
        import wandb
        dataset_tag = os.path.splitext(os.path.basename(cfg["dataset_path"]))[0]
        run_name = cfg.get("run_name", "simpler_wm") + "_" + dataset_tag
        wandb.init(project="gpc_world_model_agilex", config=cfg, name=run_name)

    # Load action stats if provided
    stats = None
    if cfg.get("action_stats_path"):
        raw = np.load(cfg["action_stats_path"], allow_pickle=True).item()
        stats = {"action": {"min": raw["action"]["min"], "max": raw["action"]["max"]}}
        if rank == 0:
            print(f"Action stats loaded from {cfg['action_stats_path']}")

    # Dataset
    dataset = AgilexWorldModelDataset(
        dataset_path=cfg["dataset_path"],
        pred_horizon=cfg["pred_horizon"],
        obs_horizon=cfg["obs_horizon"],
        action_horizon=cfg["action_horizon"],
        resize_scale=cfg.get("resize_scale", 96),
        stats=stats,
    )
    if rank == 0:
        print(f"Dataset: {len(dataset)} samples")

    # Save stats if not provided (for inference later), rank 0 only
    if stats is None and rank == 0:
        stats_path = os.path.join(cfg["models_save_dir"], "action_stats.npy")
        os.makedirs(cfg["models_save_dir"], exist_ok=True)
        np.save(stats_path, {"action": dataset.stats["action"]})
        print(f"Action stats saved to {stats_path}")

    # Dataloader with DistributedSampler
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
        shuffle=(sampler is None),
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True,
    )

    # Model
    action_dim = cfg.get("action_dim", 14)
    denoiser = Denoiser(
        n_cond=cfg.get("num_steps_conditioning", 4),
        cond_ch=cfg.get("cond_channels", 256),
        depths=cfg.get("depths", [2,2,2,2]),
        channels=cfg.get("channels", [96,96,96,96]),
        attn_depths=cfg.get("attn_depths", [0,0,1,1]),
        action_dim=action_dim,
        sigma_data=cfg.get("sigma_data", 0.5),
        sigma_offset_noise=cfg.get("sigma_offset_noise", 0.1),
        noise_previous_obs=cfg.get("noise_previous_obs", True),
    ).to(device)
    denoiser.setup_sigma_sampling()

    # Load phase one checkpoint for phase two
    if cfg.get("phase_one_checkpoint"):
        ckpt = torch.load(cfg["phase_one_checkpoint"], map_location=device)
        denoiser.load_state_dict(ckpt)
        if rank == 0:
            print(f"Phase one checkpoint loaded: {cfg['phase_one_checkpoint']}")

    # Wrap with DDP
    if distributed:
        denoiser = DDP(denoiser, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=cfg.get("lr", 1e-4))

    num_epochs = cfg["num_epochs"]
    save_dir = cfg["models_save_dir"]

    pbar = tqdm(range(1, num_epochs + 1), desc="Epoch") if rank == 0 else range(1, num_epochs + 1)
    for epoch in pbar:
        if distributed:
            sampler.set_epoch(epoch)

        epoch_losses = []
        for batch in dataloader:
            loss, metrics = denoiser(batch, device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())

            if cfg.get("wandb") and rank == 0:
                import wandb
                wandb.log({"loss": loss.item(), "epoch": epoch})

        avg = np.mean(epoch_losses)
        if rank == 0:
            if isinstance(pbar, tqdm):
                pbar.set_postfix(loss=f"{avg:.4f}")

            # Save checkpoints
            if epoch % cfg.get("save_every", 5) == 0 or epoch == num_epochs:
                save_checkpoint(denoiser, os.path.join(save_dir, f"epoch_{epoch}"))

        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()

    if rank == 0:
        print("Training complete!")


if __name__ == "__main__":
    main()
