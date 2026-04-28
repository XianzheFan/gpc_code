"""
GPC-RANK Real Robot Inference for Agilex (pi05 Base Policy)

Algorithm (from "Inference-Time Enhancement of Generative Robot Policies
via Predictive World Modeling", Qi et al. 2026):
  1. Sample N candidate action chunks from pi05 policy
  2. For each candidate, autoregressively rollout with a predictive world model
     to generate predicted future frames
  3. Score each predicted trajectory with a learned reward predictor
  4. Execute the highest-ranked (lowest cost) action chunk on the robot

Usage:
  # 1. Start pi05 policy server (e.g. via openpi serve_policy.py)
  # 2. Run GPC-RANK inference:
  python gpc_rank_agilex_infer.py \
      --task towel --host <pi05_host> --port 8000 \
      --gpc_config configs/gpc_rank_agilex_config.yml \
      --world_model_ckpt <world_model.pth> \
      --reward_predictor_ckpt <reward_predictor.pth>
"""

import argparse
import collections
import logging
import math
import os
import signal
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from functools import partial as functools_partial
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.v2 as v2
import yaml

import rospy

# ---------------------------------------------------------------------------
# Path setup: agilex robot utilities
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_THIS_DIR, '..', 'openpi', 'agilex'))
from clients import OpenpiClient
from agilex_utils import (InferenceDataRecorder, check_keyboard_input,
                           get_config, get_inference_observation,
                           get_rollout_observation, handle_interactive_mode,
                           process_action)
from ros_operator import RosOperator
from scipy.spatial.transform import Rotation as _R


def quat_2_euler(quat):
    return _R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler('xyz')


def _rot6d_to_matrix(r6):
    a = np.asarray(r6[:3], dtype=np.float64)
    b = np.asarray(r6[3:6], dtype=np.float64)
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_p = b - np.dot(a_n, b) * a_n
    b_n = b_p / (np.linalg.norm(b_p) + 1e-8)
    c_n = np.cross(a_n, b_n)
    return np.stack([a_n, b_n, c_n], axis=-1)


def abs_6d_2_abs_euler(act):
    """Convert 20-d ee6d action [pos3, rot6, grip1] x 2  ->  14-d euler form."""
    a = np.asarray(act, dtype=np.float64)
    pos_l, rot6_l, grip_l = a[0:3], a[3:9], a[9:10]
    pos_r, rot6_r, grip_r = a[10:13], a[13:19], a[19:20]
    eul_l = _R.from_matrix(_rot6d_to_matrix(rot6_l)).as_euler('xyz')
    eul_r = _R.from_matrix(_rot6d_to_matrix(rot6_r)).as_euler('xyz')
    return np.concatenate([pos_l, eul_l, grip_l, pos_r, eul_r, grip_r])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
#  World Model Building Blocks
#  (Self-contained from gpc_rank_evaluation/diffusion, adapted for agilex
#   action_dim.  Avoids importing training-only deps like zarr.)
###############################################################################

# -- constants ---------------------------------------------------------------
GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

Conv1x1 = functools_partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = functools_partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)


class WMGroupNorm(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x):
        return self.norm(x)


class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x, cond):
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM):
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        self.norm = WMGroupNorm(in_channels)
        self.qkv = Conv1x1(in_channels, in_channels * 3)
        self.out = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        n, c, h, w = x.shape
        x_normed = self.norm(x)
        qkv = self.qkv(x_normed).view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous()
        q, k, v = qkv.chunk(3, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(2, 3).reshape(n, c, h, w)
        return x + self.out(y)


class FourierFeatures(nn.Module):
    def __init__(self, cond_channels: int):
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))

    def forward(self, x):
        assert x.ndim == 1
        f = 2 * math.pi * x.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = Conv3x3(c, c)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_ch, attn):
        super().__init__()
        self.proj = Conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()
        self.n1 = AdaGroupNorm(in_ch, cond_ch)
        self.c1 = Conv3x3(in_ch, out_ch)
        self.n2 = AdaGroupNorm(out_ch, cond_ch)
        self.c2 = Conv3x3(out_ch, out_ch)
        self.attn = SelfAttention2d(out_ch) if attn else nn.Identity()
        nn.init.zeros_(self.c2.weight)

    def forward(self, x, cond):
        r = self.proj(x)
        x = self.c1(F.silu(self.n1(x, cond)))
        x = self.c2(F.silu(self.n2(x, cond)))
        return self.attn(x + r)


class ResBlocks(nn.Module):
    def __init__(self, list_in, list_out, cond_ch, attn):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlock(i, o, cond_ch, attn) for i, o in zip(list_in, list_out)
        ])

    def forward(self, x, cond, to_cat=None):
        outputs = []
        for i, rb in enumerate(self.resblocks):
            if to_cat is not None:
                x = torch.cat((x, to_cat[i]), dim=1)
            x = rb(x, cond)
            outputs.append(x)
        return x, outputs


class UNet(nn.Module):
    def __init__(self, cond_ch, depths, channels, attn_depths):
        super().__init__()
        self._num_down = len(channels) - 1
        db, ub = [], []
        for i, n in enumerate(depths):
            c1, c2 = channels[max(0, i - 1)], channels[i]
            db.append(ResBlocks([c1] + [c2] * (n - 1), [c2] * n, cond_ch, attn_depths[i]))
            ub.append(ResBlocks([2 * c2] * n + [c1 + c2], [c2] * n + [c1], cond_ch, attn_depths[i]))
        self.db = nn.ModuleList(db)
        self.ub = nn.ModuleList(reversed(ub))
        self.mid = ResBlocks([channels[-1]] * 2, [channels[-1]] * 2, cond_ch, True)
        self.ds = nn.ModuleList([nn.Identity()] + [Downsample(c) for c in channels[:-1]])
        self.us = nn.ModuleList([nn.Identity()] + [Upsample(c) for c in reversed(channels[:-1])])

    def forward(self, x, cond):
        *_, h, w = x.size()
        n = self._num_down
        ph = math.ceil(h / 2 ** n) * 2 ** n - h
        pw = math.ceil(w / 2 ** n) * 2 ** n - w
        x = F.pad(x, (0, pw, 0, ph))
        d_outputs = []
        for block, down in zip(self.db, self.ds):
            x_d = down(x)
            x, bo = block(x_d, cond)
            d_outputs.append((x_d, *bo))
        x, _ = self.mid(x, cond)
        u_outputs = []
        for block, up, skip in zip(self.ub, self.us, reversed(d_outputs)):
            x_u = up(x)
            x, bo = block(x_u, cond, skip[::-1])
            u_outputs.append((x_u, *bo))
        return x[..., :h, :w], d_outputs, u_outputs


# -- InnerModel (configurable action_dim for agilex) -------------------------

class InnerModel(nn.Module):
    def __init__(self, img_channels, num_steps_conditioning, cond_channels,
                 depths, channels, attn_depths, action_dim, is_upsampler=False):
        super().__init__()
        self.noise_emb = FourierFeatures(cond_channels)
        self.noise_cond_emb = FourierFeatures(cond_channels)
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, cond_channels // num_steps_conditioning),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
        )
        n_in = (num_steps_conditioning + int(is_upsampler) + 1) * img_channels
        self.conv_in = Conv3x3(n_in, channels[0])
        self.unet = UNet(cond_channels, depths, channels, attn_depths)
        self.norm_out = WMGroupNorm(channels[0])
        self.conv_out = Conv3x3(channels[0], img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_next_obs, c_noise, c_noise_cond, obs, act):
        act_emb = self.act_emb(act) if act is not None else 0
        cond = self.cond_proj(self.noise_emb(c_noise) + self.noise_cond_emb(c_noise_cond) + act_emb)
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        return self.conv_out(F.silu(self.norm_out(x)))


# -- Denoiser ----------------------------------------------------------------

def _add_dims(t, n):
    return t.reshape(t.shape + (1,) * (n - t.ndim))


@dataclass
class DenoiserConfig:
    img_channels: int = 3
    num_steps_conditioning: int = 4
    cond_channels: int = 256
    depths: list = None
    channels: list = None
    attn_depths: list = None
    action_dim: int = 14
    sigma_data: float = 0.5
    sigma_offset_noise: float = 0.1
    noise_previous_obs: bool = True

    def __post_init__(self):
        if self.depths is None:
            self.depths = [2, 2, 2, 2]
        if self.channels is None:
            self.channels = [96, 96, 96, 96]
        if self.attn_depths is None:
            self.attn_depths = [0, 0, 1, 1]


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig):
        super().__init__()
        self.cfg = cfg
        self.inner_model = InnerModel(
            img_channels=cfg.img_channels,
            num_steps_conditioning=cfg.num_steps_conditioning,
            cond_channels=cfg.cond_channels,
            depths=cfg.depths,
            channels=cfg.channels,
            attn_depths=cfg.attn_depths,
            action_dim=cfg.action_dim,
        )
        self.sample_sigma_training = None

    @property
    def _device(self):
        return self.inner_model.noise_emb.weight.device

    def setup_sigma_sampling(self, loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=20):
        def _sample(n, dev):
            s = torch.randn(n, device=dev) * scale + loc
            return s.exp().clip(sigma_min, sigma_max)
        self.sample_sigma_training = _sample

    def apply_noise(self, x, sigma, sigma_offset):
        b, c, _, _ = x.shape
        offset = sigma_offset * torch.randn(b, c, 1, 1, device=self._device)
        return x + offset + torch.randn_like(x) * _add_dims(sigma, x.ndim)

    def _conditioners(self, sigma, sigma_cond=None):
        sd = self.cfg.sigma_data
        so = self.cfg.sigma_offset_noise
        sigma = (sigma ** 2 + so ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + sd ** 2).sqrt()
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        c_noise_cond = sigma_cond.log() / 4 if sigma_cond is not None else torch.zeros_like(c_noise)
        return tuple(_add_dims(c, n) for c, n in zip(
            (c_in, c_out, c_skip, c_noise, c_noise_cond), (4, 4, 4, 1, 1)))

    @torch.no_grad()
    def denoise(self, noisy, sigma, sigma_cond, obs, act):
        c_in, c_out, c_skip, c_noise, c_noise_cond = self._conditioners(sigma, sigma_cond)
        rescaled_obs = obs / self.cfg.sigma_data
        model_out = self.inner_model(noisy * c_in, c_noise, c_noise_cond, rescaled_obs, act)
        d = c_skip * noisy + c_out * model_out
        return d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)


# -- DiffusionSampler --------------------------------------------------------

@dataclass
class SamplerConfig:
    num_steps: int = 3
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1
    s_cond: float = 0


def _build_sigmas(n, smin, smax, rho, dev):
    mi = smin ** (1 / rho)
    mx = smax ** (1 / rho)
    l = torch.linspace(0, 1, n, device=dev)
    sigmas = (mx + l * (mi - mx)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: SamplerConfig):
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = _build_sigmas(cfg.num_steps, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser._device)

    @torch.no_grad()
    def sample(self, prev_obs, prev_act):
        """
        Args:
            prev_obs: (B, T, C, H, W) conditioning frames
            prev_act: (B, T, action_dim) conditioning actions
        Returns:
            predicted next frame: (B, C, H, W)
        """
        dev = prev_obs.device
        b, t, c, h, w = prev_obs.size()
        prev_obs_flat = prev_obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=dev)
        gamma_ = min(self.cfg.s_churn / max(len(self.sigmas) - 1, 1), 2 ** 0.5 - 1)
        x = torch.randn(b, c, h, w, device=dev)
        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                x = x + torch.randn_like(x) * self.cfg.s_noise * (sigma_hat ** 2 - sigma ** 2) ** 0.5
            sigma_cond = None
            obs_input = prev_obs_flat
            if self.cfg.s_cond > 0:
                sigma_cond = torch.full((b,), self.cfg.s_cond, device=dev)
                obs_input = self.denoiser.apply_noise(obs_input, sigma_cond, sigma_offset_noise=0)
            denoised = self.denoiser.denoise(x, sigma * s_in, sigma_cond, obs_input, prev_act)
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.cfg.order == 1 or next_sigma == 0:
                x = x + d * dt
            else:
                x2 = x + d * dt
                denoised2 = self.denoiser.denoise(x2, next_sigma * s_in, sigma_cond, obs_input, prev_act)
                d2 = (x2 - denoised2) / next_sigma
                x = x + (d + d2) / 2 * dt
        return x


###############################################################################
#  Reward Predictor (GPC-RANK paper: ResNet18 + MLP)
###############################################################################

class RewardPredictor(nn.Module):
    """
    Predicts a scalar task-completion score from a single image.
    Lower = closer to goal.  Architecture follows GPC-RANK paper.
    """

    def __init__(self, output_dim: int = 1):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.mlp(self.resnet18(x))


###############################################################################
#  GPC-RANK Selector
###############################################################################

class GPCRankSelector:
    """
    Core GPC-RANK:
      - Maintain observation & action history
      - Given N candidates, batch rollout via world model
      - Score predicted final frames, return best candidate
    """

    def __init__(self, sampler: DiffusionSampler, reward_predictor: RewardPredictor,
                 num_candidates=10, rollout_steps=8, n_cond=4,
                 img_size=96, action_dim=14, action_stats=None):
        self.sampler = sampler
        self.reward_predictor = reward_predictor
        self.num_candidates = num_candidates
        self.rollout_steps = rollout_steps
        self.n_cond = n_cond
        self.img_size = img_size
        self.action_dim = action_dim
        self.action_stats = action_stats

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(img_size),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.obs_history: List[np.ndarray] = []
        self.action_history: List[np.ndarray] = []

    def reset(self):
        self.obs_history.clear()
        self.action_history.clear()

    def update_obs(self, img_hwc_uint8: np.ndarray):
        t = self.transform(img_hwc_uint8).numpy()   # (3, H, W) [0,1]
        self.obs_history.append(t * 2.0 - 1.0)      # -> [-1, 1]

    def update_action(self, action: np.ndarray):
        self.action_history.append(self._norm_act(action))

    def _norm_act(self, a):
        if self.action_stats is None:
            return a.astype(np.float32)
        lo, hi = self.action_stats['min'], self.action_stats['max']
        return (2.0 * (a - lo) / (hi - lo + 1e-8) - 1.0).astype(np.float32)

    @torch.no_grad()
    def rank(self, candidates: List[np.ndarray]) -> Tuple[int, List[float]]:
        """
        Rank candidates (each shape (T, action_dim)) via world-model rollout.
        Returns (best_idx, per-candidate scores).
        """
        N = len(candidates)
        n = self.n_cond

        if len(self.obs_history) < n:
            return 0, [0.0] * N

        # -- conditioning frames: (N, n, 3, H, W) --
        frames = torch.tensor(np.stack(self.obs_history[-n:]),
                              dtype=torch.float32, device=device)
        frames = frames.unsqueeze(0).expand(N, -1, -1, -1, -1)

        # -- conditioning actions: (N, n, action_dim) --
        if len(self.action_history) >= n:
            act_hist = np.stack(self.action_history[-n:])
        else:
            pad = n - len(self.action_history)
            existing = np.stack(self.action_history) if self.action_history else np.zeros((0, self.action_dim), np.float32)
            act_hist = np.concatenate([np.zeros((pad, self.action_dim), np.float32), existing])
        act_t = torch.tensor(act_hist, dtype=torch.float32, device=device).unsqueeze(0).expand(N, -1, -1)

        # -- candidate actions: (N, rollout, action_dim) --
        rollout = min(self.rollout_steps, min(c.shape[0] for c in candidates))
        cand = torch.stack([
            torch.tensor(self._norm_act(c[:rollout]), dtype=torch.float32, device=device)
            for c in candidates
        ])  # (N, rollout, action_dim)

        # -- autoregressive rollout --
        pred = frames.clone()    # (N, n, 3, H, W)
        acts = act_t.clone()     # (N, n, action_dim)

        for s in range(rollout):
            next_frame = self.sampler.sample(pred[:, -n:], acts[:, -n:])  # (N, 3, H, W)
            pred = torch.cat([pred, next_frame.unsqueeze(1)], dim=1)
            acts = torch.cat([acts, cand[:, s:s+1]], dim=1)

        # -- score final frames --
        final = (pred[:, -1] + 1.0) / 2.0   # (N, 3, H, W) in [0,1]
        scores = self.reward_predictor(final).squeeze(-1).cpu().numpy()

        best = int(np.argmin(scores))
        return best, scores.tolist()


###############################################################################
#  Observation window & ROS helpers (from agilex_infer.py)
###############################################################################

observation_window = None
observation_window_lock = threading.Lock()
shutdown_event = threading.Event()


def _on_sigint(signum, frame):
    try:
        shutdown_event.set()
    except Exception:
        pass
    try:
        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass


def reset_observation_window():
    global observation_window
    with observation_window_lock:
        observation_window = None


def update_observation_window(args, config, ros_operator):
    global observation_window
    with observation_window_lock:
        if observation_window is None:
            observation_window = collections.deque(maxlen=2)
            observation_window.append({
                "qpos": None,
                "images": {cn: None for cn in config["camera_names"]},
                "eef_pose": None,
            })

    observation = get_inference_observation(args, config, ros_operator)
    if observation is None:
        return False

    with observation_window_lock:
        observation_window.append(observation)

    return True


def _get_obs_and_state(args, config):
    with observation_window_lock:
        imgs = [observation_window[-1]["images"][cn] for cn in config["camera_names"]]
        if args.ctrl_type in ("joint", "ee6d"):
            state = observation_window[-1]["qpos"]
        elif args.ctrl_type == "eef":
            state = observation_window[-1]["eef_pose"]
        else:
            raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")
    return imgs, state


###############################################################################
#  GPC-RANK inference step
###############################################################################

def gpc_rank_inference(args, config, policy, ranker: GPCRankSelector, ros_operator):
    """
    One GPC-RANK decision:
      1. Read observation
      2. Sample N candidates from pi05
      3. Rank via world model
      4. Return best action chunk
    """
    if not update_observation_window(args, config, ros_operator):
        return None, None
    imgs, state = _get_obs_and_state(args, config)

    # Push front-camera into world-model history
    if imgs[0] is not None:
        ranker.update_obs(imgs[0])

    payload = {
        "top": imgs[0], "left": imgs[1], "right": imgs[2],
        "instruction": config["language_instruction"],
        "state": state, "action_prefix": None, "delay": None,
    }

    # Step 1: sample N candidates from pi05
    t0 = time.perf_counter()
    candidates = [np.array(policy.predict_action(payload))
                  for _ in range(ranker.num_candidates)]
    t_sample = (time.perf_counter() - t0) * 1000

    # Trim to exec_horizon
    exec_h = config["exec_horizon"]
    trimmed = [c[:exec_h] for c in candidates]

    # Spread candidates around the mean (GPC-RANK paper, Eq. in eval_baseline)
    spread = config.get("spread_factor", 1.01)
    arr = np.stack(trimmed)
    mean = arr.mean(axis=0, keepdims=True)
    arr = mean + spread * (arr - mean)
    trimmed = [arr[i] for i in range(len(candidates))]

    # Step 2: rank
    t1 = time.perf_counter()
    best_idx, scores = ranker.rank(trimmed)
    t_rank = (time.perf_counter() - t1) * 1000

    logging.info(
        f"[GPC-RANK] sample={t_sample:.0f}ms  rank={t_rank:.0f}ms  "
        f"best={best_idx}  score={scores[best_idx]:.4f}"
    )
    return trimmed[best_idx], scores


###############################################################################
#  Main control loop
###############################################################################

def model_inference(args, config, ros_operator):
    # -- pi05 policy --
    policy = OpenpiClient(host=args.host, port=args.port)

    # -- world model --
    wm = yaml.safe_load(open(args.gpc_config))
    action_dim = wm.get("action_dim", 14)
    config["spread_factor"] = wm.get("spread_factor", 1.01)

    dcfg = DenoiserConfig(
        img_channels=3,
        num_steps_conditioning=wm.get("num_steps_conditioning", 4),
        cond_channels=wm.get("cond_channels", 256),
        depths=wm.get("depths", [2, 2, 2, 2]),
        channels=wm.get("channels", [96, 96, 96, 96]),
        attn_depths=wm.get("attn_depths", [0, 0, 1, 1]),
        action_dim=action_dim,
        sigma_data=wm.get("sigma_data", 0.5),
        sigma_offset_noise=wm.get("sigma_offset_noise", 0.1),
        noise_previous_obs=wm.get("noise_previous_obs", True),
    )
    denoiser = Denoiser(dcfg).to(device)
    denoiser.setup_sigma_sampling(
        loc=wm.get("sigma_loc", -1.2), scale=wm.get("sigma_scale", 1.2),
        sigma_min=wm.get("sigma_min", 2e-3), sigma_max=wm.get("sigma_max", 20),
    )
    denoiser.load_state_dict(torch.load(args.world_model_ckpt, map_location=device))
    denoiser.eval()
    logging.info(f"World model loaded: {args.world_model_ckpt}")

    scfg = SamplerConfig(
        num_steps=wm.get("num_steps_denoising", 3),
        sigma_min=wm.get("sampler_sigma_min", 2e-3),
        sigma_max=wm.get("sampler_sigma_max", 5),
    )
    wm_sampler = DiffusionSampler(denoiser, scfg)

    # -- reward predictor --
    reward_pred = RewardPredictor(output_dim=wm.get("reward_output_dim", 1)).to(device)
    reward_pred.load_state_dict(torch.load(args.reward_predictor_ckpt, map_location=device))
    reward_pred.eval()
    logging.info(f"Reward predictor loaded: {args.reward_predictor_ckpt}")

    # -- action stats --
    action_stats = None
    if args.action_stats_path and os.path.exists(args.action_stats_path):
        raw = np.load(args.action_stats_path, allow_pickle=True).item()
        action_stats = {"min": raw["action"]["min"], "max": raw["action"]["max"]}

    # -- GPC ranker --
    ranker = GPCRankSelector(
        sampler=wm_sampler, reward_predictor=reward_pred,
        num_candidates=args.num_candidates,
        rollout_steps=args.rollout_steps,
        n_cond=wm.get("num_steps_conditioning", 4),
        img_size=wm.get("img_size", 96),
        action_dim=action_dim,
        action_stats=action_stats,
    )

    # -- robot --
    max_step = config["episode_len"]
    left0, right0 = config["left0"], config["right0"]
    print(config)

    ros_operator.follower_arm_publish_continuous(left0, right0)
    print("Warmup pi05 server...")
    policy.warmup(rtc=False, streaming=False)
    print("Server warmed up")

    input("Press enter to start")
    task_time = time.time()
    ros_operator.follower_arm_publish_continuous(left0, right0)
    recorder = InferenceDataRecorder(args, config, shutdown_event=shutdown_event)

    try:
        while not rospy.is_shutdown():
            t = 0
            rate = rospy.Rate(args.publish_rate)

            reset_observation_window()
            ranker.reset()

            # First GPC-RANK inference
            chunk, _ = gpc_rank_inference(args, config, policy, ranker, ros_operator)
            if chunk is None:
                break
            idx = 0
            last_act = None
            episode_closed = False

            while t < max_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                print(f"[Step {t:4d}] action_idx={idx}/{len(chunk)}")

                key = check_keyboard_input()
                if key == " ":
                    result = handle_interactive_mode(task_time)
                    if result == "reset":
                        recorder.save_episode()
                        episode_closed = True
                        ros_operator.follower_arm_publish_continuous(left0, right0)
                        input("Press enter to continue")
                        task_time = time.time()
                        break
                    elif result == "quit":
                        recorder.save_episode()
                        return

                # Re-plan when chunk exhausted
                if idx >= len(chunk):
                    chunk, _ = gpc_rank_inference(args, config, policy, ranker, ros_operator)
                    if chunk is None:
                        break
                    idx = 0

                act = chunk[idx]
                idx += 1

                # Update histories
                ranker.update_action(act)
                if not update_observation_window(args, config, ros_operator):
                    break
                with observation_window_lock:
                    front = observation_window[-1]["images"].get(config["camera_names"][0])
                if front is not None:
                    ranker.update_obs(front)

                observation_to_save = (
                    get_rollout_observation(args, config, ros_operator)
                    if recorder.enabled else None
                )
                if recorder.enabled and observation_to_save is None:
                    break

                # Execute
                if args.ctrl_type == "joint":
                    la, ra = process_action(config["task"], act)
                    ros_operator.follower_arm_publish(la, ra)
                elif args.ctrl_type == "ee6d":
                    la, ra = process_action(config["task"], abs_6d_2_abs_euler(act))
                    ros_operator.follower_arm_pose_publish(la, ra)
                elif args.ctrl_type == "eef":
                    la, ra = process_action(config["task"], act)
                    ros_operator.follower_arm_pose_publish(la, ra)

                if args.use_robot_base:
                    ros_operator.robot_base_publish(act[14:16])

                action_to_save = np.concatenate((la, ra), axis=0)
                recorder.add_step(observation_to_save, action_to_save)

                t += 1
                last_act = act
                rate.sleep()

            if not episode_closed:
                recorder.save_episode()
            if shutdown_event.is_set():
                return

    finally:
        ros_operator.follower_arm_publish_continuous(left0, right0)


###############################################################################
#  CLI
###############################################################################

def get_arguments():
    p = argparse.ArgumentParser(
        description="GPC-RANK real-robot inference for Agilex (pi05 base policy)")

    # ROS topics
    p.add_argument("--max_publish_step", type=int, default=10000)
    p.add_argument("--img_front_topic", default="/camera_f/color/image_raw")
    p.add_argument("--img_left_topic", default="/camera_l/color/image_raw")
    p.add_argument("--img_right_topic", default="/camera_r/color/image_raw")
    p.add_argument("--img_front_depth_topic", default="/camera_f/depth/image_raw")
    p.add_argument("--img_left_depth_topic", default="/camera_l/depth/image_raw")
    p.add_argument("--img_right_depth_topic", default="/camera_r/depth/image_raw")
    p.add_argument("--leader_arm_left_topic", default="/leader/joint_left")
    p.add_argument("--leader_arm_right_topic", default="/leader/joint_right")
    p.add_argument("--follower_arm_left_topic", default="/follower/joint_left")
    p.add_argument("--follower_arm_right_topic", default="/follower/joint_right")
    p.add_argument("--pos_cmd_left_topic", default="/follower/pos_cmd_left")
    p.add_argument("--pos_cmd_right_topic", default="/follower/pos_cmd_right")
    p.add_argument("--follower_arm_left_pose_topic", default="/follower/end_pose_euler_left")
    p.add_argument("--follower_arm_right_pose_topic", default="/follower/end_pose_euler_right")
    p.add_argument("--robot_base_topic", default="/odom_raw")
    p.add_argument("--robot_base_cmd_topic", default="/cmd_vel")
    p.add_argument("--use_robot_base", action="store_true", default=False)
    p.add_argument("--use_depth_image", action="store_true", default=False)
    p.add_argument("--save_rollout", action="store_true", default=False,
                   help="Save rollout observations/actions to HDF5 episodes")
    p.add_argument("--save_dir", type=str, default="/home/sail/data_rollout",
                   help="Directory used when --save_rollout is set")

    # Robot control
    p.add_argument("--publish_rate", type=int, default=30)
    p.add_argument("--chunk_size", type=int, default=50)
    p.add_argument("--arm_steps_length", type=float, nargs=7,
                   default=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.2])
    p.add_argument("--ctrl_type", choices=["joint", "eef", "ee6d"], default="joint")
    p.add_argument("--task", required=True,
                   help="Task name; must be a key in task_configs.yaml")
    p.add_argument("--exec_horizon", type=int, default=25)

    # pi05 policy server
    p.add_argument("--host", default="10.0.0.1", help="pi05 server host")
    p.add_argument("--port", type=int, default=8000, help="pi05 server port")

    # GPC-RANK
    p.add_argument("--gpc_config", required=True,
                   help="YAML config for world model / reward predictor")
    p.add_argument("--world_model_ckpt", required=True,
                   help="World model checkpoint (.pth)")
    p.add_argument("--reward_predictor_ckpt", required=True,
                   help="Reward predictor checkpoint (.pth)")
    p.add_argument("--action_stats_path", default=None,
                   help="Action normalization stats (.npy)")
    p.add_argument("--num_candidates", type=int, default=10,
                   help="Number of action candidates to sample from pi05")
    p.add_argument("--rollout_steps", type=int, default=8,
                   help="World-model rollout horizon")

    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = get_arguments()
    ros_operator = RosOperator(args, mode="inference")
    config = get_config(args)

    signal.signal(signal.SIGINT, _on_sigint)

    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        model_inference(args, config, ros_operator)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
