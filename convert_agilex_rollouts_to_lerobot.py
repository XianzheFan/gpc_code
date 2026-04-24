"""
Convert Agilex rollout HDF5 episodes (from openpi agilex_infer.py --save_rollout)
into a LeRobot v2.1 dataset with a per-frame `success` column, ready for
train_reward_predictor.py --robot_type agilex.

Input layout (produced by openpi/agilex/agilex_utils.py::save_inference_data):
  <input_dir>/
    episode_0.hdf5
    episode_1.hdf5
    ...
  Each file has:
    root.attrs["rollout"]  = True
    root.attrs["success"]  = bool       # s/f label, written by wait_save_choice
    /observations/images/cam_high        (T, 480, 640, 3) uint8
    /observations/images/cam_left_wrist  (T, 480, 640, 3) uint8
    /observations/images/cam_right_wrist (T, 480, 640, 3) uint8
    /observations/qpos                   (T, 14) float
    /observations/eef_pose               (T, 14) float
    /action                              (T, 14) float

Output layout (matches openpi/data/pnp_cup_0415 schema):
  <output_dir>/
    meta/
      info.json
      episodes.jsonl
      tasks.jsonl
    data/chunk-000/episode_000000.parquet    # + success column
    videos/chunk-000/observation.images.cam_high/episode_000000.mp4
    videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4
    videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4

Usage:
  python convert_agilex_rollouts_to_lerobot.py \
      --input_dir ~/data/gpc_perturbation_rollouts_pnp_cup \
      --output_dir data/agilex_pnp_cup_exploration_lerobot \
      --task "Pick up the paper cup and put it into the cup sleeve." \
      --fps 30
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
CHUNK_SIZE = 1000
IMG_H, IMG_W = 480, 640


def list_episodes(input_dir: Path):
    files = sorted(glob.glob(str(input_dir / "episode_*.hdf5")),
                   key=lambda p: int(re.search(r"episode_(\d+)\.hdf5", p).group(1)))
    if not files:
        sys.exit(f"[error] no episode_*.hdf5 under {input_dir}")
    return [Path(f) for f in files]


def encode_video(frames: np.ndarray, out_path: Path, fps: int):
    """Pipe (T, H, W, 3) uint8 RGB into ffmpeg, encode libx264 yuv420p mp4."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, _ = frames.shape
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "23",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(frames.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def write_parquet(episode_idx: int, global_start: int, qpos: np.ndarray, action: np.ndarray,
                   success: bool, fps: int, out_path: Path, task_index: int = 0):
    T = len(action)
    df = pd.DataFrame({
        "observation.state": [row.astype(np.float32) for row in qpos],
        "action":            [row.astype(np.float32) for row in action],
        "timestamp":         np.arange(T, dtype=np.float32) / float(fps),
        "frame_index":       np.arange(T, dtype=np.int64),
        "episode_index":     np.full(T, episode_idx, dtype=np.int64),
        "index":             np.arange(global_start, global_start + T, dtype=np.int64),
        "task_index":        np.full(T, task_index, dtype=np.int64),
        "success":           np.full(T, bool(success), dtype=bool),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def build_info_json(total_episodes: int, total_frames: int, fps: int):
    cam_feat = lambda: {
        "dtype": "video",
        "shape": [IMG_H, IMG_W, 3],
        "names": ["height", "width", "channel"],
        "video_info": {"video.fps": float(fps), "video.codec": "h264",
                       "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                       "has_audio": False},
        "info": {"video.height": IMG_H, "video.width": IMG_W, "video.codec": "h264",
                 "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                 "video.fps": fps, "video.channels": 3, "has_audio": False},
    }
    features = {f"observation.images.{c}": cam_feat() for c in CAMERAS}
    features.update({
        "observation.state": {"dtype": "float32", "shape": [14]},
        "action":            {"dtype": "float32", "shape": [14]},
        "timestamp":         {"dtype": "float32", "shape": [1], "names": None},
        "frame_index":       {"dtype": "int64",   "shape": [1], "names": None},
        "episode_index":     {"dtype": "int64",   "shape": [1], "names": None},
        "index":             {"dtype": "int64",   "shape": [1], "names": None},
        "task_index":        {"dtype": "int64",   "shape": [1], "names": None},
        "success":           {"dtype": "bool",    "shape": [1], "names": None},
    })
    return {
        "codebase_version": "v2.1",
        "robot_type": "agilex",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * len(CAMERAS),
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="Dir containing episode_N.hdf5 rollouts (openpi save_rollout output)")
    ap.add_argument("--output_dir", required=True,
                    help="Destination LeRobot v2.1 dataset dir")
    ap.add_argument("--task", required=True,
                    help="Task description written into tasks.jsonl / episodes.jsonl")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--default_success", choices=["true", "false"], default="true",
                    help="Fallback label when hdf5 has no success attr (older rollouts)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if output_dir.exists() and any(output_dir.iterdir()):
        sys.exit(f"[error] output_dir {output_dir} is not empty — refusing to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = list_episodes(input_dir)
    print(f"[info] found {len(episodes)} hdf5 episodes in {input_dir}")

    chunk = "chunk-000"
    global_idx = 0
    meta_episodes = []
    n_success, n_fail = 0, 0

    for new_idx, ep_path in enumerate(episodes):
        with h5py.File(ep_path, "r") as f:
            if "success" in f.attrs:
                success = bool(f.attrs["success"])
            else:
                success = (args.default_success == "true")
            qpos   = np.asarray(f["/observations/qpos"])         # (T, 14)
            action = np.asarray(f["/action"])                     # (T, 14)
            imgs = {c: np.asarray(f[f"/observations/images/{c}"]) for c in CAMERAS}

        T = len(action)
        assert qpos.shape == (T, 14) and action.shape == (T, 14), \
            f"bad shapes in {ep_path}: qpos={qpos.shape}, action={action.shape}"
        for c in CAMERAS:
            assert imgs[c].shape == (T, IMG_H, IMG_W, 3), \
                f"bad img shape {c} in {ep_path}: {imgs[c].shape}"

        # parquet
        parquet_path = output_dir / "data" / chunk / f"episode_{new_idx:06d}.parquet"
        write_parquet(new_idx, global_idx, qpos, action, success, args.fps, parquet_path)

        # videos (one per camera)
        for c in CAMERAS:
            vid_path = output_dir / "videos" / chunk / f"observation.images.{c}" / f"episode_{new_idx:06d}.mp4"
            encode_video(imgs[c], vid_path, args.fps)

        meta_episodes.append({
            "episode_index": new_idx,
            "tasks": [args.task],
            "length": T,
            "success": success,
        })
        global_idx += T
        n_success += int(success)
        n_fail += int(not success)
        print(f"  [{new_idx:3d}] {ep_path.name}  T={T}  success={success}")

    # meta files
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": args.task}) + "\n"
    )
    with (meta_dir / "episodes.jsonl").open("w") as fh:
        for ep in meta_episodes:
            fh.write(json.dumps(ep) + "\n")
    (meta_dir / "info.json").write_text(
        json.dumps(build_info_json(len(episodes), global_idx, args.fps), indent=4)
    )

    print(f"\n[done] wrote {len(episodes)} episodes, {global_idx} frames to {output_dir}")
    print(f"       success={n_success}, fail={n_fail}")


if __name__ == "__main__":
    main()
