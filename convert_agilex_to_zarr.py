"""
Convert Agilex LeRobot dataset (parquet + mp4) to zarr format
for GPC world model training.

Input:  fold_towel_0109_agilex/
        ├── data/chunk-000/episode_XXXXXX.parquet  (actions, state)
        ├── videos/chunk-000/observation.images.cam_high/episode_XXXXXX.mp4
        └── meta/info.json, stats.json, ...

Output: agilex_fold_towel.zarr/
        ├── data/
        │   ├── img      (N, 96, 96, 3) uint8   — front camera frames
        │   ├── action   (N, 14)         float32 — joint actions
        │   └── state    (N, 14)         float32 — joint states
        └── meta/
            └── episode_ends  (num_episodes,) int64

Usage:
  python convert_agilex_to_zarr.py \
      --dataset_dir ../DreamDojo/datasets/fold_towel_0109_agilex \
      --output_path agilex_fold_towel.zarr \
      --camera cam_high \
      --img_size 96
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


def read_video_frames(video_path: str, img_size: int) -> np.ndarray:
    """Read all frames from mp4, resize to (img_size, img_size, 3) uint8 RGB.

    Uses system ffmpeg via subprocess to avoid OpenCV AV1 decoding issues.
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", video_path,
        "-vf", f"scale={img_size}:{img_size}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {video_path}: {result.stderr.decode()}")
    raw = result.stdout
    frame_size = img_size * img_size * 3
    if len(raw) == 0:
        return np.zeros((0, img_size, img_size, 3), dtype=np.uint8)
    n_frames = len(raw) // frame_size
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, img_size, img_size, 3)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True,
                        help="Path to fold_towel_0109_agilex dataset")
    parser.add_argument("--output_path", required=True,
                        help="Output zarr path")
    parser.add_argument("--camera", default="cam_high",
                        choices=["cam_high", "cam_left_wrist", "cam_right_wrist"],
                        help="Which camera to use for world model")
    parser.add_argument("--img_size", type=int, default=96,
                        help="Resize images to this resolution")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit number of episodes (default: all)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    camera_key = f"observation.images.{args.camera}"

    # Discover episodes
    parquet_dir = dataset_dir / "data" / "chunk-000"
    video_dir = dataset_dir / "videos" / "chunk-000" / camera_key

    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))
    if args.max_episodes:
        parquet_files = parquet_files[:args.max_episodes]

    num_episodes = len(parquet_files)
    print(f"Found {num_episodes} episodes")

    # First pass: count total frames
    total_frames = 0
    episode_lengths = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        episode_lengths.append(len(df))
        total_frames += len(df)

    print(f"Total frames: {total_frames}")

    # Allocate zarr arrays
    store = zarr.DirectoryStore(args.output_path)
    root = zarr.group(store=store, overwrite=True)
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    img_arr = data_group.create_dataset(
        "img",
        shape=(total_frames, args.img_size, args.img_size, 3),
        dtype=np.uint8,
        chunks=(256, args.img_size, args.img_size, 3),
    )
    action_arr = data_group.create_dataset(
        "action",
        shape=(total_frames, 14),
        dtype=np.float32,
        chunks=(256, 14),
    )
    state_arr = data_group.create_dataset(
        "state",
        shape=(total_frames, 14),
        dtype=np.float32,
        chunks=(256, 14),
    )

    # Second pass: fill data
    episode_ends = []
    offset = 0

    for ep_idx, pf in enumerate(parquet_files):
        ep_name = pf.stem  # e.g. episode_000042
        video_path = video_dir / f"{ep_name}.mp4"

        # Read parquet
        df = pd.read_parquet(pf)
        n = len(df)

        actions = np.stack(df["action"].values).astype(np.float32)       # (n, 14)
        states = np.stack(df["observation.state"].values).astype(np.float32)  # (n, 14)

        # Read video
        if video_path.exists():
            frames = read_video_frames(str(video_path), args.img_size)
            # Video may have slightly different frame count than parquet
            min_n = min(n, len(frames))
            if min_n < n:
                print(f"  Warning: {ep_name} video has {len(frames)} frames, parquet has {n}. Using {min_n}.")
                n = min_n
                actions = actions[:n]
                states = states[:n]
                frames = frames[:n]
        else:
            print(f"  Warning: video not found for {ep_name}, skipping")
            continue

        img_arr[offset:offset + n] = frames
        action_arr[offset:offset + n] = actions
        state_arr[offset:offset + n] = states

        offset += n
        episode_ends.append(offset)

        if (ep_idx + 1) % 10 == 0 or ep_idx == num_episodes - 1:
            print(f"  Processed {ep_idx + 1}/{num_episodes} episodes ({offset} frames)")

    # Trim if some episodes were skipped
    if offset < total_frames:
        print(f"Trimming arrays from {total_frames} to {offset} frames")
        img_arr.resize(offset, args.img_size, args.img_size, 3)
        action_arr.resize(offset, 14)
        state_arr.resize(offset, 14)

    meta_group.create_dataset("episode_ends", data=np.array(episode_ends, dtype=np.int64))

    # Also save action stats for normalization
    all_actions = action_arr[:]
    all_states = state_arr[:]
    stats = {
        "action": {
            "min": all_actions.min(axis=0).tolist(),
            "max": all_actions.max(axis=0).tolist(),
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
        },
        "state": {
            "min": all_states.min(axis=0).tolist(),
            "max": all_states.max(axis=0).tolist(),
        },
    }
    stats_path = args.output_path.replace(".zarr", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")

    # Also save as .npy for easy loading in inference
    np.save(args.output_path.replace(".zarr", "_stats.npy"), {
        "action": {
            "min": all_actions.min(axis=0),
            "max": all_actions.max(axis=0),
        },
        "state": {
            "min": all_states.min(axis=0),
            "max": all_states.max(axis=0),
        },
    })

    print(f"\nDone! Zarr dataset saved to {args.output_path}")
    print(f"  Episodes: {len(episode_ends)}")
    print(f"  Frames:   {offset}")
    print(f"  Image:    ({offset}, {args.img_size}, {args.img_size}, 3) uint8")
    print(f"  Action:   ({offset}, 14) float32")
    print(f"  State:    ({offset}, 14) float32")


if __name__ == "__main__":
    main()
