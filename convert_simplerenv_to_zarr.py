"""
Convert SimplerEnv LeRobot dataset (parquet + mp4) to zarr format
for GPC world model training.

Input (produced by collect_finetune_dreamdojo_data.py):
    <dataset_dir>/
        data/chunk-000/episode_XXXXXX.parquet   (action 7D, observation.state 8D)
        videos/chunk-000/
            observation.images.image/           (Google Fractal)
                episode_XXXXXX.mp4
            observation.images.image_0/         (WidowX Bridge)
                episode_XXXXXX.mp4
        meta/stats.json  (optional, for reference)

Output:
    <output_path>.zarr/
        data/
            img      (N, 96, 96, 3) uint8   — camera frames
            action   (N, 7)         float32  — 7D EE actions
            state    (N, 8)         float32  — 8D EE states
        meta/
            episode_ends  (num_episodes,) int64

    <output_path>_stats.npy   — action normalization stats for inference

Usage:
    # Google Fractal data
    python convert_simplerenv_to_zarr.py \
        --dataset_dir data/simpler_env_dreamdojo \
        --output_path data/simpler_env_google.zarr \
        --robot_type google \
        --img_size 96

    # WidowX Bridge data
    python convert_simplerenv_to_zarr.py \
        --dataset_dir data/simpler_env_dreamdojo \
        --output_path data/simpler_env_widowx.zarr \
        --robot_type widowx \
        --img_size 96

    # Only convert successful episodes
    python convert_simplerenv_to_zarr.py \
        --dataset_dir data/simpler_env_dreamdojo \
        --output_path data/simpler_env_google_success.zarr \
        --robot_type google \
        --only_success
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


# SimplerEnv constants
ACTION_DIM = 7   # x, y, z, roll, pitch, yaw, gripper
STATE_DIM = 8    # Google: eef(3)+quat(4)+gripper(1)  WidowX: eef(3)+rpy(3)+pad(1)+gripper(1)

# Camera video key per robot type (must match collect_finetune_dreamdojo_data.py)
CAMERA_KEY = {
    "google": "observation.images.image",
    "widowx": "observation.images.image_0",
}


def read_video_frames(video_path: str, img_size: int) -> np.ndarray:
    """Read all frames from mp4, resize to (img_size, img_size, 3) uint8 RGB.

    Uses system ffmpeg via subprocess to avoid OpenCV AV1 decoding issues.
    """
    # Use ffmpeg to decode video and scale to target size, output raw RGB24
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


def discover_episodes(dataset_dir: Path, camera_key: str, only_success: bool):
    """Find all episode parquet+video pairs and return sorted list of
    (parquet_path, video_path) tuples."""
    episodes = []

    # Scan all chunk directories
    data_root = dataset_dir / "data"
    video_root = dataset_dir / "videos"

    for chunk_dir in sorted(data_root.glob("chunk-*")):
        chunk_name = chunk_dir.name  # e.g. "chunk-000"
        for pq_path in sorted(chunk_dir.glob("episode_*.parquet")):
            ep_name = pq_path.stem  # e.g. "episode_000042"
            vid_path = video_root / chunk_name / camera_key / f"{ep_name}.mp4"

            # Filter by success if requested
            if only_success:
                df = pd.read_parquet(pq_path)
                if "success" in df.columns and not bool(df["success"].iloc[-1]):
                    continue

            episodes.append((pq_path, vid_path))

    return episodes


def main():
    parser = argparse.ArgumentParser(
        description="Convert SimplerEnv LeRobot dataset to zarr for GPC world model training")
    parser.add_argument("--dataset_dir", required=True,
                        help="Path to SimplerEnv LeRobot dataset (output of collect_finetune_dreamdojo_data.py)")
    parser.add_argument("--output_path", required=True,
                        help="Output zarr path (e.g. data/simpler_env_google.zarr)")
    parser.add_argument("--robot_type", required=True, choices=["google", "widowx"],
                        help="Robot type (determines camera key)")
    parser.add_argument("--img_size", type=int, default=96,
                        help="Resize images to this resolution (default: 96)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit number of episodes (default: all)")
    parser.add_argument("--only_success", action="store_true",
                        help="Only include successful episodes")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    camera_key = CAMERA_KEY[args.robot_type]

    # Discover episodes
    episodes = discover_episodes(dataset_dir, camera_key, args.only_success)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]

    num_episodes = len(episodes)
    if num_episodes == 0:
        print("No episodes found. Check --dataset_dir and --robot_type.")
        return
    print(f"Found {num_episodes} episodes (robot_type={args.robot_type}, "
          f"only_success={args.only_success})")

    # First pass: count total frames
    total_frames = 0
    episode_lengths = []
    for pq_path, _ in episodes:
        df = pd.read_parquet(pq_path)
        episode_lengths.append(len(df))
        total_frames += len(df)

    print(f"Total frames: {total_frames}")

    # Allocate zarr arrays
    root = zarr.open_group(args.output_path, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    img_arr = data_group.create_array(
        "img",
        shape=(total_frames, args.img_size, args.img_size, 3),
        dtype=np.uint8,
        chunks=(256, args.img_size, args.img_size, 3),
    )
    action_arr = data_group.create_array(
        "action",
        shape=(total_frames, ACTION_DIM),
        dtype=np.float32,
        chunks=(256, ACTION_DIM),
    )
    state_arr = data_group.create_array(
        "state",
        shape=(total_frames, STATE_DIM),
        dtype=np.float32,
        chunks=(256, STATE_DIM),
    )

    # Second pass: fill data
    episode_ends = []
    offset = 0
    skipped = 0

    for ep_idx, (pq_path, vid_path) in enumerate(episodes):
        df = pd.read_parquet(pq_path)
        n = len(df)

        # Extract actions and states from parquet
        actions = np.stack(df["action"].values).astype(np.float32)               # (n, 7)
        states = np.stack(df["observation.state"].values).astype(np.float32)     # (n, 8)

        # Read video frames
        if not vid_path.exists():
            print(f"  Warning: video not found for {pq_path.stem}, skipping")
            skipped += 1
            continue

        frames = read_video_frames(str(vid_path), args.img_size)

        # Align frame counts (video may differ slightly from parquet)
        min_n = min(n, len(frames))
        if min_n < n:
            print(f"  Warning: {pq_path.stem} video has {len(frames)} frames, "
                  f"parquet has {n}. Using {min_n}.")
            n = min_n
            actions = actions[:n]
            states = states[:n]
            frames = frames[:n]

        img_arr[offset:offset + n] = frames
        action_arr[offset:offset + n] = actions
        state_arr[offset:offset + n] = states

        offset += n
        episode_ends.append(offset)

        if (ep_idx + 1) % 20 == 0 or ep_idx == num_episodes - 1:
            print(f"  Processed {ep_idx + 1}/{num_episodes} episodes ({offset} frames)")

    # Trim if some episodes were skipped
    if offset < total_frames:
        print(f"Trimming arrays from {total_frames} to {offset} frames "
              f"({skipped} episodes skipped)")
        img_arr.resize(offset, args.img_size, args.img_size, 3)
        action_arr.resize(offset, ACTION_DIM)
        state_arr.resize(offset, STATE_DIM)

    ep_ends_data = np.array(episode_ends, dtype=np.int64)
    meta_group.create_array("episode_ends", data=ep_ends_data)

    # Compute and save action/state stats for normalization
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
            "mean": all_states.mean(axis=0).tolist(),
            "std": all_states.std(axis=0).tolist(),
        },
    }

    # Save as JSON (human-readable)
    stats_json_path = args.output_path.replace(".zarr", "_stats.json")
    with open(stats_json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats (JSON) saved to {stats_json_path}")

    # Save as .npy (for train_world_model_agilex.py --action_stats_path)
    stats_npy_path = args.output_path.replace(".zarr", "_stats.npy")
    np.save(stats_npy_path, {
        "action": {
            "min": all_actions.min(axis=0),
            "max": all_actions.max(axis=0),
        },
        "state": {
            "min": all_states.min(axis=0),
            "max": all_states.max(axis=0),
        },
    })
    print(f"Stats (npy) saved to {stats_npy_path}")

    print(f"\nDone! Zarr dataset saved to {args.output_path}")
    print(f"  Episodes:   {len(episode_ends)}")
    print(f"  Frames:     {offset}")
    print(f"  Image:      ({offset}, {args.img_size}, {args.img_size}, 3) uint8")
    print(f"  Action:     ({offset}, {ACTION_DIM}) float32")
    print(f"  State:      ({offset}, {STATE_DIM}) float32")
    print(f"  Robot type: {args.robot_type}")


if __name__ == "__main__":
    main()
