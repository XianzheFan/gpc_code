"""
Merge multiple zarr datasets (expert + exploration) into one for world model training.

Usage:
    python merge_zarr_datasets.py \
        --inputs data/fractal_google.zarr data/fractal_google_explore.zarr \
        --output data/fractal_google_combined.zarr
"""

import argparse
import numpy as np
import zarr


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple zarr datasets into one")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input zarr dataset paths")
    parser.add_argument("--output", required=True,
                        help="Output merged zarr path")
    args = parser.parse_args()

    # First pass: count total frames and collect metadata
    total_frames = 0
    datasets = []
    for path in args.inputs:
        root = zarr.open(path, mode="r")
        ep_ends = root["meta"]["episode_ends"][:]
        n_frames = ep_ends[-1]
        n_episodes = len(ep_ends)
        img_shape = root["data"]["img"].shape[1:]  # (H, W, 3)
        action_dim = root["data"]["action"].shape[1]
        datasets.append({
            "path": path,
            "root": root,
            "n_frames": n_frames,
            "n_episodes": n_episodes,
            "episode_ends": ep_ends,
        })
        total_frames += n_frames
        print(f"  {path}: {n_episodes} episodes, {n_frames} frames")

    total_episodes = sum(d["n_episodes"] for d in datasets)
    print(f"\nTotal: {total_episodes} episodes, {total_frames} frames")

    # Create output zarr
    out = zarr.open_group(args.output, mode="w")
    data_group = out.create_group("data")
    meta_group = out.create_group("meta")

    img_arr = data_group.create_array(
        "img", shape=(total_frames, *img_shape),
        dtype=np.uint8, chunks=(256, *img_shape),
    )
    action_arr = data_group.create_array(
        "action", shape=(total_frames, action_dim),
        dtype=np.float32, chunks=(256, action_dim),
    )
    # Copy state array if present in first dataset
    has_state = "state" in datasets[0]["root"]["data"]
    if has_state:
        state_dim = datasets[0]["root"]["data"]["state"].shape[1]
        state_arr = data_group.create_array(
            "state", shape=(total_frames, state_dim),
            dtype=np.float32, chunks=(256, state_dim),
        )

    # Second pass: copy data
    frame_offset = 0
    all_episode_ends = []

    for i, d in enumerate(datasets):
        root = d["root"]
        n = d["n_frames"]

        print(f"Copying {d['path']} ({n} frames)...")

        # Copy in chunks to avoid memory issues
        chunk_size = 10000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            img_arr[frame_offset + start:frame_offset + end] = \
                root["data"]["img"][start:end]
            action_arr[frame_offset + start:frame_offset + end] = \
                root["data"]["action"][start:end]
            if has_state and "state" in root["data"]:
                state_arr[frame_offset + start:frame_offset + end] = \
                    root["data"]["state"][start:end]

        # Shift episode ends by frame offset
        shifted_ends = d["episode_ends"] + frame_offset
        all_episode_ends.extend(shifted_ends.tolist())

        frame_offset += n

    meta_group.create_array(
        "episode_ends",
        data=np.array(all_episode_ends, dtype=np.int64),
    )

    print(f"\nMerged dataset saved to {args.output}")
    print(f"  Episodes: {len(all_episode_ends)}")
    print(f"  Frames:   {frame_offset}")


if __name__ == "__main__":
    main()
