"""One-off script: write the missing episode_ends into an existing zarr."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from convert_simplerenv_to_zarr import CAMERA_KEY, discover_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_path", required=True, help="Existing zarr path")
    parser.add_argument("--robot_type", required=True, choices=["google", "widowx"])
    parser.add_argument("--only_success", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()

    camera_key = CAMERA_KEY[args.robot_type]
    episodes = discover_episodes(Path(args.dataset_dir), camera_key, args.only_success)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    # Reconstruct episode_ends from parquet lengths
    # (same logic the main script used, minus video reading)
    offset = 0
    episode_ends = []
    skipped = 0
    for pq_path, vid_path in episodes:
        if not vid_path.exists():
            skipped += 1
            continue
        df = pd.read_parquet(pq_path)
        n = len(df)
        offset += n
        episode_ends.append(offset)

    ep_ends_data = np.array(episode_ends, dtype=np.int64)

    root = zarr.open_group(args.output_path, mode="r+")
    meta_group = root["meta"]
    meta_group.create_array("episode_ends", data=ep_ends_data)

    print(f"Written episode_ends: {len(episode_ends)} episodes, last offset={offset}")
    print(f"Skipped {skipped} episodes (missing video)")


if __name__ == "__main__":
    main()
