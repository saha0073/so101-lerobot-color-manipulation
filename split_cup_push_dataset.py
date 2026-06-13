"""Split so101_cup_push into separate single-colour datasets.

Blue: episodes 30-59 + 90-119  →  subhodipsaha/so101_cup_push_blue
Red:  episodes 0-29  + 60-89   →  subhodipsaha/so101_cup_push_red

Usage:
    python split_cup_push_dataset.py
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

# ── Config ────────────────────────────────────────────────────────────────────
SRC = Path.home() / ".cache/huggingface/lerobot/subhodipsaha/so101_cup_push"
OUT_BASE = Path.home() / ".cache/huggingface/lerobot/subhodipsaha"

SPLITS = {
    "so101_cup_push_blue": {
        "episodes": list(range(30, 60)) + list(range(90, 120)),
        "task": "Move to the blue object",
        "repo_id": "subhodipsaha/so101_cup_push_blue",
        # old video file_index → new video file_index
        "video_file_map": {1: 0, 4: 1},
    },
    "so101_cup_push_red": {
        "episodes": list(range(0, 30)) + list(range(60, 90)),
        "task": "Move to the red object",
        "repo_id": "subhodipsaha/so101_cup_push_red",
        # old video file_index → new video file_index
        "video_file_map": {0: 0, 2: 1, 3: 2},
    },
}

# ── Load source data ──────────────────────────────────────────────────────────
print("Loading source data parquets...")
data_parts = []
for i in range(5):
    data_parts.append(pd.read_parquet(SRC / f"data/chunk-000/file-00{i}.parquet"))
df_src = pd.concat(data_parts, ignore_index=True)
df_src = df_src.sort_values("index").reset_index(drop=True)
print(f"  Total frames: {len(df_src)}, episodes: {df_src['episode_index'].nunique()}")

print("Loading source episodes metadata...")
eps_parts = []
for i in range(5):
    eps_parts.append(pd.read_parquet(SRC / f"meta/episodes/chunk-000/file-00{i}.parquet"))
df_eps_src = pd.concat(eps_parts, ignore_index=True)
df_eps_src = df_eps_src.sort_values("episode_index").reset_index(drop=True)

with open(SRC / "meta/info.json") as f:
    src_info = json.load(f)

with open(SRC / "meta/stats.json") as f:
    src_stats = json.load(f)


# ── Process each split ────────────────────────────────────────────────────────
for dataset_name, cfg in SPLITS.items():
    src_episodes = cfg["episodes"]
    task_str = cfg["task"]
    repo_id = cfg["repo_id"]
    video_file_map = cfg["video_file_map"]
    n_eps = len(src_episodes)

    print(f"\n{'='*60}")
    print(f"Building {dataset_name}  ({n_eps} episodes)")
    print(f"  Source episodes: {src_episodes[:3]}...{src_episodes[-3:]}")

    out_dir = OUT_BASE / dataset_name
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # ── 1. Filter + re-index data parquet ─────────────────────────────────────
    ep_map = {old: new for new, old in enumerate(src_episodes)}  # old→new ep idx
    df = df_src[df_src["episode_index"].isin(src_episodes)].copy()
    df["episode_index"] = df["episode_index"].map(ep_map)
    df["task_index"] = 0  # single task

    # Re-index globally from 0 (preserving per-episode frame_index order)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    df["index"] = df.index

    print(f"  Data: {len(df)} frames, {df['episode_index'].nunique()} episodes")

    # ── 2. Build episodes metadata ─────────────────────────────────────────────
    df_eps = df_eps_src[df_eps_src["episode_index"].isin(src_episodes)].copy()
    df_eps["episode_index"] = df_eps["episode_index"].map(ep_map)
    df_eps = df_eps.sort_values("episode_index").reset_index(drop=True)

    # Update task label
    df_eps["tasks"] = [[task_str]] * len(df_eps)

    # Update video file_index
    vid_col = "videos/observation.images.phone/file_index"
    df_eps[vid_col] = df_eps[vid_col].map(video_file_map)

    # Update data/file_index → all in file-000 now
    df_eps["data/file_index"] = 0

    # Update dataset_from_index / dataset_to_index from new data frame
    ep_bounds = (
        df.groupby("episode_index")["index"]
        .agg(["min", "max"])
        .reset_index()
    )
    ep_bounds.columns = ["episode_index", "from_idx", "to_idx"]
    df_eps = df_eps.merge(ep_bounds, on="episode_index", how="left")
    df_eps["dataset_from_index"] = df_eps["from_idx"].astype(int)
    df_eps["dataset_to_index"] = df_eps["to_idx"].astype(int) + 1
    df_eps.drop(columns=["from_idx", "to_idx"], inplace=True)

    # Update per-episode episode_index stats (they mirror episode_index value)
    for stat in ["min", "max", "mean", "q01", "q10", "q50", "q90", "q99"]:
        col = f"stats/episode_index/{stat}"
        if col in df_eps.columns:
            df_eps[col] = df_eps["episode_index"].astype(float)
    for stat in ["std"]:
        col = f"stats/episode_index/{stat}"
        if col in df_eps.columns:
            df_eps[col] = 0.0

    # Update index stats (global index range per episode)
    for stat in ["min", "q01", "q10"]:
        col = f"stats/index/{stat}"
        if col in df_eps.columns:
            df_eps[col] = df_eps["dataset_from_index"].astype(float)
    for stat in ["max", "q90", "q99"]:
        col = f"stats/index/{stat}"
        if col in df_eps.columns:
            df_eps[col] = (df_eps["dataset_to_index"] - 1).astype(float)
    for stat in ["mean", "q50"]:
        col = f"stats/index/{stat}"
        if col in df_eps.columns:
            df_eps[col] = ((df_eps["dataset_from_index"] + df_eps["dataset_to_index"] - 1) / 2.0)
    if "stats/index/std" in df_eps.columns:
        length = df_eps["dataset_to_index"] - df_eps["dataset_from_index"]
        df_eps["stats/index/std"] = (length / np.sqrt(12)).astype(float)

    # ── 3. Write data parquet ──────────────────────────────────────────────────
    data_out = out_dir / "data/chunk-000"
    data_out.mkdir(parents=True)
    df.to_parquet(data_out / "file-000.parquet", index=False)
    print(f"  Wrote data/chunk-000/file-000.parquet  ({len(df)} rows)")

    # ── 4. Write episodes metadata ─────────────────────────────────────────────
    eps_out = out_dir / "meta/episodes/chunk-000"
    eps_out.mkdir(parents=True)
    df_eps.to_parquet(eps_out / "file-000.parquet", index=False)
    print(f"  Wrote meta/episodes/chunk-000/file-000.parquet  ({len(df_eps)} rows)")

    # ── 5. Copy video files ────────────────────────────────────────────────────
    vid_src = SRC / "videos/observation.images.phone/chunk-000"
    vid_out = out_dir / "videos/observation.images.phone/chunk-000"
    vid_out.mkdir(parents=True)
    for old_idx, new_idx in sorted(video_file_map.items()):
        src_mp4 = vid_src / f"file-{old_idx:03d}.mp4"
        dst_mp4 = vid_out / f"file-{new_idx:03d}.mp4"
        shutil.copy2(str(src_mp4), str(dst_mp4))
        size_mb = dst_mp4.stat().st_size / 1e6
        print(f"  Copied video file-{old_idx:03d}.mp4 → file-{new_idx:03d}.mp4  ({size_mb:.1f} MB)")

    # ── 6. Write tasks.parquet ─────────────────────────────────────────────────
    df_tasks = pd.DataFrame({"task": [task_str], "task_index": [0]}).set_index("task")
    meta_out = out_dir / "meta"
    meta_out.mkdir(exist_ok=True)
    df_tasks.to_parquet(meta_out / "tasks.parquet")
    print(f"  Wrote meta/tasks.parquet")

    # ── 7. Write info.json ─────────────────────────────────────────────────────
    total_frames = int(len(df))
    info = dict(src_info)
    info["total_episodes"] = n_eps
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    info["splits"] = {"train": f"0:{n_eps}"}
    with open(meta_out / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    print(f"  Wrote meta/info.json  (total_episodes={n_eps}, total_frames={total_frames})")

    # ── 8. Write stats.json (reuse source stats — same robot, same motions) ───
    shutil.copy2(str(SRC / "meta/stats.json"), str(meta_out / "stats.json"))
    print(f"  Copied meta/stats.json from source")

    # ── 9. Push to HuggingFace ─────────────────────────────────────────────────
    print(f"\n  Pushing {repo_id} to HuggingFace Hub...")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Done: https://huggingface.co/datasets/{repo_id}")

print("\nAll done!")
