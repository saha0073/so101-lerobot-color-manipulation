"""Merge local parquet files and upload so101_cup_push dataset to HuggingFace.

Local cache has 5 data + 5 episode parquet files (due to multiple recording sessions).
This script merges them into single files and uploads everything cleanly.
Videos stay as separate files (0-4) since episodes reference them by file_index.
"""

import json
import io
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "subhodipsaha/so101_cup_push"
TAG = "v3.0"
LOCAL = Path("~/.cache/huggingface/lerobot/subhodipsaha/so101_cup_push").expanduser()

api = HfApi()

# ── 1. Merge data parquets ────────────────────────────────────────────────────
print("Merging data parquets...")
data_files = sorted((LOCAL / "data/chunk-000").glob("*.parquet"))
data = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
print(f"  Total frames: {len(data)}, task_index counts: {data['task_index'].value_counts().to_dict()}")

# ── 2. Merge episodes parquets ────────────────────────────────────────────────
print("Merging episodes parquets...")
eps_files = sorted((LOCAL / "meta/episodes/chunk-000").glob("*.parquet"))
eps = pd.concat([pd.read_parquet(f) for f in eps_files], ignore_index=True)
print(f"  Total episodes: {len(eps)}")

# Update data/file_index to 0 (all frames now in file-000)
eps["data/file_index"] = 0
eps["data/chunk_index"] = 0
# meta/episodes file_index also 0
eps["meta/episodes/file_index"] = 0
eps["meta/episodes/chunk_index"] = 0

from collections import Counter
tasks_flat = [t for arr in eps["tasks"] for t in arr]
print(f"  Task counts: {Counter(tasks_flat)}")

# ── 3. Update info.json ───────────────────────────────────────────────────────
print("Updating info.json...")
info_path = LOCAL / "meta/info.json"
with open(info_path) as f:
    info = json.load(f)
info["total_episodes"] = len(eps)
info["total_frames"] = len(data)
print(f"  total_episodes={info['total_episodes']}, total_frames={info['total_frames']}")

# ── 4. Upload merged data parquet ─────────────────────────────────────────────
print("Uploading data/chunk-000/file-000.parquet...")
buf = io.BytesIO()
data.to_parquet(buf, index=False)
buf.seek(0)
api.upload_file(path_or_fileobj=buf, path_in_repo="data/chunk-000/file-000.parquet",
    repo_id=REPO_ID, repo_type="dataset", commit_message="Upload merged data parquet (120 eps)")

# Delete old data files 1-4
for i in range(1, len(data_files)):
    print(f"  Deleting data/chunk-000/file-{i:03d}.parquet...")
    api.delete_file(path_in_repo=f"data/chunk-000/file-{i:03d}.parquet",
        repo_id=REPO_ID, repo_type="dataset")

# ── 5. Upload merged episodes parquet ─────────────────────────────────────────
print("Uploading meta/episodes/chunk-000/file-000.parquet...")
buf = io.BytesIO()
eps.to_parquet(buf, index=False)
buf.seek(0)
api.upload_file(path_or_fileobj=buf, path_in_repo="meta/episodes/chunk-000/file-000.parquet",
    repo_id=REPO_ID, repo_type="dataset", commit_message="Upload merged episodes parquet (120 eps)")

# Delete old episodes files 1-4
for i in range(1, len(eps_files)):
    print(f"  Deleting meta/episodes/chunk-000/file-{i:03d}.parquet...")
    api.delete_file(path_in_repo=f"meta/episodes/chunk-000/file-{i:03d}.parquet",
        repo_id=REPO_ID, repo_type="dataset")

# ── 6. Upload video files ──────────────────────────────────────────────────────
video_dir = LOCAL / "videos/observation.images.phone/chunk-000"
video_files = sorted(video_dir.glob("*.mp4"))
print(f"Uploading {len(video_files)} video files...")
for vf in video_files:
    repo_path = f"videos/observation.images.phone/chunk-000/{vf.name}"
    print(f"  Uploading {repo_path} ({vf.stat().st_size / 1e6:.1f} MB)...")
    api.upload_file(path_or_fileobj=vf, path_in_repo=repo_path,
        repo_id=REPO_ID, repo_type="dataset", commit_message=f"Upload {vf.name}")

# ── 7. Upload info.json ───────────────────────────────────────────────────────
print("Uploading meta/info.json...")
api.upload_file(
    path_or_fileobj=io.BytesIO(json.dumps(info, indent=2).encode()),
    path_in_repo="meta/info.json",
    repo_id=REPO_ID, repo_type="dataset", commit_message="Update info.json for 120 episodes")

# ── 8. Move v3.0 tag ──────────────────────────────────────────────────────────
sha = api.repo_info(repo_id=REPO_ID, repo_type="dataset").sha
api.delete_tag(REPO_ID, tag=TAG, repo_type="dataset")
api.create_tag(REPO_ID, tag=TAG, revision=sha, repo_type="dataset")
print(f"Tag {TAG} -> {sha}")
print("\nDone! Dataset has 120 episodes.")
