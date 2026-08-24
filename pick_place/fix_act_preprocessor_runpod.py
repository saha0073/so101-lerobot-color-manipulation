"""[DEPRECATED - no longer needed]

The rename fix is now handled on the client side in eval_act_runpod_pick_place.py
by setting client.policy_config.rename_map before start(). The server always
overrides the preprocessor rename_map with what the client sends, so patching
the cache has no effect.

Kept here for reference only.
"""

import json
import os
from huggingface_hub import hf_hub_download

MODEL = "subhodipsaha/act_pick_place_07_16"

path = hf_hub_download(MODEL, "policy_preprocessor.json")
print(f"Found: {path}")

cfg = json.load(open(path))

patched = False
for step in cfg["steps"]:
    if step["registry_name"] == "rename_observations_processor":
        step["config"]["rename_map"] = {"observation.state": "observation.environment_state"}
        patched = True
        break

if not patched:
    print("ERROR: rename_observations_processor step not found in config!")
    raise SystemExit(1)

# HF cache files are symlinks to read-only blobs — resolve before writing
real_path = os.path.realpath(path)
os.chmod(real_path, 0o644)
with open(real_path, "w") as f:
    json.dump(cfg, f, indent=2)

print("Patched: observation.state -> observation.environment_state")
print("\nNow start the server:")
print("  python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080 --fps=25")
