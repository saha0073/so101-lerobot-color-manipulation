# RunPod Setup for Async Inference

## Every new pod: create venv in /workspace (survives pod restarts on same volume)

```bash
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
```

## ACT / SmolVLA install

```bash
pip install 'lerobot[async]'
huggingface-cli login   # paste token from https://huggingface.co/settings/tokens
```

## Pi0.5 install

```bash
pip install 'lerobot[pi0]' 'lerobot[pi]' 'lerobot[async]'
huggingface-cli login
```

---

## Pi0.5 — requires one patch before starting server

Make sure port 8080 is exposed when creating the pod (check TCP Port Mappings in RunPod dashboard).

### Patch: Fix vision tower key mismatch

The Pi0.5 base checkpoint uses `vision_tower.vision_model.*` key paths but the current
LeRobot code expects `vision_tower.*`. Without this patch the entire vision encoder loads
with random weights → robot shakes at inference.

```bash
python3 -c "
path = '/usr/local/lib/python3.12/dist-packages/lerobot/policies/pi05/modeling_pi05.py'
src = open(path).read()
target = '            # Handle vision tower embedding layer potential differences'
fix = '            if \"vision_tower.vision_model.\" in new_key:\n                new_key = new_key.replace(\"vision_tower.vision_model.\", \"vision_tower.\")\n\n'
if 'new_key.replace(\"vision_tower.vision_model.\"' not in src:
    open(path, 'w').write(src.replace(target, fix + target))
    print('Vision tower key fix applied')
else:
    print('Already patched')
"
```

### Start server

On RunPod:
```bash
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080 --fps=25
```

Update `RUNPOD_SERVER` in `config.py` with the pod's TCP port for :8080, then locally:
```bash
python stack_cubes/eval_pi05_runpod_stack_three_cubes_08_19.py
```

---

## SmolVLA — no patches needed

```bash
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080 --fps=25
```

Locally:
```bash
python pick_place/eval_smolvla_runpod_pick_place.py
```

---

## ACT — requires two patches before starting server

### Patch 1: Remove `observation.environment_state` from the normalizer

The ACT model was trained with `observation.environment_state` as a feature type ENV,
but the normalizer on PyPI crashes if that key is missing from the observation.
Fix: remove it from the normalizer's features in the cached preprocessor config.

```bash
# Step 1: find the cached file (hash may differ on each pod)
find /workspace/.cache/huggingface/hub/models--subhodipsaha--act_pick_place_07_16/ \
     -name "policy_preprocessor.json" | head -5

# Check which snapshot refs/main points to:
cat /workspace/.cache/huggingface/hub/models--subhodipsaha--act_pick_place_07_16/refs/main

# Step 2: patch that snapshot (replace HASH with the hash from refs/main)
python3 -c "
import json, os
HASH = 'c9d30aee0c57e7a1b7bd31183a6ab5beaedbf126'  # update if different
path = f'/workspace/.cache/huggingface/hub/models--subhodipsaha--act_pick_place_07_16/snapshots/{HASH}/policy_preprocessor.json'
real_path = os.path.realpath(path)
cfg = json.load(open(real_path))
for step in cfg['steps']:
    if step['registry_name'] == 'normalizer_processor':
        removed = step['config']['features'].pop('observation.environment_state', None)
        print('Removed:', removed)
        break
os.chmod(real_path, 0o644)
with open(real_path, 'w') as f:
    json.dump(cfg, f, indent=2)
print('Patch 1 done')
"
```

### Patch 2: Fix `batch[OBS_STATE].device` in modeling_act.py

The PyPI version of ACT hardcodes `batch['observation.state']` for device lookup,
but after rename the key is `observation.environment_state`. Fix:

```bash
python3 -c "
path = '/usr/local/lib/python3.12/dist-packages/lerobot/policies/act/modeling_act.py'
src = open(path).read()
old = 'batch[OBS_STATE].device'
new = 'batch.get(OBS_ENV_STATE, batch.get(OBS_STATE)).device'
assert old in src, 'Pattern not found — already patched or version changed'
open(path, 'w').write(src.replace(old, new))
print('Patch 2 done')
"
```

### Start server and run client

On RunPod:
```bash
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080 --fps=25
```

Update `RUNPOD_SERVER` in `config.py` with the new pod's TCP port for :8080, then locally:
```bash
python pick_place/eval_act_runpod_pick_place.py
```

---

## Updating config.py when pod changes

The TCP port mapping changes every time a pod is restarted/recreated.
Check RunPod dashboard → Connect → TCP ports, then update:

```python
# config.py
RUNPOD_SERVER = "69.30.85.220:22171"   # ← update this
```

## Notes

- Always STOP (not TERMINATE) the pod to preserve the HF model cache (~500MB for ACT).
- Terminating loses the cache; next run re-downloads and needs patches re-applied.
- SmolVLA cache is larger (~2GB); same rule applies.
- Idle disk cost: ~$0.014/hr ($0.33/day) on RunPod.
