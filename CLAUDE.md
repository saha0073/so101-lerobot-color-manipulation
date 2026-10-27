# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a local checkout of **LeRobot** (v0.5.2) by Hugging Face — a PyTorch library for real-world robotics combining imitation learning, reinforcement learning, and vision-language-action models with hardware control. The actual source lives in `lerobot/`.

## Setup & Installation

```bash
# Development install (from lerobot/)
uv sync --locked --extra dev

# Install pre-commit hooks
pre-commit install

# Install with specific extras
uv sync --locked --extra test        # Test dependencies
pip install 'lerobot[all]'           # Everything
```

## Commands

### Testing

```bash
# From lerobot/
pytest -sv ./tests                                  # Full test suite
pytest -sv tests/test_specific_feature.py           # Single file
make DEVICE=cpu test-end-to-end                     # E2E (ACT, Diffusion, TDMPC, SmolVLA)
make DEVICE=cpu test-act-ete-train                  # Train ACT on test dataset
make DEVICE=cpu test-act-ete-eval                   # Evaluate ACT policy
```

### Linting & Quality

```bash
pre-commit run --all-files    # ruff format/lint, mypy, bandit, typos, prettier
```

### CLI Entry Points

```bash
lerobot-train          # Train a policy
lerobot-eval           # Evaluate a policy
lerobot-record         # Record robot demonstrations
lerobot-replay         # Replay recorded data
lerobot-teleoperate    # Teleoperate a robot
lerobot-calibrate      # Calibrate hardware
lerobot-find-cameras   # Discover connected cameras
lerobot-find-port      # Find serial ports for motors
lerobot-setup-motors   # Configure motor parameters
```

## Architecture

The codebase is under `lerobot/src/lerobot/`. Key modules:

| Module | Purpose |
|--------|---------|
| `scripts/` | CLI entry points (`lerobot_train.py`, `lerobot_eval.py`, `lerobot_record.py`, etc.) |
| `policies/` | 15+ policy implementations (ACT, Diffusion, SmolVLA, Pi0/Pi0.5, Groot, SAC, etc.) |
| `datasets/` | Dataset loading/creation; `lerobot_dataset.py` is the core class |
| `robots/` | Unified hardware interface for SO-100, SO-200, LeKiwi, Reachy2, Koch, etc. |
| `motors/` | Low-level motor drivers (Dynamixel, Feetech, Damiao, RobStride) |
| `cameras/` | Camera interfaces (OpenCV, RealSense, RealSense, ZMQ) |
| `teleoperators/` | Input devices: gamepads, keyboards, phones, leader arms |
| `processor/` | Observation/action preprocessing pipelines |
| `configs/` | Draccus-based configuration system; `default.py` defines top-level config |
| `envs/` | Gymnasium-compatible simulation wrappers (ALOHA, PUSHT, MetaWorld, LIBERO) |
| `common/` | Training loop utilities, wandb integration, control helpers |
| `async_inference/` | Policy server + client for decoupled inference |
| `transport/` | gRPC communication layer |
| `rl/` | Reinforcement learning base classes |

### Data Flow (Training)

```
lerobot-train --policy.type=act --dataset.repo_id=...
    │
    ├── Load config (draccus)
    ├── make_dataset()       → LeRobotDataset (Parquet + video/image features)
    ├── make_pre_post_processors() → image encoding, normalization
    ├── make_policy()        → instantiates the policy model (policies/factory.py)
    └── Training loop → batch → preprocess → forward → loss → backward → checkpoint
```

Policies are selected via `--policy.type=<name>` and instantiated through `policies/factory.py`. Each policy lives in its own subdirectory (e.g., `policies/act/`, `policies/diffusion/`).

### Configuration System

Uses **draccus** (dataclass-based). Top-level configs are in `configs/`. Training is configured via CLI flags that map to dataclass fields — no YAML/JSON config files by default.

### Dataset Format

Datasets follow the [LeRobot dataset format](https://github.com/huggingface/lerobot/blob/main/docs/source/dataset_format.mdx): Parquet files for metadata/actions/states, MP4 or image files for visual observations. Hosted on Hugging Face Hub or stored locally.

## Key Files

- `lerobot/pyproject.toml` — dependencies (organized as extras), tool configs (ruff, mypy, pytest, bandit), and all 18 CLI entry points
- `lerobot/.pre-commit-config.yaml` — pre-commit hook definitions
- `lerobot/Makefile` — end-to-end test targets (accepts `DEVICE=cpu` or `DEVICE=cuda`)
- `lerobot/tests/conftest.py` — pytest fixtures and configuration
- `commands.txt` — example training commands for reference

---

## Current Project State (as of 2026-10-27)

### Hardware

- **SO-101 follower arm** — on Windows: `COM3`, on Ubuntu: likely `/dev/ttyACM0`
- **SO-101 leader arm** (teleoperation) — on Windows: `COM4`, on Ubuntu: likely `/dev/ttyACM1`
- **Camera** — IP Webcam app (Android) over HTTP MJPEG: `http://192.168.1.2:8080/video`
  - Resolution set to **640×480** in the IP Webcam app
  - USB tethering (phone WiFi ON, laptop WiFi OFF) routes stream over USB → faster, ~8 Hz with ACT

### Completed Tasks

1. **Jar-push task** — 20 episodes recorded, jar placed at varying positions each episode
   - Dataset: `subhodipsaha/so101_jar_push` (on HuggingFace Hub)
   - Single task string: `"Push the jar"`

2. **ACT policy trained** — 10,000 steps, loss ~0.295
   - Checkpoint: `outputs/train/act_jar_push/checkpoints/010000/pretrained_model`
   - Inference script: `eval_jar_push.py`
   - Runs at ~8 Hz on RTX 3060 (Windows, with USB tethering)

3. **SmolVLA policy trained** — 5,000 steps on Google Colab T4, loss ~0.221
   - Model pushed to HuggingFace: `subhodipsaha/smolvla_jar_push`
   - Inference script: `eval_smolvla_jar_push.py`
   - Benchmark script: `benchmark_smolvla.py` (uses `lerobot/smolvla_base`, dummy inputs, measures Hz + GPU%)
   - **Windows (after torch fix)**: full model on GPU (`{'cuda'}`), 11–35% GPU utilization, ~1.8 Hz on RTX 3060 6 GB
   - Previous Windows issue (1.5 Hz, 0% GPU) was caused by `pip install -e ".[smolvla]"` pulling in CPU-only torch (`2.10.0+cpu`). Fixed by reinstalling CUDA torch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

4. **Reach-object task** (June 2025) — arm approaches object with gripper open, no grasping
   - Two cameras: phone (global, portrait) + wrist (USB, /dev/video0)
   - Scripts: `record_reach_object_06_25.py`, `record_reach_object_06_25_more.py`, `train_act_reach_object_06_25.py`, `eval_act_reach_object_06_25.py`

5. **Single-cup rotation task** (June 2025) — arm rotates toward a cup placed at varying positions
   - One red cup placed left/right at varying distances each episode
   - Scripts: `record_single_cup_06_22.py`, `train_act_single_cup.py`, `train_smolvla_single_cup.py`, `eval_smolvla_single_cup.py`, `eval_xvla_single_cup.py`

6. **Grab-and-release task** (July 2025) — arm grasps object then opens gripper to release
   - Two cameras: phone (global, portrait) + wrist (USB, /dev/video2)
   - Scripts: `record_grab_release_07_01.py`, `record_grab_release_07_01_more.py`, `train_act_grab_release_07_01.py`, `train_smolvla_grab_release_07_01.py`, `eval_act_grab_release_07_01.py`, `eval_smolvla_grab_release_07_01.py`

7. **Pick-and-place task** (July 2025) — pick object from one position, place at another
   - Evaluated ACT, SmolVLA, Pi0.5, MolmoAct2 on this task
   - RunPod used for Pi0.5 async inference (server/client architecture): `pick_place/run_pi05_client.py`, `pick_place/RUNPOD_SETUP.md`
   - ACT tuning notes: `n_action_steps=50` works better than 25; `num_steps=20` reduces jittering vs 10
   - See `pick_place/` folder for all scripts and notes

8. **Stack-three-cubes task** — MolmoAct2 v1 and v2
   - Dataset: `subhodipsaha/so101_stack_three_cubes_08_19` (119 episodes)
   - **Dataset breakdown:**
     - 59 episodes: full stack from scratch (varied cube rotation and position)
     - 14 episodes: 1 cube near target, 2 at initial positions
     - 21 episodes: 1 placed at target, 1 near target, 1 at initial
     - 25 episodes: 2 stacked, fixing the last cube near target
   - v1 (39 episodes, no recovery): doesn't recover from mid-task failures
   - v2 (119 episodes, recovery data): actively tries to correct mid-task
   - Fine-tuned model (v2): `subhodipsaha/molmoact2_stack_three_cubes_v2`
   - Base checkpoint: `allenai/MolmoAct2-SO100_101`
   - Scripts: `molmoact2/eval_molmoact2_so101.py`, `molmoact2/CALIBRATION.md`
   - **Key finding**: robot-specific pretraining (SO-101 pretrained base) outperformed Pi0.5 on the same task with same episode count

### Current Setup (Ubuntu)

Now running on Ubuntu with the SO-101 arms connected via udev symlinks (see port names table below). All inference scripts use Linux paths. Windows setup notes kept for reference but Ubuntu is the primary environment.

### Ubuntu Setup Checklist

```bash
# 1. Find serial ports (plug in arms, then check)
ls /dev/ttyUSB*
# Usually: follower = /dev/ttyACM0, leader = /dev/ttyACM1
# Add user to dialout group if permission denied:
sudo usermod -a -G dialout $USER   # then log out/in

# 2. Install conda/miniconda if not present
# https://docs.conda.io/en/latest/miniconda.html

# 3. Create environment
conda create -n lerobot python=3.12 -y
conda activate lerobot

# 4. Install lerobot with smolvla extra
# From the lerobot/ subdirectory of this repo:
cd /path/to/so101/lerobot
pip install -e ".[smolvla]"

# 5. Verify CUDA torch (must be CUDA build, not CPU-only)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.x.x+cu12x  True
# If CPU-only, reinstall:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 6. Verify accelerate sees GPU
python -c "from accelerate import Accelerator; a = Accelerator(); print(a.device)"
# Expected: cuda:0  (not cpu)

# 7. Update port names in eval scripts
# Edit eval_smolvla_jar_push.py and eval_jar_push.py:
#   port="COM3"  →  port="/dev/ttyACM0"   (follower)
#   port="COM4"  →  port="/dev/ttyACM1"   (leader)

# 8. Run SmolVLA inference
python eval_smolvla_jar_push.py
```

### Scripts in This Directory

| Script | Purpose |
|--------|---------|
| `record_jar_push.py` | Record jar-push episodes (leader arm teleoperation) |
| `record_reach_object_06_25.py` | Record reach-object episodes |
| `record_grab_release_07_01.py` | Record grab-and-release episodes |
| `record_single_cup_06_22.py` | Record single-cup rotation episodes |
| `train_jar_push.py` | Train ACT on local GPU |
| `train_smolvla_jar_push.py` | Train SmolVLA (better on Colab T4 — ~1h54m for 5k steps) |
| `train_act_reach_object_06_25.py` | Train ACT on reach-object task |
| `train_act_grab_release_07_01.py` | Train ACT on grab-and-release task |
| `train_smolvla_grab_release_07_01.py` | Train SmolVLA on grab-and-release task |
| `train_act_single_cup.py` | Train ACT on single-cup task |
| `train_smolvla_single_cup.py` | Train SmolVLA on single-cup task |
| `train_xvla_single_cup.py` | Train xVLA on single-cup task |
| `eval_jar_push.py` | Run ACT inference on follower arm |
| `eval_smolvla_jar_push.py` | Run SmolVLA inference on follower arm |
| `eval_act_reach_object_06_25.py` | Run ACT inference for reach-object |
| `eval_act_grab_release_07_01.py` | Run ACT inference for grab-and-release |
| `eval_smolvla_grab_release_07_01.py` | Run SmolVLA inference for grab-and-release |
| `eval_smolvla_single_cup.py` | Run SmolVLA inference for single-cup |
| `eval_xvla_single_cup.py` | Run xVLA inference for single-cup |
| `preextract_frames_jar_push.py` | Pre-extract video frames to JPEG cache for fast training |
| `preextract_frames.py` | General frame pre-extraction |
| `plot_jar_push.py` | Plot dataset quality (states, actions, images) |
| `benchmark_smolvla.py` | Benchmark SmolVLA Hz + GPU% |
| `benchmark_diffusion.py` | Benchmark diffusion policy |
| `benchmark_pi0fast.py` | Benchmark Pi0-fast |
| `benchmark_xvla.py` | Benchmark xVLA |
| `download_pi0fast.py` | Download Pi0-fast weights |
| `download_xvla.py` | Download xVLA weights |
| `livestream_camera.py` | Livestream USB wrist camera feed |
| `livestream_phone.py` | Livestream phone camera feed |

**Subdirectories:**

| Folder | Purpose |
|--------|---------|
| `pick_place/` | Pick-and-place task: ACT, SmolVLA, Pi0.5 async inference, MolmoAct2, RunPod setup |
| `molmoact2/` | MolmoAct2 stack-cubes: Colab fine-tune notebooks, inference, calibration notes |
| `mujoco_sim/` | MuJoCo SO-101 model, BC policy, sim-to-real exploration |
| `experiments/` | Camera calibration, visual odometry, sensor fusion experiments |
| `calibration/` | SO-101 follower and leader calibration JSON backups |

### Known Issues & Fixes

- **`observation.state` vs `observation.environment_state`**: ACT internally expects `observation.environment_state` but LeRobot stores `observation.state`. The `eval_jar_push.py` and `train_jar_push.py` scripts include a monkey-patch to rename the key. SmolVLA does NOT need this patch.
- **Windows symlink error during training**: `update_last_checkpoint` tries to create a symlink. Patched out in training scripts with `_train_module.update_last_checkpoint = lambda *args, **kwargs: None`. Not needed on Ubuntu (symlinks work natively).
- **HF cache conflict**: If a previous partial run left a dataset in `~/.cache/huggingface/lerobot/subhodipsaha/so101_jar_push`, delete it before re-recording: `rm -rf ~/.cache/huggingface/lerobot/subhodipsaha/so101_jar_push`
- **`video_utils.py` hardcoded cache path**: `lerobot/src/lerobot/datasets/video_utils.py` has `_cache_dir` set to `outputs/frame_cache/so101_jar_push`. If training on a new task, update this path to match the new task name.
- **Ubuntu port permissions**: If `Permission denied` on `/dev/ttyUSB*`, run `sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1` or add user to `dialout` group.

### Port Names: Windows → Ubuntu

| Device | Windows | Ubuntu |
|--------|---------|--------|
| Follower arm | `COM3` | `/dev/so101_follower` (symlink via udev, serial `5B3D048734`) |
| Leader arm | `COM4` | `/dev/so101_leader` (symlink via udev, serial `5B61034422`) |

Update `port=` in all scripts when switching OS. Also update `pretrained_path` to use Linux-style absolute paths if needed.
