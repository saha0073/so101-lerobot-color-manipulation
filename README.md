# SO-101 Manipulation with ACT, SmolVLA, Pi0.5 and MolmoAct2

Imitation learning experiments on an SO-101 robot arm using ACT, SmolVLA, Pi0.5 and MolmoAct2 policies. Three tasks:

1. **Color-conditioned cup push** — arm moves toward a specific colored cup (blue or red) from varying positions
2. **Pick and place** — arm picks up an object and places it at a target location, object at a different position every episode
3. **Stack three cubes** — arm picks up three cubes one by one and stacks them into a tower, cubes placed at slightly different positions each episode

---

## Task 2: Pick and Place

Pick up an object and place it at a target location. Object placed at a different position each episode to test spatial generalization.

**New vs Task 1:**
- Wrist camera added — close-up gripper view needed for grasp timing
- Uniform yellow backdrop — improves generalization across rooms/environments

### Datasets

| Dataset | Episodes | HuggingFace |
|---------|----------|-------------|
| Pick and place | 40 | [so101_pick_place_07_16](https://huggingface.co/datasets/subhodipsaha/so101_pick_place_07_16) |

### Models

| Model | Policy | HuggingFace |
|-------|--------|-------------|
| ACT | ACT (deterministic) | [act_pick_place_07_16](https://huggingface.co/subhodipsaha/act_pick_place_07_16) |
| SmolVLA | SmolVLA (flow matching) | [smolvla_pick_place_07_16](https://huggingface.co/subhodipsaha/smolvla_pick_place_07_16) |
| MolmoAct2 | MolmoAct2 (action expert fine-tune) | [molmoact2_pick_place](https://huggingface.co/subhodipsaha/molmoact2_pick_place) |

### Scripts

```bash
# Record episodes (leader arm teleoperation)
python pick_place/record_pick_place_07_16.py

# Train (local)
python pick_place/train_act_pick_place_07_16.py       # ~1h44m on RTX 3060
python pick_place/train_smolvla_pick_place_07_16.py   # ~2h on RTX 3060

# Fine-tune MolmoAct2 (Colab L4)
# open pick_place/molmoact_finetune_colab.ipynb

# Inference (ACT / SmolVLA — local)
python pick_place/eval_act_pick_place_07_16.py
python pick_place/eval_smolvla_pick_place_07_16.py

# Inference (MolmoAct2 — requires policy server on Colab L4)
# 1. Run pick_place/molmoact_finetuned_inference_colab.ipynb on Colab L4
# 2. Paste the ngrok URL into config.py as MOLMOACT2_SERVER
python pick_place/eval_molmoact2_pick_place.py
```

### Training Details

| | ACT | SmolVLA | MolmoAct2 |
|--|-----|---------|-----------|
| Dataset | 40 episodes | 40 episodes | 40 episodes |
| Cameras | phone + wrist | phone + wrist | phone + wrist |
| Steps | 20,000 | 20,000 | 2,500 |
| Batch size | 8 | 4 | 16 |
| Mixed precision | BF16 | BF16 | BF16 |
| Trainable params | ~90M | ~450M | ~1B / 7B (action expert, VLM frozen) |
| Training hardware | RTX 3060 (local) | RTX 3060 (local) | Colab L4 |
| Final loss | ~5.0 | ~0.047 | ~0.055 (flow matching) |
| Training time | ~1h44m | ~2h | ~2h |

### MolmoAct2 Async Inference Setup

MolmoAct2 (7B params) is too large to run locally alongside the robot. Uses a FastAPI server on Colab L4 with ngrok HTTP tunnel:

- **Policy server**: `pick_place/molmoact_finetuned_inference_colab.ipynb` — loads the fine-tuned model, injects QUANTILE normalization stats, serves `/predict` returning 30-step action chunks
- **Robot client**: `pick_place/eval_molmoact2_pick_place.py` — sends phone + wrist images and joint state, executes returned action chunks at 25 Hz

Update `MOLMOACT2_SERVER` in `config.py` with the ngrok URL each session.

**Key implementation details:**
- Base model processor loaded from `allenai/MolmoAct2-SO100_101` (fine-tuned repo has no processor files)
- QUANTILE normalization stats from the pick-place dataset are injected at inference time as a custom `norm_tag`
- Server returns 30-step chunks at ~1.1s latency; client uses look-ahead state (last queued action) for the next chunk request to improve trajectory continuity

### Results

ACT: smooth, stable, picks consistently from varied positions.
SmolVLA: works but shows more jitter in the trajectory.
MolmoAct2: follows the correct motion trajectory; chunk-boundary retraction present due to 1.1s server latency.

Same pattern as Task 1 — ACT more reliable on precision tasks with small datasets. MolmoAct2 shows promise as a general-purpose VLA but async inference latency limits trajectory continuity.

---

## Task 3: Stack Three Cubes

Pick up three cubes one by one and stack them into a tower. Significantly harder than pick-and-place — requires three sequential grasps and recovering if any cube slips.

Trained ACT locally. Fine-tuned Pi0.5 (4B) and MolmoAct2 (7B) on Colab with async inference. MolmoAct2 v2 uses a 119-episode dataset with deliberate recovery scenarios (60 recovery episodes) and outperforms Pi0.5.

### Datasets

| Dataset | Episodes | HuggingFace |
|---------|----------|-------------|
| Stack three cubes v2 (with recovery) | 119 | [so101_stack_three_cubes_08_19](https://huggingface.co/datasets/subhodipsaha/so101_stack_three_cubes_08_19) |

### Models

| Model | Policy | HuggingFace |
|-------|--------|-------------|
| ACT | ACT | [act_stack_three_cubes_08_19](https://huggingface.co/subhodipsaha/act_stack_three_cubes_08_19) |
| Pi0.5 v2 | Pi0.5 (action expert only) | [pi05_stack_three_cubes_08_19_v2](https://huggingface.co/subhodipsaha/pi05_stack_three_cubes_08_19_v2) |
| MolmoAct2 v1 | MolmoAct2 (39 eps, no recovery) | [molmoact2_stack_three_cubes](https://huggingface.co/subhodipsaha/molmoact2_stack_three_cubes) |
| MolmoAct2 v2 | MolmoAct2 (119 eps, recovery data) | [molmoact2_stack_three_cubes](https://huggingface.co/subhodipsaha/molmoact2_stack_three_cubes) |

### Scripts

```bash
# Record episodes
python stack_cubes/record_stack_three_cubes_08_19.py

# Train
python stack_cubes/train_act_stack_three_cubes_08_19.py        # ~1h45m on RTX 3060
python stack_cubes/train_pi05_runpod_stack_three_cubes_08_19.py  # ~8h on Colab L4

# Fine-tune MolmoAct2 (Colab L4)
# open stack_cubes/molmoact_stack_cubes_finetune_colab.ipynb

# Inference (ACT — local)
python stack_cubes/eval_act_stack_three_cubes_08_19.py

# Inference (Pi0.5 — async, requires policy server running on Colab/RunPod)
python stack_cubes/eval_pi05_runpod_stack_three_cubes_08_19.py

# Inference (MolmoAct2 — requires policy server on Colab L4)
# 1. Run stack_cubes/molmoact_stack_cubes_inference_colab.ipynb on Colab L4
# 2. Paste the ngrok URL into config.py as MOLMOACT2_STACK_SERVER
python stack_cubes/eval_molmoact2_stack_three_cubes.py
python stack_cubes/eval_molmoact2_stack_three_cubes.py --record   # saves phone + wrist MP4 per episode
```

### Training Details

| | ACT | Pi0.5 | MolmoAct2 v1 | MolmoAct2 v2 |
|--|-----|-------|--------------|--------------|
| Dataset | 40 eps | 40 eps | 39 eps | 119 eps (60 recovery) |
| Cameras | phone + wrist | phone + wrist | phone + wrist | phone + wrist |
| Steps | 20,000 | 5,000 | 2,500 | 5,000 |
| Batch size | 8 | 16 | 16 | 16 |
| Mixed precision | BF16 | BF16 | BF16 | BF16 |
| Trainable params | ~90M | 693M / 4B | ~1B / 7B | ~1B / 7B |
| Training hardware | RTX 3060 | Colab L4 | Colab L4 | Colab L4 |
| Training time | ~1h45m | ~8h | ~6h | ~12h |
| Final loss | ~3.9 (L1) | ~0.050 (flow matching) | ~0.019 (flow matching) | ~0.021 (flow matching) |

Pi0.5 uses PaliGemma as VLM backbone. VLM frozen, only action expert fine-tuned. Gradient checkpointing used to reduce memory from ~37 GB to ~15 GB.

### Pi0.5 Async Inference Setup

Pi0.5 (4B params) is too large to run alongside the robot locally. Uses LeRobot's async inference architecture:
- **Policy server**: runs on Colab/RunPod GPU, exposes gRPC endpoint
- **Robot client**: runs on local machine, connects via ngrok TCP tunnel
- Robot streams camera observations → server returns action chunks → robot executes

Update `RUNPOD_SERVER` in `config.py` with the ngrok address each session.

### Results

ACT stacks reliably across varied cube positions.

Pi0.5 v2 struggled to generalize consistently despite lower training loss — loss saturated by 5K steps.

MolmoAct2 v2 outperforms Pi0.5 on this task. Starting from `allenai/MolmoAct2-SO100_101` (pretrained on SO-101 data) gives a head start over Pi0.5's general pretraining. Recovery data (60/119 episodes starting from mid-task positions) enables the model to correct mid-task failures. MolmoAct2 v1 (39 eps, no recovery data) does not recover; v2 does.

---

## Task 1: Color-Conditioned Cup Push

Training an SO-101 robot arm to move toward a specific colored cup using imitation learning. Two separate policies — one for a blue cup, one for a red cup — trained on 60 teleoperation demonstrations each.

The arm isn't doing real-time color detection. It learned the motion from demonstrations of that specific colored cup. Swap cup positions — the arm still finds the right color.

## Hardware

| Component | Details |
|-----------|---------|
| Robot arm | SO-101 follower (Feetech motors) |
| Teleoperation | SO-101 leader arm |
| Camera (phone) | Android phone running IP Webcam app (MJPEG over USB tethering) |
| Camera (wrist) | WowRobo USB camera, `/dev/video2` (added for pick-and-place) |
| GPU | RTX 3060 6GB |

**Serial ports (Ubuntu, via udev rules):**
- Follower: `/dev/so101_follower`
- Leader: `/dev/so101_leader`
- Camera: `http://192.168.1.3:8080/video` (640×480, ~25 fps)

Install udev rules for persistent port names:
```bash
sudo cp 99-so101.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Datasets

| Dataset | Episodes | HuggingFace |
|---------|----------|-------------|
| Full (blue + red) | 120 | [so101_cup_push](https://huggingface.co/datasets/subhodipsaha/so101_cup_push) |
| Blue only | 60 | [so101_cup_push_blue](https://huggingface.co/datasets/subhodipsaha/so101_cup_push_blue) |
| Red only | 60 | [so101_cup_push_red](https://huggingface.co/datasets/subhodipsaha/so101_cup_push_red) |

Episodes 0–29, 60–89 = red cup. Episodes 30–59, 90–119 = blue cup.

## Models

| Model | Policy | HuggingFace |
|-------|--------|-------------|
| ACT Blue | ACT (deterministic) | [act_cup_push_blue](https://huggingface.co/subhodipsaha/act_cup_push_blue) |
| ACT Red | ACT (deterministic) | [act_cup_push_red](https://huggingface.co/subhodipsaha/act_cup_push_red) |
| SmolVLA Blue | SmolVLA (flow matching) | [smolvla_cup_push_blue](https://huggingface.co/subhodipsaha/smolvla_cup_push_blue) |
| SmolVLA Red | SmolVLA (flow matching) | [smolvla_cup_push_red](https://huggingface.co/subhodipsaha/smolvla_cup_push_red) |

## Setup

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot

# Install LeRobot with ACT + SmolVLA
cd lerobot/
pip install -e ".[smolvla]"

# Verify CUDA torch (must NOT be CPU-only)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.x.x+cu12x  True
# If CPU-only, reinstall:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Enable BF16 mixed precision (recommended)
accelerate config  # select bf16, or copy default_config.yaml manually
```

## Scripts

### Data Collection
```bash
# Record blue cup episodes (leader arm teleoperation)
python cup_push/record_cup_push_blue.py

# Record red cup episodes
python cup_push/record_cup_push_red.py

# Upload combined dataset to HuggingFace Hub
python cup_push/upload_cup_push.py

# Split combined dataset into blue/red subsets
python cup_push/split_cup_push_dataset.py
```

### Training

**ACT (recommended — deterministic, reliable with small datasets)**
```bash
python cup_push/train_act_blue.py   # ~2-3 hrs on RTX 3060
python cup_push/train_act_red.py
```

**SmolVLA (stochastic — needs more data for consistency)**
```bash
python cup_push/train_smolvla_blue.py   # best run on Colab T4
python cup_push/train_smolvla_red.py
```

### Inference
```bash
# ACT inference
python cup_push/eval_act_cup_push.py --task "Move to the blue object" --policy subhodipsaha/act_cup_push_blue --episodes 5 --episode-time 20
python cup_push/eval_act_cup_push.py --task "Move to the red object"  --policy subhodipsaha/act_cup_push_red  --episodes 5 --episode-time 20

# SmolVLA inference
python cup_push/eval_smolvla_cup_push.py --policy subhodipsaha/smolvla_cup_push_blue
python cup_push/eval_smolvla_cup_push.py --policy subhodipsaha/smolvla_cup_push_red

# Benchmark SmolVLA inference speed (Hz + GPU%)
python cup_push/benchmark_smolvla.py
```

## Training Details

### ACT

| | Blue | Red |
|--|------|-----|
| Dataset | 60 episodes | 60 episodes |
| Steps | 10,000 | 10,000 |
| Batch size | 16 | 16 |
| Epochs | ~8 | ~8 |
| Learning rate | 1e-5 | 1e-5 |
| Final loss | ~0.20 | ~0.22 |
| Training time | ~2-3 hrs (RTX 3060) | ~2-3 hrs (RTX 3060) |
| Mixed precision | BF16 | BF16 |

### SmolVLA

| | Blue | Red |
|--|------|-----|
| Dataset | 60 episodes | 60 episodes |
| Steps | 10,000 | 10,000 |
| Batch size | 16 | 16 |
| Final loss | ~0.14 | ~0.16 |
| Training time | ~1h54m (Colab T4) | ~1h54m (Colab T4) |
| Mixed precision | BF16 | BF16 |

## ACT vs SmolVLA

SmolVLA reached **lower training loss** (~0.14) than ACT (~0.20), yet ACT was far more reliable in practice.

**Why:** SmolVLA uses flow matching for action prediction — it samples random noise at inference time. Same scene, different trajectory each run. With only 60 episodes, the data isn't diverse enough to average out that stochasticity.

ACT is deterministic at inference — the CVAE encoder is discarded and the latent is fixed to zero. Same input = same output every time.

**Takeaway:** For manipulation with ~60 demonstrations, deterministic policies outperform stochastic generative ones.

## Known Issues

- `observation.state` → `observation.environment_state`: ACT internally expects `observation.environment_state` but LeRobot stores `observation.state`. The training and eval scripts include a monkey-patch to handle this — SmolVLA does not need it.
- Ubuntu port permissions: if `Permission denied`, run `sudo chmod 666 /dev/so101_follower /dev/so101_leader` or install the udev rules above.
- HF cache conflict: if re-recording, delete the old cache first: `rm -rf ~/.cache/huggingface/lerobot/subhodipsaha/so101_cup_push`

## Built With

- [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face
- [ACT](https://arxiv.org/abs/2304.13705) — Action Chunking with Transformers
- [SmolVLA](https://huggingface.co/blog/smolvla) — Small Vision-Language-Action model
- [MolmoAct2](https://huggingface.co/allenai/MolmoAct2-SO100_101) — 7B VLA by Allen AI, pretrained on SO-100/101
- [Pi0.5](https://www.physicalintelligence.company/blog/pi05) — 4B VLA by Physical Intelligence
