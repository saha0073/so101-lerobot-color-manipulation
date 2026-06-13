# SO-101 Color-Conditioned Manipulation

Training an SO-101 robot arm to move toward a specific colored cup using imitation learning. Two separate policies — one for a blue cup, one for a red cup — trained on 60 teleoperation demonstrations each.

## Demo

The arm isn't doing real-time color detection. It learned the motion from demonstrations of that specific colored cup. Swap cup positions — the arm still finds the right color.

## Hardware

| Component | Details |
|-----------|---------|
| Robot arm | SO-101 follower (Feetech motors) |
| Teleoperation | SO-101 leader arm |
| Camera | Android phone running IP Webcam app (MJPEG over USB tethering) |
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
python record_cup_push_blue.py

# Record red cup episodes
python record_cup_push_red.py

# Upload combined dataset to HuggingFace Hub
python upload_cup_push.py

# Split combined dataset into blue/red subsets
python split_cup_push_dataset.py
```

### Training

**ACT (recommended — deterministic, reliable with small datasets)**
```bash
python train_act_blue.py   # ~2-3 hrs on RTX 3060
python train_act_red.py
```

**SmolVLA (stochastic — needs more data for consistency)**
```bash
python train_smolvla_blue.py   # best run on Colab T4
python train_smolvla_red.py
```

### Inference
```bash
# ACT inference
python eval_act_cup_push.py --task "Move to the blue object" --policy subhodipsaha/act_cup_push_blue --episodes 5 --episode-time 20
python eval_act_cup_push.py --task "Move to the red object"  --policy subhodipsaha/act_cup_push_red  --episodes 5 --episode-time 20

# SmolVLA inference
python eval_smolvla_cup_push.py --policy subhodipsaha/smolvla_cup_push_blue
python eval_smolvla_cup_push.py --policy subhodipsaha/smolvla_cup_push_red

# Benchmark SmolVLA inference speed (Hz + GPU%)
python benchmark_smolvla.py
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
