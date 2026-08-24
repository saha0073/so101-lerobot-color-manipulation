"""Fine-tune Pi0.5 on stack-three-cubes dataset — run this ON RunPod.

Setup on RunPod pod:
    pip install 'lerobot[pi0]'
    huggingface-cli login

Then run:
    python train_pi05_runpod_stack_three_cubes_08_19.py
"""

import subprocess
import sys

cmd = [
    sys.executable, "-m", "lerobot.scripts.lerobot_train",
    "--policy.type=pi05",
    "--policy.repo_id=subhodipsaha/pi05_stack_three_cubes_08_19",
    "--dataset.repo_id=subhodipsaha/so101_stack_three_cubes_08_19",
    "--dataset.image_transforms.enable=true",
    "--batch_size=16",
    "--steps=20000",
    "--log_freq=100",
    "--save_freq=5000",
    "--output_dir=outputs/train/pi05_stack_three_cubes_08_19",
    "--policy.device=cuda",
    "--wandb.enable=false",
]

print("Starting Pi0.5 fine-tuning for stack-three-cubes...")
print("Dataset: subhodipsaha/so101_stack_three_cubes_08_19 (39 episodes)")
print("Output:  subhodipsaha/pi05_stack_three_cubes_08_19\n")
subprocess.run(cmd, check=True)
