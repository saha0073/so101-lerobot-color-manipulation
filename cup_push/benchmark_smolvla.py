"""Benchmark SmolVLA inference speed and GPU utilization with dummy inputs.

Uses lerobot/smolvla_base (official pretrained checkpoint).
"""

import sys
sys.argv = ["benchmark"]

import pandas  # must import before lerobot on Windows to avoid pyarrow DLL conflict

import subprocess
import time

import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

MODEL_ID = "lerobot/smolvla_base"
N_WARMUP = 3
N_RUNS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_KEY = "observation.images.camera1"
IMAGE_H, IMAGE_W = 256, 256
STATE_DIM = 6


def gpu_stats() -> str:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        util, used, total = r.stdout.strip().split(", ")
        return f"GPU {util}%  VRAM {used}/{total} MB"
    except Exception:
        mb = torch.cuda.memory_allocated() // 1024**2
        return f"VRAM {mb} MB"


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading {MODEL_ID} ...")
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy.eval()

    param_devices = {p.device.type for p in policy.parameters()}
    print(f"Parameter devices: {param_devices}")
    print(f"After load  → {gpu_stats()}\n")

    preprocess, _ = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
    )

    # Dummy raw observation — uint8 image (H, W, C) + float32 joint state
    raw_obs = {
        IMAGE_KEY: np.random.randint(0, 256, (IMAGE_H, IMAGE_W, 3), dtype=np.uint8),
        "observation.state": np.zeros(STATE_DIM, dtype=np.float32),
    }

    def make_batch():
        obs = prepare_observation_for_inference(
            dict(raw_obs),  # copy — function mutates the dict
            device=DEVICE,
            task="pick the object",
        )
        return preprocess(obs)

    # Warmup
    print(f"Warmup ({N_WARMUP} steps)...")
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            policy.reset()
            batch = make_batch()
            policy.select_action(batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
    print(f"After warmup → {gpu_stats()}\n")

    # Timed runs
    print(f"{'Run':>4}  {'ms':>7}  {'Hz':>6}  GPU stats")
    print("-" * 52)
    times = []
    with torch.inference_mode():
        for i in range(N_RUNS):
            policy.reset()
            batch = make_batch()
            t0 = time.perf_counter()
            action = policy.select_action(batch)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)
            print(f"{i+1:>4}  {dt*1000:>7.1f}  {1/dt:>6.1f}  {gpu_stats()}")

    print("\n--- Summary ---")
    print(f"  Mean   {np.mean(times)*1000:.1f} ms  =  {1/np.mean(times):.1f} Hz")
    print(f"  Median {np.median(times)*1000:.1f} ms  =  {1/np.median(times):.1f} Hz")
    print(f"  Min    {np.min(times)*1000:.1f} ms  =  {1/np.min(times):.1f} Hz")
    print(f"  Max    {np.max(times)*1000:.1f} ms  =  {1/np.max(times):.1f} Hz")
    print(f"\nFinal GPU state → {gpu_stats()}")


if __name__ == "__main__":
    main()
