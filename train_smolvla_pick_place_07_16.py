"""Train SmolVLA on pick-place dataset (40 episodes, two cameras, 20K steps)."""

import sys
sys.argv = ["lerobot-train"]

import pandas  # must import before lerobot to avoid pyarrow DLL conflict

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.train import TrainPipelineConfig, DatasetConfig
from lerobot.scripts.lerobot_train import train
from pathlib import Path

if __name__ == "__main__":
    policy_cfg = SmolVLAConfig(
        repo_id="subhodipsaha/smolvla_pick_place_07_16",
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.phone": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
        load_vlm_weights=True,
        freeze_vision_encoder=True,
        train_expert_only=True,
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="subhodipsaha/so101_pick_place_07_16",
        ),
        policy=policy_cfg,
        output_dir=Path("outputs/train/smolvla_pick_place_07_16"),
        steps=20000,
        batch_size=4,
        num_workers=2,
        log_freq=100,
        save_freq=2500,
    )

    train(cfg)
