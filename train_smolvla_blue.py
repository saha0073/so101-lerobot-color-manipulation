"""Train SmolVLA on blue-only cup-push dataset (60 episodes).

Dataset: subhodipsaha/so101_cup_push_blue
Task: "Move to the blue object"
"""

import sys
sys.argv = ["lerobot-train"]

import pandas  # must import before lerobot to avoid pyarrow DLL conflict

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.train import TrainPipelineConfig, DatasetConfig
from lerobot.scripts.lerobot_train import train
from pathlib import Path
import lerobot.scripts.lerobot_train as _train_module

_train_module.update_last_checkpoint = lambda *args, **kwargs: None

if __name__ == "__main__":
    policy_cfg = SmolVLAConfig(
        repo_id="subhodipsaha/smolvla_cup_push_blue",
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.phone": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
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
            repo_id="subhodipsaha/so101_cup_push_blue",
        ),
        policy=policy_cfg,
        output_dir=Path("outputs/train/smolvla_cup_push_blue"),
        steps=10000,
        batch_size=16,
        num_workers=2,
        log_freq=50,
        save_freq=2000,
    )

    train(cfg)
