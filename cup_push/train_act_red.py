"""Train ACT on red-only cup-push dataset (60 episodes)."""

import sys
sys.argv = ["lerobot-train"]

import pandas  # must import before lerobot to avoid pyarrow DLL conflict

from lerobot.utils.feature_utils import PolicyFeature, FeatureType
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.train import TrainPipelineConfig, DatasetConfig
from lerobot.scripts.lerobot_train import train
from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE
from pathlib import Path
import lerobot.scripts.lerobot_train as _train_module

_train_module.update_last_checkpoint = lambda *args, **kwargs: None

_orig_act_forward = ACTPolicy.forward
def _patched_act_forward(self, batch):
    if OBS_STATE in batch and OBS_ENV_STATE not in batch:
        batch = dict(batch)
        batch[OBS_ENV_STATE] = batch[OBS_STATE]
    return _orig_act_forward(self, batch)
ACTPolicy.forward = _patched_act_forward

if __name__ == "__main__":
    policy_cfg = ACTConfig(
        repo_id="subhodipsaha/act_cup_push_red",
        input_features={
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(6,)),
            "observation.images.phone": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="subhodipsaha/so101_cup_push_red",
        ),
        policy=policy_cfg,
        output_dir=Path("outputs/train/act_cup_push_red"),
        steps=10000,
        batch_size=16,
        num_workers=4,
        log_freq=50,
        save_freq=2000,
    )

    train(cfg)
