"""Train ACT on pick-place dataset — 40 episodes, 20000 steps, BF16."""

import sys
sys.argv = ["lerobot-train"]

import pandas  # must import before lerobot to avoid pyarrow DLL conflict

from lerobot.utils.feature_utils import PolicyFeature, FeatureType
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.train import TrainPipelineConfig, DatasetConfig
from lerobot.configs.types import NormalizationMode
from lerobot.scripts.lerobot_train import train
from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE
from pathlib import Path
import lerobot.scripts.lerobot_train as _train_module
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

_train_module.update_last_checkpoint = lambda *args, **kwargs: None

_orig_act_forward = ACTPolicy.forward
def _patched_act_forward(self, batch):
    if OBS_STATE in batch and OBS_ENV_STATE not in batch:
        batch = dict(batch)
        batch[OBS_ENV_STATE] = batch[OBS_STATE]
    return _orig_act_forward(self, batch)
ACTPolicy.forward = _patched_act_forward

if __name__ == "__main__":
    # 40 episodes × 449 frames (18s@25fps) / batch_size 8 = ~2245 steps/epoch × ~9 epochs = 20000 steps
    policy_cfg = ACTConfig(
        repo_id="subhodipsaha/act_pick_place_07_16",
        input_features={
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(6,)),
            "observation.images.phone": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE":  NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.IDENTITY,
        },
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="subhodipsaha/so101_pick_place_07_16",
        ),
        policy=policy_cfg,
        output_dir=Path("outputs/train/act_pick_place_07_16"),
        steps=20000,
        batch_size=8,
        num_workers=4,
        log_freq=100,
        save_freq=2500,
    )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )

    train(cfg, accelerator=accelerator)
