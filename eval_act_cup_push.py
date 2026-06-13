"""Run ACT cup-push policy on the SO-101 follower arm.

Usage:
    python eval_act_cup_push.py --task "Move to the blue object" --policy subhodipsaha/act_cup_push_blue
    python eval_act_cup_push.py --task "Move to the red object"  --policy subhodipsaha/act_cup_push_red
"""

import sys
import argparse

_real_argv = sys.argv[1:]
sys.argv = ["lerobot-record"]

FOLLOWER_PORT = "/dev/so101_follower"
LEADER_PORT   = "/dev/so101_leader"

import pandas  # must import before lerobot to avoid pyarrow DLL conflict
from config import CAMERA_URL

import lerobot.scripts.lerobot_record as _record_module

# Rename observation.state -> observation.environment_state for ACTPolicy
_orig_make_prepost = _record_module.make_pre_post_processors
def _patched_make_prepost(policy_cfg, pretrained_path=None, **kwargs):
    overrides = dict(kwargs.get("preprocessor_overrides") or {})
    overrides["rename_observations_processor"] = {
        "rename_map": {"observation.state": "observation.environment_state"}
    }
    kwargs["preprocessor_overrides"] = overrides
    return _orig_make_prepost(policy_cfg, pretrained_path, **kwargs)
_record_module.make_pre_post_processors = _patched_make_prepost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Move to the blue object",
                        choices=["Move to the red object", "Move to the blue object"])
    parser.add_argument("--policy", type=str, default="subhodipsaha/act_cup_push_blue")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-time", type=int, default=20)
    parser.add_argument("--reset-time", type=int, default=8)
    args = parser.parse_args(_real_argv)

    from pathlib import Path
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.scripts.lerobot_record import record, RecordConfig, DatasetRecordConfig
    from lerobot.configs import PreTrainedConfig

    policy = PreTrainedConfig.from_pretrained(args.policy)
    policy.pretrained_path = args.policy

    task_slug = args.task.lower().replace(" ", "_")
    eval_repo_id = f"subhodipsaha/eval_act_{task_slug}"

    cfg = RecordConfig(
        robot=SOFollowerRobotConfig(
            port=FOLLOWER_PORT,
            cameras={
                "phone": OpenCVCameraConfig(
                    index_or_path=CAMERA_URL,
                    fps=25,
                    width=640,
                    height=480,
                )
            },
        ),
        teleop=SOLeaderTeleopConfig(port=LEADER_PORT),
        dataset=DatasetRecordConfig(
            repo_id=eval_repo_id,
            single_task=args.task,
            num_episodes=args.episodes,
            episode_time_s=args.episode_time,
            reset_time_s=args.reset_time,
            fps=25,
        ),
        policy=policy,
        display_data=False,
    )

    print(f"Task:   {args.task}")
    print(f"Policy: {args.policy}")
    record(cfg)
