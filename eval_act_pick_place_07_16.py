"""Run ACT pick-place policy on the SO-101 follower arm (two cameras).

Usage:
    python eval_act_pick_place_07_16.py
    python eval_act_pick_place_07_16.py --policy subhodipsaha/act_pick_place_07_16 --episodes 5
"""

import sys
import argparse

_real_argv = sys.argv[1:]
sys.argv = ["lerobot-record"]

import pandas  # must import before lerobot to avoid pyarrow DLL conflict
from config import FOLLOWER_PORT, LEADER_PORT

PHONE_URL = "http://192.168.0.3:8080/video"
WRIST_CAM = 2   # /dev/video2 (WowRobo USB camera)
POLICY_ID = "subhodipsaha/act_pick_place_07_16"
TASK      = "Pick up the object and place it at the target location"

import lerobot.scripts.lerobot_record as _record_module

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
    parser.add_argument("--policy", type=str, default=POLICY_ID)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-time", type=int, default=18)
    parser.add_argument("--reset-time", type=int, default=10)
    args = parser.parse_args(_real_argv)

    from pathlib import Path
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.scripts.lerobot_record import record, RecordConfig, DatasetRecordConfig
    from lerobot.configs import PreTrainedConfig

    policy = PreTrainedConfig.from_pretrained(args.policy)
    policy.pretrained_path = args.policy

    cfg = RecordConfig(
        robot=SOFollowerRobotConfig(
            port=FOLLOWER_PORT,
            cameras={
                "phone": OpenCVCameraConfig(
                    index_or_path=PHONE_URL,
                    fps=25,
                    width=480,
                    height=640,
                ),
                "wrist": OpenCVCameraConfig(
                    index_or_path=WRIST_CAM,
                    fps=25,
                    width=640,
                    height=480,
                    fourcc="MJPG",
                ),
            },
        ),
        teleop=SOLeaderTeleopConfig(port=LEADER_PORT),
        dataset=DatasetRecordConfig(
            repo_id="subhodipsaha/eval_act_pick_place_07_16",
            single_task=TASK,
            num_episodes=args.episodes,
            episode_time_s=args.episode_time,
            reset_time_s=args.reset_time,
            fps=25,
        ),
        policy=policy,
        display_data=False,
    )

    print(f"Policy: {args.policy}")
    print(f"Task:   {TASK}")
    record(cfg)
