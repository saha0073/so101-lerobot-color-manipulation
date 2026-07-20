"""Run SmolVLA pick-place policy on the SO-101 follower arm (two cameras).

Usage:
    python eval_smolvla_pick_place_07_16.py
    python eval_smolvla_pick_place_07_16.py --policy subhodipsaha/smolvla_pick_place_07_16 --episodes 5
"""

import sys
import argparse

_real_argv = sys.argv[1:]
sys.argv = ["lerobot-record"]

import pandas  # must import before lerobot to avoid pyarrow DLL conflict
from config import FOLLOWER_PORT, LEADER_PORT

PHONE_URL = "http://192.168.0.3:8080/video"
WRIST_CAM = 2   # /dev/video2 (WowRobo USB camera)
POLICY_ID = "subhodipsaha/smolvla_pick_place_07_16"
TASK      = "Pick up the object and place it at the target location"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default=POLICY_ID)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-time", type=int, default=18)
    parser.add_argument("--reset-time", type=int, default=10)
    args = parser.parse_args(_real_argv)

    import torch
    import time
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.scripts.lerobot_record import record, RecordConfig, DatasetRecordConfig
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION

    _orig_select_action = SmolVLAPolicy.select_action
    _call_count = [0]
    def _timed_select_action(self, batch):
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = _orig_select_action(self, batch)
        elapsed_ms = (time.perf_counter() - t) * 1000
        _call_count[0] += 1
        queue_len = len(self._queues.get(ACTION, []))
        print(f"[select_action #{_call_count[0]}] {elapsed_ms:.1f}ms  queue_remaining={queue_len}")
        return result
    SmolVLAPolicy.select_action = _timed_select_action

    policy = PreTrainedConfig.from_pretrained(args.policy)
    policy.pretrained_path = args.policy
    policy.num_steps = 10

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
            repo_id="subhodipsaha/eval_smolvla_pick_place_07_16",
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
