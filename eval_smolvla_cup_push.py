"""Run the trained SmolVLA cup-push policy on the SO-101 follower arm.

Usage:
    python eval_smolvla_cup_push.py --task "Move to the red object"
    python eval_smolvla_cup_push.py --task "Move to the blue object"
"""

import sys
import argparse

# Save real args before lerobot clobbers sys.argv
_real_argv = sys.argv[1:]
sys.argv = ["lerobot-record"]

OS = "ubuntu"  # Set to "windows" or "ubuntu"

FOLLOWER_PORT = "COM3" if OS == "windows" else "/dev/so101_follower"
LEADER_PORT   = "COM4" if OS == "windows" else "/dev/so101_leader"

import pandas  # must import before lerobot to avoid pyarrow DLL conflict
from config import CAMERA_URL

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="Move to the red object",
        choices=["Move to the red object", "Move to the blue object"],
        help="Language instruction for the policy",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-time", type=int, default=10)
    parser.add_argument("--reset-time", type=int, default=8)
    parser.add_argument("--policy", type=str, default="subhodipsaha/smolvla_cup_push_v3",
                        help="HuggingFace repo or local path to pretrained policy")
    args = parser.parse_args(_real_argv)

    import torch
    from pathlib import Path
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.scripts.lerobot_record import record, RecordConfig, DatasetRecordConfig
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    import time
    from lerobot.utils.constants import ACTION

    # Diagnostic: time each select_action call and show queue state
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

    policy_path = args.policy
    policy = PreTrainedConfig.from_pretrained(policy_path)
    policy.pretrained_path = policy_path
    policy.num_steps = 10

    # repo_id slug: replace spaces with underscores, lowercase
    task_slug = args.task.lower().replace(" ", "_")
    eval_repo_id = f"subhodipsaha/eval_smolvla_{task_slug}"

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

    print(f"Task: {args.task}")
    print(f"Saving eval episodes to: {eval_repo_id}")
    record(cfg)
