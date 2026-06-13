"""Record cup-push episodes — 30 episodes pushing the RED cup.

Both red and blue cups are always visible on the table.
Teleoperate arm to push the RED cup, ignore the blue one.

Suggested episode positions (vary cup positions to prevent position memorization):
  0  → Red: far left,     Blue: right
  1  → Red: right,        Blue: far left
  2  → Red: near left,    Blue: near right
  3  → Red: near right,   Blue: near left
  4  → Red: far left,     Blue: center-right
  5  → Red: center-left,  Blue: far right
  6  → Red: right,        Blue: left
  7  → Red: left,         Blue: right
  8  → Red: far left,     Blue: near right
  9  → Red: near right,   Blue: far left
  10 → Red: center-left,  Blue: center-right
  11 → Red: far right,    Blue: left
  12 → Red: left,         Blue: far right
  13 → Red: near left,    Blue: far right
  14 → Red: far right,    Blue: near left
  15 → Red: far left,     Blue: right
  16 → Red: right,        Blue: far left
  17 → Red: near left,    Blue: near right
  18 → Red: near right,   Blue: near left
  19 → Red: center-right, Blue: far left
  20 → Red: far left,     Blue: center-right
  21 → Red: right,        Blue: left
  22 → Red: left,         Blue: right
  23 → Red: far left,     Blue: near right
  24 → Red: near right,   Blue: far left
  25 → Red: center-left,  Blue: far right
  26 → Red: far right,    Blue: left
  27 → Red: left,         Blue: far right
  28 → Red: near left,    Blue: far right
  29 → Red: far right,    Blue: near left
"""

import sys
sys.argv = ["lerobot-record"]

from config import CAMERA_URL, FOLLOWER_PORT, LEADER_PORT


import pandas  # must import before lerobot to avoid pyarrow DLL conflict
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.scripts.lerobot_record import record, RecordConfig, DatasetRecordConfig

if __name__ == "__main__":
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
            repo_id="subhodipsaha/so101_cup_push",
            root="/home/subho/.cache/huggingface/lerobot/subhodipsaha/so101_cup_push",
            single_task="Move to the red object",
            num_episodes=30,
            episode_time_s=10,
            reset_time_s=10,
            push_to_hub=False,
        ),
        display_data=False,
        resume=True,
    )
    record(cfg)
