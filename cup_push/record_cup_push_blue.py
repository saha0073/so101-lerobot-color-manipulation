"""Record cup-push episodes — 30 episodes pushing the BLUE cup.

Run this AFTER record_cup_push_red.py completes.
Both red and blue cups are always visible on the table.
Teleoperate arm to push the BLUE cup, ignore the red one.

Suggested episode positions (vary cup positions to prevent position memorization):
  0  → Blue: far left,     Red: right
  1  → Blue: right,        Red: far left
  2  → Blue: near left,    Red: near right
  3  → Blue: near right,   Red: near left
  4  → Blue: far left,     Red: center-right
  5  → Blue: center-left,  Red: far right
  6  → Blue: right,        Red: left
  7  → Blue: left,         Red: right
  8  → Blue: far left,     Red: near right
  9  → Blue: near right,   Red: far left
  10 → Blue: center-left,  Red: center-right
  11 → Blue: far right,    Red: left
  12 → Blue: left,         Red: far right
  13 → Blue: near left,    Red: far right
  14 → Blue: far right,    Red: near left
  15 → Blue: far left,     Red: right
  16 → Blue: right,        Red: far left
  17 → Blue: near left,    Red: near right
  18 → Blue: near right,   Red: near left
  19 → Blue: center-right, Red: far left
  20 → Blue: far left,     Red: center-right
  21 → Blue: right,        Red: left
  22 → Blue: left,         Red: right
  23 → Blue: far left,     Red: near right
  24 → Blue: near right,   Red: far left
  25 → Blue: center-left,  Red: far right
  26 → Blue: far right,    Red: left
  27 → Blue: left,         Red: far right
  28 → Blue: near left,    Red: far right
  29 → Blue: far right,    Red: near left
"""

import sys
sys.argv = ["lerobot-record"]

from config import CAMERA_URL, FOLLOWER_PORT, LEADER_PORT


from pathlib import Path
import pandas  # must import before lerobot to avoid pyarrow DLL conflict
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.scripts.lerobot_record import record, RecordConfig, DatasetRecordConfig

if __name__ == "__main__":
    cfg = RecordConfig(
        resume=True,
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
            single_task="Move to the blue object",
            num_episodes=30,
            episode_time_s=10,
            reset_time_s=10,
            push_to_hub=False,
        ),
        display_data=False,
    )
    record(cfg)
