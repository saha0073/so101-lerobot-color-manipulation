"""Record pick-and-place episodes — arm picks up the object and places it at a target location.

Task: arm moves to object, grasps it, lifts and moves to target location, then releases.
Vary object start position and/or target position slightly each episode.
Two cameras: phone (global, portrait) + wrist (USB, /dev/video2).
"""

import sys
from pathlib import Path
sys.argv = ["lerobot-record"]

import pandas  # must import before lerobot to avoid pyarrow DLL conflict
from config import FOLLOWER_PORT, LEADER_PORT
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.scripts.lerobot_record import record, RecordConfig, DatasetRecordConfig

PHONE_URL = "http://192.168.0.3:8080/video"
WRIST_CAM = 2   # /dev/video2 (WowRobo USB camera)

if __name__ == "__main__":
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
            repo_id="subhodipsaha/so101_pick_place_07_16",
            single_task="Pick up the object and place it at the target location",
            num_episodes=40,
            episode_time_s=18,
            reset_time_s=10,
            fps=25,
            root=str(Path.home() / ".cache/huggingface/lerobot/subhodipsaha/so101_pick_place_07_16"),
        ),
        resume=False,
        display_data=False,
    )
    record(cfg)
