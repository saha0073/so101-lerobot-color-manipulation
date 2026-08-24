"""Record 3-cube stacking episodes — arm picks and stacks cubes one by one to build a tower.

Task: stack all three cubes into a tower (pick cube 1, place on cube 2; pick that stack, place on cube 3).
Vary cube positions slightly each episode.
Two cameras: phone (global, portrait) + wrist (USB, /dev/video2).
"""

import sys
from pathlib import Path
sys.argv = ["lerobot-record"]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
            repo_id="subhodipsaha/so101_stack_three_cubes_08_19",
            single_task="Stack all three cubes into a tower",
            num_episodes=40,
            episode_time_s=30,
            reset_time_s=12,
            fps=25,
            root=str(Path.home() / ".cache/huggingface/lerobot/subhodipsaha/so101_stack_three_cubes_08_19"),
        ),
        resume=False,
        display_data=False,
    )
    record(cfg)
