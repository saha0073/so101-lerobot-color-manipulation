"""Record cube-stacking episodes — arm picks up one cube and stacks it on top of another.

Task: arm moves to the first cube, grasps it, lifts and places it on top of the second cube.
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
            repo_id="subhodipsaha/so101_stack_two_cubes_08_12",
            single_task="Stack one cube on top of the other cube",
            num_episodes=39,
            episode_time_s=25,
            reset_time_s=10,
            fps=25,
            root=str(Path.home() / ".cache/huggingface/lerobot/subhodipsaha/so101_stack_two_cubes_08_12"),
        ),
        resume=True,
        display_data=False,
    )
    record(cfg)
