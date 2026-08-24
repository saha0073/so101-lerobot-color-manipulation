"""SmolVLA pick-place async inference client for SO-101 via RunPod.

Before running:
1. SSH into RunPod and start the policy server:
   python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080 --fps=25
2. Run this script: python pick_place/eval_smolvla_runpod_pick_place.py
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lerobot" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FOLLOWER_PORT, CAMERA_URL, RUNPOD_SERVER

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.import_utils import register_third_party_plugins

TASK      = "Pick up the object and place it at the target location"
SERVER    = RUNPOD_SERVER
POLICY    = "smolvla"
MODEL     = "subhodipsaha/smolvla_pick_place_07_16"
WRIST_CAM = 2

robot_cfg = SOFollowerRobotConfig(
    port=FOLLOWER_PORT,
    cameras={
        "phone": OpenCVCameraConfig(
            index_or_path=CAMERA_URL,
            fps=25, width=480, height=640,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=WRIST_CAM,
            fps=30, width=640, height=480, fourcc="MJPG",
        ),
    },
)

cfg = RobotClientConfig(
    robot=robot_cfg,
    policy_type=POLICY,
    pretrained_name_or_path=MODEL,
    server_address=SERVER,
    policy_device="cuda",
    client_device="cpu",
    actions_per_chunk=20,
    task=TASK,
    fps=25,
)

register_third_party_plugins()
client = RobotClient(cfg)

if client.start():
    print(f"Connected to server at {SERVER}")
    print(f"Policy: {MODEL}")
    print(f"Task:   {TASK}")
    action_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_thread.start()
    try:
        client.control_loop(task=TASK)
    finally:
        client.stop()
        action_thread.join()
else:
    print("Failed to connect to policy server. Is the server running on RunPod?")
