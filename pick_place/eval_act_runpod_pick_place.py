"""ACT pick-place async inference client for SO-101 via RunPod.

Before running:
1. Apply RunPod patches (see RUNPOD_SETUP.md)
2. Start server: python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080 --fps=25
3. Run this script: python pick_place/eval_act_runpod_pick_place.py
"""

import sys
import time
import threading
from pathlib import Path
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lerobot" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FOLLOWER_PORT, CAMERA_URL, RUNPOD_SERVER

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.import_utils import register_third_party_plugins

TASK          = "Pick up the object and place it at the target location"
SERVER        = RUNPOD_SERVER
POLICY        = "act"
MODEL         = "subhodipsaha/act_pick_place_07_16"
WRIST_CAM     = 2
NUM_EPISODES  = 10
EPISODE_TIME  = 30   # seconds per episode
RESET_TIME    = 10   # seconds to reset scene between episodes

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
    actions_per_chunk=50,
    task=TASK,
    fps=25,
)

register_third_party_plugins()
client = RobotClient(cfg)
# ACT's env_state_feature uses observation.environment_state internally;
# robot produces observation.state — rename so the model finds it.
client.policy_config.rename_map = {"observation.state": "observation.environment_state"}

if client.start():
    print(f"Connected to server at {SERVER}")
    print(f"Policy: {MODEL}")
    print(f"Task:   {TASK}")
    print(f"Episodes: {NUM_EPISODES} × {EPISODE_TIME}s, {RESET_TIME}s reset between\n")
    try:
        for ep in range(NUM_EPISODES):
            print(f"=== Episode {ep + 1}/{NUM_EPISODES} — {EPISODE_TIME}s ===")

            # Reset per-episode state so stale actions don't carry over
            client.shutdown_event.clear()
            client.must_go.set()
            client.action_queue = Queue()
            client.latest_action = -1
            client.action_chunk_size = -1
            client.start_barrier = threading.Barrier(2)

            action_thread = threading.Thread(target=client.receive_actions, daemon=True)
            loop_thread = threading.Thread(
                target=client.control_loop, kwargs={"task": TASK}, daemon=True
            )
            action_thread.start()
            loop_thread.start()

            time.sleep(EPISODE_TIME)

            # Signal both threads to stop
            client.shutdown_event.set()
            loop_thread.join(timeout=5)
            action_thread.join(timeout=5)

            print(f"Episode {ep + 1} done.")
            if ep < NUM_EPISODES - 1:
                print(f"Reset the scene — next episode starts in {RESET_TIME}s...")
                time.sleep(RESET_TIME)

    finally:
        client.stop()
else:
    print("Failed to connect to policy server. Is the server running on RunPod?")
