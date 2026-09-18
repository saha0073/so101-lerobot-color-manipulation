"""MolmoAct2 stack-three-cubes client — fine-tuned on subhodipsaha/so101_stack_three_cubes_08_19.

Before running:
  1. Start stack_cubes/molmoact_stack_cubes_inference_colab.ipynb on Colab L4
  2. Paste the ngrok URL into config.py as MOLMOACT2_STACK_SERVER
  3. python stack_cubes/eval_molmoact2_stack_three_cubes.py
"""

import argparse
import base64
import io
import sys
import time
import threading
from pathlib import Path
from queue import Queue, Empty

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lerobot" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FOLLOWER_PORT, CAMERA_URL, MOLMOACT2_STACK_SERVER

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_URL       = MOLMOACT2_STACK_SERVER
WRIST_CAM        = 2
CONTROL_HZ       = 25
# Server latency ~1.1s = 27 steps at 25Hz.
# REFILL_THRESHOLD ≥ 28 ensures new chunk arrives before queue empties.
REFILL_THRESHOLD = 28
NUM_EPISODES     = 5
EPISODE_TIME     = 180  # stacking 3 cubes needs more time than pick-and-place
RESET_TIME       = 30
TASK             = "Stack all three cubes into a tower"
# ─────────────────────────────────────────────────────────────────────────────

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Mean starting state from training data (so101_stack_three_cubes_08_19, 39 episodes)
HOME_POS = {
    "shoulder_pan":   -3.9,
    "shoulder_lift": -98.7,
    "elbow_flex":     96.5,
    "wrist_flex":     77.4,
    "wrist_roll":    -86.4,
    "gripper":        41.7,
}

NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}


def go_home(robot: "SOFollower", tol_deg: float = 2.0):
    """Drive arm to the training-data starting pose before each episode."""
    print("  Going to home pose...")
    dt = 1.0 / CONTROL_HZ
    while True:
        obs = robot.get_observation()
        max_err = 0.0
        cmd = {}
        for m in MOTOR_NAMES:
            cur = float(obs[f"{m}.pos"])
            err = HOME_POS[m] - cur
            max_err = max(max_err, abs(err))
            cmd[f"{m}.pos"] = HOME_POS[m]
        robot.send_action(cmd)
        if max_err < tol_deg:
            break
        time.sleep(dt)
    print("  Home reached.")


def encode_bgr_frame(frame_bgr: np.ndarray) -> str:
    img = Image.fromarray(frame_bgr[..., ::-1])  # BGR → RGB
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def obs_to_server_inputs(obs: dict) -> tuple[list[str], list[float]]:
    images_b64 = []
    for cam_key in ("phone", "wrist"):
        if cam_key in obs:
            images_b64.append(encode_bgr_frame(obs[cam_key]))
    state = [float(obs[f"{m}.pos"]) for m in MOTOR_NAMES]
    return images_b64, state


def actions_to_robot(actions_row: np.ndarray) -> dict:
    cmd = {f"{m}.pos": float(actions_row[i]) for i, m in enumerate(MOTOR_NAMES)}
    cmd["gripper.pos"] = max(cmd["gripper.pos"], 0.0)
    return cmd


def fetch_chunk(images_b64: list, state: list, task: str) -> np.ndarray:
    resp = requests.post(
        f"{SERVER_URL}/predict",
        json={"images_b64": images_b64, "state": state, "task": task},
        headers=NGROK_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    actions = np.array(data["actions"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[np.newaxis, :]
    print(f"  [server] {data['inference_ms']:.0f}ms | chunk={actions.shape}")
    return actions


def check_server() -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/health", headers=NGROK_HEADERS, timeout=5)
        info = r.json()
        print(f"Server healthy — GPU mem: {info.get('gpu_mem_gb', '?')} GB")

        if not info.get('warmed_up', True):
            print("Warming up CUDA caches (first call ~1-2s)...")
            dummy_phone = Image.new('RGB', (480, 640))
            dummy_wrist = Image.new('RGB', (640, 480))
            def _enc(img):
                buf = io.BytesIO(); img.save(buf, format='JPEG'); return base64.b64encode(buf.getvalue()).decode()
            requests.post(f"{SERVER_URL}/predict", headers=NGROK_HEADERS, timeout=30,
                json={"images_b64": [_enc(dummy_phone), _enc(dummy_wrist)],
                      "state": list(HOME_POS.values()),
                      "task": TASK})
            print("Server warmed up.")
        return True
    except Exception as e:
        print(f"Server not reachable: {e}")
        return False


def run_episode(robot: SOFollower, task: str, episode_time: float):
    action_queue: Queue = Queue()
    refilling = threading.Event()
    last_queued = [None]  # last action put in queue = predicted end-of-chunk position

    def refill_worker(images_b64, state):
        # Send the predicted end-of-chunk position as state rather than the stale
        # observed state from 28 steps ago.
        predicted_state = last_queued[0].tolist() if last_queued[0] is not None else state
        try:
            actions = fetch_chunk(images_b64, predicted_state, task)
            for row in actions:
                last_queued[0] = row.copy()
                action_queue.put(row)
        except Exception as e:
            print(f"  [refill error] {e}")
        finally:
            refilling.clear()

    def maybe_refill(images_b64, state):
        if not refilling.is_set() and action_queue.qsize() <= REFILL_THRESHOLD:
            refilling.set()
            threading.Thread(target=refill_worker, args=(images_b64, state), daemon=True).start()

    print("  Fetching initial action chunk...")
    obs = robot.get_observation()
    images_b64, state = obs_to_server_inputs(obs)
    print(f"  [debug] ncams={len(images_b64)} state={dict(zip(MOTOR_NAMES, [round(s,2) for s in state]))}")
    for row in fetch_chunk(images_b64, state, task):
        last_queued[0] = row.copy()
        action_queue.put(row)

    dt = 1.0 / CONTROL_HZ
    t_end = time.time() + episode_time

    while time.time() < t_end:
        t_step = time.time()

        obs = robot.get_observation()
        images_b64, state = obs_to_server_inputs(obs)
        maybe_refill(images_b64, state)

        try:
            action_row = action_queue.get(timeout=2.0)
        except Empty:
            print("  [warn] queue empty — holding position")
            time.sleep(dt)
            continue

        robot.send_action(actions_to_robot(action_row))

        elapsed = time.time() - t_step
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=TASK)
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--episode-time", type=int, default=EPISODE_TIME)
    parser.add_argument("--reset-time", type=int, default=RESET_TIME)
    args = parser.parse_args()

    print(f"Task:   {args.task}")
    print(f"Server: {SERVER_URL}\n")

    if not check_server():
        print("Start the Colab server first and update MOLMOACT2_STACK_SERVER in config.py.")
        return

    robot_cfg = SOFollowerRobotConfig(
        port=FOLLOWER_PORT,
        max_relative_target=None,
        cameras={
            "phone": OpenCVCameraConfig(
                index_or_path=CAMERA_URL, fps=25, width=480, height=640,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=WRIST_CAM, fps=30, width=640, height=480, fourcc="MJPG",
            ),
        },
    )

    robot = SOFollower(robot_cfg)
    robot.connect()
    print("Robot connected.\n")

    try:
        for ep in range(args.episodes):
            print(f"=== Episode {ep + 1}/{args.episodes} — {args.episode_time}s ===")
            go_home(robot)
            run_episode(robot, args.task, args.episode_time)
            print(f"Episode {ep + 1} done.")

            if ep < args.episodes - 1:
                robot.bus.disable_torque()
                print(f"Torque OFF — reset scene. Next episode in {args.reset_time}s...")
                time.sleep(args.reset_time)
                robot.bus.enable_torque()
                print("Torque ON\n")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
