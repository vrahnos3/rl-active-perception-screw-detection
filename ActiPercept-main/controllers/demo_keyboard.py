"""
Keyboard demo runner for the Webots ScrewDetectorEnv.

Usage
-----
1) Start Webots and open your world.
2) Run this controller script via Webots (as a controller) OR run it from the same
   environment where Webots controller python is available.

Controls
--------
- Move camera: W/A/S/D and Q/E
- Refine a corner: 1..4 (TL, TR, BR, BL)
- ESC / close the window to stop.
"""
import os

from stable_baselines3.common.monitor import Monitor

from screw_detector_env import ScrewDetectorEnv


def run_demo(detection_model_path: str, log_dir: str = "./logs") -> None:
    """
    Keyboard demo runner for the Webots ScrewDetectorEnv.

    Usage
    -----
    1) Start Webots and open your world.
    2) Run this controller script via Webots (as a controller) OR run it from the same
       environment where Webots controller python is available.

    Controls
    --------
    - Move camera: W/A/S/D and Q/E
    - Refine a corner: 1..4 (TL, TR, BR, BL)
    - ESC / close the window to stop.
    """
    os.makedirs(log_dir, exist_ok=True)

    env = ScrewDetectorEnv(
        model_path=detection_model_path,
        visualize_detections=True,
        max_steps= 180 #2048,
        control_mode="keyboard",
        scripted_return_on_success=False,
    )

    env = Monitor(env, log_dir)

    env.reset()
    while True:
        # action is ignored in keyboard mode
        _, _, terminated, truncated, _ = env.step(None)
        if terminated or truncated:
            break


if __name__ == "__main__":
    # run_demo(detection_model_path="./pc_screw_detection.pt")
    run_demo(detection_model_path="./best.pt")
