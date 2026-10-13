import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import yaml
import cv2
from stable_baselines3 import PPO


def add_webots_python_controller_path() -> None:
    webots_home = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")
    controller_python_path = Path(webots_home) / "lib" / "controller" / "python"
    controller_dll_path = Path(webots_home) / "lib" / "controller"

    if controller_python_path.exists():
        sys.path.insert(0, str(controller_python_path))
    if controller_dll_path.exists():
        os.environ["PATH"] = f"{controller_dll_path}{os.pathsep}{os.environ.get('PATH', '')}"


add_webots_python_controller_path()

from screw_env_v3 import ScrewDetectorEnv


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        path = Path(__file__).resolve().parent / config_path

    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_env(config: dict):
    eval_config = config["eval"]
    env_config = eval_config["env"]

    workspace = Path(config["workspace_path"])
    experiment_name = config["experiment_name"]
    log_dir = workspace / config["log_dir"] / experiment_name
    yolo_path = workspace / config["yolo_model"]
    laptop_metadata_path = workspace / config["laptop_metadata_path"]
    mode = env_config["mode"]

    random_luminosity = env_config['random_luminosity']
    set_luminosity = env_config['set_luminosity']

    env = ScrewDetectorEnv(
        yolo_path=str(yolo_path),
        visualize_detections=env_config["visualize_detections"],
        max_steps=env_config["max_steps"],
        control_mode=env_config["control_mode"],
        mode=mode,
        laptop_metadata_path=str(laptop_metadata_path)
    )

    return env, log_dir, experiment_name


def load_laptop_database(config: dict) -> Dict[str, Any]:
    workspace = Path(config["workspace_path"])
    database_path = workspace / config["laptop_metadata_path"]
    with open(database_path, "r") as f:
        return json.load(f)


def make_reset_options(
    config: dict,
    laptop_name: Optional[str],
    laptop_version_id: Optional[str],
    scale: Optional[float],
    luminosity: Optional[float],
) -> Dict[str, Any]:
    env_config = config["eval"]["env"]
    options: Dict[str, Any] = {}

    selected_laptop = laptop_name or env_config.get("laptop_name")
    selected_version = laptop_version_id or env_config.get("laptop_version_id")
    selected_scale = scale if scale is not None else env_config.get("scale")
    selected_luminosity = (
        luminosity if luminosity is not None else env_config.get("set_luminosity")
    )

    if (selected_laptop is None) != (selected_version is None):
        raise ValueError("laptop_name and laptop_version_id must be set together")

    if selected_laptop is not None:
        database = load_laptop_database(config)
        if selected_laptop not in database["laptops"]:
            available = ", ".join(database["laptops"].keys())
            raise ValueError(f"Unknown laptop '{selected_laptop}'. Available: {available}")
        if selected_version not in database["laptops"][selected_laptop]["version_objs"]:
            available = ", ".join(database["laptops"][selected_laptop]["version_objs"].keys())
            raise ValueError(
                f"Version '{selected_version}' is not available for '{selected_laptop}'. "
                f"Available: {available}"
            )
        options["laptop_name"] = selected_laptop
        options["laptop_version_id"] = selected_version

    initial_screws = env_config.get("initial_confirmed_screws")
    if initial_screws is not None:
        options["initial_confirmed_screws"] = initial_screws
    if selected_scale is not None:
        options["scale"] = selected_scale
    if selected_luminosity is not None:
        options["luminosity"] = selected_luminosity

    return options


def evaluate_model(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    env, log_dir, experiment_name = build_env(config)
    control_mode = config["eval"]["env"]["control_mode"]
    model = None
    if control_mode == "policy":
        checkpoint = Path(args.checkpoint) if args.checkpoint else (
            log_dir / f"{experiment_name}_{config['train']['total_timesteps']}_steps"
        )
        model = PPO.load(str(checkpoint), env=env)

    reset_options = make_reset_options(
        config=config,
        laptop_name=args.laptop,
        laptop_version_id=args.version,
        scale=args.scale,
        luminosity=args.luminosity,
    )

    try:
        obs, reset_info = env.reset(seed=args.seed, options=reset_options or None)
        print(
            "Evaluation setup: "
            f"laptop={reset_info['current_laptop_name']}, "
            f"version={reset_info['current_laptop_version_id']}, "
            f"scale={reset_info['current_scale']}, "
            f"luminosity={reset_info['current_luminosity']}, "
            f"control={control_mode}"
        )

        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if control_mode == "keyboard":
                action = None
            else:
                action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            step += 1

        print(f"Evaluation completed. Total Reward: {total_reward:.2f}, Steps: {step}")
    finally:
        env.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one laptop with policy or keyboard control.")
    parser.add_argument("--config", default="configs/config_v3.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--laptop", default=None, help="Laptop name from laptop_database.json")
    parser.add_argument("--version", default=None, help="Four-character version ID, e.g. 0001")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--luminosity", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_model(parse_args())
