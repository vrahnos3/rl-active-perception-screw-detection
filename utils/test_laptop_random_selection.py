from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import random

import cv2
import yaml

from screw_env_v3 import ScrewDetectorEnv


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_env(config: dict) -> ScrewDetectorEnv:
    workspace = Path(config["workspace_path"])
    env_config = config["train"]["env"]

    return ScrewDetectorEnv(
        yolo_path=str(workspace / config["yolo_model"]),
        visualize_detections=False,
        max_steps=env_config["max_steps"],
        control_mode=env_config["control_mode"],
        mode=env_config["mode"],
        laptop_metadata_path=str(workspace / "controllers" / "laptop_database.json"),
    )


def print_reset_result(env: ScrewDetectorEnv, reset_index: int, info: dict) -> tuple[str, str]:
    laptop_name = info["current_laptop_name"]
    laptop_version_id = info["current_laptop_version_id"]
    obj_url = env.laptop_model_url.getMFString(0)
    gt_classes = {name: env.screws[name].gt_class for name in env.target_names}
    
    # Grab the current scale
    current_scale = info["current_scale"]  

    # Grab the target that the environment actually selected
    active_target_name = env.current_target_name
    sample_target = env.screws[active_target_name].point if active_target_name else None

    print(f"\nReset {reset_index:03d}")
    print(f"  laptop:  {laptop_name}")
    print(f"  version: {laptop_version_id}")
    print(f"  scale:   {current_scale:.3f}")           # See the random 0.85 - 1.15 value
    print(f"  target:  {sample_target}")               # See the math updating the World coords
    print(f"  obj_url: {obj_url}")
    print(f"  classes: {gt_classes}")

    return laptop_name, laptop_version_id


def test_random_selection(config_path: str, resets: int, seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)

    config = load_config(config_path)
    env = build_env(config)
    counts = Counter()

    try:
        for reset_index in range(1, resets + 1):
            _, info = env.reset()
            laptop_name, laptop_version_id = print_reset_result(env, reset_index, info)
            counts[(laptop_name, laptop_version_id)] += 1

        print("\nSelection counts")
        for (laptop_name, laptop_version_id), count in sorted(counts.items()):
            print(f"  {laptop_name} / {laptop_version_id}: {count}")
    finally:
        env.close()
        cv2.destroyAllWindows()


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Test random laptop/version selection in screw_env_v3.")
    parser.add_argument(
        "--config",
        default="C:/Users/panosvrachnos/Documents/thesis_env/controllers/config_v2.yaml",
        help="Path to config_v2.yaml.",
    )
    parser.add_argument(
        "--resets",
        type=int,
        default=10,
        help="Number of resets to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional Python random seed for repeatable selection.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_random_selection(
        config_path=args.config,
        resets=args.resets,
        seed=args.seed,
    )
