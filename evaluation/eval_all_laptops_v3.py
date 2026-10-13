import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml
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


def build_env(config: dict) -> ScrewDetectorEnv:
    env_config = config["eval"]["env"]
    workspace = Path(config["workspace_path"])

    return ScrewDetectorEnv(
        yolo_path=str(workspace / config["yolo_model"]),
        visualize_detections=env_config["visualize_detections"],
        max_steps=env_config["max_steps"],
        control_mode=env_config["control_mode"],
        mode=env_config["mode"],
        laptop_metadata_path=str(workspace / config["laptop_metadata_path"]),
    )


def default_checkpoint_path(config: dict) -> Path:
    workspace = Path(config["workspace_path"])
    experiment_name = config["experiment_name"]
    log_dir = workspace / config["log_dir"] / experiment_name
    return log_dir / f"{experiment_name}_{config['train']['total_timesteps']}_steps"


def load_laptop_database(config: dict) -> Dict[str, Any]:
    workspace = Path(config["workspace_path"])
    metadata_path = workspace / config["laptop_metadata_path"]
    with open(metadata_path, "r") as f:
        return json.load(f)


def count_present_screws(database: Dict[str, Any], version_id: str) -> int:
    classes = database["versions"][version_id]["classes"].values()
    return sum(class_name == "screw" for class_name in classes)


def make_reset_options(
    laptop_name: str,
    laptop_version_id: str,
    initial_confirmed_screws: Optional[List[bool]],
) -> Dict[str, Any]:
    options = {
        "laptop_name": laptop_name,
        "laptop_version_id": laptop_version_id,
    }
    if initial_confirmed_screws is not None:
        options["initial_confirmed_screws"] = initial_confirmed_screws
    return options


def run_episode(
    env: ScrewDetectorEnv,
    model: PPO,
    laptop_name: str,
    laptop_version_id: str,
    initial_confirmed_screws: Optional[List[bool]],
) -> Dict[str, Any]:
    obs, _ = env.reset(
        options=make_reset_options(
            laptop_name=laptop_name,
            laptop_version_id=laptop_version_id,
            initial_confirmed_screws=initial_confirmed_screws,
        )
    )

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info: Dict[str, Any] = {}

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    confirmed_screws = info.get("confirmed_screws", [])
    return {
        "laptop_name": laptop_name,
        "laptop_version_id": laptop_version_id,
        "success": bool(info.get("all_done", False)),
        "steps": steps,
        "reward": total_reward,
        "confirmed_count": int(sum(confirmed_screws)),
        "confirmed_screws": confirmed_screws,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def summarize_rows(rows: List[Dict[str, Any]], database: Dict[str, Any]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["laptop_name"], []).append(row)

    summary = []
    for laptop_name, laptop_rows in sorted(grouped.items()):
        total = len(laptop_rows)
        successes = sum(row["success"] for row in laptop_rows)
        summary.append(
            {
                "laptop_name": laptop_name,
                "episodes": total,
                "successes": successes,
                "success_rate": successes / total if total else 0.0,
                "avg_reward": float(np.mean([row["reward"] for row in laptop_rows])) if total else 0.0,
                "avg_steps": float(np.mean([row["steps"] for row in laptop_rows])) if total else 0.0,
                "avg_confirmed_count": float(
                    np.mean([row["confirmed_count"] for row in laptop_rows])
                ) if total else 0.0,
                "timeouts": sum(row["truncated"] for row in laptop_rows),
                "present_screws_tested": sorted(
                    {count_present_screws(database, row["laptop_version_id"]) for row in laptop_rows}
                ),
            }
        )
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(excel_safe_row(row) for row in rows)


def excel_safe_row(row: Dict[str, Any]) -> Dict[str, Any]:
    csv_row = dict(row)
    version_id = csv_row.get("laptop_version_id")
    if version_id is not None:
        csv_row["laptop_version_id"] = f'=\"{version_id}\"'
    return csv_row


def evaluate_all_laptops(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    database = load_laptop_database(config)
    env = build_env(config)

    checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(config)
    model = PPO.load(str(checkpoint), env=env)

    initial_screws = config["eval"]["env"].get("initial_confirmed_screws")
    if initial_screws is None:
        initial_screws = [False] * len(database["screws"])

    laptops = database["laptops"]
    selected_laptops = [args.laptop] if args.laptop else sorted(laptops.keys())
    rows: List[Dict[str, Any]] = []
    start_time = time.time()

    for laptop_name in selected_laptops:
        if laptop_name not in laptops:
            raise ValueError(f"Unknown laptop '{laptop_name}'. Available: {sorted(laptops.keys())}")

        versions = sorted(laptops[laptop_name]["version_objs"].keys())
        if args.version:
            if args.version not in versions:
                raise ValueError(f"Laptop '{laptop_name}' does not have version '{args.version}'")
            versions = [args.version]

        for version_id in versions:
            for episode_idx in range(args.episodes_per_version):
                result = run_episode(
                    env=env,
                    model=model,
                    laptop_name=laptop_name,
                    laptop_version_id=version_id,
                    initial_confirmed_screws=initial_screws,
                )
                result["episode_idx"] = episode_idx + 1
                result["present_screws"] = count_present_screws(database, version_id)
                rows.append(result)

                status = "SUCCESS" if result["success"] else "FAIL"
                print(
                    f"{status} | {laptop_name} | version={version_id} | "
                    f"episode={episode_idx + 1}/{args.episodes_per_version} | "
                    f"confirmed={result['confirmed_count']}/{len(database['screws'])} | "
                    f"steps={result['steps']} | reward={result['reward']:.2f}"
                )

    env.close()
    cv2.destroyAllWindows()

    summary = summarize_rows(rows, database)
    total_episodes = len(rows)
    total_successes = sum(row["success"] for row in rows)
    elapsed = time.time() - start_time

    workspace = Path(config["workspace_path"])
    output_dir = Path(args.output_dir) if args.output_dir else workspace / config["log_dir"] / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

    detail_csv = output_dir / f"eval_all_laptops_detail_{timestamp}.csv"
    summary_csv = output_dir / f"eval_all_laptops_summary_{timestamp}.csv"
    summary_json = output_dir / f"eval_all_laptops_summary_{timestamp}.json"

    write_csv(
        detail_csv,
        rows,
        [
            "laptop_name",
            "laptop_version_id",
            "episode_idx",
            "present_screws",
            "success",
            "steps",
            "reward",
            "confirmed_count",
            "confirmed_screws",
            "terminated",
            "truncated",
        ],
    )
    write_csv(
        summary_csv,
        summary,
        [
            "laptop_name",
            "episodes",
            "successes",
            "success_rate",
            "avg_reward",
            "avg_steps",
            "avg_confirmed_count",
            "timeouts",
            "present_screws_tested",
        ],
    )
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint),
                "episodes": total_episodes,
                "successes": total_successes,
                "success_rate": total_successes / total_episodes if total_episodes else 0.0,
                "elapsed_seconds": elapsed,
                "summary_by_laptop": summary,
            },
            f,
            indent=2,
        )

    print("\nEvaluation complete")
    print(f"Episodes: {total_episodes}")
    print(f"Successes: {total_successes}")
    print(f"Success rate: {(total_successes / total_episodes * 100.0) if total_episodes else 0.0:.1f}%")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Detailed CSV: {detail_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO agent on every laptop in laptop_database.json.")
    parser.add_argument("--config", default="configs/config_v3.yaml", help="Path to the YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional PPO checkpoint path.")
    parser.add_argument("--episodes-per-version", type=int, default=1, help="Episodes for each laptop/version pair.")
    parser.add_argument("--laptop", default=None, help="Optional laptop name to evaluate instead of all laptops.")
    parser.add_argument("--version", default=None, help="Optional version id to evaluate instead of all versions.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for CSV/JSON outputs.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_all_laptops(parse_args())
