import argparse
import csv
import json
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml
from stable_baselines3 import PPO
from tqdm import tqdm


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


AVAILABLE_SCALES = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
AVAILABLE_LUMINOSITIES = [0.3, 1.5, 3.0]
DETAIL_FIELDS = [
    "laptop_name",
    "laptop_version_id",
    "scale",
    "luminosity",
    "episode_idx",
    "seed",
    "execution_sequence",
    "present_screws",
    "success",
    "failure_reason",
    "steps",
    "reward",
    "episode_seconds",
    "confirmed_count",
    "confirmed_screws",
    "terminated",
    "truncated",
]
SUMMARY_FIELDS = [
    "laptop_name",
    "laptop_version_id",
    "scale",
    "luminosity",
    "episodes",
    "successes",
    "success_rate",
    "avg_reward",
    "std_reward",
    "avg_steps",
    "std_steps",
    "avg_confirmed_count",
    "timeouts",
    "out_of_bounds",
]


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        path = Path(__file__).resolve().parent / config_path

    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_env(config: dict, verbose: bool = False) -> ScrewDetectorEnv:
    eval_config = config["eval"]
    env_config = eval_config["env"]

    workspace = Path(config["workspace_path"])
    yolo_path = workspace / config["yolo_model"]
    laptop_metadata_path = workspace / config["laptop_metadata_path"]

    return ScrewDetectorEnv(
        yolo_path=str(yolo_path),
        visualize_detections=env_config["visualize_detections"],
        max_steps=env_config["max_steps"],
        control_mode=env_config["control_mode"],
        mode=env_config["mode"],
        laptop_metadata_path=str(laptop_metadata_path),
        verbose=verbose,
    )


def default_checkpoint_path(config: dict) -> Path:
    workspace = Path(config["workspace_path"])
    experiment_name = config["experiment_name"]
    log_dir = workspace / config["log_dir"] / experiment_name
    return log_dir / f"{experiment_name}_{config['train']['total_timesteps']}_steps"


def load_laptop_database(config: dict) -> Dict[str, Any]:
    workspace = Path(config["workspace_path"])
    laptop_metadata_path = workspace / config["laptop_metadata_path"]
    with open(laptop_metadata_path, "r") as f:
        return json.load(f)


def count_present_screws(database: Dict[str, Any], version_id: str) -> int:
    classes = database["versions"][version_id]["classes"].values()
    return sum(class_name == "screw" for class_name in classes)


def make_reset_options(
    laptop_name: str,
    laptop_version_id: str,
    initial_confirmed_screws: Optional[List[bool]],
    scale: Optional[float],
    luminosity: Optional[float],
) -> Dict[str, Any]:
    options = {
        "laptop_name": laptop_name,
        "laptop_version_id": laptop_version_id,
    }
    if scale is not None:
        options["scale"] = scale
    if luminosity is not None:
        options["luminosity"] = luminosity
    if initial_confirmed_screws is not None:
        options["initial_confirmed_screws"] = initial_confirmed_screws
    return options


def run_episode(
    env: ScrewDetectorEnv,
    model: PPO,
    laptop_name: str,
    laptop_version_id: str,
    initial_confirmed_screws: Optional[List[bool]],
    scale: Optional[float],
    luminosity: Optional[float],
    seed: int,
    deterministic: bool,
) -> Dict[str, Any]:
    model.set_random_seed(seed)
    episode_start = time.time()
    obs, reset_info = env.reset(
        seed=seed,
        options=make_reset_options(
            laptop_name=laptop_name,
            laptop_version_id=laptop_version_id,
            initial_confirmed_screws=initial_confirmed_screws,
            scale=scale,
            luminosity=luminosity,
        )
    )

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info: Dict[str, Any] = dict(reset_info)

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    confirmed_screws = info.get("confirmed_screws", [])
    confirmed_count = int(sum(confirmed_screws))
    success = bool(info.get("all_done", False))
    if success:
        failure_reason = ""
    elif truncated:
        failure_reason = "timeout"
    else:
        failure_reason = "out_of_bounds"

    return {
        "laptop_name": laptop_name,
        "laptop_version_id": laptop_version_id,
        "scale": reset_info.get("current_scale"),
        "luminosity": reset_info.get("current_luminosity"),
        "seed": seed,
        "execution_sequence": reset_info.get("execution_sequence", []),
        "success": success,
        "failure_reason": failure_reason,
        "steps": steps,
        "reward": total_reward,
        "episode_seconds": time.time() - episode_start,
        "confirmed_count": confirmed_count,
        "confirmed_screws": confirmed_screws,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def run_episode_with_optional_logs(
    show_step_logs: bool,
    **episode_kwargs: Any,
) -> Dict[str, Any]:
    if show_step_logs:
        return run_episode(**episode_kwargs)

    with open(os.devnull, "w") as output_sink, redirect_stdout(output_sink):
        return run_episode(**episode_kwargs)


def summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["laptop_name"],
            row["laptop_version_id"],
            row["scale"],
            row["luminosity"],
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for condition, condition_rows in sorted(grouped.items()):
        laptop_name, version_id, scale, luminosity = condition
        total = len(condition_rows)
        successes = sum(row["success"] for row in condition_rows)
        rewards = [row["reward"] for row in condition_rows]
        steps = [row["steps"] for row in condition_rows]
        summary.append(
            {
                "laptop_name": laptop_name,
                "laptop_version_id": version_id,
                "scale": scale,
                "luminosity": luminosity,
                "episodes": total,
                "successes": successes,
                "success_rate": successes / total if total else 0.0,
                "avg_reward": float(np.mean(rewards)) if total else 0.0,
                "std_reward": float(np.std(rewards)) if total else 0.0,
                "avg_steps": float(np.mean(steps)) if total else 0.0,
                "std_steps": float(np.std(steps)) if total else 0.0,
                "avg_confirmed_count": float(np.mean([row["confirmed_count"] for row in condition_rows])) if total else 0.0,
                "timeouts": sum(row["failure_reason"] == "timeout" for row in condition_rows),
                "out_of_bounds": sum(row["failure_reason"] == "out_of_bounds" for row in condition_rows),
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


def append_csv_row(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(excel_safe_row(row))


def evaluate_all_laptops(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    database = load_laptop_database(config)
    env = build_env(config, verbose=args.show_step_logs)

    checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(config)
    model = PPO.load(str(checkpoint), env=env)

    initial_screws = config["eval"]["env"].get("initial_confirmed_screws")
    if initial_screws is None:
        initial_screws = [False] * len(database["screws"])

    laptops = database["laptops"]
    selected_laptops = [args.laptop] if args.laptop else sorted(laptops.keys())
    if args.all_scales and args.scale is not None:
        raise ValueError("Use either --all-scales or --scale, not both.")
    if args.all_luminosities and args.luminosity is not None:
        raise ValueError("Use either --all-luminosities or --luminosity, not both.")

    scales = AVAILABLE_SCALES if args.all_scales else [args.scale]
    luminosities = AVAILABLE_LUMINOSITIES if args.all_luminosities else [args.luminosity]
    episodes_per_combination = args.episodes_per_combination
    if episodes_per_combination <= 0:
        raise ValueError("--episodes-per-combination must be greater than zero.")

    workspace = Path(config["workspace_path"])
    output_dir = Path(args.output_dir) if args.output_dir else workspace / config["log_dir"] / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    detail_csv = output_dir / f"eval_all_laptops_detail_{timestamp}.csv"
    summary_csv = output_dir / f"eval_all_laptops_summary_{timestamp}.csv"
    summary_json = output_dir / f"eval_all_laptops_summary_{timestamp}.json"

    planned_conditions = 0
    for laptop_name in selected_laptops:
        if laptop_name not in laptops:
            raise ValueError(f"Unknown laptop '{laptop_name}'. Available: {sorted(laptops.keys())}")
        version_count = 1 if args.version else len(laptops[laptop_name]["version_objs"])
        planned_conditions += version_count * len(scales) * len(luminosities)
    planned_episodes = planned_conditions * episodes_per_combination

    print(f"Planned conditions: {planned_conditions}")
    print(f"Planned episodes: {planned_episodes}")
    print(f"Detailed results will be appended to: {detail_csv}")
    print(f"Completed-condition summaries will be appended to: {summary_csv}")

    rows: List[Dict[str, Any]] = []
    start_time = time.time()
    episode_number = 0
    progress = tqdm(
        total=planned_episodes,
        desc="Stress test",
        unit="episode",
        dynamic_ncols=True,
        ascii=True,
        disable=False,
        leave=True,
        file=sys.stderr,
    )

    for laptop_name in selected_laptops:
        if laptop_name not in laptops:
            raise ValueError(f"Unknown laptop '{laptop_name}'. Available: {sorted(laptops.keys())}")

        versions = sorted(laptops[laptop_name]["version_objs"].keys())
        if args.version:
            if args.version not in versions:
                raise ValueError(f"Laptop '{laptop_name}' does not have version '{args.version}'")
            versions = [args.version]

        for version_id in versions:
            for scale in scales:
                for luminosity in luminosities:
                    condition_rows: List[Dict[str, Any]] = []
                    for episode_idx in range(episodes_per_combination):
                        seed = args.base_seed + episode_number
                        episode_number += 1
                        result = run_episode_with_optional_logs(
                            show_step_logs=args.show_step_logs,
                            env=env,
                            model=model,
                            laptop_name=laptop_name,
                            laptop_version_id=version_id,
                            initial_confirmed_screws=initial_screws,
                            scale=scale,
                            luminosity=luminosity,
                            seed=seed,
                            deterministic=args.deterministic,
                        )
                        result["episode_idx"] = episode_idx + 1
                        result["present_screws"] = count_present_screws(database, version_id)
                        rows.append(result)
                        condition_rows.append(result)
                        append_csv_row(detail_csv, result, DETAIL_FIELDS)

                        progress.set_postfix(
                            laptop=laptop_name,
                            version=version_id,
                            scale=f"{result['scale']:.2f}",
                            light=f"{result['luminosity']:.1f}",
                            result="OK" if result["success"] else "FAIL",
                            refresh=False,
                        )
                        progress.update(1)

                    condition_summary = summarize_rows(condition_rows)[0]
                    append_csv_row(summary_csv, condition_summary, SUMMARY_FIELDS)

    progress.close()
    env.close()
    cv2.destroyAllWindows()

    summary = summarize_rows(rows)
    total_episodes = len(rows)
    total_successes = sum(row["success"] for row in rows)
    elapsed = time.time() - start_time

    write_csv(summary_csv, summary, SUMMARY_FIELDS)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint),
                "episodes": total_episodes,
                "successes": total_successes,
                "success_rate": total_successes / total_episodes if total_episodes else 0.0,
                "elapsed_seconds": elapsed,
                "base_seed": args.base_seed,
                "deterministic_policy": args.deterministic,
                "summary_by_condition": summary,
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
    parser = argparse.ArgumentParser(description="Stress-test a PPO agent across laptop, version, scale, and lighting combinations.")
    parser.add_argument("--config", default="configs/config_v3.yaml", help="Path to the YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional PPO checkpoint path. Defaults to config experiment final checkpoint.")
    parser.add_argument("--episodes-per-combination", type=int, default=1, help="Episodes for each laptop/version/scale/light combination.")
    parser.add_argument("--laptop", default=None, help="Optional laptop name to evaluate instead of all laptops.")
    parser.add_argument("--version", default=None, help="Optional version id to evaluate instead of all versions.")
    parser.add_argument("--scale", type=float, default=None, help="Evaluate one fixed scale. Omit for a random scale per episode.")
    parser.add_argument("--all-scales", action="store_true", help="Evaluate all seven training scales from 0.85 to 1.15.")
    parser.add_argument("--luminosity", type=float, default=None, help="Evaluate one fixed light luminosity. Omit for a random value per episode.")
    parser.add_argument("--all-luminosities", action="store_true", help="Evaluate luminosities 0.3, 1.5, and 3.0.")
    parser.add_argument("--base-seed", type=int, default=20260715, help="First episode seed; subsequent episodes use consecutive unique seeds.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic PPO actions. By default, stochastic actions are seeded per episode.")
    parser.add_argument("--show-step-logs", action="store_true", help="Show verbose environment output; hidden by default so the progress bar stays visible.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for CSV/JSON outputs.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_all_laptops(parse_args())
