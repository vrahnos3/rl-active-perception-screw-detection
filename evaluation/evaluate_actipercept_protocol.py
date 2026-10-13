"""Run a correctness-aware ActiPercept-style evaluation.

The script is intentionally method-at-a-time because a Webots external
controller process owns one Supervisor connection.  Run it once with
``--method ppo`` and once with ``--method actipercept`` using the same
``--run-id`` and scenario arguments.  The two outputs then share an episode
schema and can be joined directly.

The paper leaves "final accuracy" and "confidence gain" underspecified.  This
runner therefore preserves accepted, last-query, best-so-far, and first-query
values and uses the following declared definitions in its summary:

* verified success: first threshold crossing is correct;
* false verification: first threshold crossing is incorrect;
* final accuracy: the accepted prediction, or the best-so-far target prediction
  on timeout, is correct;
* confidence gain: maximum observed target confidence minus a standardized,
  evaluator-only target score acquired at the home pose. A separate observed
  gain subtracts the controller's first detector-query score;
* steps: environment/control steps, averaged over verified successes for the
  paper-compatible ``steps_verified_mean`` metric.

Raw episode rows are flushed to CSV as they finish so a long Webots run still
leaves useful partial results if it is interrupted.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
from importlib import metadata as importlib_metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# This entry point lives below controllers, but the environment imports the
# sibling test and utils packages. Prefer those over installed namesakes.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import evaluation_sequential as sequential


PAPER_LUMINOSITIES: Tuple[float, ...] = (0.5, 0.7, 1.0, 1.3, 1.5, 2.0)
TRAINING_SCALES: Tuple[float, ...] = (0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15)
CORNER_TO_TARGET: Mapping[str, str] = {
    "TL": "left_up",
    "TR": "right_up",
    "BL": "left_down",
    "BR": "right_down",
}
# ActiPercept's internal corner order is [TR, BR, TL, BL].
CORNER_TO_ACTIPERCEPT_INDEX: Mapping[str, int] = {
    "TR": 0,
    "BR": 1,
    "TL": 2,
    "BL": 3,
}


DETAIL_FIELDS: Tuple[str, ...] = (
    "run_id",
    "method",
    "setting",
    "episode_id",
    "episode_index",
    "seed",
    "laptop_name",
    "laptop_version_id",
    "target_corner",
    "target_name",
    "ground_truth_class",
    "scale",
    "luminosity",
    "actual_luminosity",
    "threshold",
    "max_steps",
    "verified_success",
    "false_verification",
    "threshold_crossed",
    "verification_outcome",
    "abstained",
    "accepted_prediction_class",
    "accepted_prediction_confidence",
    "accepted_prediction_step",
    "final_prediction_class",
    "final_prediction_confidence",
    "final_prediction_correct",
    "last_prediction_class",
    "last_prediction_confidence",
    "last_prediction_correct",
    "best_prediction_class",
    "best_prediction_confidence",
    "best_prediction_correct",
    "best_prediction_step",
    "initial_target_confidence",
    "initial_probe_prediction_class",
    "initial_probe_prediction_correct",
    "initial_probe_inference_time_s",
    "first_observed_target_confidence",
    "last_target_confidence",
    "max_target_confidence",
    "confidence_gain",
    "confidence_gain_raw",
    "positive_confidence_gain",
    "observed_confidence_gain_raw",
    "observed_positive_confidence_gain",
    "steps",
    "steps_to_verification",
    "detector_calls",
    "detection_action_count",
    "movement_action_count",
    "no_op_action_count",
    "path_length",
    "return_home_path_length",
    "total_path_length",
    "detector_inference_time_s",
    "detector_fps",
    "detector_call_fraction",
    "effective_detection_hz",
    "ppo_inference_time_s",
    "ppo_inference_fps",
    "active_wall_time_s",
    "return_home_steps",
    "return_home_time_s",
    "total_wall_time_s",
    "processing_fps",
    "active_simulated_time_s",
    "total_simulated_time_s",
    "terminated",
    "truncated",
    "failure_reason",
    "raw_info_json",
)


SUMMARY_FIELDS: Tuple[str, ...] = (
    "grouping",
    "method",
    "setting",
    "luminosity",
    "target_corner",
    "ground_truth_class",
    "episodes",
    "verified_successes",
    "verified_success_rate",
    "verified_success_ci95_low",
    "verified_success_ci95_high",
    "false_verifications",
    "false_verification_rate",
    "false_verification_ci95_low",
    "false_verification_ci95_high",
    "accepted_episodes",
    "acceptance_rate",
    "selective_accuracy",
    "final_correct",
    "final_accuracy",
    "final_accuracy_ci95_low",
    "final_accuracy_ci95_high",
    "last_correct",
    "last_accuracy",
    "best_correct",
    "best_accuracy",
    "confidence_gain_mean",
    "confidence_gain_std",
    "positive_confidence_gain_mean",
    "observed_confidence_gain_mean",
    "steps_verified_mean",
    "steps_verified_std",
    "steps_all_mean",
    "detector_calls_mean",
    "detection_action_count_mean",
    "movement_action_count_mean",
    "path_length_mean",
    "processing_fps",
    "detector_fps",
    "detector_call_fraction",
    "effective_detection_hz",
    "ppo_inference_fps",
    "timeouts",
    "out_of_bounds",
)


@dataclass(frozen=True)
class Scenario:
    setting: str
    laptop_name: str
    laptop_version_id: str
    target_corner: str
    target_name: str
    ground_truth_class: str
    scale: float
    luminosity: float
    episode_index: int
    seed: int
    include_scale_in_id: bool = False

    @property
    def episode_id(self) -> str:
        light = f"{self.luminosity:.2f}".replace(".", "p")
        scale_id = f"_X{str(self.scale).replace('.', 'p')}" if self.include_scale_in_id else ""
        return (
            f"{self.setting}_{self.laptop_name}_{self.laptop_version_id}_"
            f"{self.target_corner}{scale_id}_L{light}_E{self.episode_index:03d}_S{self.seed}"
        )


def add_webots_python_controller_path() -> None:
    webots_home = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")
    controller_python_path = Path(webots_home) / "lib" / "controller" / "python"
    controller_dll_path = Path(webots_home) / "lib" / "controller"
    if controller_python_path.exists():
        sys.path.insert(0, str(controller_python_path))
    if controller_dll_path.exists():
        os.environ["PATH"] = f"{controller_dll_path}{os.pathsep}{os.environ.get('PATH', '')}"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def resolve_from_workspace(workspace: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def sha256_or_none(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def collect_runtime_metadata() -> Dict[str, Any]:
    runtime: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "webots_home": os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots"),
        "packages": {
            name: package_version(name)
            for name in ("gymnasium", "numpy", "stable-baselines3", "torch", "ultralytics")
        },
    }
    try:
        import torch

        runtime["cuda_available"] = bool(torch.cuda.is_available())
        runtime["cuda_version"] = torch.version.cuda
        runtime["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception as exc:
        runtime["torch_probe_error"] = str(exc)
    return runtime


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def first_present(info: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in info and info[key] is not None:
            return info[key]
    return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def as_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def mean_or_none(values: Iterable[Any]) -> Optional[float]:
    numbers = [number for value in values if (number := as_optional_float(value)) is not None]
    return statistics.fmean(numbers) if numbers else None


def std_or_none(values: Iterable[Any]) -> Optional[float]:
    numbers = [number for value in values if (number := as_optional_float(value)) is not None]
    return statistics.stdev(numbers) if len(numbers) >= 2 else (0.0 if numbers else None)


def ratio_or_none(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator > 0 else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[Optional[float], Optional[float]]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - radius, center + radius


def normalize_outcome(info: Mapping[str, Any]) -> str:
    explicit = first_present(info, "verification_outcome", "outcome")
    if explicit is not None:
        return str(explicit)
    if as_bool(first_present(info, "verified_success", default=False)):
        return "correct_verification"
    if as_bool(first_present(info, "false_verification", default=False)):
        return "false_verification"
    return "not_verified"


def first_trace_confidence(info: Mapping[str, Any]) -> float:
    explicit = first_present(info, "first_observed_target_confidence")
    if explicit is not None:
        return as_float(explicit)
    trace = info.get("prediction_trace", [])
    if isinstance(trace, Sequence) and not isinstance(trace, (str, bytes)):
        for item in trace:
            if isinstance(item, Mapping):
                return as_float(first_present(item, "confidence", "predicted_confidence", default=0.0))
    return 0.0


def build_detail_row(
    *,
    run_id: str,
    method: str,
    scenario: Scenario,
    threshold: float,
    max_steps: int,
    info: Mapping[str, Any],
    terminated: bool,
    truncated: bool,
    loop_steps: int,
    loop_wall_time_s: float,
    ppo_inference_time_s: float = 0.0,
) -> Dict[str, Any]:
    outcome = normalize_outcome(info)
    verified_success = as_bool(first_present(
        info,
        "verified_success",
        "correct_verification",
        default=outcome in {"correct", "correct_verification", "verified_success"},
    ))
    false_verification = as_bool(first_present(
        info,
        "false_verification",
        default=outcome in {"false", "incorrect", "false_verification"},
    ))
    threshold_crossed = as_bool(first_present(
        info,
        "threshold_crossed",
        "detection_threshold_crossed",
        default=verified_success or false_verification,
    ))

    accepted_class = first_present(
        info,
        "accepted_prediction_class",
        "accepted_predicted_class",
        "accepted_pred_class",
        "accepted_class",
    )
    accepted_confidence = as_optional_float(first_present(
        info,
        "accepted_prediction_confidence",
        "accepted_predicted_confidence",
        "accepted_pred_confidence",
        "accepted_confidence",
    ))
    accepted_step_value = first_present(
        info,
        "accepted_prediction_step",
        "verification_step",
        "threshold_crossing_step",
    )
    accepted_step = int(accepted_step_value) if accepted_step_value is not None else None

    last_class = first_present(
        info,
        "last_prediction_class",
        "predicted_class",
        "detection_predicted_class",
    )
    last_confidence = as_float(first_present(
        info,
        "last_prediction_confidence",
        "last_target_confidence",
        "predicted_confidence",
        "detection_predicted_confidence",
        default=0.0,
    ))
    last_correct_value = first_present(
        info,
        "last_prediction_correct",
        "prediction_correct",
        "detection_correct",
    )
    last_correct = (
        as_bool(last_correct_value)
        if last_correct_value is not None
        else last_class is not None and str(last_class) == scenario.ground_truth_class
    )

    best_class = first_present(
        info,
        "best_prediction_class",
        "best_class",
    )
    best_confidence = as_float(first_present(
        info,
        "best_prediction_confidence",
        "max_target_confidence",
        "best_confidence",
        "best_conf",
        default=last_confidence,
    ))
    best_correct_value = first_present(info, "best_prediction_correct", "best_correct")
    best_correct = (
        as_bool(best_correct_value)
        if best_correct_value is not None
        else best_class is not None and str(best_class) == scenario.ground_truth_class
    )
    best_step_value = first_present(info, "best_prediction_step", "best_confidence_step")
    best_step = int(best_step_value) if best_step_value is not None else None

    # For the paper-compatible final decision, use the accepted candidate when
    # one crossed the threshold; otherwise use the best-so-far candidate.  Both
    # the last and best variants remain in the raw row because the paper does
    # not define Final Accuracy precisely.
    if threshold_crossed and accepted_class is not None:
        final_class = accepted_class
        final_confidence = accepted_confidence if accepted_confidence is not None else 0.0
    elif best_class is not None:
        final_class = best_class
        final_confidence = best_confidence
    else:
        final_class = last_class
        final_confidence = last_confidence
    # The runner owns this declared definition.  Do not reuse an environment's
    # boolean if it describes a different candidate (for example the last query
    # while final_class is the best-so-far query).
    final_prediction_correct = (
        final_class is not None and str(final_class) == scenario.ground_truth_class
    )

    # The paper does not state Gain's baseline. The evaluator supplies the same
    # selected-region detector probe at the home pose for both controllers.
    initial_confidence = as_float(first_present(
        info,
        "initial_home_target_confidence",
        "initial_target_confidence",
        default=0.0,
    ))
    initial_probe_class = first_present(info, "initial_home_prediction_class")
    initial_probe_correct = as_bool(first_present(
        info,
        "initial_home_prediction_correct",
        default=(
            initial_probe_class is not None
            and str(initial_probe_class) == scenario.ground_truth_class
        ),
    ))
    initial_probe_time = as_float(first_present(
        info,
        "initial_home_probe_inference_time_s",
        default=0.0,
    ))
    first_observed_confidence = first_trace_confidence(info)
    last_target_confidence = as_float(first_present(
        info,
        "last_target_confidence",
        "final_prediction_confidence",
        "predicted_confidence",
        default=last_confidence,
    ))
    max_confidence = as_float(first_present(
        info,
        "max_target_confidence",
        "best_confidence",
        "best_conf",
        default=max(initial_confidence, best_confidence, last_target_confidence),
    ))
    confidence_gain_raw = max_confidence - initial_confidence
    confidence_gain = max(0.0, confidence_gain_raw)
    observed_confidence_gain_raw = max_confidence - first_observed_confidence

    steps = int(first_present(info, "env_step_count", "step_count", default=loop_steps))
    detector_calls = int(first_present(info, "detector_calls", "detector_call_count", default=0))
    detection_action_count = int(first_present(
        info,
        "detection_action_count",
        default=detector_calls if method == "ppo" else 0,
    ))
    movement_action_count = int(first_present(
        info,
        "movement_action_count",
        "motion_action_count",
        default=0,
    ))
    no_op_action_count = int(first_present(info, "no_op_action_count", default=0))
    path_length = as_float(first_present(
        info,
        "path_length",
        "motion_path_length",
        "camera_path_length",
        default=0.0,
    ))
    return_home_path_length = as_float(first_present(info, "return_home_path_length", default=0.0))
    total_path_length = as_float(first_present(
        info,
        "total_path_length",
        default=path_length + return_home_path_length,
    ))
    detector_time = as_float(first_present(
        info,
        "detector_inference_time_s",
        "cumulative_yolo_inference_time_s",
        default=0.0,
    ))
    detector_fps = as_float(first_present(
        info,
        "detector_fps",
        default=ratio_or_none(detector_calls, detector_time) or 0.0,
    ))
    return_home_steps = int(first_present(
        info,
        "return_home_steps",
        "return_home_motion_count",
        default=0,
    ))
    return_home_time = as_float(first_present(info, "return_home_time_s", default=0.0))
    total_wall_time = as_float(first_present(
        info,
        "elapsed_wall_time_s",
        "wall_time_s",
        default=loop_wall_time_s,
    ))
    active_wall_time = as_float(first_present(
        info,
        "active_wall_time_s",
        default=max(0.0, total_wall_time - return_home_time),
    ))
    processing_fps = as_float(first_present(
        info,
        "processing_fps",
        "episode_fps",
        default=ratio_or_none(steps, active_wall_time) or 0.0,
    ))
    ppo_inference_fps = (
        ratio_or_none(steps, ppo_inference_time_s) or 0.0
        if method == "ppo"
        else 0.0
    )
    timestep_ms = as_float(first_present(info, "timestep_ms", default=64.0), default=64.0)
    active_simulated_time = as_float(first_present(
        info,
        "active_simulated_time_s",
        "simulated_time_s",
        default=steps * timestep_ms / 1000.0,
    ))
    total_simulated_time = as_float(first_present(
        info,
        "total_simulated_time_s",
        "simulated_time_s",
        default=(steps + return_home_steps) * timestep_ms / 1000.0,
    ))

    failure_reason = str(first_present(info, "failure_reason", default=""))
    if not failure_reason and not verified_success:
        if false_verification:
            failure_reason = "false_verification"
        elif truncated:
            failure_reason = "timeout"
        elif as_bool(first_present(info, "out_of_bounds", default=False)):
            failure_reason = "out_of_bounds"
        else:
            failure_reason = "not_verified"

    return {
        "run_id": run_id,
        "method": method,
        "setting": scenario.setting,
        "episode_id": scenario.episode_id,
        "episode_index": scenario.episode_index,
        "seed": scenario.seed,
        "laptop_name": scenario.laptop_name,
        "laptop_version_id": scenario.laptop_version_id,
        "target_corner": scenario.target_corner,
        "target_name": scenario.target_name,
        "ground_truth_class": scenario.ground_truth_class,
        "scale": scenario.scale,
        "luminosity": scenario.luminosity,
        "actual_luminosity": as_float(first_present(info, "current_luminosity", "actual_luminosity", default=scenario.luminosity)),
        "threshold": threshold,
        "max_steps": max_steps,
        "verified_success": verified_success,
        "false_verification": false_verification,
        "threshold_crossed": threshold_crossed,
        "verification_outcome": outcome,
        "abstained": not threshold_crossed,
        "accepted_prediction_class": accepted_class or "",
        "accepted_prediction_confidence": accepted_confidence if accepted_confidence is not None else "",
        "accepted_prediction_step": accepted_step if accepted_step is not None else "",
        "final_prediction_class": final_class or "",
        "final_prediction_confidence": final_confidence,
        "final_prediction_correct": final_prediction_correct,
        "last_prediction_class": last_class or "",
        "last_prediction_confidence": last_confidence,
        "last_prediction_correct": last_correct,
        "best_prediction_class": best_class or "",
        "best_prediction_confidence": best_confidence,
        "best_prediction_correct": best_correct,
        "best_prediction_step": best_step if best_step is not None else "",
        "initial_target_confidence": initial_confidence,
        "initial_probe_prediction_class": initial_probe_class or "",
        "initial_probe_prediction_correct": initial_probe_correct,
        "initial_probe_inference_time_s": initial_probe_time,
        "first_observed_target_confidence": first_observed_confidence,
        "last_target_confidence": last_target_confidence,
        "max_target_confidence": max_confidence,
        "confidence_gain": confidence_gain,
        "confidence_gain_raw": confidence_gain_raw,
        "positive_confidence_gain": confidence_gain,
        "observed_confidence_gain_raw": observed_confidence_gain_raw,
        "observed_positive_confidence_gain": max(0.0, observed_confidence_gain_raw),
        "steps": steps,
        "steps_to_verification": (accepted_step if accepted_step is not None else steps) if verified_success else "",
        "detector_calls": detector_calls,
        "detection_action_count": detection_action_count,
        "movement_action_count": movement_action_count,
        "no_op_action_count": no_op_action_count,
        "path_length": path_length,
        "return_home_path_length": return_home_path_length,
        "total_path_length": total_path_length,
        "detector_inference_time_s": detector_time,
        "detector_fps": detector_fps,
        "detector_call_fraction": ratio_or_none(detector_calls, steps) or 0.0,
        "effective_detection_hz": ratio_or_none(detector_calls, active_wall_time) or 0.0,
        "ppo_inference_time_s": ppo_inference_time_s,
        "ppo_inference_fps": ppo_inference_fps,
        "active_wall_time_s": active_wall_time,
        "return_home_steps": return_home_steps,
        "return_home_time_s": return_home_time,
        "total_wall_time_s": total_wall_time,
        "processing_fps": processing_fps,
        "active_simulated_time_s": active_simulated_time,
        "total_simulated_time_s": total_simulated_time,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "failure_reason": failure_reason,
        "raw_info_json": json.dumps(json_safe(info), sort_keys=True),
    }


def summarize_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty row group")
    total = len(rows)
    successes = sum(as_bool(row["verified_success"]) for row in rows)
    false_verifications = sum(as_bool(row["false_verification"]) for row in rows)
    accepted = successes + false_verifications
    final_correct = sum(as_bool(row["final_prediction_correct"]) for row in rows)
    last_correct = sum(as_bool(row["last_prediction_correct"]) for row in rows)
    best_correct = sum(as_bool(row["best_prediction_correct"]) for row in rows)
    ci_low, ci_high = wilson_interval(successes, total)
    false_ci_low, false_ci_high = wilson_interval(false_verifications, total)
    final_ci_low, final_ci_high = wilson_interval(final_correct, total)

    total_active_time = sum(as_float(row["active_wall_time_s"]) for row in rows)
    total_steps = sum(as_float(row["steps"]) for row in rows)
    total_detector_time = sum(as_float(row["detector_inference_time_s"]) for row in rows)
    total_detector_calls = sum(as_float(row["detector_calls"]) for row in rows)
    total_ppo_time = sum(as_float(row["ppo_inference_time_s"]) for row in rows)

    return {
        "grouping": "",
        "method": rows[0]["method"],
        "setting": rows[0]["setting"],
        "luminosity": rows[0]["luminosity"],
        "target_corner": rows[0]["target_corner"],
        "ground_truth_class": rows[0]["ground_truth_class"],
        "episodes": total,
        "verified_successes": successes,
        "verified_success_rate": successes / total,
        "verified_success_ci95_low": ci_low,
        "verified_success_ci95_high": ci_high,
        "false_verifications": false_verifications,
        "false_verification_rate": false_verifications / total,
        "false_verification_ci95_low": false_ci_low,
        "false_verification_ci95_high": false_ci_high,
        "accepted_episodes": accepted,
        "acceptance_rate": accepted / total,
        "selective_accuracy": ratio_or_none(successes, accepted),
        "final_correct": final_correct,
        "final_accuracy": final_correct / total,
        "final_accuracy_ci95_low": final_ci_low,
        "final_accuracy_ci95_high": final_ci_high,
        "last_correct": last_correct,
        "last_accuracy": last_correct / total,
        "best_correct": best_correct,
        "best_accuracy": best_correct / total,
        "confidence_gain_mean": mean_or_none(row["confidence_gain"] for row in rows),
        "confidence_gain_std": std_or_none(row["confidence_gain"] for row in rows),
        "positive_confidence_gain_mean": mean_or_none(row["positive_confidence_gain"] for row in rows),
        "observed_confidence_gain_mean": mean_or_none(
            row["observed_confidence_gain_raw"] for row in rows
        ),
        "steps_verified_mean": mean_or_none(
            row["steps_to_verification"] for row in rows if as_bool(row["verified_success"])
        ),
        "steps_verified_std": std_or_none(
            row["steps_to_verification"] for row in rows if as_bool(row["verified_success"])
        ),
        "steps_all_mean": mean_or_none(row["steps"] for row in rows),
        "detector_calls_mean": mean_or_none(row["detector_calls"] for row in rows),
        "detection_action_count_mean": mean_or_none(row["detection_action_count"] for row in rows),
        "movement_action_count_mean": mean_or_none(row["movement_action_count"] for row in rows),
        "path_length_mean": mean_or_none(row["path_length"] for row in rows),
        "processing_fps": ratio_or_none(total_steps, total_active_time),
        "detector_fps": ratio_or_none(total_detector_calls, total_detector_time),
        "detector_call_fraction": ratio_or_none(total_detector_calls, total_steps),
        "effective_detection_hz": ratio_or_none(total_detector_calls, total_active_time),
        "ppo_inference_fps": ratio_or_none(total_steps, total_ppo_time) if total_ppo_time > 0 else None,
        "timeouts": sum(str(row["failure_reason"]) == "timeout" for row in rows),
        "out_of_bounds": sum(str(row["failure_reason"]) == "out_of_bounds" for row in rows),
    }


def grouped_summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return paper table, Figure 3, and diagnostic summary scopes."""
    specifications: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("overall", ("method",)),
        ("setting", ("method", "setting")),
        ("setting_luminosity", ("method", "setting", "luminosity")),
        ("setting_corner", ("method", "setting", "target_corner")),
        ("setting_ground_truth", ("method", "setting", "ground_truth_class")),
        (
            "setting_luminosity_corner",
            ("method", "setting", "luminosity", "target_corner"),
        ),
    )
    output: List[Dict[str, Any]] = []
    display_fields = ("setting", "luminosity", "target_corner", "ground_truth_class")
    for grouping, keys in specifications:
        grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
        for row in rows:
            key = tuple(row[field] for field in keys)
            grouped.setdefault(key, []).append(row)
        for key in sorted(grouped, key=lambda value: tuple(map(str, value))):
            summary = summarize_group(grouped[key])
            summary["grouping"] = grouping
            key_values = dict(zip(keys, key))
            for field in display_fields:
                summary[field] = key_values.get(field, "all")
            output.append(summary)
    return output


def append_csv_row(path: Path, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
        stream.flush()


def append_prediction_trace(
    path: Path,
    *,
    run_id: str,
    method: str,
    scenario: Scenario,
    info: Mapping[str, Any],
) -> None:
    trace = info.get("prediction_trace", [])
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        return
    with path.open("a", encoding="utf-8") as stream:
        for query_index, event in enumerate(trace, start=1):
            if not isinstance(event, Mapping):
                continue
            record = {
                "run_id": run_id,
                "method": method,
                "episode_id": scenario.episode_id,
                "setting": scenario.setting,
                "target_corner": scenario.target_corner,
                "target_name": scenario.target_name,
                "ground_truth_class": scenario.ground_truth_class,
                "luminosity": scenario.luminosity,
                "seed": scenario.seed,
                "query_index": query_index,
                **json_safe(event),
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def resolve_versions(requested: Sequence[str], database: Mapping[str, Any]) -> List[str]:
    available = sorted(str(value) for value in database["versions"].keys())
    if len(requested) == 1 and requested[0].lower() == "all":
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown version(s) {unknown}; available versions: {available}")
    return list(requested)


def resolve_laptops(requested: Sequence[str], database: Mapping[str, Any]) -> List[str]:
    """Resolve exact names, shell-style patterns, or 'all' in database order."""
    available = list(database["laptops"])
    selected = set()
    for value in requested:
        matches = available if value == "all" else fnmatch.filter(available, value)
        if not matches:
            raise ValueError(f"No laptop matches {value!r}; available: {available}")
        selected.update(matches)
    if not selected:
        raise ValueError("Select at least one laptop")
    return [name for name in available if name in selected]


def build_scenario_matrix(
    *, database: Mapping[str, Any], laptop_names: Sequence[str],
    scales: Sequence[float], **protocol: Any,
) -> List[Scenario]:
    """Extend the legacy matrix without changing any single-block seeds/order."""
    if not scales or any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("Scales must be finite and positive")
    if len(set(scales)) != len(scales):
        raise ValueError("Duplicate scales would repeat evaluation cases")
    for name in ("versions", "corners", "illumination_values"):
        values = protocol[name]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be nonempty and contain no duplicates")
    lights = protocol["illumination_values"]
    if any(not math.isfinite(value) or value < 0 for value in lights):
        raise ValueError("Luminosities must be finite and non-negative")
    nominal = protocol["nominal_luminosity"]
    if not math.isfinite(nominal) or nominal < 0:
        raise ValueError("Nominal luminosity must be finite and non-negative")
    rows: List[Scenario] = []
    base_seed = protocol.pop("base_seed")
    for laptop_name in laptop_names:
        for scale in scales:
            block = build_scenarios(
                database=database, laptop_name=laptop_name, scale=scale,
                base_seed=base_seed + len(rows), **protocol,
            )
            if len(scales) > 1:
                block = [Scenario(**{**asdict(row), "include_scale_in_id": True}) for row in block]
            rows.extend(block)
    if not rows or rows[0].seed < 0 or rows[-1].seed >= 2**32:
        raise ValueError("The full matrix must have seeds in [0, 2**32 - 1]")
    if len({row.episode_id for row in rows}) != len(rows):
        raise ValueError("Duplicate episode identifiers in scenario matrix")
    return rows


def scenario_digest(scenarios: Sequence[Scenario]) -> str:
    digest = hashlib.sha256()
    for scenario in scenarios:
        digest.update(json.dumps(asdict(scenario), sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_resume_rows(
    detail_csv: Path, trace_jsonl: Path, scenarios: Sequence[Scenario],
    *, run_id: str, method: str, threshold: float, max_steps: int,
    fields: Sequence[str] = DETAIL_FIELDS,
) -> List[Dict[str, Any]]:
    """Read only a complete, consistent episode prefix; never repair raw logs."""
    rows: List[Dict[str, Any]] = []
    csv.field_size_limit(64 * 1024 * 1024)
    expected_traces: Dict[str, int] = {}
    for path in (detail_csv, trace_jsonl):
        if path.exists() and path.stat().st_size:
            with path.open("rb") as stream:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    raise ValueError(f"Cannot resume: incomplete final line in {path}; files left unchanged")
    if detail_csv.exists():
        with detail_csv.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, strict=True)
            if reader.fieldnames != list(fields):
                raise ValueError("Cannot resume: episode CSV schema differs or is incomplete")
            for index, row in enumerate(reader):
                if index >= len(scenarios) or None in row or any(value is None for value in row.values()):
                    raise ValueError("Cannot resume: extra or incomplete episode CSV row")
                scenario = scenarios[index]
                expected = {
                    "run_id": run_id, "method": method, "episode_id": scenario.episode_id,
                    "setting": scenario.setting, "laptop_name": scenario.laptop_name,
                    "laptop_version_id": scenario.laptop_version_id,
                    "target_corner": scenario.target_corner, "target_name": scenario.target_name,
                    "ground_truth_class": scenario.ground_truth_class,
                }
                if any(row[key] != value for key, value in expected.items()):
                    raise ValueError(f"Cannot resume: scenario mismatch at CSV row {index + 2}")
                for key, value in {
                    "seed": scenario.seed, "episode_index": scenario.episode_index,
                    "scale": scenario.scale, "luminosity": scenario.luminosity,
                    "threshold": threshold, "max_steps": max_steps,
                }.items():
                    if float(row[key]) != value:
                        raise ValueError(f"Cannot resume: {key} mismatch at CSV row {index + 2}")
                info = json.loads(row["raw_info_json"])
                expected_traces[scenario.episode_id] = sum(
                    isinstance(event, Mapping) for event in info.get("prediction_trace", [])
                )
                # Keep the summary JSON's original numeric grouping type.
                row["luminosity"] = float(row["luminosity"])
                # All summary calculations already accept numeric CSV strings.
                # Do not retain the potentially large raw trace twice in memory.
                rows.append({key: value for key, value in row.items() if key not in {"raw_info_json", "targets_json"}})
    if not trace_jsonl.exists():
        raise ValueError("Cannot resume: prediction trace file is missing")
    seen: Dict[str, int] = {}
    with trace_jsonl.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            episode_id = record["episode_id"]
            if episode_id not in expected_traces or record["method"] != method or record["run_id"] != run_id:
                raise ValueError("Cannot resume: trace has an uncommitted or mismatched episode")
            seen[episode_id] = seen.get(episode_id, 0) + 1
            if record["query_index"] != seen[episode_id]:
                raise ValueError("Cannot resume: duplicate or out-of-order detector trace")
    if any(seen.get(key, 0) != count for key, count in expected_traces.items()):
        raise ValueError("Cannot resume: incomplete detector trace; original files left unchanged")
    return rows


def validate_resume_manifest(previous: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    for key in ("resume_protocol_version", "run_id", "method", "scenario_count", "scenario_sha256", "sha256", "runtime"):
        if previous.get(key) != current.get(key):
            raise ValueError(f"Cannot resume: {key} differs from the recorded run")
    # Matrix equivalence is checked above, so singular/list selectors can differ.
    for key in ("threshold", "max_steps", "device", "warmup_calls", "visualize", "return_home"):
        if previous["arguments"].get(key) != current["arguments"].get(key):
            raise ValueError(f"Cannot resume: argument {key} differs from the recorded run")
    for key, default in (("inspection_mode", "single"), ("target_order", "shuffled")):
        if previous["arguments"].get(key, default) != current["arguments"].get(key, default):
            raise ValueError(f"Cannot resume: argument {key} differs from the recorded run")


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Replace metadata atomically so an interruption cannot half-write it."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_scenarios(
    *,
    database: Mapping[str, Any],
    laptop_name: str,
    versions: Sequence[str],
    corners: Sequence[str],
    setting: str,
    episodes_per_corner: int,
    base_seed: int,
    scale: float,
    nominal_luminosity: float,
    illumination_values: Sequence[float],
) -> List[Scenario]:
    if laptop_name not in database["laptops"]:
        raise ValueError(f"Unknown laptop {laptop_name!r}")
    laptop_versions = database["laptops"][laptop_name]["version_objs"]
    missing_versions = [version for version in versions if version not in laptop_versions]
    if missing_versions:
        raise ValueError(f"Laptop {laptop_name!r} lacks versions {missing_versions}")
    if episodes_per_corner <= 0:
        raise ValueError("episodes_per_corner must be positive")

    setting_lights: List[Tuple[str, float]] = []
    if setting in {"nominal", "both"}:
        setting_lights.append(("nominal", float(nominal_luminosity)))
    if setting in {"illumination", "both"}:
        setting_lights.extend(("illumination", float(value)) for value in illumination_values)

    scenarios: List[Scenario] = []
    next_seed = int(base_seed)
    for setting_name, luminosity in setting_lights:
        for version in versions:
            version_classes = database["versions"][version]["classes"]
            for corner in corners:
                target_name = CORNER_TO_TARGET[corner]
                ground_truth = str(version_classes[target_name])
                for episode_index in range(1, episodes_per_corner + 1):
                    scenarios.append(
                        Scenario(
                            setting=setting_name,
                            laptop_name=laptop_name,
                            laptop_version_id=version,
                            target_corner=corner,
                            target_name=target_name,
                            ground_truth_class=ground_truth,
                            scale=float(scale),
                            luminosity=luminosity,
                            episode_index=episode_index,
                            seed=next_seed,
                        )
                    )
                    next_seed += 1
    return scenarios


def import_actipercept_environment(source_path: Path):
    controller_dir = source_path.parent
    sys.path.insert(0, str(controller_dir))
    spec = importlib.util.spec_from_file_location("local_actipercept_environment", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load ActiPercept environment from {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ScrewDetectorEnv


def configure_actipercept_scene(
    env: Any,
    database: Mapping[str, Any],
    scenario: Scenario,
) -> None:
    """Apply the same CAD version, scale, and luminosity used by the PPO env.

    This requires the common thesis ``world.wbt`` (with DEF LAPTOP_TRANSFORM,
    LAPTOP_CAD, and light) to be open in Webots.  Failing loudly avoids silently
    comparing different scenes.
    """
    supervisor = env.supervisor
    transform_node = supervisor.getFromDef("LAPTOP_TRANSFORM")
    cad_node = supervisor.getFromDef("LAPTOP_CAD")
    light_node = supervisor.getFromDef("light")
    if transform_node is None or cad_node is None or light_node is None:
        raise RuntimeError(
            "ActiPercept comparison requires the common thesis world with DEF "
            "LAPTOP_TRANSFORM, LAPTOP_CAD, and light. Open thesis_env/worlds/world.wbt."
        )

    obj_url = database["laptops"][scenario.laptop_name]["version_objs"][scenario.laptop_version_id]
    cad_node.getField("url").setMFString(0, str(obj_url))
    transform_node.getField("scale").setSFVec3f([scenario.scale, 1.0, scenario.scale])
    luminosity_field = light_node.getField("luminosity")
    if luminosity_field is None:
        raise RuntimeError("DEF light has no luminosity field")
    luminosity_field.setSFFloat(scenario.luminosity)

    # Let URL/scale/light changes reach the rendered camera before evaluation.
    for _ in range(3):
        supervisor.step(env.timestep)


def _detection_values(detection: Any) -> Tuple[Optional[str], float, Optional[Tuple[float, float]]]:
    if isinstance(detection, Mapping):
        class_name = detection.get("class")
        confidence = as_float(detection.get("confidence"))
        center = detection.get("bbox_center")
    else:
        class_name = getattr(detection, "cls", None)
        confidence = as_float(getattr(detection, "confidence", 0.0))
        center = getattr(detection, "bbox_center", None)
    if center is None or len(center) < 2:
        return (str(class_name) if class_name is not None else None, confidence, None)
    return (
        str(class_name) if class_name is not None else None,
        confidence,
        (float(center[0]), float(center[1])),
    )


def _pixel_corner(x: float, y: float, width: float, height: float) -> str:
    if x >= width / 2.0:
        return "BR" if y >= height / 2.0 else "TR"
    return "BL" if y >= height / 2.0 else "TL"


def initial_home_probe(env: Any, scenario: Scenario) -> Dict[str, Any]:
    """Score the requested paper region at home without charging the controller."""
    frame = env._get_camera_frame()
    started_at = time.perf_counter()
    detections, _ = env.run_yolo_detection(frame)
    elapsed = time.perf_counter() - started_at
    width, height = float(env.camera_intrinsics[0]), float(env.camera_intrinsics[1])
    best_class: Optional[str] = None
    best_confidence = 0.0
    for detection in detections:
        class_name, confidence, center = _detection_values(detection)
        if class_name not in {"screw", "noscrew"} or center is None:
            continue
        if _pixel_corner(center[0], center[1], width, height) != scenario.target_corner:
            continue
        if confidence > best_confidence:
            best_class = class_name
            best_confidence = confidence

    # run_yolo_detection owns the normal cost counters. The probe is evaluator
    # instrumentation, so clear only those counters before controller timing.
    for attribute, value in (
        ("detector_call_count", 0),
        ("detector_inference_time_s", 0.0),
        ("last_detector_inference_time_s", 0.0),
        ("_last_detector_inference_time_s", 0.0),
    ):
        if hasattr(env, attribute):
            setattr(env, attribute, value)
    return {
        "initial_home_target_confidence": best_confidence,
        "initial_home_prediction_class": best_class,
        "initial_home_prediction_correct": (
            best_class is not None and best_class == scenario.ground_truth_class
        ),
        "initial_home_probe_inference_time_s": elapsed,
    }


def run_ppo(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    database: Mapping[str, Any],
    detector_path: Path,
    checkpoint_path: Path,
    scenarios: Sequence[Scenario],
    run_id: str,
    detail_csv: Path,
    trace_jsonl: Path,
) -> List[Dict[str, Any]]:
    is_sequential = args.inspection_mode == "sequential"
    add_webots_python_controller_path()
    from stable_baselines3 import PPO
    from screw_env_v3 import ScrewDetectorEnv

    env = ScrewDetectorEnv(
        yolo_path=str(detector_path),
        visualize_detections=args.visualize,
        max_steps=args.max_steps,
        control_mode="policy",
        mode="eval",
        laptop_metadata_path=str(args.database_path),
        verbose=args.show_step_logs,
        scripted_return_on_success=args.return_home,
        verification_threshold=args.threshold,
        device=args.device,
    )
    model = PPO.load(str(checkpoint_path), env=env)
    rows: List[Dict[str, Any]] = []
    try:
        if scenarios and args.warmup_calls > 0:
            warmup = scenarios[0]
            observation, _ = env.reset(
                seed=warmup.seed,
                options=sequential.reset_options(warmup, args.target_order) if is_sequential else {
                    "laptop_name": warmup.laptop_name,
                    "laptop_version_id": warmup.laptop_version_id,
                    "scale": warmup.scale,
                    "luminosity": warmup.luminosity,
                    "evaluation_target_name": warmup.target_name,
                },
            )
            for _ in range(args.warmup_calls):
                model.predict(observation, deterministic=False)
            frame = env._get_camera_frame()
            for _ in range(args.warmup_calls):
                env.run_yolo_detection(frame)
            print(f"Completed {args.warmup_calls} unscored PPO/detector warm-up calls")

        for number, scenario in enumerate(scenarios, start=1):
            model.set_random_seed(scenario.seed)
            episode_started_utc = datetime.now(timezone.utc).isoformat()
            reset_options = sequential.reset_options(scenario, args.target_order) if is_sequential else {
                "laptop_name": scenario.laptop_name,
                "laptop_version_id": scenario.laptop_version_id,
                "scale": scenario.scale,
                "luminosity": scenario.luminosity,
                "evaluation_target_name": scenario.target_name,
            }
            observation, reset_info = env.reset(seed=scenario.seed, options=reset_options)
            if is_sequential:
                probes = sequential.home_probe(sys.modules[__name__], env, scenario, database)
                probe_info = {}
            else:
                probe_info = initial_home_probe(env, scenario)
            if hasattr(env, "start_evaluation_timing"):
                env.start_evaluation_timing()
            else:
                env._episode_start_time = time.perf_counter()
            terminated = False
            truncated = False
            info: Dict[str, Any] = {}
            loop_steps = 0
            ppo_inference_time_s = 0.0
            episode_start = time.perf_counter()
            active_started_utc = datetime.now(timezone.utc).isoformat()
            while not (terminated or truncated):
                inference_start = time.perf_counter()
                action, _ = model.predict(observation, deterministic=False)
                ppo_inference_time_s += time.perf_counter() - inference_start
                observation, _, terminated, truncated, info = env.step(action)
                loop_steps += 1
                if loop_steps > args.max_steps + 1:
                    raise RuntimeError("PPO environment exceeded its configured episode limit")
            loop_wall_time_s = time.perf_counter() - episode_start
            episode_ended_utc = datetime.now(timezone.utc).isoformat()
            info = {**info, **probe_info}
            if is_sequential:
                # Keep trace metadata target-specific instead of labelling every query ALL.
                classes = database["versions"][scenario.laptop_version_id]["classes"]
                corners = {v: k for k, v in CORNER_TO_TARGET.items()}
                info = {**info, "prediction_trace": [
                    {**e, "target_name": e["target"], "target_corner": corners[e["target"]],
                     "ground_truth_class": classes[e["target"]]}
                    for e in info.get("prediction_trace", [])
                ]}
            row_options = dict(
                run_id=run_id,
                method="ppo",
                scenario=scenario,
                threshold=args.threshold,
                max_steps=args.max_steps,
                info=info,
                terminated=terminated,
                truncated=truncated,
                loop_steps=loop_steps,
                loop_wall_time_s=loop_wall_time_s,
                ppo_inference_time_s=ppo_inference_time_s,
            )
            if is_sequential:
                row = sequential.build_row(
                    sys.modules[__name__], order=reset_info["execution_sequence"], classes=classes,
                    probes=probes, timestamps=dict(episode_started_utc=episode_started_utc,
                        active_started_utc=active_started_utc, episode_ended_utc=episode_ended_utc),
                    **row_options,
                )
            else:
                row = build_detail_row(**row_options)
            rows.append({key: value for key, value in row.items() if key not in {"raw_info_json", "targets_json"}})
            append_csv_row(detail_csv, row, sequential.DETAIL_FIELDS if is_sequential else DETAIL_FIELDS)
            append_prediction_trace(
                trace_jsonl,
                run_id=run_id,
                method="ppo",
                scenario=scenario,
                info=info,
            )
            target_progress = (
                f" accepted={row['accepted_target_count']}/4"
                f" correct={row['correct_verification_count']} wrong={row['false_verification_count']}"
                if is_sequential else ""
            )
            print(
                f"[{number}/{len(scenarios)}] PPO {scenario.episode_id}: "
                f"{row['verification_outcome']} steps={row['steps']} "
                f"detect={row['detector_calls']} fps={row['processing_fps']:.2f}{target_progress}"
            )
    finally:
        env.close()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
    return rows


def run_actipercept(
    *,
    args: argparse.Namespace,
    database: Mapping[str, Any],
    detector_path: Path,
    scenarios: Sequence[Scenario],
    run_id: str,
    detail_csv: Path,
    trace_jsonl: Path,
) -> List[Dict[str, Any]]:
    add_webots_python_controller_path()
    Environment = import_actipercept_environment(args.actipercept_env_path)
    env = Environment(
        model_path=str(detector_path),
        visualize_detections=args.visualize,
        max_steps=args.max_steps,
        control_mode="policy",
        scripted_return_on_success=args.return_home,
        conf_threshold=args.threshold,
        screw_success_threshold=args.threshold,
        device=args.device,
    )
    rows: List[Dict[str, Any]] = []
    try:
        if scenarios and args.warmup_calls > 0:
            warmup = scenarios[0]
            env.reset(
                seed=warmup.seed,
                options={
                    "target_corner": CORNER_TO_ACTIPERCEPT_INDEX[warmup.target_corner],
                    "ground_truth_class": warmup.ground_truth_class,
                    "luminosity": warmup.luminosity,
                },
            )
            configure_actipercept_scene(env, database, warmup)
            frame = env._get_camera_frame()
            for _ in range(args.warmup_calls):
                env.run_yolo_detection(frame)
            print(f"Completed {args.warmup_calls} unscored detector warm-up calls")

        for number, scenario in enumerate(scenarios, start=1):
            reset_options = {
                "target_corner": CORNER_TO_ACTIPERCEPT_INDEX[scenario.target_corner],
                "ground_truth_class": scenario.ground_truth_class,
                "luminosity": scenario.luminosity,
            }
            _, _ = env.reset(seed=scenario.seed, options=reset_options)
            configure_actipercept_scene(env, database, scenario)
            probe_info = initial_home_probe(env, scenario)
            # Scene changes and their settling ticks are setup, not decision
            # time. Prefer the environment's public hook when available.
            if hasattr(env, "start_evaluation_timing"):
                env.start_evaluation_timing()
            else:
                env._episode_start_time = time.perf_counter()

            terminated = False
            truncated = False
            info: Dict[str, Any] = {}
            loop_steps = 0
            episode_start = time.perf_counter()
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step(None)
                loop_steps += 1
                if loop_steps > args.max_steps + 1:
                    raise RuntimeError("ActiPercept environment exceeded its configured episode limit")
            loop_wall_time_s = time.perf_counter() - episode_start
            info = {**info, **probe_info}
            row = build_detail_row(
                run_id=run_id,
                method="actipercept",
                scenario=scenario,
                threshold=args.threshold,
                max_steps=args.max_steps,
                info=info,
                terminated=terminated,
                truncated=truncated,
                loop_steps=loop_steps,
                loop_wall_time_s=loop_wall_time_s,
            )
            rows.append({key: value for key, value in row.items() if key != "raw_info_json"})
            append_csv_row(detail_csv, row, DETAIL_FIELDS)
            append_prediction_trace(
                trace_jsonl,
                run_id=run_id,
                method="actipercept",
                scenario=scenario,
                info=info,
            )
            print(
                f"[{number}/{len(scenarios)}] ActiPercept {scenario.episode_id}: "
                f"{row['verification_outcome']} steps={row['steps']} "
                f"detect={row['detector_calls']} fps={row['processing_fps']:.2f}"
            )
    finally:
        env.close()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-compatible single-corner protocol for PPO or ActiPercept."
    )
    parser.add_argument("--method", choices=("ppo", "actipercept"), required=True)
    parser.add_argument("--inspection-mode", choices=("single", "sequential"), default="single",
                        help="Single-corner protocol (default), or all four targets per PPO episode.")
    parser.add_argument("--target-order", choices=sequential.TARGET_ORDER_MODES, default="shuffled",
                        help="Sequential mode only: shuffled (default), fixed BL/TL/BR/TR, "
                             "or a seeded starting corner followed clockwise, counterclockwise, "
                             "or around the perimeter with a seeded random direction. "
                             "All seeded modes share the shuffled starting corner.")
    parser.add_argument("--setting", choices=("nominal", "illumination", "both"), default="both")
    parser.add_argument("--config", type=Path, default=Path("configs/config_v3.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--detector", type=Path, default=None)
    parser.add_argument(
        "--database-path",
        type=Path,
        default=Path("laptop_database.json"),
        help="Laptop database used to set the scene and obtain target ground truth.",
    )
    parser.add_argument("--actipercept-env-path", type=Path, default=Path("ActiPercept-main/controllers/screw_detector_env.py"))
    laptop_group = parser.add_mutually_exclusive_group()
    laptop_group.add_argument("--laptop", default=None, help="One exact laptop name (default: dell_laptop; legacy option).")
    laptop_group.add_argument("--laptops", nargs="+", help="Exact names, quoted patterns, or 'all' from the selected database.")
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["0110"],
        help="One or more four-character version IDs, or 'all'.",
    )
    parser.add_argument("--corners", nargs="+", choices=tuple(CORNER_TO_TARGET), default=list(CORNER_TO_TARGET))
    parser.add_argument("--episodes-per-corner", "--episodes-per-condition", dest="episodes_per_corner", type=int, default=10,
                        help="Repetitions per condition; per corner in single mode, per four-target episode in sequential mode.")
    parser.add_argument("--base-seed", type=int, default=20260903)
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument("--scale", type=float, default=None, help="One scale (default: 1.0; legacy option).")
    scale_group.add_argument("--scales", nargs="+", type=float, help="Evaluate each of these scales for each laptop.")
    scale_group.add_argument("--training-scales", action="store_true", help="Use 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15.")
    parser.add_argument("--nominal-luminosity", type=float, default=1.0)
    parser.add_argument("--illumination-values", nargs="+", type=float, default=list(PAPER_LUMINOSITIES))
    parser.add_argument("--threshold", type=float, default=0.88)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Whole-episode limit; defaults to 180 single / 1024 sequential.")
    parser.add_argument("--run-id", default=None, help="Use the same ID for the PPO and ActiPercept invocations.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="YOLO device for both methods (for example cuda or cpu); omit for automatic selection.",
    )
    parser.add_argument(
        "--warmup-calls",
        type=int,
        default=3,
        help="Unscored detector and PPO inference warm-up calls before timing episodes.",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--show-step-logs", action="store_true")
    parser.add_argument("--no-return-home", dest="return_home", action="store_false")
    parser.set_defaults(return_home=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the scenario matrix without starting Webots.")
    parser.add_argument("--resume", action="store_true", help="Validate the same run and append only unfinished episodes; requires an explicit --run-id.")
    args = parser.parse_args(argv)
    if args.inspection_mode == "sequential":
        if args.method != "ppo":
            parser.error("Sequential inspection is currently supported only for PPO")
        if set(args.corners) != set(CORNER_TO_TARGET) or len(args.corners) != 4:
            parser.error("Sequential inspection requires all four corners")
    if args.max_steps is None:
        args.max_steps = 1024 if args.inspection_mode == "sequential" else 180
    # Apply legacy defaults after conflict checking, including explicitly
    # supplied values equal to those defaults.
    if args.laptop is None:
        args.laptop = "dell_laptop"
    if args.scale is None:
        args.scale = 1.0
    return args


def main() -> None:
    args = parse_args()
    if args.resume and not args.run_id:
        raise ValueError("--resume requires the original explicit --run-id")
    args.config = args.config.resolve()
    args.database_path = args.database_path.resolve()
    args.actipercept_env_path = args.actipercept_env_path.resolve()
    config = load_yaml(args.config)
    workspace = Path(config["workspace_path"])
    database_path_from_config = resolve_from_workspace(workspace, config["laptop_metadata_path"])
    if args.database_path == Path("laptop_database.json").resolve() and database_path_from_config.exists():
        args.database_path = database_path_from_config
    with args.database_path.open("r", encoding="utf-8") as stream:
        database = json.load(stream)

    versions = resolve_versions(args.versions, database)
    laptops = resolve_laptops(args.laptops or [args.laptop], database)
    scales = list(TRAINING_SCALES) if args.training_scales else args.scales or [args.scale]
    scenario_options = dict(
        database=database,
        laptop_names=laptops,
        versions=versions,
        corners=args.corners,
        setting=args.setting,
        episodes_per_corner=args.episodes_per_corner,
        base_seed=args.base_seed,
        scales=scales,
        nominal_luminosity=args.nominal_luminosity,
        illumination_values=args.illumination_values,
    )
    is_sequential = args.inspection_mode == "sequential"
    scenarios = (sequential.build_scenarios(sys.modules[__name__], **scenario_options)
                 if is_sequential else build_scenario_matrix(**scenario_options))

    detector_path = (
        args.detector.resolve()
        if args.detector is not None
        else resolve_from_workspace(workspace, config["yolo_model"])
    )
    checkpoint_path: Optional[Path] = None
    if args.method == "ppo":
        checkpoint_path = (
            args.checkpoint.resolve()
            if args.checkpoint is not None
            else workspace
            / config["log_dir"]
            / config["experiment_name"]
            / f"{config['experiment_name']}_{config['train']['total_timesteps']}_steps"
        )

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else workspace / config["log_dir"] / "actipercept_comparison"
    )
    run_dir = output_root / run_id

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.warmup_calls < 0:
        raise ValueError("--warmup-calls must be non-negative")
    print(f"Method: {args.method}")
    print(f"Run ID: {run_id}")
    print(f"Laptops ({len(laptops)}): {', '.join(laptops)}")
    print(f"Scales ({len(scales)}): {scales}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Inspection: {args.inspection_mode}; whole-episode step budget: {args.max_steps}")
    if is_sequential:
        print(f"Target order: {args.target_order}; targets per episode: 4")
    print(f"Detector: {detector_path}")
    if checkpoint_path is not None:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {run_dir}")

    if args.dry_run:
        counts: Dict[Tuple[str, float], int] = {}
        for scenario in scenarios:
            key = (scenario.setting, scenario.luminosity)
            counts[key] = counts.get(key, 0) + 1
        for key, count in sorted(counts.items()):
            print(f"  setting={key[0]} luminosity={key[1]} episodes={count}")
        return

    if not detector_path.exists():
        raise FileNotFoundError(f"Detector weights not found: {detector_path}")
    if args.method == "ppo" and (checkpoint_path is None or not checkpoint_path.with_suffix(".zip").exists() and not checkpoint_path.exists()):
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint_path}")
    if args.method == "actipercept" and not args.actipercept_env_path.exists():
        raise FileNotFoundError(f"ActiPercept environment not found: {args.actipercept_env_path}")

    detail_csv = run_dir / f"{args.method}_episodes.csv"
    summary_csv = run_dir / f"{args.method}_summary.csv"
    summary_json = run_dir / f"{args.method}_summary.json"
    target_csv = run_dir / f"{args.method}_targets.csv"
    trace_jsonl = run_dir / f"{args.method}_prediction_trace.jsonl"
    manifest_json = run_dir / f"{args.method}_manifest.json"
    if not args.resume and any(path.exists() for path in (detail_csv, trace_jsonl, manifest_json, summary_csv, summary_json, target_csv)):
        raise FileExistsError(
            f"Refusing to overwrite existing method results in {run_dir}. Use a new --run-id or --resume."
        )
    if args.resume and not manifest_json.exists():
        raise FileNotFoundError(f"Cannot resume: manifest not found: {manifest_json}")

    manifest = {
        "run_id": run_id,
        "method": args.method,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "arguments": json_safe(vars(args)),
        "scenario_count": len(scenarios),
        "resume_protocol_version": 1,
        "scenario_sha256": scenario_digest(scenarios),
        "resolved_laptops": laptops,
        "resolved_scales": scales,
        "metric_definitions": {
            "verified_success": "first confidence-threshold crossing has the correct target class",
            "false_verification": "first confidence-threshold crossing has the wrong target class",
            "final_accuracy": "accepted class, or best-so-far target class at timeout, equals ground truth",
            "last_accuracy": "terminal detector-query target class equals ground truth; missing is incorrect",
            "best_accuracy": "highest-confidence target-associated class equals ground truth",
            "confidence_gain": "positive improvement from a standardized evaluator-only selected-region detector probe at the home pose; the paper does not specify its baseline",
            "observed_confidence_gain": "maximum observed confidence minus the first detector-query target score",
            "steps_verified_mean": "control steps averaged only over correctly verified episodes, matching the paper",
            "processing_fps": "control steps divided by active wall time, excluding return-home time when available",
            "detector_fps": "detector calls divided by cumulative detector inference time",
            "detector_call_fraction": "detector calls divided by control steps",
            "effective_detection_hz": "detector calls divided by active episode wall time",
            "path_length": "pre-decision camera translation distance; return-home distance is separate",
            "active_simulated_time_s": "decision/control simulation time excluding setup, pauses, and return home",
            "initial_home_probe": "one selected-region detector query before controller timing; excluded from controller detector-call and inference-time counters",
            "ppo_action_selection": "stochastic model.predict(deterministic=False), reseeded for each recorded scenario",
        },
        "files": {
            "config": str(args.config),
            "database": str(args.database_path),
            "detector": str(detector_path),
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "actipercept_environment": str(args.actipercept_env_path) if args.method == "actipercept" else None,
        },
        "sha256": {
            "evaluator": sha256_or_none(Path(__file__)),
            "ppo_environment": sha256_or_none(Path(__file__).with_name("screw_env_v3.py")) if args.method == "ppo" else None,
            "config": sha256_or_none(args.config),
            "database": sha256_or_none(args.database_path),
            "detector": sha256_or_none(detector_path),
            "checkpoint": sha256_or_none(
                checkpoint_path if checkpoint_path and checkpoint_path.exists() else checkpoint_path.with_suffix(".zip") if checkpoint_path else None
            ),
            "actipercept_environment": sha256_or_none(args.actipercept_env_path) if args.method == "actipercept" else None,
        },
        "runtime": collect_runtime_metadata(),
    }
    if is_sequential:
        manifest["resume_protocol_version"] = 2
        manifest["metric_definitions"].update(sequential.METRIC_DEFINITIONS)
        manifest["sha256"]["sequential_helpers"] = sha256_or_none(Path(sequential.__file__))
        # Preserve the exact declared classes and coordinates alongside the run.
        manifest["selected_database"] = {
            "laptops": {name: database["laptops"][name] for name in laptops},
            "versions": {version: database["versions"][version] for version in versions},
        }
    previous_rows: List[Dict[str, Any]] = []
    if args.resume:
        previous = json.loads(manifest_json.read_text(encoding="utf-8"))
        validate_resume_manifest(previous, manifest)
        previous_rows = load_resume_rows(
            detail_csv, trace_jsonl, scenarios, run_id=run_id, method=args.method,
            threshold=args.threshold, max_steps=args.max_steps,
            fields=sequential.DETAIL_FIELDS if is_sequential else DETAIL_FIELDS,
        )
        print(f"Resume: {len(previous_rows)} completed; {len(scenarios) - len(previous_rows)} remaining")
        if len(previous_rows) == len(scenarios) and previous["status"] == "complete":
            print("This method is already complete; no files changed and Webots was not started.")
            return
        manifest = previous
        manifest["status"] = "running"
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_jsonl.touch(exist_ok=False)
    # Preserve legacy timestamps, but record individual invocation windows so
    # GPU analysis of a resumed run can exclude the gaps between invocations.
    session = {
        "started_utc": datetime.now(timezone.utc).isoformat() if args.resume else manifest["created_utc"],
        "completed_before": len(previous_rows),
    }
    manifest.setdefault("sessions", []).append(session)
    write_manifest(manifest_json, manifest)

    try:
        pending_scenarios = scenarios[len(previous_rows):]
        if not pending_scenarios:
            rows = []
        elif args.method == "ppo":
            assert checkpoint_path is not None
            rows = run_ppo(
                args=args,
                config=config,
                database=database,
                detector_path=detector_path,
                checkpoint_path=checkpoint_path,
                scenarios=pending_scenarios,
                run_id=run_id,
                detail_csv=detail_csv,
                trace_jsonl=trace_jsonl,
            )
        else:
            rows = run_actipercept(
                args=args,
                database=database,
                detector_path=detector_path,
                scenarios=pending_scenarios,
                run_id=run_id,
                detail_csv=detail_csv,
                trace_jsonl=trace_jsonl,
            )

        rows = previous_rows + rows
        summaries = (sequential.grouped_summaries(sys.modules[__name__], rows)
                     if is_sequential else grouped_summary_rows(rows))
        write_csv(summary_csv, summaries, tuple(summaries[0]) if is_sequential else SUMMARY_FIELDS)
        if is_sequential:
            sequential.export_targets(target_csv, detail_csv)
        overall = next(row for row in summaries if row["grouping"] == "overall")
        summary_document = {
            "run_id": run_id,
            "method": args.method,
            "overall": overall,
            "groups": summaries,
            "metric_definitions": manifest["metric_definitions"],
        }
        summary_json.write_text(json.dumps(json_safe(summary_document), indent=2), encoding="utf-8")
        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        session.update(ended_utc=manifest["completed_utc"], status="complete")
        manifest["outputs"] = {
            "episodes_csv": str(detail_csv),
            "prediction_trace_jsonl": str(trace_jsonl),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
        }
        if is_sequential:
            manifest["outputs"]["targets_csv"] = str(target_csv)
        write_manifest(manifest_json, manifest)
        print(f"Episode results: {detail_csv}")
        print(f"Summary CSV: {summary_csv}")
        print(f"Summary JSON: {summary_json}")
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["interrupted_utc"] = datetime.now(timezone.utc).isoformat()
        session.update(ended_utc=manifest["interrupted_utc"], status="interrupted")
        write_manifest(manifest_json, manifest)
        raise
    except Exception:
        manifest["status"] = "failed"
        manifest["failed_utc"] = datetime.now(timezone.utc).isoformat()
        session.update(ended_utc=manifest["failed_utc"], status="failed")
        write_manifest(manifest_json, manifest)
        raise


if __name__ == "__main__":
    main()
