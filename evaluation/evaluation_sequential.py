"""Sequential-only scoring helpers; no Webots imports or changes to training rules.

An episode CSV is the durable record, including four target records as JSON.
The flat target CSV is derived at completion, including after a resumed run.
"""
import csv
from dataclasses import replace
import json
import random


TARGET_ORDER_MODES = ("fixed", "shuffled", "clockwise", "counterclockwise", "perimeter")
# Clockwise in the named top/bottom, left/right view of the laptop.
CLOCKWISE_TARGETS = ("left_up", "right_up", "right_down", "left_down")


IDENTITY_FIELDS = (
    "run_id", "method", "setting", "episode_id", "episode_index", "seed",
    "laptop_name", "laptop_version_id", "target_corner", "target_name",
    "ground_truth_class", "scale", "luminosity", "actual_luminosity",
    "threshold", "max_steps",
)
COST_FIELDS = (
    "steps", "detector_calls", "detection_action_count", "movement_action_count",
    "no_op_action_count", "path_length", "return_home_path_length", "total_path_length",
    "detector_inference_time_s", "detector_fps", "detector_call_fraction",
    "effective_detection_hz", "ppo_inference_time_s", "ppo_inference_fps",
    "active_wall_time_s", "return_home_steps", "return_home_time_s", "total_wall_time_s",
    "processing_fps", "active_simulated_time_s", "total_simulated_time_s",
    "terminated", "truncated",
)
COUNT_FIELDS = (
    "accepted_target_count", "correct_verification_count", "false_verification_count",
    "unverified_target_count", "attempted_target_count", "unreached_target_count",
    "final_correct_count", "last_correct_count", "best_correct_count",
)
DETAIL_FIELDS = IDENTITY_FIELDS + COST_FIELDS + (
    "inspection_mode", "execution_sequence", "completed", "verified_success",
    "false_verification", "verification_outcome", "failure_reason", "timeout",
    "out_of_bounds", "steps_to_verification",
) + COUNT_FIELDS + (
    "final_accuracy", "confidence_gain", "episode_started_utc", "active_started_utc",
    "episode_ended_utc", "targets_json", "raw_info_json",
)
TARGET_FIELDS = (
    "target_name", "target_corner", "ground_truth_class", "order_index", "attempted",
    "accepted", "correct_verification", "false_verification", "outcome", "start_step",
    "end_step", "steps", "detector_calls", "accepted_prediction_class",
    "accepted_prediction_confidence", "accepted_prediction_step", "best_prediction_class",
    "best_prediction_confidence", "best_prediction_step", "last_prediction_class",
    "last_prediction_confidence", "final_prediction_class", "final_prediction_confidence",
    "final_prediction_correct", "best_prediction_correct", "last_prediction_correct",
    "initial_probe_prediction_class", "initial_target_confidence", "confidence_gain",
    "observed_confidence_gain_raw",
)
METRIC_DEFINITIONS = {
    "verified_success": "all four targets accepted with correct classes",
    "completion_rate": "episodes accepting all four targets, including incorrect acceptances / episodes",
    "false_verification": "episode contains at least one incorrect acceptance; can coexist with timeout",
    "final_accuracy": "correct per-target accepted or best-so-far classes / all required targets (4 per episode); missing and unreached are incorrect",
    "last_accuracy": "correct terminal detector-query classes per target / all required targets; missing and unreached are incorrect",
    "best_accuracy": "correct highest-confidence target-associated classes / all required targets; missing and unreached are incorrect",
    "correct_target_verification_rate": "correctly accepted targets / all required targets",
    "false_target_verification_rate": "incorrectly accepted targets / all required targets",
    "unverified_target_rate": "targets without an acceptance / all required targets",
    "confidence_gain": "mean of four per-target positive best-confidence improvements over one shared evaluator-only home image probe; missing controller predictions have confidence zero",
    "steps_verified_mean": "total control steps averaged only over episodes with four correct acceptances",
    "target_order": "fixed BL, TL, BR, TR; shuffled; or seeded-start clockwise/counterclockwise/perimeter (random direction). Seeded modes share the shuffled starting corner and do not consume policy random state; actual order saved per episode",
    "step_budget": "one shared episode budget, not a separate allowance for each target",
    "initial_home_probe": "one evaluator-only YOLO call scoring all four image quadrants; excluded from controller timing and call counts",
}


def build_scenarios(api, **kwargs):
    # Reuse validation, scale enumeration and unique seeds, but ONE case per version.
    kwargs["corners"] = ["TL"]
    return [replace(s, target_corner="ALL", target_name="all", ground_truth_class="all")
            for s in api.build_scenario_matrix(**kwargs)]


def reset_options(scenario, order_mode):
    if order_mode not in TARGET_ORDER_MODES:
        raise ValueError(f"Unknown target order mode: {order_mode!r}")
    order = sorted(("left_down", "left_up", "right_down", "right_up"))
    if order_mode != "fixed":
        rng = random.Random(scenario.seed)
        rng.shuffle(order)
        if order_mode in ("clockwise", "counterclockwise", "perimeter"):
            # Match the legacy shuffled start for paired order comparisons.
            start = CLOCKWISE_TARGETS.index(order[0])
            direction = -1 if order_mode == "counterclockwise" else 1
            if order_mode == "perimeter":
                direction = rng.choice((1, -1))
            order = [CLOCKWISE_TARGETS[(start + direction * i) % 4] for i in range(4)]
    return dict(laptop_name=scenario.laptop_name, laptop_version_id=scenario.laptop_version_id,
                scale=scenario.scale, luminosity=scenario.luminosity,
                initial_confirmed_screws=[False] * 4, execution_sequence=order)


def home_probe(api, env, scenario, database):
    detections, _ = env.run_yolo_detection(env._get_camera_frame())
    result = {name: {"class": None, "confidence": 0.0} for name in api.CORNER_TO_TARGET.values()}
    width, height = env.camera_intrinsics[:2]
    for detection in detections:
        cls, confidence, center = api._detection_values(detection)
        if cls not in {"screw", "noscrew"} or center is None:
            continue
        target = api.CORNER_TO_TARGET[api._pixel_corner(*center, width, height)]
        if confidence > result[target]["confidence"]:
            result[target] = {"class": cls, "confidence": confidence}
    for attr, value in (("detector_call_count", 0), ("detector_inference_time_s", 0.0),
                        ("last_detector_inference_time_s", 0.0), ("_last_detector_inference_time_s", 0.0)):
        if hasattr(env, attr):
            setattr(env, attr, value)
    return result


def target_records(api, info, order, classes, probes, steps):
    history = info.get("detection_history", [])
    records = []
    previous_accepted = True
    previous_end = 0
    corners = {v: k for k, v in api.CORNER_TO_TARGET.items()}
    for index, name in enumerate(order, 1):
        events = [e for e in history if e["target_name"] == name]
        candidates = [e for e in events if e.get("predicted_class") in {"screw", "noscrew"}]
        best = max(candidates, key=lambda e: float(e["predicted_confidence"]), default={})
        accepted = next((e for e in events if e.get("verification_accepted")), {})
        last = events[-1] if events else {}
        attempted = previous_accepted and previous_end < steps
        if events and not attempted:
            raise ValueError("Target trace violates sequential target order")
        end = int(accepted["step"]) if accepted else steps
        target_steps = end - previous_end if attempted else 0
        final = accepted or best
        cls = classes[name]
        correct = bool(accepted) and accepted.get("predicted_class") == cls
        false = bool(accepted) and not correct
        baseline = probes.get(name, {"class": None, "confidence": 0.0})
        best_conf = float(best.get("predicted_confidence") or 0)
        record = dict(
            target_name=name, target_corner=corners[name], ground_truth_class=cls,
            order_index=index, attempted=attempted, accepted=bool(accepted),
            correct_verification=correct, false_verification=false,
            outcome=("correct_verification" if correct else "false_verification" if false
                     else "unverified" if attempted else "not_reached"),
            start_step=previous_end + 1 if attempted else "", end_step=end if attempted else "",
            steps=target_steps, detector_calls=sum(bool(e.get("detector_called")) for e in events),
            accepted_prediction_class=accepted.get("predicted_class") or "",
            accepted_prediction_confidence=accepted.get("predicted_confidence", ""),
            accepted_prediction_step=accepted.get("step", ""),
            best_prediction_class=best.get("predicted_class") or "",
            best_prediction_confidence=best_conf, best_prediction_step=best.get("step", ""),
            last_prediction_class=last.get("predicted_class") or "",
            last_prediction_confidence=float(last.get("predicted_confidence") or 0),
            final_prediction_class=final.get("predicted_class") or "",
            final_prediction_confidence=float(final.get("predicted_confidence") or 0),
            final_prediction_correct=final.get("predicted_class") == cls,
            best_prediction_correct=best.get("predicted_class") == cls,
            last_prediction_correct=last.get("predicted_class") == cls,
            initial_probe_prediction_class=baseline["class"] or "",
            initial_target_confidence=baseline["confidence"],
            confidence_gain=max(0.0, best_conf - baseline["confidence"]),
            observed_confidence_gain_raw=best_conf - float(events[0].get("predicted_confidence") or 0) if events else 0.0,
        )
        records.append(record)
        previous_accepted = bool(accepted)
        previous_end = end
    return records


def build_row(api, *, order, classes, probes, timestamps, **kwargs):
    # Reuse ONLY metadata and cost extraction; scalar single-target scores are discarded.
    legacy = api.build_detail_row(**kwargs)
    row = {k: legacy[k] for k in IDENTITY_FIELDS + COST_FIELDS}
    info = kwargs["info"]
    targets = target_records(api, info, order, classes, probes, row["steps"])
    accepted = sum(t["accepted"] for t in targets)
    correct = sum(t["correct_verification"] for t in targets)
    false = sum(t["false_verification"] for t in targets)
    completed = accepted == 4
    success = correct == 4
    out = bool(info.get("out_of_bounds"))
    timeout = bool(kwargs["truncated"])
    outcome = ("correct_verification" if success else "completed_with_errors" if completed
               else "out_of_bounds" if out else "timeout" if timeout else "incomplete")
    row.update(inspection_mode="sequential", execution_sequence=json.dumps(order),
               completed=completed, verified_success=success, false_verification=false > 0,
               verification_outcome=outcome, failure_reason="" if success else outcome,
               timeout=timeout, out_of_bounds=out, steps_to_verification=row["steps"] if success else "",
               accepted_target_count=accepted, correct_verification_count=correct,
               false_verification_count=false, unverified_target_count=4-accepted,
               attempted_target_count=sum(t["attempted"] for t in targets),
               unreached_target_count=sum(not t["attempted"] for t in targets),
               final_correct_count=sum(t["final_prediction_correct"] for t in targets),
               best_correct_count=sum(t["best_prediction_correct"] for t in targets),
               last_correct_count=sum(t["last_prediction_correct"] for t in targets),
               final_accuracy=sum(t["final_prediction_correct"] for t in targets)/4,
               confidence_gain=sum(t["confidence_gain"] for t in targets)/4,
               targets_json=json.dumps(targets), raw_info_json=legacy["raw_info_json"], **timestamps)
    if sum(t["steps"] for t in targets) != row["steps"]:
        raise ValueError("Per-target control steps do not sum to episode steps")
    if sum(t["detector_calls"] for t in targets) != row["detector_calls"]:
        raise ValueError("Per-target detector calls do not sum to episode detector calls")
    return row


def summarize(api, rows):
    n = len(rows)
    num = api.as_float
    count = lambda key: sum(num(r[key]) for r in rows)
    flag = lambda key: sum(api.as_bool(r[key]) for r in rows)
    success, complete = flag("verified_success"), flag("completed")
    required = 4*n
    accepted, correct, false = (count(k) for k in ("accepted_target_count", "correct_verification_count", "false_verification_count"))
    low, high = api.wilson_interval(success, n)
    result = dict(episodes=n, verified_successes=success, verified_success_rate=success/n,
                  verified_success_ci95_low=low, verified_success_ci95_high=high,
                  completed_episodes=complete, completion_rate=complete/n,
                  episodes_with_false_verification=flag("false_verification"),
                  false_verification_rate=flag("false_verification")/n,
                  required_targets=required, correct_target_verification_rate=correct/required,
                  false_target_verification_rate=false/required,
                  unverified_target_rate=(required-accepted)/required,
                  selective_accuracy=api.ratio_or_none(correct, accepted),
                  final_accuracy=count("final_correct_count")/required,
                  last_accuracy=count("last_correct_count")/required,
                  best_accuracy=count("best_correct_count")/required,
                  confidence_gain_mean=count("confidence_gain")/n,
                  steps_verified_mean=api.mean_or_none(r["steps"] for r in rows if api.as_bool(r["verified_success"])),
                  timeouts=flag("timeout"), out_of_bounds=flag("out_of_bounds"))
    for key in COUNT_FIELDS:
        result[key] = int(count(key))
    for key in ("steps", "detector_calls", "detection_action_count", "movement_action_count", "path_length", "active_wall_time_s", "detector_inference_time_s", "total_wall_time_s"):
        result[key + "_mean"] = count(key)/n
    for output, numerator, denominator in (
        ("processing_fps", "steps", "active_wall_time_s"),
        ("detector_fps", "detector_calls", "detector_inference_time_s"),
        ("detector_call_fraction", "detector_calls", "steps"),
        ("effective_detection_hz", "detector_calls", "active_wall_time_s"),
        ("ppo_inference_fps", "steps", "ppo_inference_time_s"),
    ):
        result[output] = api.ratio_or_none(count(numerator), count(denominator))
    return result


def grouped_summaries(api, rows):
    output = []
    dimensions = ("setting", "laptop_name", "laptop_version_id", "scale", "luminosity")
    for grouping, keys in (("overall", ()), ("laptop", ("laptop_name",)),
                           ("scale", ("scale",)), ("luminosity", ("luminosity",)),
                           ("configuration", ("laptop_version_id",)),
                           ("condition", dimensions)):
        groups = {}
        for row in rows:
            groups.setdefault(tuple(str(row[k]) for k in keys), []).append(row)
        for key, members in groups.items():
            values = dict(zip(keys, key))
            output.append(dict(grouping=grouping, method="ppo",
                               **{k: values.get(k, "all") for k in dimensions},
                               **summarize(api, members)))
    return output


def export_targets(path, detail_csv):
    identity = tuple(k for k in IDENTITY_FIELDS if k not in {"target_corner", "target_name", "ground_truth_class"})
    csv.field_size_limit(64 * 1024 * 1024)
    with detail_csv.open(newline="", encoding="utf-8") as source, path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=identity + TARGET_FIELDS)
        writer.writeheader()
        for row in csv.DictReader(source):
            for target in json.loads(row["targets_json"]):
                writer.writerow({**{k: row[k] for k in identity}, **target})
