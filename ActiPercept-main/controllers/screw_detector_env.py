"""
Webots + Gymnasium environment for screw detection active perception.

This environment is intended for two use-cases:
1) Keyboard demo: a human chooses a laptop corner to refine (1-4)
    and can also manually move the camera.
2) Policy control: an RL policy can provide actions.

High-level behavior
- In free-motion mode, you can move the camera with WASD + QE.
- Press 1-4 to enter "refine" mode for a specific laptop corner (TR, BR, TL, BL).
- In refine mode, the controller will:
    * Prefer tracking the "holder" class in the selected corner (cached for a short TTL).
    * Otherwise fall back to tracking "screw"/"noscrew" in that corner.
    * Move the target toward the image center, then move closer (DOWN) to increase confidence.
    * Exit refine mode and return "home" once confidence >= conf_threshold.

Notes
- This environment assumes the Webots world defines a node DEF 'camera' and a device named 'camera'.
- YOLO model is expected to output classes including: 'screw', 'noscrew', and optionally 'holder'.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import gymnasium as gym
import numpy as np

from abc import ABC

from controller import Supervisor
from ultralytics import YOLO

from utils_vision import draw_boxes_on_image


class Key(IntEnum):
    """Webots keyboard codes for the demo."""
    A = 65
    D = 68
    W = 87
    S = 83
    E = 69
    Q = 81
    ONE = 49
    TWO = 50
    THREE = 51
    FOUR = 52


class Action(IntEnum):
    """Discrete action space."""
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    BACKWARD = 3
    UP = 4
    DOWN = 5
    TOP_RIGHT = 6
    TOP_LEFT = 7
    BOTTOM_RIGHT = 8
    BOTTOM_LEFT = 9


CornerIdx = int  # 0..3 in order [TR, BR, TL, BL]

CORNER_NAMES: Tuple[str, ...] = ("top_right", "bottom_right", "top_left", "bottom_left")
VALID_TARGET_CLASSES: Tuple[str, ...] = ("screw", "noscrew")


@dataclass
class Detection:
    cls: str
    confidence: float
    bbox_center: Tuple[float, float]
    xyxy: Tuple[int, int, int, int]


class ScrewDetectorEnv(gym.Env, ABC):
    """
    Gymnasium environment wrapping a Webots Supervisor camera.

    Observation:
      - image: uint8 RGB image (H,W,3)
      - state: camera translation (x,y,z)

    Action:
      - 0..5: basic translations (LEFT/RIGHT/FORWARD/BACKWARD/UP/DOWN)
      - 6..9: select a corner to refine (TOP_RIGHT/TOP_LEFT/BOTTOM_RIGHT/BOTTOM_LEFT)

    Automated single-corner evaluation:
      reset(options={"target_corner": "top_right",
                     "ground_truth_class": "screw",
                     "luminosity": 1.0})

      A target supplied to reset enters refine mode immediately and makes the
      first confidence-threshold crossing terminal. ``info`` contains the raw
      selected class/confidence, correctness-aware outcome, confidence gain,
      control/motion/detector counters, and timing/FPS measurements.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        model_path: str = "./pc_screw_detection.pt",
        visualize_detections: bool = True,
        max_steps: Optional[int] = None,
        control_mode: str = "keyboard",  # "keyboard" | "policy"
        scripted_return_on_success: bool = False,
        move_step: float = 0.01,
        center_epsilon_px: int = 12,
        conf_threshold: float = 0.88,
        screw_success_threshold: float = 0.88,
        holder_cache_ttl: int = 5,
        device: Optional[str] = None,  # "cuda" | "cpu" | None(auto)
        terminate_on_verification: Optional[bool] = None,
    ):
        super().__init__()

        if control_mode not in ("keyboard", "policy"):
            raise ValueError('control_mode must be "keyboard" or "policy"')
        if not isinstance(scripted_return_on_success, bool):
            raise TypeError("scripted_return_on_success must be bool")

        self.control_mode = control_mode
        self.scripted_return_on_success = scripted_return_on_success
        # The keyboard demo historically stays alive after one verification so
        # another corner can be selected. Policy/evaluation episodes should end
        # at the first threshold crossing (correct *or* incorrect).
        self._default_terminate_on_verification = (
            control_mode == "policy"
            if terminate_on_verification is None
            else bool(terminate_on_verification)
        )
        self.terminate_on_verification = self._default_terminate_on_verification
        self.visualize_detections = bool(visualize_detections)
        self.max_steps = max_steps
        self.move_step = float(move_step)

        self.center_epsilon_px = int(center_epsilon_px)
        self.conf_threshold = float(conf_threshold)
        self.screw_success_threshold = float(screw_success_threshold)

        # --- Webots supervisor / devices ---
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getSelf()
        self.timestep = int(self.supervisor.getBasicTimeStep())

        self.camera_node = self.supervisor.getFromDef("camera")
        self.camera = self.supervisor.getDevice("camera")
        self.camera.enable(self.timestep)

        self.supervisor.keyboard.enable(self.timestep)

        # --- Workspace / home pose ---
        self.initial_pos = np.array([5.0, 5.37, 8.0], dtype=float)
        # self.workspace_bounds = np.array([4.55, 5.50, 5.05, 5.68, 6.85, 9.0], dtype=float)
        self.workspace_bounds: List[float] = [4.45, 5.60, 4.95, 5.85, 6.85, 9]

        # --- Laptop geometry (world coords) ---
        self.laptop_data: Dict[str, List[List[float]]] = {
            "laptop_1": [
                [5.367, 5.603, 6.687],  # right up
                [5.36, 5.132, 6.687],   # right down
                [4.708, 5.132, 6.687],  # left down
                [4.702, 5.603, 6.687],  # left up
            ]
        }
        self.laptop_id = "laptop_1"

        # --- Model ---
        self._device = device
        self.model = YOLO(model_path)
        self._set_model_device()

        # --- Camera intrinsics / image space ---
        self.camera_intrinsics = self._compute_camera_intrinsics()
        width, height, *_ = self.camera_intrinsics
        self.camera_shape = (height, width, 3)

        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(low=0, high=255, shape=self.camera_shape, dtype=np.uint8),
                "state": gym.spaces.Box(
                    low=np.array([self.workspace_bounds[0], self.workspace_bounds[2],
                                  self.workspace_bounds[4]], dtype=np.float32),
                    high=np.array([self.workspace_bounds[1], self.workspace_bounds[3],
                                    self.workspace_bounds[5]], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(len(Action))

        # --- Keyboard mapping ---
        self.key_to_action = {
            Key.A: Action.LEFT,
            Key.D: Action.RIGHT,
            Key.W: Action.FORWARD,
            Key.S: Action.BACKWARD,
            Key.E: Action.UP,
            Key.Q: Action.DOWN,
            Key.ONE: Action.TOP_LEFT,      # 1 -> TL
            Key.TWO: Action.TOP_RIGHT,     # 2 -> TR
            Key.THREE: Action.BOTTOM_RIGHT,# 3 -> BR
            Key.FOUR: Action.BOTTOM_LEFT,  # 4 -> BL
        }

        # --- Active perception state ---
        self.step_count = 0
        self.refining: bool = False
        self.target_corner: Optional[CornerIdx] = None  # 0..3 [TR, BR, TL, BL]
        self.current_target_best_confidence: float = 0.0
        self.corner_confidences: List[float] = [0.0, 0.0, 0.0, 0.0]  # [TR, BR, TL, BL]

        self.search_toggle = 1  # simple jitter when target not found

        # Holder cache per corner (center, confidence, step)
        self.holder_cache: List[Optional[Dict[str, object]]] = [None, None, None, None]
        self.holder_cache_ttl = int(holder_cache_ttl)

        # --- Evaluation/instrumentation state ---
        # These fields are reset in reset(). They are initialized here as well
        # so attributes are available to wrappers before the first reset.
        self.selected_target_corner: Optional[CornerIdx] = None
        self.ground_truth_class: Optional[str] = None
        self.initial_target_confidence: Optional[float] = None
        self.max_target_confidence: float = 0.0
        self.final_target_confidence: float = 0.0
        self.current_predicted_class: Optional[str] = None
        self.final_predicted_class: Optional[str] = None
        self.max_confidence_predicted_class: Optional[str] = None
        self.max_confidence_prediction_step: Optional[int] = None
        self.accepted_predicted_class: Optional[str] = None
        self.accepted_predicted_confidence: Optional[float] = None
        self.threshold_crossing_step: Optional[int] = None
        self.threshold_crossed: bool = False
        self.correct_verification: bool = False
        self.false_verification: bool = False
        self.termination_reason: Optional[str] = None
        self.episode_terminated: bool = False
        self.episode_truncated: bool = False

        self.detector_call_count: int = 0
        self.detector_inference_time_s: float = 0.0
        self._last_detector_inference_time_s: float = 0.0
        self.prediction_trace: List[Dict[str, object]] = []
        self.motion_action_count: int = 0
        self.motion_path_length: float = 0.0
        self.return_home_motion_count: int = 0
        self.return_home_call_count: int = 0
        self.return_home_path_length: float = 0.0
        self.simulator_step_count: int = 0
        self.return_home_simulator_step_count: int = 0
        self.pause_simulator_step_count: int = 0
        self.step_processing_time_s: float = 0.0
        self.return_home_time_s: float = 0.0
        self._episode_start_time: float = perf_counter()
        self._current_step_started_at: Optional[float] = None
        self.current_luminosity: Optional[float] = None
        self.episode_seed: Optional[int] = None

    # ----------------------------- Model/device -----------------------------
    def _set_model_device(self) -> None:
        """Move YOLO model to the requested/available device."""
        # Ultralytics accepts device strings. Default to cuda if available.
        dev = self._device
        if dev is None:
            try:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev = "cpu"
        self._device = dev
        try:
            self.model.to(dev)
        except Exception:
            pass

    # ----------------------------- Webots helpers -----------------------------
    def set_robot_position(self, pos: Sequence[float]) -> None:
        self.robot.getField("translation").setSFVec3f(list(map(float, pos)))

    def get_robot_position(self) -> np.ndarray:
        return np.asarray(self.robot.getField("translation").getSFVec3f(), dtype=float)

    # ----------------------------- Camera -----------------------------
    def _compute_camera_intrinsics(self):
        width = int(self.camera.getWidth())
        height = int(self.camera.getHeight())
        fov_x = float(self.camera.getFov())
        fov_y = 2 * np.arctan(np.tan(fov_x * 0.5) * (height / width))

        cx = width / 2.0
        cy = height / 2.0
        fx = cx / np.tan(fov_x / 2.0)
        fy = cy / np.tan(fov_y / 2.0)

        intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        return width, height, fov_x, fov_y, fx, fy, cx, cy, intrinsic_matrix

    def _get_camera_frame(self) -> np.ndarray:
        camera_frame = self.camera.getImage()
        width, height, *_ = self.camera_intrinsics

        if not camera_frame:
            return np.zeros(self.camera_shape, dtype=np.uint8)

        try:
            frame = np.frombuffer(camera_frame, np.uint8).reshape((height, width, 4))[:, :, :3]
            return frame.astype(np.uint8).copy()
        except Exception:
            return np.zeros(self.camera_shape, dtype=np.uint8)

    # ----------------------------- YOLO -----------------------------
    def run_yolo_detection(self, image: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        inference_started_at = perf_counter()
        try:
            results = self.model.predict(
                source=image,
                show=False,
                show_labels=False,
                save=False,
                verbose=False,
                device=self._device or "cpu",
            )
        finally:
            # Count attempted calls as detector work even if the detector raises;
            # this keeps the cost accounting honest in failed evaluations.
            inference_time_s = perf_counter() - inference_started_at
            self.detector_call_count += 1
            self.detector_inference_time_s += inference_time_s
            self._last_detector_inference_time_s = inference_time_s

        detections: List[Detection] = []
        img_with_boxes = image.copy()

        for result in results:
            if result.boxes is None:
                continue

            img_with_boxes = draw_boxes_on_image(result.orig_img, result.boxes, self.model.names)

            for box in result.boxes:
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = str(self.model.names[class_id])

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_center_x = (x1 + x2) / 2.0
                box_center_y = (y1 + y2) / 2.0

                detections.append(
                    Detection(
                        cls=class_name,
                        confidence=conf,
                        bbox_center=(box_center_x, box_center_y),
                        xyxy=(x1, y1, x2, y2),
                    )
                )

        return detections, img_with_boxes

    # ----------------------------- Corner logic -----------------------------
    def _corner_from_pixel(self, x: float, y: float) -> CornerIdx:
        """Map a pixel to one of 4 corners using the image center: [TR, BR, TL, BL]."""
        width, height, *_ = self.camera_intrinsics
        cx, cy = width / 2.0, height / 2.0

        right = x >= cx
        bottom = y >= cy

        if right and not bottom:
            return 0  # TR
        if right and bottom:
            return 1  # BR
        if (not right) and (not bottom):
            return 2  # TL
        return 3  # BL

    def _update_corner_confidences(self, detections: List[Detection]) -> List[float]:
        """Update corner_confidences as max screw/noscrew confidence per corner."""
        confs = [0.0, 0.0, 0.0, 0.0]
        for det in detections:
            if det.cls not in ("screw", "noscrew"):
                continue
            corner = self._corner_from_pixel(*det.bbox_center)
            confs[corner] = max(confs[corner], float(det.confidence))
        self.corner_confidences = confs
        return confs

    def _select_best_in_corner(
        self, detections: List[Detection], corner_idx: CornerIdx, class_whitelist: Tuple[str, ...]
    ) -> Tuple[Optional[Detection], float]:
        best: Optional[Detection] = None
        best_conf = -1.0
        for det in detections:
            if det.cls not in class_whitelist:
                continue
            if self._corner_from_pixel(*det.bbox_center) != corner_idx:
                continue
            if det.confidence > best_conf:
                best_conf = det.confidence
                best = det
        return best, float(best_conf)

    def _update_holder_cache(self, detections: List[Detection]) -> None:
        for corner_idx in range(4):
            det, conf = self._select_best_in_corner(
                detections, corner_idx, class_whitelist=("holder",))
            if det is None:
                continue
            self.holder_cache[corner_idx] = {"bbox_center": det.bbox_center,
                                             "confidence": conf, "step": int(self.step_count)}

    def _get_cached_holder(self, corner_idx: CornerIdx) -> Optional[Tuple[float, float]]:
        entry = self.holder_cache[corner_idx]
        if entry is None:
            return None
        age = int(self.step_count) - int(entry["step"])  # type: ignore[index]
        if age <= self.holder_cache_ttl:
            return entry["bbox_center"]  # type: ignore[return-value]
        return None

    # ----------------------------- Motion policy -----------------------------
    def _active_motion_from_bbox(self, bbox_center: Tuple[float, float]) -> Action:
        """
        Rule-based motion:
          - Move bbox center to image center (LEFT/RIGHT + FORWARD/BACKWARD).
          - Once centered, move DOWN (approach) to increase resolution.
        """
        width, height, *_ = self.camera_intrinsics
        cx, cy = width / 2.0, height / 2.0
        x, y = bbox_center
        ex = x - cx
        ey = y - cy

        if abs(ex) > self.center_epsilon_px:
            return Action.RIGHT if ex > 0 else Action.LEFT

        if abs(ey) > self.center_epsilon_px:
            # Assumes FORWARD/BACKWARD changes vertical pixel error in your setup.
            return Action.BACKWARD if ey > 0 else Action.FORWARD

        return Action.DOWN

    # ----------------------------- Rendering -----------------------------
    def _visualize(self, image: Optional[np.ndarray]) -> None:
        if not self.visualize_detections or cv2 is None:
            return
        frame = self._get_camera_frame() if image is None else image
        display_obs = cv2.resize(frame, (600, 400))
        cv2.imshow("Screw Detector (Webots)", display_obs)
        cv2.waitKey(1)

    # ----------------------------- Action decoding -----------------------------
    def _get_action(self, action: Optional[int]) -> Optional[Action]:
        if self.control_mode == "keyboard":
            key = self.supervisor.keyboard.getKey()
            if key == -1:
                return None
            try:
                return self.key_to_action.get(Key(key))
            except Exception:
                return None

        if action is None:
            return None
        try:
            return Action(int(action))
        except Exception:
            return None

    # ----------------------------- Bounds / safety -----------------------------
    def _is_out_of_bounds(self) -> bool:
        pos = self.get_robot_position()
        min_x, max_x, min_y, max_y, min_z, max_z = self.workspace_bounds
        return not (min_x <= pos[0] <= max_x and
                    min_y <= pos[1] <= max_y and min_z <= pos[2] <= max_z)

    # ----------------------------- Evaluation helpers -----------------------------
    @staticmethod
    def _parse_corner(value: object) -> CornerIdx:
        """Parse a reset target as a corner index or a readable corner name."""
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            corner_idx = int(value)
            if 0 <= corner_idx < len(CORNER_NAMES):
                return corner_idx
            raise ValueError("target_corner index must be in [0, 3]")

        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
            aliases = {
                "tr": 0, "top_right": 0, "right_up": 0,
                "br": 1, "bottom_right": 1, "right_down": 1,
                "tl": 2, "top_left": 2, "left_up": 2,
                "bl": 3, "bottom_left": 3, "left_down": 3,
            }
            if normalized in aliases:
                return aliases[normalized]

        raise ValueError(
            "target_corner must be an index 0..3 or one of "
            "top_right, bottom_right, top_left, bottom_left"
        )

    def set_light_luminosity(self, luminosity: float) -> bool:
        """Set DEF ``light`` luminosity when the current world exposes it."""
        light_node = self.supervisor.getFromDef("light")
        if light_node is None:
            return False
        luminosity_field = light_node.getField("luminosity")
        if luminosity_field is None:
            return False
        self.current_luminosity = float(luminosity)
        luminosity_field.setSFFloat(self.current_luminosity)
        return True

    def _reset_evaluation_state(self) -> None:
        self.selected_target_corner = None
        self.ground_truth_class = None
        self.initial_target_confidence = None
        self.max_target_confidence = 0.0
        self.final_target_confidence = 0.0
        self.current_predicted_class = None
        self.final_predicted_class = None
        self.max_confidence_predicted_class = None
        self.max_confidence_prediction_step = None
        self.accepted_predicted_class = None
        self.accepted_predicted_confidence = None
        self.threshold_crossing_step = None
        self.threshold_crossed = False
        self.correct_verification = False
        self.false_verification = False
        self.termination_reason = None
        self.episode_terminated = False
        self.episode_truncated = False

        self.detector_call_count = 0
        self.detector_inference_time_s = 0.0
        self._last_detector_inference_time_s = 0.0
        self.prediction_trace = []
        self.motion_action_count = 0
        self.motion_path_length = 0.0
        self.return_home_motion_count = 0
        self.return_home_call_count = 0
        self.return_home_path_length = 0.0
        self.simulator_step_count = 0
        self.return_home_simulator_step_count = 0
        self.pause_simulator_step_count = 0
        self.step_processing_time_s = 0.0
        self.return_home_time_s = 0.0
        self._current_step_started_at = None
        self._episode_start_time = perf_counter()

    def _record_target_detection(self, detection: Optional[Detection]) -> None:
        """Record the unfiltered detector decision for correctness-aware scoring."""
        confidence = float(detection.confidence) if detection is not None else 0.0
        predicted_class = detection.cls if detection is not None else None

        if self.initial_target_confidence is None:
            self.initial_target_confidence = confidence

        self.current_predicted_class = predicted_class
        self.final_predicted_class = predicted_class
        self.final_target_confidence = confidence
        if confidence > self.max_target_confidence:
            self.max_target_confidence = confidence
            self.max_confidence_predicted_class = predicted_class
            self.max_confidence_prediction_step = self.step_count

        # Keep the historical attribute synchronized for existing callers.
        self.current_target_best_confidence = self.max_target_confidence

        self._append_prediction_trace(detection)

    def _append_prediction_trace(self, detection: Optional[Detection]) -> None:
        """Append one compact target-decision record for the latest YOLO call."""
        confidence = float(detection.confidence) if detection is not None else 0.0
        predicted_class = detection.cls if detection is not None else None

        candidate_correct: Optional[bool] = None
        if predicted_class is not None and self.ground_truth_class is not None:
            candidate_correct = predicted_class == self.ground_truth_class
        selected_corner = self.target_corner
        self.prediction_trace.append(
            {
                "step": int(self.step_count),
                "corner": selected_corner,
                "corner_name": (
                    CORNER_NAMES[selected_corner] if selected_corner is not None else None
                ),
                "predicted_class": predicted_class,
                "confidence": confidence,
                "correct": candidate_correct,
                "threshold_crossed": confidence >= self.conf_threshold,
                "inference_time_s": float(self._last_detector_inference_time_s),
                "camera_position": self.get_robot_position().astype(float).tolist(),
            }
        )

    def _start_refinement(self, corner_idx: CornerIdx) -> None:
        """Start a new selected-corner decision without resetting cost counters."""
        self.target_corner = corner_idx
        self.selected_target_corner = corner_idx
        self.refining = True
        self.current_target_best_confidence = 0.0
        self.initial_target_confidence = None
        self.max_target_confidence = 0.0
        self.final_target_confidence = 0.0
        self.current_predicted_class = None
        self.final_predicted_class = None
        self.max_confidence_predicted_class = None
        self.max_confidence_prediction_step = None
        self.accepted_predicted_class = None
        self.accepted_predicted_confidence = None
        self.threshold_crossing_step = None
        self.threshold_crossed = False
        self.correct_verification = False
        self.false_verification = False
        self.termination_reason = None

    def _score_threshold_crossing(self) -> None:
        """Score, but never filter, the detector's first threshold crossing."""
        self.threshold_crossed = True
        self.accepted_predicted_class = self.current_predicted_class
        self.accepted_predicted_confidence = self.final_target_confidence
        self.threshold_crossing_step = self.step_count
        if self.ground_truth_class is None:
            return
        self.correct_verification = self.current_predicted_class == self.ground_truth_class
        self.false_verification = not self.correct_verification

    def _advance_simulation(self, *, return_home: bool = False, pause: bool = False) -> int:
        result = self.supervisor.step(self.timestep)
        self.simulator_step_count += 1
        if return_home:
            self.return_home_simulator_step_count += 1
        if pause:
            self.pause_simulator_step_count += 1
        return result

    def _verification_outcome(self) -> str:
        if self.threshold_crossed:
            if self.ground_truth_class is None:
                return "unscored_verification"
            return "correct_verification" if self.correct_verification else "false_verification"
        if self.termination_reason in ("timeout", "out_of_bounds"):
            return "not_verified"
        return "in_progress"

    def _build_info(self, *, refining: Optional[bool] = None) -> Dict[str, object]:
        now = perf_counter()
        total_elapsed_wall_time_s = max(0.0, now - self._episode_start_time)
        active_elapsed_wall_time_s = max(
            0.0, total_elapsed_wall_time_s - self.return_home_time_s
        )
        initial_confidence = (
            float(self.initial_target_confidence)
            if self.initial_target_confidence is not None
            else 0.0
        )
        confidence_gain_raw = self.max_target_confidence - initial_confidence
        confidence_gain = max(0.0, confidence_gain_raw)
        final_prediction_correct: Optional[bool] = None
        best_prediction_correct: Optional[bool] = None
        accepted_prediction_correct: Optional[bool] = None
        if self.ground_truth_class is not None:
            final_prediction_correct = self.final_predicted_class == self.ground_truth_class
            best_prediction_correct = (
                self.max_confidence_predicted_class == self.ground_truth_class
            )
            if self.accepted_predicted_class is not None:
                accepted_prediction_correct = (
                    self.accepted_predicted_class == self.ground_truth_class
                )

        selected_corner = self.selected_target_corner
        active_simulator_step_count = max(
            0,
            self.simulator_step_count
            - self.return_home_simulator_step_count
            - self.pause_simulator_step_count,
        )
        camera_position = self.get_robot_position().astype(float)
        at_home = bool(np.all(np.abs(camera_position - self.initial_pos) <= 0.01))
        return {
            # Task and detector decision.
            "target_corner": selected_corner,
            "target_corner_name": (
                CORNER_NAMES[selected_corner] if selected_corner is not None else None
            ),
            "ground_truth_class": self.ground_truth_class,
            "predicted_class": self.current_predicted_class,
            "predicted_confidence": float(self.final_target_confidence),
            "selected_predicted_class": self.current_predicted_class,
            "selected_predicted_confidence": float(self.final_target_confidence),
            "last_candidate_class": self.current_predicted_class,
            "last_candidate_confidence": float(self.final_target_confidence),
            "last_prediction_class": self.current_predicted_class,
            "last_prediction_confidence": float(self.final_target_confidence),
            "last_prediction_correct": final_prediction_correct,
            "final_predicted_class": self.final_predicted_class,
            "final_prediction_correct": final_prediction_correct,
            "final_accuracy": final_prediction_correct,
            "initial_target_confidence": initial_confidence,
            "max_target_confidence": float(self.max_target_confidence),
            "max_confidence_predicted_class": self.max_confidence_predicted_class,
            "best_candidate_class": self.max_confidence_predicted_class,
            "best_candidate_confidence": float(self.max_target_confidence),
            "best_prediction_class": self.max_confidence_predicted_class,
            "best_prediction_confidence": float(self.max_target_confidence),
            "best_prediction_correct": best_prediction_correct,
            "best_prediction_step": self.max_confidence_prediction_step,
            "final_target_confidence": float(self.final_target_confidence),
            # Explicit definition used by this implementation: max - initial.
            "confidence_gain": float(confidence_gain),
            "confidence_gain_raw": float(confidence_gain_raw),
            "confidence_gain_positive": float(confidence_gain),
            "corner_confidences": list(self.corner_confidences),
            "target_conf": float(self.final_target_confidence),
            "best_conf": float(self.max_target_confidence),
            "threshold": float(self.conf_threshold),
            "threshold_crossed": bool(self.threshold_crossed),
            "accepted_predicted_class": self.accepted_predicted_class,
            "accepted_predicted_confidence": self.accepted_predicted_confidence,
            "accepted_prediction_correct": accepted_prediction_correct,
            "threshold_crossing_step": self.threshold_crossing_step,
            "steps_to_threshold": self.threshold_crossing_step,
            "correct_verification": bool(self.correct_verification),
            "false_verification": bool(self.false_verification),
            "verified_success": bool(self.correct_verification),
            "verified": bool(self.correct_verification),
            "steps_to_verified_success": (
                int(self.step_count) if self.correct_verification else None
            ),
            "verification_outcome": self._verification_outcome(),
            "termination_reason": self.termination_reason,
            "terminated": bool(self.episode_terminated),
            "truncated": bool(self.episode_truncated),
            "episode_done": bool(self.episode_terminated or self.episode_truncated),
            "timed_out": self.termination_reason == "timeout",
            "out_of_bounds": self.termination_reason == "out_of_bounds",
            "refining": self.refining if refining is None else bool(refining),
            "current_luminosity": self.current_luminosity,
            "camera_position": camera_position.tolist(),
            "at_home": at_home,
            "return_home_completed": bool(
                self.return_home_call_count > 0 and at_home
            ),
            "prediction_trace": [dict(item) for item in self.prediction_trace],
            "seed": self.episode_seed,
            "max_steps": self.max_steps,
            # Comparable cost counters. A control step is one public step() call;
            # a simulator step also includes pause and return-home ticks.
            "step_count": int(self.step_count),
            "control_step_count": int(self.step_count),
            "detector_calls": int(self.detector_call_count),
            "motion_action_count": int(self.motion_action_count),
            "motion_path_length": float(self.motion_path_length),
            "return_home_motion_count": int(self.return_home_motion_count),
            "return_home_call_count": int(self.return_home_call_count),
            "return_home_path_length": float(self.return_home_path_length),
            "simulator_step_count": int(self.simulator_step_count),
            "active_simulator_step_count": int(active_simulator_step_count),
            "return_home_simulator_step_count": int(self.return_home_simulator_step_count),
            "pause_simulator_step_count": int(self.pause_simulator_step_count),
            "elapsed_wall_time_s": float(total_elapsed_wall_time_s),
            "total_elapsed_wall_time_s": float(total_elapsed_wall_time_s),
            "active_elapsed_wall_time_s": float(active_elapsed_wall_time_s),
            "step_processing_time_s": float(self.step_processing_time_s),
            "detector_inference_time_s": float(self.detector_inference_time_s),
            "return_home_time_s": float(self.return_home_time_s),
            # Return-home time is reported separately and excluded here because
            # the paper's steps-to-success metric ends at verification.
            "processing_fps": (
                float(self.step_count / active_elapsed_wall_time_s)
                if active_elapsed_wall_time_s > 0.0 else 0.0
            ),
            "internal_processing_fps": (
                float(self.step_count / self.step_processing_time_s)
                if self.step_processing_time_s > 0.0 else 0.0
            ),
            "detector_fps": (
                float(self.detector_call_count / self.detector_inference_time_s)
                if self.detector_inference_time_s > 0.0 else 0.0
            ),
            "simulated_time_s": float(self.simulator_step_count * self.timestep / 1000.0),
            "active_simulated_time_s": float(
                active_simulator_step_count * self.timestep / 1000.0
            ),
        }

    def _finish_step_info(
        self,
        *,
        refining: Optional[bool] = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> Dict[str, object]:
        self.episode_terminated = bool(terminated)
        self.episode_truncated = bool(truncated)
        if self._current_step_started_at is not None:
            self.step_processing_time_s += perf_counter() - self._current_step_started_at
            self._current_step_started_at = None
        return self._build_info(refining=refining)

    # ----------------------------- Core Gym API -----------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.supervisor.simulationReset()
        self.supervisor.step(self.timestep)

        # simulationReset() invalidates Webots Node/Field handles. Re-acquire
        # them so repeated episodes in one evaluator process remain safe.
        self.robot = self.supervisor.getSelf()
        self.camera_node = self.supervisor.getFromDef("camera")
        self.camera = self.supervisor.getDevice("camera")
        self.camera.enable(self.timestep)
        self.supervisor.keyboard.enable(self.timestep)
        self.camera_intrinsics = self._compute_camera_intrinsics()

        self.step_count = 0
        self.refining = False
        self.target_corner = None
        self.current_target_best_confidence = 0.0
        self.corner_confidences = [0.0, 0.0, 0.0, 0.0]
        self.search_toggle = 1
        self.holder_cache = [None, None, None, None]

        self._reset_evaluation_state()
        self.episode_seed = seed
        self.terminate_on_verification = self._default_terminate_on_verification
        self.current_luminosity = None

        reset_options = options or {}
        if "terminate_on_verification" in reset_options:
            self.terminate_on_verification = bool(reset_options["terminate_on_verification"])

        ground_truth = reset_options.get("ground_truth_class")
        if ground_truth is not None:
            normalized_ground_truth = str(ground_truth).strip().lower()
            if normalized_ground_truth not in VALID_TARGET_CLASSES:
                raise ValueError("ground_truth_class must be 'screw' or 'noscrew'")
            self.ground_truth_class = normalized_ground_truth

        requested_corner = reset_options.get(
            "target_corner", reset_options.get("target_corner_name")
        )
        if requested_corner is not None:
            # Supplying a target at reset denotes an automated single-corner
            # episode. Allow an explicit option to override this default.
            if "terminate_on_verification" not in reset_options:
                self.terminate_on_verification = True
            corner_idx = self._parse_corner(requested_corner)
            self._start_refinement(corner_idx)

        requested_luminosity = reset_options.get("luminosity")
        if requested_luminosity is not None:
            self.set_light_luminosity(float(requested_luminosity))
        else:
            light_node = self.supervisor.getFromDef("light")
            if light_node is not None:
                luminosity_field = light_node.getField("luminosity")
                if luminosity_field is not None:
                    self.current_luminosity = float(luminosity_field.getSFFloat())

        # Refresh the re-enabled camera and let any light change reach the next
        # frame. Reset setup ticks do not count as evaluation control steps.
        self.supervisor.step(self.timestep)

        obs = self._make_observation()
        # Reset setup and its first camera acquisition are outside the measured
        # evaluation episode.
        self._episode_start_time = perf_counter()
        info = self._build_info()
        return obs, info

    def _make_observation(self) -> Dict[str, np.ndarray]:
        img = self._get_camera_frame()
        state = self.get_robot_position().astype(np.float32)
        return {"image": img, "state": state}

    def step(self, action: Optional[int]):
        self._current_step_started_at = perf_counter()
        self.step_count += 1
        reward = 0.0  # placeholder (demo environment)

        # 1) Observe + detect
        obs_img = self._get_camera_frame()
        detections, img_with_boxes = self.run_yolo_detection(obs_img)
        self._update_holder_cache(detections)
        self._update_corner_confidences(detections)

        # 2) Visualize
        self._visualize(img_with_boxes)

        # 3) Decode action
        a = self._get_action(action)

        # Corner selection actions -> enter refine mode
        corner_action_to_idx: Dict[Action, CornerIdx] = {
            Action.TOP_RIGHT: 0,
            Action.BOTTOM_RIGHT: 1,
            Action.TOP_LEFT: 2,
            Action.BOTTOM_LEFT: 3,
        }
        if a in corner_action_to_idx:
            self._start_refinement(corner_action_to_idx[a])  # type: ignore[index]

        # 4) Execute motion
        if not self.refining:
            # There was a detector call, but no selected target with which to
            # associate a screw/noscrew decision.
            self._append_prediction_trace(None)
            if a is not None and int(a) < 6:
                self._apply_translation(a)

            terminated = False
            truncated = False

            # Safety
            out_of_bounds = self._is_out_of_bounds()
            timed_out = self.max_steps is not None and self.step_count >= self.max_steps
            if out_of_bounds:
                self.termination_reason = "out_of_bounds"
                self._scripted_return_home()
                terminated = self.terminate_on_verification
            elif timed_out:
                self.termination_reason = "timeout"
                self._scripted_return_home()
                truncated = True

            self._advance_simulation()
            obs = self._make_observation()
            info = self._finish_step_info(
                refining=False, terminated=terminated, truncated=truncated
            )
            return obs, reward, terminated, truncated, info

        # --- Refine mode ---
        assert self.target_corner is not None

        # Termination confidence comes from screw/noscrew in the selected corner
        target_det, _ = self._select_best_in_corner(
            detections,
            self.target_corner,
            class_whitelist=VALID_TARGET_CLASSES,
        )
        self._record_target_detection(target_det)
        target_conf = self.final_target_confidence

        # Stop: confidence reached -> return home and exit refine mode
        if target_conf >= self.conf_threshold:
            # The accepted class comes directly from YOLO. Ground truth is used
            # only here for scoring, never to suppress a detection.
            self._score_threshold_crossing()
            self.termination_reason = "threshold_crossed"

            # Retain the visual pause for the interactive demo, but do not add
            # twelve unproductive ticks to automated evaluation episodes.
            if self.control_mode == "keyboard" and not self.terminate_on_verification:
                for _ in range(12):
                    self._advance_simulation(pause=True)
            if self.scripted_return_on_success or self.control_mode == "keyboard":
                self._scripted_return_home()
            self.refining = False
            self.target_corner = None

            self._advance_simulation()
            obs = self._make_observation()
            terminated = self.terminate_on_verification
            truncated = False
            info = self._finish_step_info(
                refining=False, terminated=terminated, truncated=truncated
            )
            return obs, reward, terminated, truncated, info

        # Movement target: holder preferred; fallback to cached holder; fallback to screw/noscrew
        move_center: Optional[Tuple[float, float]] = None

        move_det, _ = self._select_best_in_corner(detections, self.target_corner,
                                                  class_whitelist=("holder",))
        if move_det is not None:
            move_center = move_det.bbox_center
        else:
            move_center = self._get_cached_holder(self.target_corner)

        if move_center is None:
            move_det, _ = self._select_best_in_corner(detections, self.target_corner,
                                                      class_whitelist=("screw", "noscrew"))
            if move_det is not None:
                move_center = move_det.bbox_center

        if move_center is not None:
            next_action = self._active_motion_from_bbox(move_center)

            # Prevent "crashing" into the desk/laptop: clamp Z
            pos = self.get_robot_position()
            min_z = float(self.workspace_bounds[4])
            if next_action == Action.DOWN and (pos[2] - self.move_step) < (min_z + 0.02):
                next_action = Action.LEFT if self.search_toggle > 0 else Action.RIGHT
                self.search_toggle *= -1

            self._apply_translation(next_action)
        else:
            # Nothing to track -> jitter search
            self._apply_translation(Action.LEFT if self.search_toggle > 0 else Action.RIGHT)
            self.search_toggle *= -1

        # Safety termination
        out_of_bounds = self._is_out_of_bounds()
        timed_out = self.max_steps is not None and self.step_count >= self.max_steps
        terminated = False
        truncated = False
        if out_of_bounds:
            self.termination_reason = "out_of_bounds"
            self._scripted_return_home()
            self.refining = False
            self.target_corner = None
            terminated = self.terminate_on_verification
        elif timed_out:
            self.termination_reason = "timeout"
            self._scripted_return_home()
            self.refining = False
            self.target_corner = None
            truncated = True

        self._advance_simulation()
        obs = self._make_observation()
        info = self._finish_step_info(terminated=terminated, truncated=truncated)
        return obs, reward, terminated, truncated, info

    def _apply_translation(self, action: Action, *, return_home: bool = False) -> None:
        previous_pos = self.get_robot_position()
        pos = previous_pos.copy()
        if action == Action.LEFT:
            pos[0] -= self.move_step
        elif action == Action.RIGHT:
            pos[0] += self.move_step
        elif action == Action.FORWARD:
            pos[1] += self.move_step
        elif action == Action.BACKWARD:
            pos[1] -= self.move_step
        elif action == Action.UP:
            pos[2] += self.move_step
        elif action == Action.DOWN:
            pos[2] -= self.move_step
        self.set_robot_position(pos)
        commanded_distance = float(np.linalg.norm(pos - previous_pos))
        if return_home:
            self.return_home_motion_count += 1
            self.return_home_path_length += commanded_distance
        else:
            self.motion_action_count += 1
            self.motion_path_length += commanded_distance

    def _scripted_return_home(self) -> None:
        """Simple proportional-ish return to the initial translation pose."""
        return_started_at = perf_counter()
        self.return_home_call_count += 1
        tolerance = 0.01
        max_iterations = 1000
        target_pos = self.initial_pos.copy()

        try:
            for _ in range(max_iterations):
                current_pos = self.get_robot_position()
                delta = target_pos - current_pos

                if abs(delta[0]) > tolerance:
                    self._apply_translation(
                        Action.RIGHT if delta[0] > 0 else Action.LEFT,
                        return_home=True,
                    )
                elif abs(delta[1]) > tolerance:
                    self._apply_translation(
                        Action.FORWARD if delta[1] > 0 else Action.BACKWARD,
                        return_home=True,
                    )
                elif abs(delta[2]) > tolerance:
                    self._apply_translation(
                        Action.UP if delta[2] > 0 else Action.DOWN,
                        return_home=True,
                    )
                else:
                    break

                self._visualize(image=None)
                self._advance_simulation(return_home=True)
        finally:
            self.return_home_time_s += perf_counter() - return_started_at

    def close(self):
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        super().close()
