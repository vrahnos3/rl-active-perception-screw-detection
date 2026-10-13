from __future__ import annotations

from controller import Supervisor
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
import numpy as np
import random
import cv2
import sys
import json
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC
from ultralytics import YOLO
from test.utils import draw_boxes_on_image
from dataclasses import dataclass
from pathlib import Path
from utils.draw_utils import draw_valid_zone, draw_workspace_bounds, draw_agent_zone, update_agent_zone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class KeyActions(Enum):
    LEFT = 65
    RIGHT = 68
    FORWARD = 87
    BACKWARD = 83
    UP = 69
    DOWN = 81
    DETECT = 32  # Spacebar for manual testing


class Actions(Enum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    BACKWARD = 3
    UP = 4
    DOWN = 5
    DETECT = 6


VALID_TARGET_CLASSES: Tuple[str, ...] = ("screw", "noscrew")


@dataclass
class Screw:
    name: str
    point: np.ndarray
    one_hot: np.ndarray
    confirmed: bool = False
    gt_class: Optional[str] = None
    max_confidence: float = 0.0
    verified_class: Optional[str] = None
    verification_confidence: float = 0.0
    verification_correct: Optional[bool] = None

    def reset(self) -> None:
        self.confirmed = False
        self.max_confidence = 0.0
        self.verified_class = None
        self.verification_confidence = 0.0
        self.verification_correct = None


class ScrewDetectorEnv(gym.Env, ABC):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self,
                 yolo_path: str = None,
                 visualize_detections: Optional[bool] = None,
                 max_steps: Optional[int] = None,
                 control_mode: Optional[str] = None,
                 mode: Optional[str] = None,
                 laptop_metadata_path: Optional[str] = None,
                 verbose: bool = True,
                 finetuning: bool = False,
                 scripted_return_on_success: bool = False,
                 verification_threshold: float = 0.8,
                 device: Optional[str] = None,
                 ) -> None:
        super(ScrewDetectorEnv, self).__init__()

        # ---------- Checks for Arguments --------------------------------------------------------
        if laptop_metadata_path is None:
            raise ValueError('Please give a metadata_path value')
        if not isinstance(visualize_detections, bool):
            raise TypeError('visualize_detections must be bool')
        if control_mode not in ('keyboard', 'policy') or control_mode is None:
            raise ValueError('control_mode must be "keyboard" or "policy"')
        if mode not in ['train', 'eval'] or mode is None:
            raise ValueError("mode must be 'train' or 'eval'")
        if not isinstance(verbose, bool):
            raise TypeError('verbose must be bool')
        if not isinstance(scripted_return_on_success, bool):
            raise TypeError('scripted_return_on_success must be bool')
        if not 0.0 <= float(verification_threshold) <= 1.0:
            raise ValueError('verification_threshold must be between 0 and 1')
        if not isinstance(finetuning, bool):
            raise TypeError("finetuning must be bool")

        self.visualize_detections: bool = visualize_detections
        self.control_mode: str = control_mode
        self.mode: str = mode
        self.verbose: bool = verbose
        self._default_scripted_return_on_success: bool = scripted_return_on_success
        self.scripted_return_on_success: bool = scripted_return_on_success
        self._device: Optional[str] = device
        self.finetuning = finetuning
        
        # ------------- Load metadata laptop database ------------------------------------------
        with open(laptop_metadata_path, 'r') as f:
            self.laptop_database: Dict[str, Any] = json.load(f)

        self.target_names: List[str] = sorted(list(self.laptop_database["screws"].keys()))
        self.num_screws: int = len(self.target_names)
        self.execution_sequence: List[str] = list(self.target_names)
        self.screws: Dict[str, Screw] = {}
        self.current_target_name: Optional[str] = None
        self.current_laptop_name: Optional[str] = None
        self.current_laptop_version_id: Optional[str] = None
        self.current_luminosity: Optional[float] = None

        # Random initialization of Screws. The fields will be filled correctly when the .obj will be loaded inside the load_laptop_version()
        for name in self.target_names:
            self.screws[name] = Screw(
                name=name,
                point=np.zeros(3, dtype=np.float32),
                one_hot=np.array(self.laptop_database["screws"][name]["one_hot"], dtype=np.float32)
                )

        # ------------- Get the Webots root node (needed for spawning objects) -----------------
        self.supervisor: Supervisor = Supervisor()
        self.root_node: Any = self.supervisor.getRoot()
        self.children_field: Any = self.root_node.getField('children')
        self.current_laptop_node: Any = None

        self.robot: Any = self.supervisor.getSelf()
        self.timestep: int = int(self.supervisor.getBasicTimeStep())
        
        self.laptop_transform_node = self.supervisor.getFromDef("LAPTOP_TRANSFORM")
        self.laptop_cad_node = self.supervisor.getFromDef("LAPTOP_CAD")

        if self.laptop_transform_node is None or self.laptop_cad_node is None:
            raise ValueError("Could not find DEF LAPTOP_TRANSFORM or LAPTOP_CAD in webots tree")
        
        self.laptop_model_url = self.laptop_cad_node.getField("url")
        #---------------------------------------------------------------------------------------
        
        self.camera_shape: Tuple[int, int, int] = (270, 480, 3)
        self.image_space = spaces.Box(low=0, high=255, shape=self.camera_shape, dtype=np.uint8)
        self.action_space: spaces.Discrete = spaces.Discrete(len(Actions))
        self.model: YOLO = YOLO(yolo_path)
        if self._device is not None:
            self.model.to(self._device)
        self.max_steps: Optional[int] = max_steps
        self.scale_factor: int = 1
        self.step_count: int = 0
        self.initial_pos: List[float] = [5., 5.37, 8]
        self.move: float = 0.01
        self.frame_skip: int = 3
        self.total_reward: float = 0.0

        # -------- Evaluation / instrumentation state -----------------------------------------
        # Keep action selections separate from detector invocations: a DETECT action with no
        # active target is still an action, but does not invoke YOLO.
        self.detection_action_count: int = 0
        self.detector_call_count: int = 0
        self.motion_action_count: int = 0
        self.no_op_action_count: int = 0
        self.motion_path_length: float = 0.0
        self.detector_inference_time_s: float = 0.0
        self.last_detector_inference_time_s: float = 0.0
        self.step_processing_time_s: float = 0.0
        self.last_step_processing_time_s: float = 0.0
        self.simulator_step_count: int = 0
        self.return_home_steps: int = 0
        self.return_home_time_s: float = 0.0
        self.return_home_path_length: float = 0.0
        self.return_home_reached: bool = False
        self.correct_verification_count: int = 0
        self.false_verification_count: int = 0
        self.verification_step: Optional[int] = None
        self.initial_target_confidence: float = 0.0
        self.max_target_confidence: float = 0.0
        self.max_confidence_predicted_class: Optional[str] = None
        self.max_confidence_prediction_correct: Optional[bool] = None
        self.max_confidence_prediction_step: Optional[int] = None
        self.last_predicted_class: Optional[str] = None
        self.last_predicted_confidence: float = 0.0
        self.last_prediction_correct: Optional[bool] = None
        self.final_predicted_class: Optional[str] = None
        self.final_predicted_confidence: float = 0.0
        self.final_prediction_correct: Optional[bool] = None
        self.detection_history: List[Dict[str, Any]] = []
        self.prediction_trace: List[Dict[str, Any]] = []
        self.evaluation_target_name: Optional[str] = None
        self._initial_confidences: Dict[str, float] = {}
        self._episode_start_time: float = perf_counter()

        self.visualize_gates: bool = False
        self.visualize_bounds: bool = False,
        self.visualize_agent_zone = False

        self.key_to_action: Dict[int, int] = {
            KeyActions.LEFT.value: Actions.LEFT.value,
            KeyActions.RIGHT.value: Actions.RIGHT.value,
            KeyActions.FORWARD.value: Actions.FORWARD.value,
            KeyActions.BACKWARD.value: Actions.BACKWARD.value,
            KeyActions.UP.value: Actions.UP.value,
            KeyActions.DOWN.value: Actions.DOWN.value,
            KeyActions.DETECT.value: Actions.DETECT.value
        }

        # -------- PARAMETERS FOR HEIGHT PRESERVATION AND CONFIDENCE--------------------
        self.z_plane: float = 6.687
        #self.z_min_offset: float = 0.563  # 7.25 - 6.687
        self.z_min_offset: float = 0.463  # 7.15 - 6.687
        #self.z_max_offset: float = 1.063  # 7.75 - 6.687
        self.z_max_offset: float = 1.163  # 7.85 - 6.687
        self.mean_z_offset: float = (self.z_min_offset + self.z_max_offset) / 2

        self.z_min_threshold: float = self.z_plane + self.z_min_offset
        self.z_max_threshold: float = self.z_plane + self.z_max_offset

        # ---------------- Camera --------------------------------------------------------------
        self.camera_node: Any = self.supervisor.getFromDef('camera')
        self.camera: Any = self.supervisor.getDevice('camera')
        self.camera.enable(self.timestep)
        self.supervisor.keyboard.enable(self.timestep)

        self.previous_robot_pos: Optional[np.ndarray] = None
        self.returning_home: bool = False

        # -------- THRESHOLDS ------------------------
        self.target_match_threshold: float = 0.03
        self.screw_success_threshold: float = float(verification_threshold)

        self.camera_intrinsics: Tuple[int, int, float, float, float, float, float, float, np.ndarray] = self._compute_camera_intrinsics()
        self.workspace_bounds: List[float] = [4.45, 5.60, 4.95, 5.85, 6.85, 9]
        #self.workspace_bounds: List[float] = [0.45, 10.60, 0.95, 10.85, 2.85, 15]

        self.observation_space: spaces.Dict = spaces.Dict({
            "image": self.image_space,
            "confirmed_screws": spaces.Box(low=0, high=1, shape=(self.num_screws,), dtype=np.float32),
            "target_vector": spaces.Box(low=0, high=1, shape=(self.num_screws,), dtype=np.float32)
        })

    def _log(self, *args: Any, **kwargs: Any) -> None:
        if self.verbose:
            print(*args, **kwargs)

    def set_robot_position(self, pos: List[float] | np.ndarray) -> None:
        self.robot.getField('translation').setSFVec3f(pos)

    def get_robot_position(self) -> List[float]:
        return self.robot.getField('translation').getSFVec3f()

    def get_camera_pose(self) -> np.ndarray:
        return np.array(self.camera_node.getPose()).reshape(4, 4)

    def _compute_camera_intrinsics(self) -> Tuple[int, int, float, float, float, float, float, float, np.ndarray]:

        width = self.camera.getWidth()
        height = self.camera.getHeight()
        fov_x = self.camera.getFov()
        fov_y = 2 * np.arctan(np.tan(fov_x * 0.5) * (height / width))

        cx = width / 2.0
        cy = height / 2.0
        fx = cx / np.tan(fov_x / 2.0)
        fy = cy / np.tan(fov_y / 2.0)

        intrinsic_matrix = np.array([[fx, 0, cx],
                                     [0, fy, cy],
                                     [0, 0,  1]])

        return width, height, fov_x, fov_y, fx, fy, cx, cy, intrinsic_matrix

    def ray_cast(self, u: float, v: float) -> np.ndarray:

        width, height, fov_x, fov_y, fx, fy, cx, cy, K = self.camera_intrinsics

        pose = np.array(self.get_camera_pose())
        camera_origin = np.array(pose[:3, 3])
        camera_rotation_matrix = np.array(pose[:3, :3])

        # Pixel u,v to camera coord
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        r_cam = np.array([-1, x_cam, y_cam])
        r_cam_direction = r_cam / np.linalg.norm(r_cam) # ray direction in camera coordinates

        r_world_direction = camera_rotation_matrix @ r_cam_direction
        distance = (self.z_plane - camera_origin[2]) / r_world_direction[2]

        intersection_point = camera_origin + distance * r_world_direction

        return intersection_point

    def run_yolo_detection(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Run YOLO detection on current camera image"""

        inference_started_at = perf_counter()
        try:
            predict_kwargs: Dict[str, Any] = {
                "source": image,
                "show": False,
                #"conf": 0.6,
                "show_labels": False,
                "save": False,
                "verbose": False,
            }
            if self._device is not None:
                predict_kwargs["device"] = self._device
            results = self.model.predict(**predict_kwargs)
            # results = self.model.track(**predict_kwargs)

        finally:
            # Count attempted detector calls as computational work, including failed calls.
            inference_time_s = perf_counter() - inference_started_at
            self.detector_call_count += 1
            self.last_detector_inference_time_s = inference_time_s
            self.detector_inference_time_s += inference_time_s
        # results = self.model.track(source=observation, show=False, show_labels=False, save=False, verbose=False)
        detections = []
        img_with_boxes = image.copy()
        for result in results:
            if result.boxes is not None:
                img_with_boxes = draw_boxes_on_image(result.orig_img, result.boxes, self.model.names)
                for box in result.boxes:
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]

                    #if class_name == 'holder':

                    # Get bounding box center
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_center_x = (x1 + x2) / 2
                    box_center_y = (y1 + y2) / 2

                    intersection_point = self.ray_cast(box_center_x, box_center_y)

                    detection = {
                        'class': class_name,
                        'confidence': conf,
                        # 'matched_index': matched_index,
                        'intersection_point': intersection_point,
                        'bbox_center': [box_center_x, box_center_y]
                    }
                    detections.append(detection)

        return detections, img_with_boxes

    def _visualize(self, image: Optional[np.ndarray]) -> None:
        if not self.visualize_detections:
            return

        if image is None:
            frame = self._get_camera_frame()
        else:
            frame = image

        display_obs = cv2.resize(frame, (600, 400))
        cv2.imshow('Training View', display_obs)
        # save_img_folder = Path(f"C:/Users/panosvrachnos/Documents/thesis_env/logs/actipercept_comparison/vis")
        # save_img_folder.mkdir(parents=True, exist_ok=True)
        # path = save_img_folder / f"frame_{self.step_count}.png"
        # cv2.imwrite(path, display_obs)
        cv2.waitKey(1)

    def get_action_id(self, action: Any) -> Optional[int]:
        if self.control_mode == 'keyboard':
            key = self.supervisor.keyboard.getKey()
            return self.key_to_action.get(key)
        else:
            if action is None:
                return None
            try:
                return int(action)
            except (ValueError, TypeError):
                return None

    @staticmethod
    def _action_name(action_id: Optional[int]) -> str:
        if action_id is None:
            return "NONE"
        try:
            return Actions(action_id).name
        except ValueError:
            return f"UNKNOWN({action_id})"

    def _get_camera_frame(self) -> np.ndarray:
        camera_frame = self.camera.getImage()
        width, height, _, _, _, _, _, _, _ = self.camera_intrinsics

        # resize_shape = (int(width / self.scale_factor), int(height / self.scale_factor))

        if camera_frame:
            try:
                frame = np.frombuffer(camera_frame, np.uint8).reshape((height, width, 4))
                frame = frame[:, :, :3]
                frame = frame.astype(np.uint8).copy()
                # frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_AREA)
            except Exception as e:
                self._log(f'Error processing camera frame {e}')
                frame = np.zeros(self.camera_shape, dtype=np.uint8)
        else:
            self._log("Warning: No image data received from camera!")
            frame = np.zeros(self.camera_shape, dtype=np.uint8)

        return frame

    def get_observation(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        # 1. Build the confirmed array
        confirmed_screws = np.array([self.screws[name].confirmed for name in self.target_names], dtype=np.float32)
        conf_vector = np.array([self.screws[name].max_confidence for name in self.target_names], dtype=np.float32)

        if self.current_target_name is not None:
            one_hot_vector = self.screws[self.current_target_name].one_hot
        else:
            one_hot_vector = np.zeros(self.num_screws, dtype=np.float32)

        if self.verbose:
            print(f"{conf_vector.round(3)} | {one_hot_vector} | {confirmed_screws.astype(bool)}")

        return {
            "image": image,
            "confirmed_screws": confirmed_screws,
            "target_vector": one_hot_vector
        }

    def get_target_detection(
            self,
            detections: List[Dict[str, Any]],
            target_point_above_screw: np.ndarray,
            required_class: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the best target-associated screw/noscrew prediction.

        Geometry associates predictions with the requested target. ``required_class`` is
        retained for the legacy training path; evaluation deliberately leaves it unset so
        that a wrong class can cross the operational verification threshold and be scored
        as a false verification afterwards.
        """

        target_point_xy = np.asarray(target_point_above_screw[:2], dtype=np.float32)
        best_detection: Optional[Dict[str, Any]] = None
        best_confidence = -1.0

        for det in detections:
            class_name = str(det['class'])
            if class_name not in VALID_TARGET_CLASSES:
                continue
            if required_class is not None and class_name != required_class:
                continue

            detection_point = det.get('intersection_point')
            if detection_point is None:
                continue

            detection_point_xy = np.asarray(detection_point[:2], dtype=np.float32)

            # Check geometric alignment
            delta_xy = np.abs(detection_point_xy - target_point_xy)
            if delta_xy[0] > self.target_match_threshold or delta_xy[1] > self.target_match_threshold:
                continue

            # If we are here, this detection is geometrically associated with our target.
            confidence = float(det['confidence'])
            if confidence > best_confidence:
                best_confidence = confidence
                best_detection = det
                self._log(f'\nTarget-associated detection --> #Detection Point: {detection_point_xy} |'
                          f' Target Point: {target_point_xy} |'
                          f' Delta: {delta_xy} |'
                          f' Confidence: {confidence}\n'
                          f' Predicted class: {class_name} \n')

        return best_detection

    def get_conf_of_correct_detection(
            self,
            detections: List[Dict[str, Any]],
            target_point_above_screw: np.ndarray,
            gt_class: str
    ) -> float:
        """Return legacy ground-truth-filtered confidence for training compatibility."""

        detection = self.get_target_detection(
            detections=detections,
            target_point_above_screw=target_point_above_screw,
            required_class=gt_class,
        )
        return float(detection['confidence']) if detection is not None else 0.0

    def set_random_light_luminosity(self, min_lum: float, max_lum: float) -> None:
        random_luminosity = round(float(self.np_random.uniform(min_lum, max_lum)), 1)
        self.set_light_luminosity(random_luminosity)

    def set_light_luminosity(self, luminosity: float) -> None:
        bg_light_node = self.supervisor.getFromDef("light")
        if bg_light_node is None:
            self._log("Warning: Could not find node with DEF 'light'. Lighting was not changed.")
            return

        luminosity_field = bg_light_node.getField("luminosity")
        if luminosity_field is None:
            self._log("Warning: 'luminosity' field not found on light.")
            return

        self.current_luminosity = float(luminosity)
        luminosity_field.setSFFloat(self.current_luminosity)

    def get_next_target(self) -> Optional[str]:
        for name in self.execution_sequence:
            if not self.screws[name].confirmed:
                return name
        return None

    def get_reward(
            self,
            current_robot_pos: List[float],
            previous_robot_pos: np.ndarray,
            current_confidence: float,
            active_screw: Optional[Screw],
            did_detect: bool,
            detection_correct: Optional[bool] = True
    ) -> Tuple[float, float, bool, bool, bool]:

        reward = 0.0
        all_done = False
        newly_confirmed = False
        if active_screw is None:
            return 0.0, 0.0, True, False, False

        current_target_point_above_screw = np.array([
            active_screw.point[0],
            active_screw.point[1],
            active_screw.point[2] + self.mean_z_offset
        ])

        # DISTANCE REWARD
        previous_distance = np.linalg.norm(current_target_point_above_screw - previous_robot_pos)
        current_distance = np.linalg.norm(current_target_point_above_screw - current_robot_pos)
        distance_reward = 10.0 * (previous_distance - current_distance)
        reward += distance_reward

        # CHECK ZONES
        is_in_z_zone = (self.z_min_threshold <= current_robot_pos[2] <= self.z_max_threshold)
        xy_dist = np.linalg.norm(current_target_point_above_screw[:2] - current_robot_pos[:2])
        is_in_xy_zone = xy_dist < 0.2
        valid_zone = is_in_z_zone and is_in_xy_zone

        # CONFIDENCE & SUCCESS REWARD
        if did_detect and not active_screw.confirmed:

            # We only update when the current_conf surpass the max (this is done inside the Screw.update_confidence())
            # Training keeps the historical spatial gate. During evaluation, geometric
            # target association is the operational gate, matching the paper-style
            # threshold criterion even if the policy is outside the reward-shaping zone.
            if valid_zone or self.mode == 'eval':
                if current_confidence > active_screw.max_confidence:
                    delta_conf = current_confidence - active_screw.max_confidence
                    confidence_reward = 5.0 * delta_conf
                    reward += confidence_reward
                    active_screw.max_confidence = current_confidence

                if current_confidence >= self.screw_success_threshold:
                    active_screw.confirmed = True
                    newly_confirmed = True
                    # A false acceptance advances the sequence like a correct one,
                    # but must never receive the explicit success bonus.
                    if detection_correct is not False:
                        reward += 20.0 + (10 * current_confidence)

                    confirmed_count = sum([s.confirmed for s in self.screws.values()])
                    self._log(f"  --> [{active_screw.name}] CONFIRMED (conf={current_confidence:.3f}) "
                              f"| {confirmed_count}/{self.num_screws} done")

            if all([s.confirmed for s in self.screws.values()]):
                if detection_correct is not False:
                    reward += 50
                all_done = True
                self._log("  --> ALL SCREWS CONFIRMED. Episode complete.")

        return reward, xy_dist, all_done, newly_confirmed, valid_zone

    

    def _get_instrumentation_info(self) -> Dict[str, Any]:
        """Return JSON-serializable cumulative episode cost/timing metrics."""

        elapsed_wall_time_s = max(0.0, perf_counter() - self._episode_start_time)
        # Return-home is deliberately excluded from processing FPS so that the policy's
        # steps-to-verification throughput can be compared independently. The raw total
        # and return-home duration are both exposed for end-to-end comparisons.
        active_wall_time_s = max(0.0, elapsed_wall_time_s - self.return_home_time_s)
        processing_fps = (
            float(self.step_count / active_wall_time_s)
            if self.step_count > 0 and active_wall_time_s > 0.0
            else 0.0
        )
        environment_fps = (
            float(self.step_count / self.step_processing_time_s)
            if self.step_count > 0 and self.step_processing_time_s > 0.0
            else 0.0
        )
        detector_fps = (
            float(self.detector_call_count / self.detector_inference_time_s)
            if self.detector_call_count > 0 and self.detector_inference_time_s > 0.0
            else 0.0
        )
        simulated_time_s = float(self.simulator_step_count * self.timestep / 1000.0)
        total_simulator_step_count = self.simulator_step_count + self.return_home_steps
        raw_confidence_gain = self.max_target_confidence - self.initial_target_confidence
        correct_only_steps = (
            self.verification_step
            if self.correct_verification_count > 0 and self.false_verification_count == 0
            else None
        )

        return {
            "env_step_count": self.step_count,
            "episode_step_count": self.step_count,
            "steps_to_verification": self.verification_step,
            "steps_to_decision": self.verification_step,
            "steps_to_correct_verification": correct_only_steps,
            "detection_action_count": self.detection_action_count,
            "detect_action_count": self.detection_action_count,
            "detector_call_count": self.detector_call_count,
            "detector_calls": self.detector_call_count,
            "actual_detector_call_count": self.detector_call_count,
            "motion_action_count": self.motion_action_count,
            "movement_action_count": self.motion_action_count,
            "no_op_action_count": self.no_op_action_count,
            "path_length": float(self.motion_path_length),
            "motion_path_length": float(self.motion_path_length),
            "return_home_steps": self.return_home_steps,
            "return_home_motion_count": self.return_home_steps,
            "return_home_time_s": float(self.return_home_time_s),
            "return_home_path_length": float(self.return_home_path_length),
            "return_home_reached": self.return_home_reached,
            "total_path_length": float(self.motion_path_length + self.return_home_path_length),
            "simulator_step_count": self.simulator_step_count,
            "return_home_simulator_step_count": self.return_home_steps,
            "total_simulator_step_count": total_simulator_step_count,
            "timestep_ms": int(self.timestep),
            "simulated_time_s": simulated_time_s,
            "active_simulated_time_s": simulated_time_s,
            "total_simulated_time_s": float(total_simulator_step_count * self.timestep / 1000.0),
            "last_step_processing_time_s": float(self.last_step_processing_time_s),
            "step_processing_time_s": float(self.step_processing_time_s),
            "last_detector_inference_time_s": float(self.last_detector_inference_time_s),
            "detector_inference_time_s": float(self.detector_inference_time_s),
            "elapsed_wall_time_s": float(elapsed_wall_time_s),
            "total_elapsed_wall_time_s": float(elapsed_wall_time_s),
            "active_wall_time_s": float(active_wall_time_s),
            "processing_fps": processing_fps,
            "fps": processing_fps,
            "environment_fps": environment_fps,
            "detector_fps": detector_fps,
            "initial_target_confidence": float(self.initial_target_confidence),
            "max_target_confidence": float(self.max_target_confidence),
            "best_target_confidence": float(self.max_target_confidence),
            "last_target_confidence": float(self.last_predicted_confidence),
            "final_target_confidence": float(self.last_predicted_confidence),
            "raw_confidence_gain": float(raw_confidence_gain),
            "positive_confidence_gain": float(max(0.0, raw_confidence_gain)),
            "last_confidence_gain": float(
                self.last_predicted_confidence - self.initial_target_confidence
            ),
            "confidence_gain": float(max(0.0, raw_confidence_gain)),
            "max_confidence_predicted_class": self.max_confidence_predicted_class,
            "best_prediction_class": self.max_confidence_predicted_class,
            "best_prediction_confidence": float(self.max_target_confidence),
            "best_prediction_correct": self.max_confidence_prediction_correct,
            "best_prediction_step": self.max_confidence_prediction_step,
            "last_predicted_class": self.last_predicted_class,
            "last_prediction_correct": self.last_prediction_correct,
            "last_prediction_class": self.last_predicted_class,
            "last_prediction_confidence": float(self.last_predicted_confidence),
            "final_predicted_class": self.final_predicted_class,
            "final_prediction_correct": self.final_prediction_correct,
            "accepted_pred_class": self.final_predicted_class,
            "accepted_pred_confidence": float(self.final_predicted_confidence),
            "accepted_step": self.verification_step,
            "accepted_prediction_class": self.final_predicted_class,
            "accepted_prediction_confidence": float(self.final_predicted_confidence),
            "accepted_prediction_correct": self.final_prediction_correct,
            "accepted_prediction_step": self.verification_step,
            "correct_verification_count": self.correct_verification_count,
            "false_verification_count": self.false_verification_count,
        }

    def step(self, action: Any) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        step_started_at = perf_counter()
        self.step_count += 1

        # 1. EXECUTE ACTION BASED ON CONTROL MODE
        action_id = self.get_action_id(action)
        did_detect = (action_id == Actions.DETECT.value)
        motion_action_values = {
            Actions.LEFT.value,
            Actions.RIGHT.value,
            Actions.FORWARD.value,
            Actions.BACKWARD.value,
            Actions.UP.value,
            Actions.DOWN.value,
        }

        if did_detect:
            self.detection_action_count += 1
        elif action_id in motion_action_values:
            self.motion_action_count += 1
        else:
            self.no_op_action_count += 1

        robot_pos_before_action = np.asarray(self.get_robot_position(), dtype=np.float64)
        if action_id is not None:
            self.execute_action(action_id)
        self.supervisor.step(self.timestep)
        self.simulator_step_count += 1

        # GET CURRENT ROBOT POSITION
        current_robot_pos = np.asarray(self.get_robot_position(), dtype=np.float64)
        if self.visualize_agent_zone:
            update_agent_zone(self)
        if action_id in motion_action_values:
            self.motion_path_length += float(np.linalg.norm(current_robot_pos - robot_pos_before_action))
        camera_frame = self._get_camera_frame()

        # Initialize previous position on first step to avoid spikes
        if self.previous_robot_pos is None:
            self.previous_robot_pos = current_robot_pos

        current_target = self.screws[self.current_target_name] if self.current_target_name else None
        detection_target_name = current_target.name if current_target is not None else None
        current_max_confidence = current_target.max_confidence if current_target is not None else 0.0
        current_confidence = 0.0
        predicted_class: Optional[str] = None
        predicted_confidence: Optional[float] = None
        prediction_correct: Optional[bool] = None
        matched_detection_found = False
        detector_called = False
        detector_time_before_call = self.detector_inference_time_s

        # PERCEPTION (Camera + YOLO)
        if did_detect and current_target is not None:
            if current_target.gt_class is None:
                raise RuntimeError(f"Ground-truth class not loaded for target {current_target.name}")

            detections, img_with_boxes = self.run_yolo_detection(camera_frame)
            detector_called = True
            matched_detection = self.get_target_detection(
                detections=detections,
                target_point_above_screw=current_target.point,
            )

            if matched_detection is not None:
                matched_detection_found = True
                predicted_class = str(matched_detection['class'])
                predicted_confidence = float(matched_detection['confidence'])
                prediction_correct = predicted_class == current_target.gt_class

                if predicted_confidence > self.max_target_confidence:
                    self.max_target_confidence = predicted_confidence
                    self.max_confidence_predicted_class = predicted_class
                    self.max_confidence_prediction_correct = prediction_correct
                    self.max_confidence_prediction_step = self.step_count

            self.last_predicted_class = predicted_class
            self.last_predicted_confidence = float(predicted_confidence or 0.0)
            self.last_prediction_correct = prediction_correct

            if self.mode == 'eval':
                # Evaluation uses the detector prediction itself. Ground truth is consulted
                # only below to score correctness, never to decide threshold acceptance.
                current_confidence = predicted_confidence if predicted_confidence is not None else 0.0
            elif prediction_correct:
                # Fast path equivalent to the legacy correct-class filter.
                current_confidence = predicted_confidence if predicted_confidence is not None else 0.0
            else:
                # If the highest target-associated prediction is wrong, a lower-confidence
                # correct prediction must still retain the historical training semantics.
                current_confidence = self.get_conf_of_correct_detection(
                    detections,
                    current_target.point,
                    current_target.gt_class,
                )
            self._visualize(image=img_with_boxes)
        else:
            self._visualize(image=camera_frame)

        reward_detection_correct = prediction_correct if self.mode == 'eval' else True
        reward, xy_dist, all_done, newly_confirmed, valid_zone = self.get_reward(
            current_robot_pos=current_robot_pos,
            previous_robot_pos=self.previous_robot_pos,
            current_confidence=current_confidence,
            active_screw=current_target,
            did_detect=did_detect,
            detection_correct=reward_detection_correct,
        )

        detection_threshold_crossed = bool(
            matched_detection_found
            and predicted_confidence is not None
            and predicted_confidence >= self.screw_success_threshold
        )

        if newly_confirmed and current_target is not None:
            current_target.verified_class = predicted_class
            current_target.verification_confidence = float(predicted_confidence or 0.0)
            current_target.verification_correct = bool(prediction_correct)
            self.verification_step = self.step_count if self.verification_step is None else self.verification_step
            self.final_predicted_class = predicted_class
            self.final_predicted_confidence = float(predicted_confidence or 0.0)
            self.final_prediction_correct = bool(prediction_correct)
            if prediction_correct:
                self.correct_verification_count += 1
            else:
                self.false_verification_count += 1

        if did_detect:
            detector_call_time_s = self.detector_inference_time_s - detector_time_before_call
            detection_event = {
                "step": self.step_count,
                "target_name": detection_target_name,
                "ground_truth_class": current_target.gt_class if current_target is not None else None,
                "detector_called": detector_called,
                "matched_detection_found": matched_detection_found,
                "predicted_class": predicted_class,
                "predicted_confidence": predicted_confidence,
                "prediction_correct": prediction_correct,
                "threshold": float(self.screw_success_threshold),
                "threshold_crossed": detection_threshold_crossed,
                "verification_accepted": newly_confirmed,
                "valid_reward_zone": valid_zone,
                "detector_inference_time_s": float(detector_call_time_s),
                "camera_position": [float(value) for value in current_robot_pos],
            }
            self.detection_history.append(detection_event)
            if detector_called:
                self.prediction_trace.append({
                    "step": self.step_count,
                    "target": detection_target_name,
                    "predicted_class": predicted_class,
                    "confidence": predicted_confidence,
                    "correct": prediction_correct,
                    "threshold_crossed": detection_threshold_crossed,
                    "inference_time_s": float(detector_call_time_s),
                    "camera_position": [float(value) for value in current_robot_pos],
                })

        if newly_confirmed:
            self.current_target_name = self.get_next_target()

            if self.current_target_name is not None:
                self._log(f"  --> Next target selected: {self.current_target_name}")

                if self.visualize_gates:
                    draw_valid_zone(
                        self, self.screws[self.current_target_name]
                    )
            else:
                # No more targets: remove the visualization
                debug_node = self.supervisor.getFromDef("DEBUG_VALID_ZONE")
                if debug_node is not None:
                    debug_node.remove()
        #---------------------------------------------------------------------------

        # # Terminate if max steps reached (Time limits)
        truncated = False
        terminated = False
        out_of_bounds = False

        if self.max_steps is not None and self.step_count >= self.max_steps and not all_done:
            self._log(f'----- Max steps [{self.max_steps}] completed. Episode finished -----')
            truncated = True

        # Terminate if out of bounds
        if self._is_out_of_bounds():
            reward = -1
            terminated = True
            out_of_bounds = True
            self._log(f'----- Out of Bounds Reward: {reward} -----')

        if all_done:
            terminated = True

        # Algorithm 1 returns to the home pose after every terminal reason, not
        # only after a successful verification. Keep this optional for training
        # and report its time/path separately from the decision phase.

        # if self.scripted_return_on_success and (all_done or truncated or out_of_bounds):
        #     self._scripted_return_home()

        self.total_reward += reward
        delta_conf = max(0.0, current_confidence - current_max_confidence) if did_detect else 0.0
        target_gt_class = current_target.gt_class if current_target is not None else None

        # status = "RUN"
        status = "-"
        if newly_confirmed and prediction_correct is False:
            status = "FALSE VERIFICATION"
        elif all_done:
            status = "ALL DONE"
        elif newly_confirmed:
            status = "CONFIRMED"
        elif out_of_bounds:
            status = "OOB"
        elif truncated:
            status = "TIMEOUT"

        if self.verbose:
            print(
                f"Step {self.step_count:4d} | "
                f"Target: {self.current_target_name} |"
                f"Class: {target_gt_class} |"
                f"Pred: {predicted_class} |"
                f"Status: {status} |"
                f"Detect: {int(did_detect)} | "
                f"StepR: {reward:7.3f} | "
                f"EpR: {self.total_reward:8.3f} | "
                f"Dist_xy: {xy_dist:6.3f} | "
                f"Robot_Z: {current_robot_pos[2]:6.3f} | "
                f"Conf: {current_confidence:6.3f} | "
                f"MaxConf: {current_max_confidence:6.3f} | "
                f"dConf: {delta_conf:7.3f} | "
            )

        if newly_confirmed:
            verification_outcome = (
                "correct_verification" if prediction_correct else "false_verification"
            )
        elif out_of_bounds:
            verification_outcome = "out_of_bounds"
        elif truncated:
            verification_outcome = "timeout"
        else:
            verification_outcome = "pending"

        self.last_step_processing_time_s = perf_counter() - step_started_at
        self.step_processing_time_s += self.last_step_processing_time_s
        verified_success = bool(
            all_done
            and self.correct_verification_count > 0
            and self.false_verification_count == 0
        )

        info = {
            "step_count": self.step_count,
            "confirmed_screws": [bool(self.screws[name].confirmed) for name in self.target_names],
            "confidence_vector": [float(self.screws[name].max_confidence) for name in self.target_names],
            "current_target": self.current_target_name,
            "detection_target": detection_target_name,
            "evaluation_target_name": self.evaluation_target_name,
            "current_laptop_name": self.current_laptop_name,
            "current_laptop_version_id": self.current_laptop_version_id,
            "all_done": all_done,
            "action_id": action_id,
            "action_name": self._action_name(action_id),
            "did_detect": did_detect,
            "detector_called": detector_called,
            "matched_detection_found": matched_detection_found,
            "predicted_class": predicted_class,
            "predicted_confidence": predicted_confidence,
            "ground_truth_class": target_gt_class,
            "target_gt_class": target_gt_class,
            "prediction_correct": prediction_correct,
            "detection_threshold_crossed": detection_threshold_crossed,
            "threshold_crossed": detection_threshold_crossed,
            "verification_threshold": float(self.screw_success_threshold),
            "verification_accepted": newly_confirmed,
            "newly_confirmed": newly_confirmed,
            "correct_verification": bool(newly_confirmed and prediction_correct),
            "new_false_verification": bool(newly_confirmed and prediction_correct is False),
            "false_verification": self.false_verification_count > 0,
            "verified_success": verified_success,
            "verification_outcome": verification_outcome,
            "termination_reason": verification_outcome if (terminated or truncated) else None,
            "out_of_bounds": out_of_bounds,
            "timeout": truncated,
            "valid_reward_zone": valid_zone,
            "current_detection_confidence": float(current_confidence),
            "detection_confidence_gain": float(delta_conf),
            "detection_history": list(self.detection_history),
            "prediction_trace": list(self.prediction_trace),
        }
        info.update(self._get_instrumentation_info())

        self.previous_robot_pos = current_robot_pos
        observation = self.get_observation(image=camera_frame)

        return observation, reward, terminated, truncated, info

    def start_evaluation_timing(self) -> None:
        """Rebase wall timing after evaluator-only setup or baseline probing."""
        self._episode_start_time = perf_counter()

    def reset(
            self,
            seed: Optional[int]=None,
            options: Optional[Dict[str, Any]]=None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:

        super().reset(seed=seed)
        reset_options = options or {}

        # Gymnasium seeds self.np_random. These legacy RNGs are also used below,
        # so seed them as well to make an episode reproducible from one seed.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.supervisor.simulationReset()

        # 2. FLUSH THE RESET immediately so it executes before we change anything
        self.supervisor.step(self.timestep)

        if self.visualize_bounds:
            draw_workspace_bounds(self)
        if self.visualize_agent_zone:
            draw_agent_zone(self)

        # 3. Safely re-fetch nodes/devices (Webots invalidates handles after reset).
        self.root_node = self.supervisor.getRoot()
        self.children_field = self.root_node.getField('children')
        self.robot = self.supervisor.getSelf()
        self.laptop_transform_node = self.supervisor.getFromDef("LAPTOP_TRANSFORM")
        self.laptop_cad_node = self.supervisor.getFromDef("LAPTOP_CAD")
        self.camera_node = self.supervisor.getFromDef('camera')
        self.camera = self.supervisor.getDevice('camera')
        self.camera.enable(self.timestep)
        self.supervisor.keyboard.enable(self.timestep)
        if self.laptop_transform_node is None or self.laptop_cad_node is None:
            raise RuntimeError(
                "Could not reacquire DEF LAPTOP_TRANSFORM or LAPTOP_CAD after reset"
            )
        self.laptop_model_url = self.laptop_cad_node.getField('url')

        # Load either a requested laptop/version for evaluation or a random one for training.
        requested_laptop = reset_options.get("laptop_name")
        requested_version = reset_options.get("laptop_version_id")
        if requested_laptop is not None and requested_version is not None:
            laptop_name, laptop_version_id = requested_laptop, requested_version
        else:
            laptop_name, laptop_version_id = self.select_laptop_version(self.finetuning and self.mode == "train")
            laptop_name = requested_laptop or laptop_name
            laptop_version_id = requested_version or laptop_version_id

        requested_scale = reset_options.get("scale")
        self.load_laptop_version(
            laptop_name=laptop_name,
            laptop_version_id=laptop_version_id,
            scale=requested_scale,
        )

        self.set_robot_position(self.initial_pos)
        for _ in range(self.frame_skip):
            self.supervisor.step(self.timestep)

        # Reset Episode Counters & State
        self.step_count = 0
        self.total_reward = 0.0
        self.returning_home = False
        self.previous_robot_pos = np.array(self.initial_pos, dtype=np.float32)
        self.detection_action_count = 0
        self.detector_call_count = 0
        self.motion_action_count = 0
        self.no_op_action_count = 0
        self.motion_path_length = 0.0
        self.detector_inference_time_s = 0.0
        self.last_detector_inference_time_s = 0.0
        self.step_processing_time_s = 0.0
        self.last_step_processing_time_s = 0.0
        self.simulator_step_count = 0
        self.return_home_steps = 0
        self.return_home_time_s = 0.0
        self.return_home_path_length = 0.0
        self.return_home_reached = False
        self.correct_verification_count = 0
        self.false_verification_count = 0
        self.verification_step = None
        self.initial_target_confidence = 0.0
        self.max_target_confidence = 0.0
        self.max_confidence_predicted_class = None
        self.max_confidence_prediction_correct = None
        self.max_confidence_prediction_step = None
        self.last_predicted_class = None
        self.last_predicted_confidence = 0.0
        self.last_prediction_correct = None
        self.final_predicted_class = None
        self.final_predicted_confidence = 0.0
        self.final_prediction_correct = None
        self.detection_history = []
        self.prediction_trace = []

        requested_return_home = reset_options.get(
            "scripted_return_on_success",
            self._default_scripted_return_on_success,
        )
        if not isinstance(requested_return_home, bool):
            raise TypeError("options['scripted_return_on_success'] must be bool")
        self.scripted_return_on_success = requested_return_home

        requested_luminosity = reset_options.get("luminosity")
        if requested_luminosity is None:
            self.set_random_light_luminosity(min_lum=0.3, max_lum=3.0)
        else:
            self.set_light_luminosity(requested_luminosity)

        for screw in self.screws.values():
            screw.reset()

        requested_target = reset_options.get("evaluation_target_name")
        if requested_target is not None:
            requested_target = str(requested_target)
            if requested_target not in self.screws:
                raise ValueError(
                    f"Unknown evaluation_target_name {requested_target!r}; "
                    f"expected one of {self.target_names}"
                )
            self.evaluation_target_name = requested_target
            self.execution_sequence = [requested_target]
            for target_name, screw in self.screws.items():
                screw.confirmed = target_name != requested_target
                if screw.confirmed:
                    screw.max_confidence = float(np.random.uniform(low=0.89, high=0.91))
        else:
            self.evaluation_target_name = None
            self.execution_sequence = list(self.target_names)
            requested_order = reset_options.get("execution_sequence")
            if requested_order is not None:
                if (not isinstance(requested_order, (list, tuple))
                        or len(requested_order) != self.num_screws
                        or sorted(requested_order) != sorted(self.target_names)):
                    raise ValueError("execution_sequence must contain every target exactly once")
                self.execution_sequence = list(requested_order)
            elif self.mode == 'train':
                random.shuffle(self.execution_sequence)

        # Backwards-compatible positional mask. An explicit evaluation target takes
        # precedence because it is unambiguous and independent of sequence order.
        if requested_target is None and "initial_confirmed_screws" in reset_options:
            initial_confirmed_screws = np.asarray(
                reset_options["initial_confirmed_screws"],
                dtype=bool,
            )
            if initial_confirmed_screws.shape != (self.num_screws,):
                raise ValueError(
                    "initial_confirmed_screws must contain exactly "
                    f"{self.num_screws} boolean values"
                )
            for pos, target_name in enumerate(self.execution_sequence):
                screw = self.screws[target_name]
                screw.confirmed = bool(initial_confirmed_screws[pos])
                if screw.confirmed:
                    screw.max_confidence = float(np.random.uniform(low=0.89, high=0.91))

        self._log(f"execution sequence: {self.execution_sequence}")

        self.current_target_name = self.get_next_target()
        self._initial_confidences = {
            name: float(screw.max_confidence) for name, screw in self.screws.items()
        }
        if self.current_target_name is not None:
            if self.visualize_gates:
                draw_valid_zone(self, self.screws[self.current_target_name])
            self.initial_target_confidence = float(
                self.screws[self.current_target_name].max_confidence
            )
        if self.current_target_name:
            self._log(f"  --> First target selected: {self.current_target_name}")

        camera_frame = self._get_camera_frame()
        observation = self.get_observation(image=camera_frame)

        self._log(f"--- RESET | Mode: {self.mode} | Targets: {self.execution_sequence} ---")
        # Exclude Webots reset/model loading/settling from episode processing FPS.
        self._episode_start_time = perf_counter()

        info = {"current_laptop_name": self.current_laptop_name,
                "current_laptop_version_id": self.current_laptop_version_id,
                "current_scale": self.current_laptop_scale,
                "current_luminosity": self.current_luminosity,
                "execution_sequence": list(self.execution_sequence),
                "current_target": self.current_target_name,
                "evaluation_target_name": self.evaluation_target_name,
                "ground_truth_class": (
                    self.screws[self.current_target_name].gt_class
                    if self.current_target_name is not None else None
                ),
                "initial_confirmed_count": int(sum(s.confirmed for s in self.screws.values())),
                "verification_threshold": float(
                    getattr(self, "screw_success_threshold", 0.88)
                ),
                "scripted_return_on_success": self.scripted_return_on_success,
                "seed": seed,
                }

        return observation, info
    
    def select_laptop_version(self, finetuning: bool = False) -> Tuple[str, str]:

        laptops = self.laptop_database["laptops"]
        if finetuning:
            new_laptops = ["test_laptop6", "test_laptop7"]
            for name in new_laptops:
                if name not in laptops:
                    raise ValueError(f"Fine-tuning laptops missing from database: {name}")
            old_laptops = [name for name in laptops if name not in new_laptops]
            if not old_laptops:
                raise ValueError("Fine-tuning requires original training laptops too.")

            candidates = (new_laptops if random.random() < 0.8 else old_laptops)
        else:
            candidates = list(laptops.keys())

        
        laptop_name = random.choice(candidates)
        # Get the data for the specific laptop
        laptop_data = laptops[laptop_name]
        #version_objs
        available_versions = list(laptop_data["version_objs"].keys())
        # Select randomly one key from version_objs 
        laptop_version_id = random.choice(available_versions)

        return laptop_name, laptop_version_id

    def load_laptop_version(
            self,
            laptop_name: str,
            laptop_version_id: str,
            scale: Optional[float] = None
    ) -> None:
        laptop_data = self.laptop_database["laptops"][laptop_name]
        version_data = self.laptop_database["versions"][laptop_version_id]

        # Swap the 3D model URL
        obj_url = laptop_data["version_objs"][laptop_version_id]
        self.laptop_model_url.setMFString(0, obj_url)

        # 1. Apply the scale via our helper method and store the returned value
        chosen_scale = self._apply_scale(scale)

        # 2. Get the laptop's root translation dynamically from Webots (World Space)
        laptop_origin = np.array(self.laptop_transform_node.getField("translation").getSFVec3f(), dtype=np.float32)

        # 3. Use the World Space vector for mathematical coordinate scaling
        # World X = Width, World Y = Depth, World Z = Height/Thickness
        world_scale_vector = np.array([chosen_scale, chosen_scale, 1.0], dtype=np.float32)

        for screw_name in self.target_names:
            screw = self.screws[screw_name]
            
            # Get original absolute point from JSON (World Space)
            original_point = np.array(laptop_data["screw_points"][screw_name], dtype=np.float32)
            
            # Calculate offset, apply World scale, and convert back to absolute World Space
            world_offset = original_point - laptop_origin
            scaled_point = laptop_origin + (world_offset * world_scale_vector)
            
            # Update the screw object
            screw.point = scaled_point
            screw.one_hot = np.array(self.laptop_database["screws"][screw_name]["one_hot"], dtype=np.float32)
            screw.gt_class = version_data["classes"][screw_name]

        self.current_laptop_name = laptop_name
        self.current_laptop_version_id = laptop_version_id

    def _apply_scale(self, scale: Optional[float] = None) -> float:
        """
        Applies a requested scale or randomly selects one for ordinary resets.
        Returns the chosen scale so the math logic can use it.
        """
        scales = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
        chosen_scale = float(scale) if scale is not None else random.choice(scales)
        if chosen_scale <= 0:
            raise ValueError("scale must be greater than zero")
        self.current_laptop_scale = chosen_scale
        
        # Apply scale to the Webots Transform node (Local Space)
        # Local X(Red) = Width, Local Y(Green) = Thickness, Local Z(Blue) = Depth
        local_scale_vector = [chosen_scale, 1.0, chosen_scale]
        self.laptop_transform_node.getField("scale").setSFVec3f(local_scale_vector)
        
        return chosen_scale

    def _is_out_of_bounds(self) -> bool:
        pos = self.get_robot_position()
        min_x, max_x, min_y, max_y, min_z, max_z = self.workspace_bounds
        if not (min_x <= pos[0] <= max_x and
                min_y <= pos[1] <= max_y and
                min_z <= pos[2] <= max_z):

            return True
        return False

    # ----------------------------------------------------------------------------------
    def _scripted_return_home(self) -> None:
        return_started_at = perf_counter()
        self.returning_home = True
        tolerance = 0.01
        max_iterations = 1000

        target_pos = np.asarray(self.initial_pos, dtype=np.float32)
        iterations = 0

        try:
            while iterations < max_iterations:
                current_pos = np.asarray(self.get_robot_position(), dtype=np.float64)
                delta = target_pos - current_pos

                if abs(delta[0]) > tolerance:
                    action = Actions.RIGHT.value if delta[0] > 0 else Actions.LEFT.value
                elif abs(delta[1]) > tolerance:
                    action = Actions.FORWARD.value if delta[1] > 0 else Actions.BACKWARD.value
                elif abs(delta[2]) > tolerance:
                    action = Actions.UP.value if delta[2] > 0 else Actions.DOWN.value
                else:
                    self.return_home_reached = True
                    break

                position_before_move = current_pos.copy()
                self.execute_action(action)
                position_after_move = np.asarray(self.get_robot_position(), dtype=np.float64)
                self.return_home_path_length += float(
                    np.linalg.norm(position_after_move - position_before_move)
                )
                self.supervisor.step(self.timestep)
                self.return_home_steps += 1
                self._visualize(image=None)
                iterations += 1
            else:
                self.return_home_reached = False
                self._log("[RETURN HOME] Max iterations reached before reaching home position.")

            if self.return_home_reached:
                self._log('----- Agent returned home -----')
        finally:
            self.return_home_time_s += perf_counter() - return_started_at
            self.returning_home = False

    # ---------------------------------------------------------------------------------- 

    def execute_action(self, action: int) -> None:

        current_robot_pos = self.get_robot_position()
        new_robot_pos = current_robot_pos.copy()

        if action == Actions.LEFT.value:
            new_robot_pos[0] -= self.move
        elif action == Actions.RIGHT.value:
            new_robot_pos[0] += self.move
        elif action == Actions.FORWARD.value:
            new_robot_pos[1] += self.move
        elif action == Actions.BACKWARD.value:
            new_robot_pos[1] -= self.move
        elif action == Actions.UP.value:
            new_robot_pos[2] += self.move
        elif action == Actions.DOWN.value:
            new_robot_pos[2] -= self.move
        elif action == KeyActions.LEFT.value:
            new_robot_pos[0] -= self.move
        elif action == KeyActions.RIGHT.value:
            new_robot_pos[0] += self.move
        elif action == KeyActions.FORWARD.value:
            new_robot_pos[1] += self.move
        elif action == KeyActions.BACKWARD.value:
            new_robot_pos[1] -= self.move
        elif action == KeyActions.UP.value:
            new_robot_pos[2] += self.move
        elif action == KeyActions.DOWN.value:
            new_robot_pos[2] -= self.move

        if not np.array_equal(new_robot_pos, current_robot_pos):
            self.set_robot_position(new_robot_pos)
