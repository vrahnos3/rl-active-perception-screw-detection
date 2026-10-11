from controller import Supervisor
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
import numpy as np
import random
import cv2
import json
from abc import ABC
from ultralytics import YOLO
from test.utils import draw_boxes_on_image
from dataclasses import dataclass


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


@dataclass
class Screw:
    name: str
    point: np.ndarray
    one_hot: np.ndarray
    confirmed: bool = False
    max_confidence: float = 0.0

    def reset(self):
        self.confirmed = False
        self.max_confidence = 0.0


class ScrewDetectorEnv(gym.Env, ABC):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self,
                 yolo_path='C:/Users/panosvrachnos/Documents/thesis_env/pc_screw_detection.pt',
                 visualize_detections=None,
                 max_steps=None,
                 control_mode=None,
                 targets_dict=None,
                 mode=None,
                 ):
        super(ScrewDetectorEnv, self).__init__()

    # ---------- Checks for Arguments --------------------------------------------------------
        if not isinstance(visualize_detections, bool):
            raise TypeError('visualize_detections must be bool')
        if control_mode not in ('keyboard', 'policy') or control_mode is None:
            raise ValueError('control_mode must be "keyboard" or "policy"')
        if targets_dict is None:
            raise ValueError('Please provide a targets_dict')
        if mode not in ['train', 'eval'] or mode is None:
            raise ValueError("mode must be 'train' or 'eval'")

        self.visualize_detections = visualize_detections
        self.control_mode = control_mode
        self.targets_dict = targets_dict
        self.mode = mode
    # ----------------------------------------------------------------------------------------
        self.target_names = sorted(self.targets_dict.keys())
        self.num_screws = len(self.target_names)
        self.execution_sequence = list(self.target_names)
        self.screws = {}
        self.current_target_name = None

        for name in self.target_names:
            data = self.targets_dict[name]
            self.screws[name] = Screw(
                name=name,
                point=np.array(data['point'], dtype=np.float32),
                one_hot=np.array(data['one_hot'], dtype=np.float32)
            )

        # ------------- Get the Webots root node (needed for spawning objects) -----------------
        self.supervisor = Supervisor()
        self.root_node = self.supervisor.getRoot()
        self.children_field = self.root_node.getField('children')
        self.current_laptop_node = None

        self.robot = self.supervisor.getSelf()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.laptop_transform_node = self.supervisor.getFromDef("LAPTOP_TRANSFORM")
        self.laptop_cad_node = self.supervisor.getFromDef("LAPTOP_CAD")
        #self.laptop_model_node = self.supervisor.getFromDef("LAPTOP_MODEL")

        if self.laptop_transform_node is None or self.laptop_cad_node is None:
            raise ValueError("Could not find DEF LAPTOP_TRANSFORM or LAPTOP_CAD in webots tree")
        self.laptop_model_url = self.laptop_cad_node.getField("url")        #----------------------------------------------------------------------------------------

        self.camera_shape = (270, 480, 3)
        self.image_space = spaces.Box(low=0, high=255, shape=self.camera_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(len(Actions))
        self.model = YOLO(yolo_path).to('cuda')
        self.max_steps = max_steps
        self.scale_factor = 1
        self.step_count = 0
        self.initial_pos = [5., 5.37, 8]
        self.move = 0.01
        self.frame_skip = 10
        self.total_reward = 0.0

        self.key_to_action = {
            KeyActions.LEFT.value: Actions.LEFT.value,
            KeyActions.RIGHT.value: Actions.RIGHT.value,
            KeyActions.FORWARD.value: Actions.FORWARD.value,
            KeyActions.BACKWARD.value: Actions.BACKWARD.value,
            KeyActions.UP.value: Actions.UP.value,
            KeyActions.DOWN.value: Actions.DOWN.value,
            KeyActions.DETECT.value: Actions.DETECT.value
        }

        # -------- PARAMETERS FOR HEIGHT PRESERVATION AND CONFIDENCE--------------------
        self.z_plane = 6.687
        self.z_min_offset = 0.563  # 7.25 - 6.687
        self.z_max_offset = 1.063  # 7.75 - 6.687
        # self.z_max_offset = 1.313  # 8 - 6.687
        self.mean_z_offset = (self.z_min_offset + self.z_max_offset) / 2

        self.z_min_threshold = self.z_plane + self.z_min_offset
        self.z_max_threshold = self.z_plane + self.z_max_offset

        # ---------------- Camera --------------------------------------------------------------
        self.camera_node = self.supervisor.getFromDef('camera')
        self.camera = self.supervisor.getDevice('camera')
        self.camera.enable(self.timestep)
        self.supervisor.keyboard.enable(self.timestep)

        self.previous_robot_pos = None
        self.returning_home = False

        # -------- THRESHOLDS ------------------------
        self.target_match_threshold = 0.03
        self.screw_success_threshold = 0.89

        self.camera_intrinsics = self._compute_camera_intrinsics()
        self.workspace_bounds = [4.55, 5.50, 5.05, 5.68, 6.85, 9]

        self.observation_space = spaces.Dict({
            "image": self.image_space,
            "confirmed_screws": spaces.Box(low=0, high=1, shape=(self.num_screws,), dtype=np.float32),
            "target_vector": spaces.Box(low=0, high=1, shape=(self.num_screws,), dtype=np.float32)
        })

    def set_robot_position(self, pos):
        self.robot.getField('translation').setSFVec3f(pos)

    def get_robot_position(self):
        return self.robot.getField('translation').getSFVec3f()

    def get_camera_pose(self):
        return np.array(self.camera_node.getPose()).reshape(4, 4)

    def _compute_camera_intrinsics(self):

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

    def ray_cast(self, u, v):

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

    def run_yolo_detection(self, image):
        """Run YOLO detection on current camera image"""

        results = self.model.predict(source=image, show=False, show_labels=False, save=False, verbose=False, device='cuda')
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

    def _visualize(self, image):
        if not self.visualize_detections:
            return

        if image is None:
            frame = self._get_camera_frame()
        else:
            frame = image

        display_obs = cv2.resize(frame, (600, 400))
        cv2.imshow('Training View', display_obs)
        cv2.waitKey(1)

    def get_action_id(self, action):
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
    def _action_name(action_id):
        if action_id is None:
            return "NONE"
        try:
            return Actions(action_id).name
        except ValueError:
            return f"UNKNOWN({action_id})"

    def _get_camera_frame(self):
        camera_frame = self.camera.getImage()
        width, height, fov_x, fov_y, fx, fy, cx, cy, intrinsic_matrix = self.camera_intrinsics

        # resize_shape = (int(width / self.scale_factor), int(height / self.scale_factor))

        if camera_frame:
            try:
                frame = np.frombuffer(camera_frame, np.uint8).reshape((height, width, 4))
                frame = frame[:, :, :3]
                frame = frame.astype(np.uint8).copy()
                # frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(f'Error processing camera frame {e}')
                frame = np.zeros(self.camera_shape, dtype=np.uint8)
        else:
            print("Warning: No image data received from camera!")
            frame = np.zeros(self.camera_shape, dtype=np.uint8)

        return frame

    def get_observation(self, image):
        # 1. Build the confirmed array
        confirmed_screws = np.array([self.screws[name].confirmed for name in self.target_names], dtype=np.float32)
        conf_vector = np.array([self.screws[name].max_confidence for name in self.target_names], dtype=np.float32)

        if self.current_target_name is not None:
            one_hot_vector = self.screws[self.current_target_name].one_hot
        else:
            one_hot_vector = np.zeros(self.num_screws, dtype=np.float32)

        print(f"{conf_vector.round(3)} | {one_hot_vector} | {confirmed_screws.astype(bool)}")

        return {
            "image": image,
            "confirmed_screws": confirmed_screws,
            "target_vector": one_hot_vector
        }

    def get_conf_of_correct_detection(self, detections, target_point_above_screw):

        target_point_xy = np.asarray(target_point_above_screw[:2], dtype=np.float32)
        best_confidence = 0.0

        for det in detections:
            class_name = det['class']
            if class_name not in ('screw', 'noscrew'):
                continue

            detection_point = det.get('intersection_point')
            if detection_point is None:
                continue

            detection_point_xy = np.asarray(detection_point[:2], dtype=np.float32)

            # Check geometric alignment
            delta_xy = np.abs(detection_point_xy - target_point_xy)
            if delta_xy[0] > self.target_match_threshold or delta_xy[1] > self.target_match_threshold:
                continue

            # If we are here, this detection IS our target.
            # We take the one with the highest confidence or closest distance
            confidence = det['confidence']
            if best_confidence == 0.0 or confidence > best_confidence:
                best_confidence = confidence
                print(f'\nTarget Verified --> #Detection Point: {detection_point_xy} |'
                      f' Target Point: {target_point_xy} |'
                      f' Delta: {delta_xy} |'
                      f' Confidence: {confidence}\n')

        return best_confidence

    def set_random_light_luminosity(self, min_lum, max_lum ):
        bg_light_node = self.supervisor.getFromDef("light")
        if bg_light_node is None:
            print("Warning: Could not find node with DEF 'light'. Lighting not randomized.")
            return

        luminosity_field = bg_light_node.getField("luminosity")
        if luminosity_field is None:
            print("Warning: 'luminosity' field not found on light.")
            return

        random_luminosity = round(self.np_random.uniform(min_lum, max_lum), 1)
        luminosity_field.setSFFloat(float(random_luminosity))

    def get_next_target(self):
        for name in self.execution_sequence:
            if not self.screws[name].confirmed:
                return name
        return None

    def get_reward(self, current_robot_pos, previous_robot_pos, current_confidence, active_screw, did_detect):

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
        is_in_xy_zone = xy_dist < 0.1
        valid_zone = is_in_z_zone and is_in_xy_zone

        # CONFIDENCE & SUCCESS REWARD
        if did_detect and not active_screw.confirmed:

            # We only update when the current_conf surpass the max (this is done inside the Screw.update_confidence())
            if valid_zone:
                if current_confidence > active_screw.max_confidence:
                    delta_conf = current_confidence - active_screw.max_confidence
                    confidence_reward = 5.0 * delta_conf
                    reward += confidence_reward
                    active_screw.max_confidence = current_confidence

                if current_confidence >= self.screw_success_threshold:
                    active_screw.confirmed = True
                    newly_confirmed = True
                    reward += 20.0

                    confirmed_count = sum([s.confirmed for s in self.screws.values()])
                    print(f"  --> [{active_screw.name}] CONFIRMED (conf={current_confidence:.3f}) "
                          f"| {confirmed_count}/{self.num_screws} done")

            if all([s.confirmed for s in self.screws.values()]):
                reward += 50
                all_done = True
                print("  --> ALL SCREWS CONFIRMED. Episode complete.")

        return reward, xy_dist, all_done, newly_confirmed, valid_zone

    def step(self, action):
        self.step_count += 1

        # 1. EXECUTE ACTION BASED ON CONTROL MODE
        action_id = self.get_action_id(action)
        did_detect = (action_id == Actions.DETECT.value)

        if action_id is not None:
            self.execute_action(action_id)
        self.supervisor.step(self.timestep)

        # GET CURRENT ROBOT POSITION
        current_robot_pos = self.get_robot_position()
        camera_frame = self._get_camera_frame()

        # Initialize previous position on first step to avoid spikes
        if self.previous_robot_pos is None:
            self.previous_robot_pos = current_robot_pos

        current_target = self.screws[self.current_target_name] if self.current_target_name else None
        current_max_confidence = current_target.max_confidence
        current_confidence = 0.0

        # PERCEPTION (Camera + YOLO)
        if did_detect and current_target is not None:
            detections, img_with_boxes = self.run_yolo_detection(camera_frame)
            current_confidence = self.get_conf_of_correct_detection(detections, current_target.point)
            self._visualize(image=img_with_boxes)
        else:
            self._visualize(image=camera_frame)

        reward, xy_dist, all_done, newly_confirmed, valid_zone = self.get_reward(
            current_robot_pos=current_robot_pos,
            previous_robot_pos=self.previous_robot_pos,
            current_confidence=current_confidence,
            active_screw=current_target,
            did_detect=did_detect
        )

        if newly_confirmed:
            self.current_target_name = self.get_next_target()
            if self.current_target_name is not None:
                print(f"  --> Next target selected: {self.current_target_name}")

        # # Terminate if max steps reached (Time limits)
        truncated = False
        terminated = False
        out_of_bounds = False

        if self.step_count >= self.max_steps:
            print(f'----- Max steps [{self.max_steps}] completed. Episode finished -----')
            truncated = True

        # Terminate if out of bounds
        if self._is_out_of_bounds():
            reward = -1
            terminated = True
            out_of_bounds = True
            print(f'----- Out of Bounds Reward: {reward} -----')

        if all_done:
            if self.mode == 'eval':
                self._scripted_return_home()
            terminated = True

        self.total_reward += reward
        delta_conf = max(0.0, current_confidence - current_max_confidence) if did_detect else 0.0

        # status = "RUN"
        status = "-"
        if all_done:
            status = "ALL DONE"
        elif newly_confirmed:
            status = "CONFIRMED"
        elif out_of_bounds:
            status = "OOB"
        elif truncated:
            status = "TIMEOUT"

        print(
            f"Step {self.step_count:4d} | "
            f"Target: {self.current_target_name} |"
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
        info = {
            "step_count": self.step_count,
            "confirmed_screws": [s.confirmed for s in self.screws.values()],
            "confidence_vector": [s.max_confidence for s in self.screws.values()],
            "current_target": self.current_target_name,
            "all_done": all_done,
        }

        self.previous_robot_pos = current_robot_pos
        observation = self.get_observation(image=camera_frame)

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.supervisor.simulationReset()
        self.set_robot_position(self.initial_pos)
        for _ in range(self.frame_skip):
            self.supervisor.step(self.timestep)

        # Reset Episode Counters & State
        self.step_count = 0
        self.total_reward = 0.0
        self.returning_home = False
        self.previous_robot_pos = np.array(self.initial_pos, dtype=np.float32)
        self.set_random_light_luminosity(min_lum=0.3, max_lum=3.0)

        random.shuffle(self.execution_sequence)
        print(self.execution_sequence)

        for screw in self.screws.values():
            screw.reset()

        if options is not None and "initial_confirmed_screws" in options:

            initial_confirmed_screws = np.array(options["initial_confirmed_screws"], dtype=bool)

            for pos, target_name in enumerate(self.execution_sequence):
                screw = self.screws[target_name]
                screw.confirmed = initial_confirmed_screws[pos]
                if screw.confirmed:
                    screw.max_confidence = np.random.uniform(low=0.89, high=0.91)

        self.current_target_name = self.get_next_target()
        if self.current_target_name:
            print(f"  --> First target selected: {self.current_target_name}")

        camera_frame = self._get_camera_frame()
        observation = self.get_observation(image=camera_frame)

        print(f"--- RESET | Mode: {self.mode} | Targets: {self.execution_sequence} ---")

        info = {}

        return observation, info

    def _is_out_of_bounds(self):
        pos = self.get_robot_position()
        min_x, max_x, min_y, max_y, min_z, max_z = self.workspace_bounds
        if not (min_x <= pos[0] <= max_x and
                min_y <= pos[1] <= max_y and
                min_z <= pos[2] <= max_z):

            return True
        return False

    # ----------------------------------------------------------------------------------
    def _scripted_return_home(self):
        self.returning_home = True
        tolerance = 0.01
        max_iterations = 1000

        target_pos = np.asarray(self.initial_pos, dtype=np.float32)
        iterations = 0

        while iterations < max_iterations:
            current_pos = np.asarray(self.get_robot_position(), dtype=np.float32)
            delta = target_pos - current_pos

            if abs(delta[0]) > tolerance:
                action = Actions.RIGHT.value if delta[0] > 0 else Actions.LEFT.value
            elif abs(delta[1]) > tolerance:
                action = Actions.FORWARD.value if delta[1] > 0 else Actions.BACKWARD.value
            elif abs(delta[2]) > tolerance:
                action = Actions.UP.value if delta[2] > 0 else Actions.DOWN.value
            else:
                break

            self.execute_action(action)
            self.supervisor.step(self.timestep)
            self._visualize(image=None)
            iterations += 1
        else:
            print("[RETURN HOME] Max iterations reached before reaching home position.")

        # self.home_reached = True
        print(f'----- Agent returned home -----')
        self.returning_home = False

    # ---------------------------------------------------------------------------------- 

    def execute_action(self, action):

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

        # self.supervisor.step(self.timestep)
