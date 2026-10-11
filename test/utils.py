import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import math

# ----------------- Robot --------------------------------------
def set_robot_position(self, pos):
    self.robot.getField('translation').setSFVec3f(pos)


def get_robot_position(self):
    return self.robot.getField('translation').getSFVec3f()


def get_robot_rotation(self):
    return self.robot.getField('rotation').getSFRotation()

# ----------------- Camera --------------------------------------
def get_camera_position(self):
    return self.camera_node.getField("translation").getSFVec3f()


def get_camera_rotation(self):
    return self.camera_node.getField("rotation").getSFRotation()


def get_camera_orientation(self):
    return np.array(self.camera_node.getOrientation()).reshape(3, 3)


def get_rotation_matrix(self, rotation):
    """
    Equals to get_orientation()
    """
    axis = rotation[:3]
    angle = rotation[3]

    r = R.from_rotvec(np.array(axis) * angle)
    rotation_matrix = r.as_matrix()

    print(f'Axis: {axis}, Angle: {angle}')

    return rotation_matrix


def get_camera_extrinsics(self):

    rotation_matrix = self.get_camera_orientation()
    translation = self.get_camera_position()

    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = translation

    return extrinsic_matrix


def draw_boxes_on_image(image, boxes, model_names):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
        class_id = int(box.cls)  # Get class id
        class_name = model_names[class_id]  # Get class name
        confidence = box.conf[0]  # Get confidence score

        if box.id is not None:
            track_id = int(box.id[0])
            label = f"{class_name}{track_id}: {confidence:.2f}"
        else:
            label = f"{class_name}: {confidence:.2f}"  # Prepare label text
        # Set color based on class id
        color = (0, 0, 255) if class_id == 0 else (255, 255, 0) if class_id == 1 else (0, 255, 0)
        # Draw label at different positions based on class
        if class_id == 0:
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=1)
        elif class_id == 1:
            cv2.putText(image, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=1)
        elif class_id == 2:
            cv2.putText(image, label, (x1 + 20, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=1)
        # Draw rectangle for bounding box
        cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=1)
    return image



def draw_boxes_and_lines_on_image(self, image, boxes, model_names):
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    box_center_x = 0
    box_center_y = 0
    box_centers = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
        class_id = int(box.cls)  # Get class id
        class_name = model_names[class_id]  # Get class name
        confidence = box.conf[0]  # Get confidence score

        track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else None
        if class_name == 'screw' and track_id is not None:
            label = f"ID: {track_id} {class_name}: {confidence:.2f}"
        else:
            label = f"{class_name}: {confidence:.2f}"

        color = (0, 0, 255) if class_id == 0 else (255, 255, 0) if class_id == 1 else (0, 255, 0)
        # Draw rectangle for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label at different positions based on class
        if class_id == 0:
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif class_id == 1:
            cv2.putText(image, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif class_id == 2:
            cv2.putText(image, label, (x1 + 20, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw a line from image center to box center
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2
            box_centers.append((box_center_x, box_center_y))
            cv2.line(image, (center_x, center_y), (box_center_x, box_center_y), color, 2)

    return image, box_centers