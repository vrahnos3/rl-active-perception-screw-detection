"""
Utility functions for the Webots screw detection demo.

Kept intentionally small:
- draw_boxes_on_image(): render YOLO boxes with class-specific labels and colors.

If you need camera pose/extrinsics helpers, consider moving those into a separate
module (e.g. geometry.py) so vision utilities stay focused.
"""
from __future__ import annotations

import math
from typing import Any

import cv2


def draw_boxes_on_image(image, boxes, model_names) -> Any:
    """
    Draw bounding boxes and labels onto an image.

    Parameters
    ----------
    image : np.ndarray
        RGB image (H,W,3)
    boxes : ultralytics.engine.results.Boxes
        YOLO boxes
    model_names : dict[int,str]
        Class id -> name mapping

    Returns
    -------
    image : np.ndarray
        Image with overlays (modified in-place and returned)
    """
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls)
        class_name = model_names[class_id]
        confidence = float(box.conf[0])
        confidence = math.floor(confidence * 100) / 100

        track_id = int(box.id[0]) if hasattr(box, "id") and box.id is not None else None
        if class_name == "screw" and track_id is not None:
            label = f"ID: {track_id} {class_name}: {confidence:.2f}"
        else:
            label = f"{class_name}: {confidence:.2f}"

        # Simple class-based color scheme
        color = (0, 0, 255) if class_id == 0 else (255, 255, 0) if class_id == 1 else (0, 255, 0)

        # Label placement varies slightly by class to reduce overlap
        if class_id == 0:
            text_org = (x1, y1 - 5)
        elif class_id == 1:
            text_org = (x1, y2 + 15)
        else:
            text_org = (x1 + 20, y2 - 5)

        cv2.putText(image, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    return image
