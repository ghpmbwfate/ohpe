"""
COCO human pose visualization utilities.
"""
import numpy as np
import cv2


# COCO 17 keypoint pairs for skeleton drawing
COCO_SKELETON = [
    [16, 14], [14, 12],  # right leg
    [17, 15], [15, 13],  # left leg
    [12, 13],            # hips
    [6,  12], [7,  13],  # hip -> shoulder
    [6,  7],             # shoulders
    [6,  8], [8,  10],   # right arm
    [7,  9], [9,  11],   # left arm
    [1,  2],             # eyes
    [0,  1], [0,  2],    # nose -> eyes
    [1,  3], [2,  4],    # ear -> eye
    [3,  5], [4,  6],    # ear -> shoulder
]

# Keypoint names for labeling (1-indexed as per COCO convention)
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

# Color palette for keypoints and skeleton lines
KEYPOINT_COLOR = (0, 255, 0)       # green for keypoints
SKELETON_COLOR = (255, 0, 0)       # blue for skeleton lines (BGR)
TEXT_COLOR = (255, 255, 255)       # white for text


def draw_keypoints(image, keypoints, radius=3, color=KEYPOINT_COLOR, thickness=-1):
    """
    Draw keypoints on an image.

    Args:
        image: np.ndarray [H, W, 3] (BGR), modified in-place.
        keypoints: np.ndarray [17, 2] or [17, 3], pixel coordinates (x, y).
                   If 3 columns, third column is confidence (ignored for drawing).
        radius: int, circle radius.
        color: tuple(B, G, R).
        thickness: int, -1 for filled circle.

    Returns:
        np.ndarray, the image with keypoints drawn.
    """
    keypoints = np.array(keypoints)
    for i, kp in enumerate(keypoints):
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.putText(image, str(i), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, TEXT_COLOR, 1)
    return image


def draw_skeleton(image, keypoints, line_color=SKELETON_COLOR, thickness=2):
    """
    Draw COCO skeleton on an image.

    Args:
        image: np.ndarray [H, W, 3] (BGR), modified in-place.
        keypoints: np.ndarray [17, 2] or [17, 3], pixel coordinates (x, y).
        line_color: tuple(B, G, R).
        thickness: int, line thickness.

    Returns:
        np.ndarray, the image with skeleton drawn.
    """
    keypoints = np.array(keypoints)
    for p1, p2 in COCO_SKELETON:
        idx1, idx2 = p1 - 1, p2 - 1  # convert 1-indexed to 0-indexed
        x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
        x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
        cv2.line(image, (x1, y1), (x2, y2), line_color, thickness)

    draw_keypoints(image, keypoints)
    return image


def denormalize_keypoints(norm_keypoints, width, height):
    """
    Convert normalized keypoints in [-1, 1] to pixel coordinates.

    Args:
        norm_keypoints: np.ndarray [17, 2] or [B, 17, 2], values in [-1, 1].
        width: int, image width in pixels.
        height: int, image height in pixels.

    Returns:
        np.ndarray, same shape, pixel coordinates.
    """
    norm_keypoints = np.array(norm_keypoints)
    pixel_kp = (norm_keypoints + 1.0) / 2.0
    pixel_kp[..., 0] *= (width - 1)
    pixel_kp[..., 1] *= (height - 1)
    return pixel_kp
