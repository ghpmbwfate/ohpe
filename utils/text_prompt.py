"""
Generate text prompts for pose occlusion descriptions.
"""

# COCO 17 keypoint indices
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

BODY_PARTS = {
    'head': [0, 1, 2, 3, 4],
    'left arm': [5, 7, 9],
    'right arm': [6, 8, 10],
    'left leg': [11, 13, 15],
    'right leg': [12, 14, 16],
}


def generate_text_prompt(visibility):
    """
    Generate a text description based on keypoint visibility.

    Args:
        visibility: array-like [17], values in {0, 1, 2}
            0 = not in image, 1 = occluded, 2 = visible

    Returns:
        str: text prompt like "The left arm of the human is occluded"
    """
    occluded_parts = []
    for part_name, indices in BODY_PARTS.items():
        # Part is occluded if all its keypoints are not fully visible
        if all(visibility[i] < 2 for i in indices):
            occluded_parts.append(part_name)

    if not occluded_parts:
        return "The body of the human is unoccluded"

    # Return description of the first occluded part
    return f"The {occluded_parts[0]} of the human is occluded"


def generate_text_prompts_batch(visibilities):
    """
    Batch version of generate_text_prompt.

    Args:
        visibilities: [B, 17]

    Returns:
        list of str
    """
    return [generate_text_prompt(v) for v in visibilities]
