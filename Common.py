import numpy as np
from typing import List, Optional
import json
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark

MO_CAP_CAMERA_TIMER = 5 #Seconds
MACOS_HANDOFF_CAMERA_OVERRIDE = True #Default true, false if you want handoff to iPhone camera

def calculate_angle(first,middle,last):
    a = np.array(first) # First
    b = np.array(middle) # Mid
    c = np.array(last) # End
        
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
        
    if angle >180.0:
        angle = 360-angle        
    return angle

def calculate_score(tracked_guy, desired_pose):
    angle_between_arms_error = calculate_error(tracked_guy.angle_between_arms, desired_pose.angle_between_arms)
    
    left_arm_straightness_angle_error = calculate_error(tracked_guy.left_arm_straightness_angle, desired_pose.left_arm_straightness_angle)
    right_arm_straightness_angle_error = calculate_error(tracked_guy.right_arm_straightness_angle, desired_pose.right_arm_straightness_angle)

    left_leg_straightness_angle_error = calculate_error(tracked_guy.left_leg_straightness_angle, desired_pose.left_leg_straightness_angle)
    right_leg_straightness_angle_error = calculate_error(tracked_guy.right_leg_straightness_angle, desired_pose.right_leg_straightness_angle)

    left_arm_raise_angle_error = calculate_error(tracked_guy.left_arm_raise_angle, desired_pose.left_arm_raise_angle)
    right_arm_raise_angle_error = calculate_error(tracked_guy.right_arm_raise_angle, desired_pose.right_arm_raise_angle)

    
    error_array = [angle_between_arms_error, left_arm_straightness_angle_error, right_arm_straightness_angle_error, left_leg_straightness_angle_error, right_leg_straightness_angle_error, left_arm_raise_angle_error, right_arm_raise_angle_error]

    score = 100 - np.mean(error_array)

    return score

def parse_pose_landmarks_from_json(file_name: str) -> List[NormalizedLandmark]:
    """Parse pose landmarks from a JSON file and return a list of NormalizedLandmark objects."""
    with open(file_name, 'r') as json_file:
        data = json.load(json_file)
    
    normalized_landmarks = []
    
    for landmark_data in data['landmarks']:
        if isinstance(landmark_data, dict):  # Ensure landmark_data is a dict
            landmark = NormalizedLandmark(
                x=landmark_data.get('x'),
                y=landmark_data.get('y'),
                z=landmark_data.get('z'),
                visibility=landmark_data.get('visibility'),
                presence=landmark_data.get('presence')  # Optional attribute
            )
            normalized_landmarks.append(landmark)
        else:
            print(f"Expected dict but got: {type(landmark_data)}")  # Error handling

    return normalized_landmarks

def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
        bg_h, bg_w, bg_channels = background.shape
        fg_h, fg_w, fg_channels = foreground.shape

        assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
        assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

        # center by default
        if x_offset is None: x_offset = (bg_w - fg_w) // 2
        if y_offset is None: y_offset = (bg_h - fg_h) // 2

        w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
        h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

        if w < 1 or h < 1: return

        # clip foreground and background images to the overlapping regions
        bg_x = max(0, x_offset)
        bg_y = max(0, y_offset)
        fg_x = max(0, x_offset * -1)
        fg_y = max(0, y_offset * -1)
        foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
        background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

        # separate alpha and color channels from the foreground image
        foreground_colors = foreground[:, :, :3]
        alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

        # construct an alpha_mask that matches the image shape
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

        # combine the background with the overlay image weighted by alpha
        composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

        # overwrite the section of the background image that has been updated
        background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

def calculate_error(a,b):
    return abs(((a - b) / b) * 100)

def create_status_bar(percentage):
    # Validate the input
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100.")

    # Define the dimensions of the image
    width, height = 1920, 300

    # Create a transparent RGBA image
    image = np.zeros((height, width, 4), dtype=np.uint8)

    # Calculate the width of the filled portion
    filled_width = int((percentage / 100) * width)

    # Fill the filled portion with white color (255, 255, 255) and full opacity (255)
    image[:, :filled_width] = [255, 255, 255, 255]  # White with full opacity

    # The rest remains transparent (0, 0, 0, 0)
    
    return image