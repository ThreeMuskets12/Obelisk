import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(first,middle,last):
    a = np.array(first) # First
    b = np.array(middle) # Mid
    c = np.array(last) # End
        
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
        
    if angle >180.0:
        angle = 360-angle        
    return angle 

angle_between_arms = calculate_angle([228,145], [171,155], [113,172])

left_arm_straightness_angle = calculate_angle([257,103], [228,145], [204,186])
right_arm_straightness_angle = calculate_angle([64,133], [113,172], [147,201])
left_leg_straightness_angle = calculate_angle([211,295], [213,347], [214,393])
right_leg_straightness_angle = calculate_angle([160,296], [163,349], [166,393])

left_arm_raise_angle = calculate_angle([228,145], [204,186], [171,155])
right_arm_raise_angle = calculate_angle([113,172], [147,201], [171,155])

left_to_right_shoulder_angle = calculate_angle([204,186], [171,155], [147,201])

print(angle_between_arms)

print(left_arm_straightness_angle)
print(right_arm_straightness_angle)
print(left_leg_straightness_angle)
print(right_leg_straightness_angle)

print(left_arm_raise_angle)
print(right_arm_raise_angle)
print(left_to_right_shoulder_angle)