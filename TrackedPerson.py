from Common import *
import mediapipe as mp
import json

class tracked_person():
    def __init__(self, nose = None, left_shoulder = None, right_shoulder = None, left_elbow = None, right_elbow = None, left_wrist = None, right_wrist = None, left_hip = None, right_hip = None, left_knee = None, right_knee = None, left_ankle = None, right_ankle = None):
        #X,Y points of our landmarks
        self.nose = nose
        self.left_shoulder = left_shoulder
        self.right_shoulder = right_shoulder
        self.left_elbow = left_elbow
        self.right_elbow = right_elbow
        self.left_wrist = left_wrist
        self.right_wrist = right_wrist
        self.left_hip = left_hip
        self.right_hip = right_hip
        self.left_knee = left_knee
        self.right_knee = right_knee
        self.left_ankle = left_ankle
        self.right_ankle = right_ankle

        #Boolean flags
        self.hands_above_head_state = False
        self.arms_straight_state = False
        self.legs_straight_state = False
        self.is_pose_1_state = False

        #angles
        self.angle_between_arms = 0

        self.left_arm_straightness_angle = 0 #TODO not sure what the default angle should be
        self.right_arm_straightness_angle = 0 #TODO not sure what the default angle should be
        self.left_leg_straightness_angle = 0
        self.right_leg_straightness_angle = 0

        self.left_arm_raise_angle = 0
        self.right_arm_raise_angle = 0

        self.left_to_right_shoulder_angle = 0

        self.mp_pose = mp.solutions.pose

    def to_dict(self):
        #All attributes as a dict
        return {
            'Nose': self.nose,
            'left_shoulder': self.left_shoulder,
            'right_elbow': self.right_elbow,
            'is_visible': self.is_visible
        }
    
    def hands_above_head(self):
        #x,y
        if(self.left_wrist[1] < self.nose[1] and self.right_wrist[1] < self.nose[1]):
            self.hands_above_head_state = True
        else:
            self.hands_above_head_state = False
    
    def arms_straight(self):
        if(self.left_arm_straightness_angle > 140 and self.right_arm_straightness_angle > 140):
            self.arms_straight_state = True
        else:
            self.arms_straight_state = False
    
    def legs_straight(self):
        if(self.left_leg_straightness_angle > 150 and self.right_leg_straightness_angle > 150):
            self.legs_straight_state = True
        else:
            self.legs_straight_state = False

    def calculate_angles(self):
        self.angle_between_arms = calculate_angle(self.left_elbow, self.nose, self.right_elbow)

        self.left_arm_straightness_angle = calculate_angle(self.left_wrist, self.left_elbow, self.left_shoulder)
        self.right_arm_straightness_angle = calculate_angle(self.right_wrist, self.right_elbow, self.right_shoulder)
        self.left_leg_straightness_angle = calculate_angle(self.left_hip, self.left_knee, self.left_ankle)
        self.right_leg_straightness_angle = calculate_angle(self.right_hip, self.right_knee, self.right_ankle)

        self.left_arm_raise_angle = calculate_angle(self.left_elbow, self.left_shoulder, self.nose)
        self.right_arm_raise_angle = calculate_angle(self.right_elbow, self.right_shoulder, self.nose)

        self.left_to_right_shoulder_angle = calculate_angle(self.left_shoulder, self.nose, self.right_shoulder)

    
    def is_pose_1(self):
        if(self.hands_above_head_state and self.arms_straight_state and self.legs_straight_state and self.angle_between_arms > 50 and self.angle_between_arms < 120):
            self.is_pose_1_state = True
        else:
            self.is_pose_1_state = False
    
    def update(self, landmarks):
        self.nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
        self.left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        self.right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        self.left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        self.right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        self.left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        self.right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        self.left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        self.right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        self.left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        self.right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        self.left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        self.right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        self.calculate_angles()

        self.hands_above_head()
        self.arms_straight()
        self.legs_straight()