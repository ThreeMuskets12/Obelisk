from Common import *
import mediapipe as mp

import json

class tracked_person():
    def __init__(self, landmarks = None):
        self.landmarks = landmarks

        #Boolean flags, ideally we can get rid of these in favor of our magical friends the angles soon
        self.hands_above_head_state = False
        self.arms_straight_state = False
        self.legs_straight_state = False
        self.is_pose_1_state = False

        #angles
        self.angle_between_arms = 0

         #TODO not sure what the default angles should be, zero might cause some / 0 bullshit
        self.left_arm_straightness_angle = 0
        self.right_arm_straightness_angle = 0
        self.left_leg_straightness_angle = 0
        self.right_leg_straightness_angle = 0

        self.left_arm_raise_angle = 0
        self.right_arm_raise_angle = 0

        self.left_to_right_shoulder_angle = 0

        self.mp_pose = mp.solutions.pose

        if (self.landmarks):
            self.update()
            self.calculate_angles()
        else:
            pass

    def to_dict(self):
        #All attributes as a dict
        return {
            self.landmarks
        }
    
    def set_landmarks(self, landmarks):
        #print(landmarks)
        self.landmarks = landmarks
    
    def update_pose_landmarks_to_dict(self):
        """Save pose landmarks to a JSON file."""
        # Create a dictionary to hold the landmark data
        landmarks_data = []

        # Assuming pose_landmarker_result contains landmarks in a certain format
        for landmark in self.landmarks:  # Adjust this based on your actual result structure
            landmark_data = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility  # If available
            }
            landmarks_data.append(landmark_data)

        # Create a final dictionary
        self.landmarks_as_dict = {
            'landmarks': landmarks_data,
            # Include any other relevant attributes from pose_landmarker_result
        }
    
    def save_to_json(self, file_name):
        with open(file_name, 'w') as json_file:
            json.dump(self.landmarks_as_dict, json_file, indent=4)

    def hands_above_head(self):
        #x,y
        if(self.landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y < self.landmarks[self.mp_pose.PoseLandmark.NOSE.value].y and self.landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y < self.landmarks[self.mp_pose.PoseLandmark.NOSE.value].y):
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
        #This one function is not even close to readable, but my previous attempt to clean this up only introduced a ton of unnecessary variables...
        self.angle_between_arms = calculate_angle([self.landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,self.landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y], [self.landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,self.landmarks[self.mp_pose.PoseLandmark.NOSE.value].y], [self.landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,self.landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])

        self.left_arm_straightness_angle = calculate_angle([self.landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,self.landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y], [self.landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,self.landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y], [self.landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,self.landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        self.right_arm_straightness_angle = calculate_angle([self.landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,self.landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y], [self.landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,self.landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y], [self.landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,self.landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        self.left_leg_straightness_angle = calculate_angle([self.landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,self.landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y], [self.landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,self.landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y], [self.landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,self.landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
        self.right_leg_straightness_angle = calculate_angle([self.landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,self.landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y], [self.landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,self.landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y], [self.landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,self.landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])

        self.left_arm_raise_angle = calculate_angle([self.landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,self.landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y], [self.landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,self.landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y], [self.landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,self.landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y])
        self.right_arm_raise_angle = calculate_angle([self.landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,self.landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y], [self.landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,self.landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y], [self.landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,self.landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y])

        self.left_to_right_shoulder_angle = calculate_angle([self.landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,self.landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y], [self.landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,self.landmarks[self.mp_pose.PoseLandmark.NOSE.value].y], [self.landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,self.landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])

    
    def is_pose_1(self):
        if(self.hands_above_head_state and self.arms_straight_state and self.legs_straight_state and self.angle_between_arms > 50 and self.angle_between_arms < 120):
            self.is_pose_1_state = True
        else:
            self.is_pose_1_state = False


    #When we get rid of the booleans (Maybe?) this can just be reduced to calculate angles
    def update(self):

        self.calculate_angles()

        self.hands_above_head()
        self.arms_straight()
        self.legs_straight()