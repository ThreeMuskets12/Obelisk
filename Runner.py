import cv2
import mediapipe as mp
import numpy as np
import logging

from Common import *
from TrackedPerson import tracked_person

#Okay so I think I am going to have a person class that keeps track of all the attributes a person can have

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

### NOW WE ARE OUT OF THE CLASS ###

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Customize the output format
    handlers=[
        logging.FileHandler('log.txt'),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

logger = logging.getLogger(__name__)
try:
    cap = cv2.VideoCapture(0)
except:
    cap = cv2.VideoCapture(0)

pose_1_image = cv2.imread('blueman2.png', cv2.IMREAD_UNCHANGED)
pose_1_image = cv2.resize(pose_1_image, (686, 1000))
pose_1_image = cv2.cvtColor(pose_1_image, cv2.COLOR_BGRA2RGBA)

mp_drawing = mp.solutions.drawing_utils

score = 0

Person = tracked_person()
Pose1 = tracked_person(landmarks=(parse_pose_landmarks_from_json('Pose1.json')))


# Setup mediapipe instance
with Person.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Extract landmarks``
        try:
            landmarks = results.pose_landmarks.landmark
            Person.set_landmarks(landmarks)
            Person.update() #There is something wrong with Person.update
            score = calculate_score(Person, Pose1)
        except:
            pass


        # Rep data
        cv2.putText(image, 'SCORE', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(score), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, Person.mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )
                       
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        add_transparent_image(image, pose_1_image, 0, 0)
        status_bar_image = create_status_bar(score)
        add_transparent_image(image, status_bar_image, 0, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


    
    cap.release()
    cv2.destroyAllWindows()