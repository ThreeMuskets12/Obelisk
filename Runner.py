import cv2
import mediapipe as mp
import numpy as np
import logging

from Common import *
from TrackedPerson import tracked_person

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Customize the output format
    handlers=[
        logging.FileHandler('log.txt'),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)
logger = logging.getLogger(__name__)

if MACOS_HANDOFF_CAMERA_OVERRIDE == True:
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(0)

pose_1_image = cv2.imread('Poses/Pose1/Pose1.png', cv2.IMREAD_UNCHANGED)
pose_1_image = cv2.resize(pose_1_image, (686, 1000))
pose_1_image = cv2.cvtColor(pose_1_image, cv2.COLOR_BGRA2RGBA)

pose_2_image = cv2.imread('Poses/Pose2/Pose2.png', cv2.IMREAD_UNCHANGED)
pose_2_image = cv2.resize(pose_2_image, (540, 1017))
pose_2_image = cv2.cvtColor(pose_2_image, cv2.COLOR_BGRA2RGBA)

mp_drawing = mp.solutions.drawing_utils

score = 0

Person = tracked_person()
Pose1 = tracked_person(landmarks= (parse_pose_landmarks_from_json('Poses/Pose1/Pose1.json')), image= pose_1_image)
Pose2 = tracked_person(landmarks= (parse_pose_landmarks_from_json('Poses/Pose2/Pose2.json')), image= pose_2_image)

PoseArray = [Pose1, Pose2]
TargetPoseArrayIndex = 0

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
            Person.update()
            score = calculate_score(Person, PoseArray[TargetPoseArrayIndex])
        except:
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, Person.mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )

        #The thought behind this flip is the webcam is facing the same direction as the display on both laptop and Obelisk
        #But if you are doing mocap or testing where the webcam faces away from the display you should probably comment this out
        image = cv2.flip(image, 1)

                        # Rep data
        cv2.putText(image, 'SCORE', (100,500), 
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(image, str(score), 
                    (100,600), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        add_transparent_image(image, PoseArray[TargetPoseArrayIndex].image, 1200, 150)
        status_bar_image = create_status_bar(score)
        add_transparent_image(image, status_bar_image, 0, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        cv2.imshow('Mediapipe Feed', image)

        if(score > 95):
            TargetPoseArrayIndex = TargetPoseArrayIndex + 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


    
    cap.release()
    cv2.destroyAllWindows()