import cv2
import mediapipe as mp
import numpy as np
import logging
import time

from Common import *
from TrackedPerson import tracked_person
from mediapipe.framework.formats.landmark_pb2 import Landmark


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

mp_drawing = mp.solutions.drawing_utils
Person = tracked_person()


pose_2_image = cv2.imread('Poses/Pose2/Pose2.png', cv2.IMREAD_UNCHANGED)
pose_2_image = cv2.resize(pose_2_image, (540, 1017))
pose_2_image = cv2.cvtColor(pose_2_image, cv2.COLOR_BGRA2RGBA)

shutter = False
segmentation_mask = np.zeros((1080, 1920), dtype=np.uint8)  # 1080 rows and 1920 columns
segmentation_mask_mono = cv2.cvtColor(segmentation_mask, cv2.COLOR_GRAY2BGR)
## Setup mediapipe instance

with Person.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True) as pose:
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
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            segmentation_mask = results.segmentation_mask
            segmentation_mask = (segmentation_mask * 255).astype(np.uint8)
            segmentation_mask_mono = cv2.cvtColor(segmentation_mask, cv2.COLOR_GRAY2BGR)
            image = segmentation_mask_mono
            print(segmentation_mask_mono.shape)
            Person.set_landmarks(landmarks)
            Person.update()
        except:
            segmentation_mask = np.zeros((1080, 1920), dtype=np.uint8)  # 1080 rows and 1920 columns
            segmentation_mask_mono = cv2.cvtColor(segmentation_mask, cv2.COLOR_GRAY2BGR)
            image = segmentation_mask_mono
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, Person.mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )
        
        image = cv2.flip(image, 1)

                # Rep data
        cv2.putText(image, 'angle_between_arms', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(Person.angle_between_arms), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        add_transparent_image(image, pose_2_image, 1300, 80)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        cv2.imshow('Mediapipe Feed', image)

        press = cv2.waitKey(10) & 0xFF

        if press == ord('q'):
            break

        if press == ord('c'):
            #Record pose
            start_time = time.time()
            shutter = True
        
        if shutter == True:
            if time.time() - start_time > MO_CAP_CAMERA_TIMER:
                Person.update_pose_landmarks_to_dict()
                Person.save_to_json('Pose2.json')
                cap.release()
                cv2.destroyAllWindows()
                break


    cap.release()
    cv2.destroyAllWindows()