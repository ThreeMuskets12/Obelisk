import cv2
import mediapipe as mp
import numpy as np
import logging

from Common import add_transparent_image, calculate_error, create_status_bar
from TrackedPerson import tracked_person

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Customize the output format
    handlers=[
        logging.FileHandler('app.log'),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

logger = logging.getLogger(__name__)
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
Person = tracked_person()
## Setup mediapipe instance

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
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            Person.update(landmarks) #There is something wrong with Person.update
        except:
            pass

        # Rep data
        cv2.putText(image, 'angle', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(Person.angle_between_arms), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, Person.mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )
                       
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        cv2.imshow('Mediapipe Feed', image)

        press = cv2.waitKey(10) & 0xFF

        if press == ord('q'):
            break

        if press == ord('c'):
            #Record pose
            #logger.info("NOSE " + str(Person))
            Person.save_to_json('tracked_person.json')


    cap.release()
    cv2.destroyAllWindows()