import cv2 
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# Initialize video capture
cap = cv2.VideoCapture(0)

# Setup MediaPipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Process with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('FitCheck AI Camera', image)
        
        
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
            print(landmarks)
        except:
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('FitCheck AI Camera', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

mp_drawing.DrawingSpec


cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
 

    cap.release()
    cv2.destroyAllWindows()
    len(landmarks)
for lndmrk in mp_pose.PoseLandmark:
    print(lndmrk)
landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]


# Function to calculate angle between three points.
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Capture video from the webcam.
cap = cv2.VideoCapture(0)

# Set up MediaPipe Pose.
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection and render.
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks.
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Calculate angles for both knees and hips.
            for side in ['LEFT', 'RIGHT']:
                hip = [landmarks[getattr(mp_pose.PoseLandmark, f'{side}_HIP').value].x,
                       landmarks[getattr(mp_pose.PoseLandmark, f'{side}_HIP').value].y]
                knee = [landmarks[getattr(mp_pose.PoseLandmark, f'{side}_KNEE').value].x,
                        landmarks[getattr(mp_pose.PoseLandmark, f'{side}_KNEE').value].y]
                ankle = [landmarks[getattr(mp_pose.PoseLandmark, f'{side}_ANKLE').value].x,
                         landmarks[getattr(mp_pose.PoseLandmark, f'{side}_ANKLE').value].y]
                shoulder = [landmarks[getattr(mp_pose.PoseLandmark, f'{side}_SHOULDER').value].x,
                            landmarks[getattr(mp_pose.PoseLandmark, f'{side}_SHOULDER').value].y]

                # Calculate knee angle.
                knee_angle = calculate_angle(hip, knee, ankle)
                cv2.putText(image, f'Knee Angle: {int(knee_angle)}', 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Calculate hip angle.
                hip_angle = calculate_angle(shoulder, hip, knee)
                cv2.putText(image, f'Hip Angle: {int(hip_angle)}', 
                            tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
        except:
            pass

        # Render detections.
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the frame.
        cv2.imshow('Squat Pose Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows() 



from collections import deque

# Set the number of frames to average over
N = 5
angle_history = deque(maxlen=N)

def get_smooth_angle(new_angle):
    angle_history.append(new_angle)
    return sum(angle_history) / len(angle_history)


angle = calculate_angle(hip, knee, ankle)
smooth_angle = get_smooth_angle(angle)

def draw_text_with_background(image, text, position, font_scale=0.5, font_thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    text_x, text_y = position
    cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 5), bg_color, -1)
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

# Inside your main loop after calculating the angle:
hip_text_position = (int(hip[0] * image.shape[1]), int(hip[1] * image.shape[0]))
knee_text_position = (int(knee[0] * image.shape[1]), int(knee[1] * image.shape[0]))

draw_text_with_background(image, f'Hip Angle: {int(hip_angle)}', hip_text_position, bg_color=(0, 0, 255))
draw_text_with_background(image, f'Knee Angle: {int(knee_angle)}', knee_text_position, bg_color=(255, 0, 0))


#SQUAT COUNTER
cap = cv2.VideoCapture(0)

# Curl counter variables
counter = 0 
stage = None

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate angle
            hip_angle = calculate_angle(shoulder, hip, knee)
            knee_angle = calculate_angle(hip, knee, ankle)
            
            # Visualize angle
            cv2.putText(image, str(hip_angle), 
                           tuple(np.multiply(hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            # Curl counter logic
            if hip_angle < 135:
                stage = "down"
            if hip_angle > 165 and stage =='down':
                stage="up"
                counter +=1
                print(counter)
                
               
                # Determine if the form is good
            if 75 <= knee_angle <= 170 and hip_angle <= 100:
                form_text = "Good Form"
                color = (0, 255, 0)  # Green
            else:
                form_text = "Needs Improvement"
                color = (0, 0, 255)  # Red
                      
        except:
            pass
        
        # Display form evaluation
        cv2.putText(image, form_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(image, f'Hip Angle: {int(hip_angle)}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f'Knee Angle: {int(knee_angle)}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


        # Show the frame
        #cv2.imshow('Squat Form Analysis', frame)
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
