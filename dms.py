import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import pygame

# --- Helper Function ---
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- Constants and Variables ---
LEFT_EYE_INDICES = list(range(42, 48))
RIGHT_EYE_INDICES = list(range(36, 42))

EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES_THRESHOLD = 20
ALERT_SOUND_PATH = "alert.mp3"

drowsy_frames_counter = 0
is_alert_playing = False 

# --- Pygame Sound Initialization ---
try:
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(ALERT_SOUND_PATH)
    sound_available = True
except Exception as e:
    print(f"Can't load sound: {ALERT_SOUND_PATH}. Error: {e}")
    sound_available = False

# --- Dlib and Model Initialization ---
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError as e:
    print(f"Error loading model: {e}")
    exit()

# --- Video Capture Initialization ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- NEW: HEAD POSE ESTIMATION SETUP ---
# Get frame size (we'll need this for the camera matrix)
ret, frame = cap.read()
size = frame.shape

# A generic 3D model of a face
# The specific points are: Nose tip, Chin, Left eye left corner, Right eye right corner, Left Mouth corner, Right mouth corner
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# A generic camera matrix
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)
# Assuming no lens distortion
dist_coeffs = np.zeros((4, 1)) 


# --- MAIN PROCESSING LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # --- Drowsiness Detection Logic (Existing) ---
        landmarks = predictor(gray, face)
        points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])
        
        # ... (all the ear calculation and alert logic remains here)
        left_eye = points[LEFT_EYE_INDICES]
        right_eye = points[RIGHT_EYE_INDICES]
        avg_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
        
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (frame.shape[1]-150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if avg_ear < EAR_THRESHOLD:
            drowsy_frames_counter += 1
            if drowsy_frames_counter >= CONSECUTIVE_FRAMES_THRESHOLD:
                if not is_alert_playing and sound_available:
                    pygame.mixer.music.play(loops=-1)
                    is_alert_playing = True
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if is_alert_playing and sound_available:
                pygame.mixer.music.stop()
            drowsy_frames_counter = 0
            is_alert_playing = False

        # --- NEW: DISTRACTION DETECTION LOGIC ---
        # Get the 2D coordinates of the same points from the landmark detector
        image_points = np.array([
            (points[30][0], points[30][1]),     # Nose tip
            (points[8][0], points[8][1]),      # Chin
            (points[36][0], points[36][1]),    # Left eye left corner
            (points[45][0], points[45][1]),    # Right eye right corner
            (points[48][0], points[48][1]),    # Left Mouth corner
            (points[54][0], points[54][1])     # Right mouth corner
        ], dtype="double")

        # Use solvePnP to find the head's rotation and translation
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

        # Project a 3D point (like the tip of a 300mm long line) to see the head direction
        # This creates a line coming out of the nose
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 300.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        # Draw the line on the image
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(frame, p1, p2, (255, 0, 0), 2)

        # Convert the rotation vector to angles and determine direction
        # This is a simplified conversion
        x_angle = rotation_vector[1] * (180/np.pi)
        y_angle = rotation_vector[0] * (180/np.pi)
        
        direction_text = ""
        if y_angle < -10:
            direction_text = "Looking Left"
        elif y_angle > 10:
            direction_text = "Looking Right"
        elif x_angle < -10:
            direction_text = "Looking Down"
        else:
            direction_text = "Looking Forward"
            
        cv2.putText(frame, direction_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    cv2.imshow("Driver Monitoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
if sound_available:
    pygame.mixer.quit()
    pygame.quit()