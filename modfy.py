import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to calculate the Euclidean distance between two points (x, y, z)
def calculate_distance_3d(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Assume real-world distance between wrist and middle finger base is around 8 cm
REAL_WORLD_REF_DISTANCE_CM = 8.0  # Approximate

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally for mirror view
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmarks
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            landmarks = []
            for lm in handslms.landmark:
                lmx = int(lm.x * frame.shape[1])
                lmy = int(lm.y * frame.shape[0])
                lmz = lm.z  # Z-coordinate (depth information)
                landmarks.append([lmx, lmy, lmz])

            # Draw hand connections
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Reference points for scaling
            wrist = landmarks[0]
            middle_base = landmarks[9]

            # Calculate 3D reference distance (including depth)
            ref_distance_3d = calculate_distance_3d(wrist, middle_base)

            if ref_distance_3d != 0:
                # Fix the scaling factor using 3D distance
                scaling_factor = REAL_WORLD_REF_DISTANCE_CM / ref_distance_3d

                # --- Thumb (from tip to 3rd point) ---
                thumb_tip = landmarks[4]
                thumb_point3 = landmarks[3]
                thumb_point2 = landmarks[2]

                # Calculate thumb length as tip -> 3 -> 2 (in 3D)
                thumb_segment1 = calculate_distance_3d(thumb_tip, thumb_point3)
                thumb_segment2 = calculate_distance_3d(thumb_point3, thumb_point2)
                thumb_total_pixels = thumb_segment1 + thumb_segment2
                thumb_length_cm = thumb_total_pixels * scaling_factor

                # Draw thumb segments
                cv2.line(frame, tuple(thumb_tip[:2]), tuple(thumb_point3[:2]), (255, 0, 0), 2)
                cv2.line(frame, tuple(thumb_point3[:2]), tuple(thumb_point2[:2]), (255, 0, 0), 2)

                # Display thumb length
                y_offset = 50
                cv2.putText(frame, f"Thumb: {thumb_length_cm:.2f} cm", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 40

                # --- Other Fingers (Index, Middle, Ring, Pinky) ---
                fingers = {
                    "Index": (landmarks[5], landmarks[8]),
                    "Middle": (landmarks[9], landmarks[12]),
                    "Ring": (landmarks[13], landmarks[16]),
                    "Pinky": (landmarks[17], landmarks[20]),
                }

                for finger_name, (base, tip) in fingers.items():
                    length_3d = calculate_distance_3d(base, tip)
                    length_cm = length_3d * scaling_factor

                    # Draw line between base and tip (2D projection)
                    cv2.line(frame, tuple(base[:2]), tuple(tip[:2]), (255, 0, 0), 2)

                    # Display text
                    text = f"{finger_name}: {length_cm:.2f} cm"
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y_offset += 40

    # Show the frame
    cv2.imshow("Finger Length Measurement", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
