import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Functions
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = np.clip(dot_product / magnitude, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad), angle_rad

# Set average hand width (in cm) to scale the distance
average_hand_width_cm = 8.0  # You can adjust based on your own hand size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.append([lm.x, lm.y, lm.z])  # Using 3D coordinates

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if len(lm_list) > 8:
                # Get 3D coordinates for thumb tip and index tip
                thumb_base = lm_list[1]
                thumb_point2 = lm_list[2]
                thumb_tip = lm_list[4]
                index_base = lm_list[5]
                index_tip = lm_list[8]

                # Calculate vectors for angle calculation
                thumb_vector = np.array([thumb_tip[0] - thumb_point2[0], thumb_tip[1] - thumb_point2[1], thumb_tip[2] - thumb_point2[2]])
                index_vector = np.array([index_tip[0] - index_base[0], index_tip[1] - index_base[1], index_tip[2] - index_base[2]])

                # Calculate 3D distance (hypotenuse)
                distance_3d = np.sqrt(
                    (thumb_tip[0] - index_tip[0]) ** 2 +
                    (thumb_tip[1] - index_tip[1]) ** 2 +
                    (thumb_tip[2] - index_tip[2]) ** 2
                )

                # Scale the distance based on hand size (average hand width)
                scaling_factor = average_hand_width_cm / 0.2  # Approx. 0.2 is hand width in normalized units
                real_distance_cm = distance_3d * scaling_factor

                # Calculate angle between thumb and index finger
                angle_deg, _ = calculate_angle(thumb_vector, index_vector)

                # Draw the lines (thumb and index fingers)
                cv2.line(frame, (int(thumb_tip[0] * 640), int(thumb_tip[1] * 480)),
                         (int(thumb_point2[0] * 640), int(thumb_point2[1] * 480)), (0, 255, 0), 3)  # Thumb Line 1
                cv2.line(frame, (int(thumb_point2[0] * 640), int(thumb_point2[1] * 480)),
                         (int(thumb_base[0] * 640), int(thumb_base[1] * 480)), (0, 255, 255), 3)  # Thumb Line 2
                cv2.line(frame, (int(index_base[0] * 640), int(index_base[1] * 480)),
                         (int(index_tip[0] * 640), int(index_tip[1] * 480)), (0, 0, 255), 3)  # Index Finger Line
                cv2.line(frame, (int(thumb_point2[0] * 640), int(thumb_point2[1] * 480)),
                         (int(index_base[0] * 640), int(index_base[1] * 480)), (255, 255, 0), 2)  # Angle Line

                # Draw the hypotenuse line (Blue)
                cv2.line(frame, (int(thumb_tip[0] * 640), int(thumb_tip[1] * 480)),
                         (int(index_tip[0] * 640), int(index_tip[1] * 480)), (255, 0, 0), 2)  # Hypotenuse

                # Display the hypotenuse length in centimeters
                cv2.putText(frame, f"Hypotenuse: {real_distance_cm:.2f} cm",
                            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

                # Display the angle between the thumb and index finger
                cv2.putText(frame, f"Angle: {angle_deg:.2f} degrees",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
