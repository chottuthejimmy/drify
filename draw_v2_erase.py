import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize the webcam
cap = cv2.VideoCapture(1)

# Get the frame dimensions
ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]

# Create a blank canvas
canvas = np.zeros((frame_height, frame_width, 3), np.uint8)

# Previous coordinates of the index finger
prev_x, prev_y = 0, 0

# Buffer for smoothing
x_buffer = deque(maxlen=5)
y_buffer = deque(maxlen=5)

# Points for curve drawing
points = []

def is_fist(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    if (thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and 
        thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y):
        return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(rgb_frame)

    # Create a copy of the canvas for this frame
    display = canvas.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if is_fist(hand_landmarks):
                canvas = np.zeros((frame_height, frame_width, 3), np.uint8)
                points = []
            else:
                # Get the coordinates of the index finger tip (Landmark 8)
                index_finger = hand_landmarks.landmark[8]
                x, y = int(index_finger.x * frame_width), int(index_finger.y * frame_height)

                # Add current coordinates to the buffer
                x_buffer.append(x)
                y_buffer.append(y)

                # Calculate weighted average for smoothing
                weights = np.linspace(1, 2, len(x_buffer))
                smooth_x = int(np.average(x_buffer, weights=weights))
                smooth_y = int(np.average(y_buffer, weights=weights))

                # Add current point to the list
                points.append([smooth_x, smooth_y])

                # If we have enough points, draw a curve
                if len(points) > 3:
                    # Convert points to numpy array
                    points_array = np.array(points, dtype=np.int32)

                    # Draw curved line
                    cv2.polylines(canvas, [points_array], False, (255, 255, 255), 2, cv2.LINE_AA)

                    # Keep only the last 4 points for smooth continuation
                    points = points[-4:]

                # Draw just the index finger tip on the display as a small red circle
                cv2.circle(display, (smooth_x, smooth_y), 5, (0, 0, 255), -1)

    # Combine the canvas with the camera feed
    result = cv2.addWeighted(frame, 1, display, 0.5, 0)

    # Show the result
    cv2.imshow('Hand Drawing', result)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
