import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Get the frame dimensions
ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]

# Create a blank canvas
canvas = np.ones((frame_height, frame_width, 3), np.uint8) * 255

# Previous coordinates of the index finger
prev_x, prev_y = 0, 0

# Buffer for smoothing
smooth_factor = 5
x_buffer = deque(maxlen=smooth_factor)
y_buffer = deque(maxlen=smooth_factor)


# New variables for interpolation
prev_smooth_x, prev_smooth_y = 0, 0
interpolation_steps = 4

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and find hands
    results = hands.process(rgb_frame)
    
    # Create a copy of the canvas for this frame
    display = canvas.copy()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of the index finger tip (landmark 8)
            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * frame_width), int(index_finger.y * frame_height)
            
            # Add current coordinates to the buffer
            x_buffer.append(x)
            y_buffer.append(y)
            
            # Calculate weighted average
            weights = np.linspace(1, 2, len(x_buffer))
            smooth_x = int(np.average(x_buffer, weights=weights))
            smooth_y = int(np.average(y_buffer, weights=weights))
            
            if prev_smooth_x != 0 and prev_smooth_y != 0:
                for i in range(1, interpolation_steps + 1):
                    alpha = i / (interpolation_steps + 1)
                    interpx = int(prev_smooth_x * (1 - alpha) + smooth_x * alpha)
                    interpy = int(prev_smooth_y * (1 - alpha) + smooth_y * alpha)
                    cv2.line(canvas, (prev_smooth_x, prev_smooth_y), (interpx, interpy), (0, 0, 0), 2)
                    prev_smooth_x, prev_smooth_y = interpx, interpy
            
            prev_smooth_x, prev_smooth_y = smooth_x, smooth_y
            
            # Draw just the index finger tip on the display as a small red circle
            cv2.circle(display, (smooth_x, smooth_y), 7, (0, 0, 255), -1)
            
    if cv2.waitKey(1) & 0xFF == ord('w'):
        canvas = np.ones((frame_height, frame_width, 3), np.uint8) * 255
    
    # Display the result
    cv2.imshow("Drawing Canvas", display)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
