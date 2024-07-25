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
canvas = np.zeros((frame_height, frame_width, 3), np.uint8)

# Previous coordinates of the index finger
prev_x, prev_y = 0, 0

# Buffer for smoothing
x_buffer = deque(maxlen=5)
y_buffer = deque(maxlen=5)

# Points for curve drawing
points = []

def is_fist(hand_landmarks):
    index_tip = hand_landmarks.landmark[8]
    thumb_base = hand_landmarks.landmark[2]
    
    # Calculate the Euclidean distance between the index finger tip and thumb base
    distance = np.sqrt((index_tip.x - thumb_base.x) ** 2 + (index_tip.y - thumb_base.y) ** 2)
    
    if distance < 0.05:
        return True
    
    return False

def show_flash_and_text(display, message="Screenshot saved!", duration=0.5):
    # White flash
    flash_screen = np.full_like(display, 255)
    cv2.imshow('Hand Drawing', flash_screen)
    cv2.waitKey(1)  # Display the flash screen for a short moment

def is_spock(hand_landmarks):
    # Define landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Check for Spock sign: index finger and middle finger should be together and ring and pinky should be together, 
    # and the thumb should be raised. There should be a gap between the middle and ring fingers.
    #Euclidean distance between the middle and ring finger tips
    distance = np.sqrt((middle_tip.x - ring_tip.x) ** 2 + (middle_tip.y - ring_tip.y) ** 2)
    #Euclidean distance between the index and middle finger tips
    distance1 = np.sqrt((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2)
    #Euclidean distance between the ring and pinky finger tips
    distance2 = np.sqrt((ring_tip.x - pinky_tip.x) ** 2 + (ring_tip.y - pinky_tip.y) ** 2)
    
    if distance > 0.05 and distance1 < 0.05 and distance2 < 0.05:
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
            elif is_spock(hand_landmarks):
                show_flash_and_text(display, "Spock sign detected!")
                cv2.imwrite('spock_canvas.png', canvas)
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
                if len(points) > 2:
                    # Convert points to numpy array
                    points_array = np.array(points, dtype=np.int32)

                    # Draw curved line
                    cv2.polylines(canvas, [points_array], False, (255, 255, 255), 2, cv2.LINE_AA)

                    # Keep only the last 4 points for smooth continuation
                    points = points[-4:]

                # Draw just the index finger tip on the display as a small red circle
                cv2.circle(display, (smooth_x, smooth_y), 5, (0, 0, 255), -1)

    # Combine the canvas with the camera feed
    result = cv2.addWeighted(frame, 0.5, display, 1, 0)

    # Show the result
    cv2.imshow('Hand Drawing', result)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        show_flash_and_text(display)
        cv2.imwrite('hand_drawing.png', canvas)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
