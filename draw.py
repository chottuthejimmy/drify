import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a blank canvas
canvas = np.zeros((480, 640, 3), np.uint8)

# Previous coordinates of the index finger
prev_x, prev_y = 0, 0

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
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of the index finger tip (landmark 8)
            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])
            
            # Draw on the canvas
            if prev_x != 0 and prev_y != 0:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 2)
            
            prev_x, prev_y = x, y
            
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    
    if frame.shape != canvas.shape:
    # Resize canvas to match the dimensions of frame
        canvas = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))
    
    # Combine the frame and the canvas
    combined = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
    
    # Display the result
    cv2.imshow("Drawing Canvas", combined)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# import cv2

# # Assuming frame and canvas are already defined
# # Check if the dimensions of frame and canvas match
# if frame.shape != canvas.shape:
#     # Resize canvas to match the dimensions of frame
#     canvas = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))

# # Now call cv2.addWeighted
# combined = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
