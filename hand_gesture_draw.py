import cv2 as cv
import numpy as np
import mediapipe as mp

# Load the mediapipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load the drawing canvas
canvas = np.zeros((720, 1080, 3), np.uint8)

# Set up the video capture
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally
    frame = cv.flip(frame, 1)
    # Convert the BGR frame to RGB
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Process the frame to get the hand landmarks
    results = hands.process(rgb_frame)
    # Get the text for left and right hands
    text = ['Left Hand', 'Right Hand']
    
    # Draw the landmarks and get the hand positions
    if results.left_hand_landmarks:
        for hand_landmarks in results.left_hand_landmarks.landmark:
            # Get the height, width, and channels of the frame
            h, w, c = frame.shape
            # Get the x and y coordinates of the landmarks
            cx, cy = int(hand_landmarks.x * w), int(hand_landmarks.y * h)
            # Draw a circle on the landmark
            cv.circle(frame, (cx, cy), 10, (0, 255, 0), cv.FILLED)
            # Display the text for the left hand
            cv.putText(frame, text[0], (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    
                       
                
     

    # Display the frame and the canvas
    cv.imshow('Frame', frame)
    cv.imshow('Canvas', canvas)

    # Check for the 'q' key to quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the video capture
cap.release()
# Destroy all OpenCV windows
cv.destroyAllWindows()
# Release the mediapipe hands model
hands.close()
# Save the canvas as an image
cv.imwrite('hand_gesture.png', canvas)
