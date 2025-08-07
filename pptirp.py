import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import pyautogui
import time

#Loading the model of choice
model = load_model('rnn1.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
gesture_labels = ['Swipe Right', 'Swipe Left', 'Full Screen', 'Window Screen']

#Help 30 frames in acordance to the training
sequence = deque(maxlen=30)
last_gesture_name = None
last_gesture_confidence = 0
last_gesture_time = time.time()
cap = cv2.VideoCapture(0)

def control_presentation(gesture):
    """Map predicted gestures to PowerPoint control actions."""
    if gesture == 0:
        pyautogui.press('right')  # Next Slide
    elif gesture == 1:
        pyautogui.press('left')  # Previous slide
    elif gesture == 2:
        pyautogui.hotkey('command', 'return')  # Full-screen 
    elif gesture == 3:
        pyautogui.press('esc')  # Window-screen
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image) 
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])
            
            sequence.append(np.array(keypoints).flatten())
            
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            h, w, c = image.shape
            x_min = min([int(landmark.x * w) for landmark in hand_landmarks.landmark])
            y_min = min([int(landmark.y * h) for landmark in hand_landmarks.landmark])
            x_max = max([int(landmark.x * w) for landmark in hand_landmarks.landmark])
            y_max = max([int(landmark.y * h) for landmark in hand_landmarks.landmark])
            
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            if len(sequence) == 30 and (time.time() - last_gesture_time > 5):  # 5 seconds threshold
                input_sequence = np.array(sequence).reshape(1, 30, 63)
                prediction = model.predict(input_sequence)
                gesture_id = np.argmax(prediction)
                last_gesture_name = gesture_labels[gesture_id]
                last_gesture_confidence = np.max(prediction) * 100  # Accuracy percentage

                # Only allow gestures to be performed if the accuracy is above 95%
                if last_gesture_confidence > 95:
                    control_presentation(gesture_id)
                    last_gesture_time = time.time()
    
    if last_gesture_name:
        cv2.putText(image, f'{last_gesture_name} ({last_gesture_confidence:.2f}%)', 
                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Gesture: {last_gesture_name}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Confidence: {last_gesture_confidence:.2f}%', 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Hand Gesture Recognition', image)
    
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
