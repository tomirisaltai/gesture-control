import cv2
import mediapipe as mp
import csv
import os

#MediaPipe Hands Solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

output_csv = 'hand_landmark1.csv'

def get_next_sequence_id(csv_file):
    if not os.path.exists(csv_file):
        return 0
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        lines = list(reader)
        if len(lines) < 2:
            return 0
        last_sequence_id = int(lines[-1][1])
        return last_sequence_id + 1
sequence_id = get_next_sequence_id(output_csv)

#Checking the file
if not os.path.exists(output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['gesture_id', 'sequence_id'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)]
        writer.writerow(header)

#Gesture ID, changed everytime a new gesture is recorded, 0, 1, 2, and 3
gesture_id = 3

recording = False
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks and recording:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            landmarks = hand_landmarks.landmark
            row = [gesture_id, sequence_id]
            for landmark in landmarks:
                row.extend([landmark.x, landmark.y, landmark.z])
            with open(output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            
            frame_count += 1

            if frame_count >= 30:  # 30 frames per each gesture
                recording = False
                frame_count = 0
                sequence_id += 1
                print(f"Recording stopped for sequence_id {sequence_id - 1}")
    cv2.imshow('Hand Pose Estimation', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('s') and not recording:  #'s' button to start recording
        recording = True
        print(f"Recording started for sequence_id {sequence_id}")

cap.release()
cv2.destroyAllWindows()
