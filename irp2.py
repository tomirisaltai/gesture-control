import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands and Pose
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

# Define the output CSV file
output_csv = 'irp2.csv'

# Function to get the next sequence_id from the existing CSV file
def get_next_sequence_id(csv_file):
    if not os.path.exists(csv_file):
        return 0
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        lines = list(reader)
        if len(lines) < 2:  # Only header or empty file
            return 0
        last_sequence_id = int(lines[-1][1])
        return last_sequence_id + 1

# Get the next sequence_id
sequence_id = get_next_sequence_id(output_csv)

# Check if file exists; if not, create and write header
if not os.path.exists(output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Define the header
        header = ['gesture_id', 'sequence_id'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)]
        header += [f'pose_x{i}' for i in range(33)] + [f'pose_y{i}' for i in range(33)] + [f'pose_z{i}' for i in range(33)]
        writer.writerow(header)

# Define a gesture ID
gesture_id = 0

# Recording state
recording = False
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame to detect hands and pose
    hand_result = hands.process(rgb_frame)
    pose_result = pose.process(rgb_frame)

    if hand_result.multi_hand_landmarks and pose_result.pose_landmarks and recording:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        # Draw pose landmarks on the frame
        mp_drawing.draw_landmarks(
            frame,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )

        # Extract keypoints and save to CSV
        for hand_landmarks in hand_result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            row = [gesture_id, sequence_id]
            for landmark in landmarks:
                row.extend([landmark.x, landmark.y, landmark.z])
            for landmark in pose_result.pose_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])
            with open(output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            
            frame_count += 1

            if frame_count >= 30:  # Stop recording after 30 frames
                recording = False
                frame_count = 0
                sequence_id += 1
                print(f"Recording stopped for sequence_id {sequence_id - 1}")

    # Display the frame
    cv2.imshow('Hand and Arm Pose Estimation', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('s') and not recording:  # Press 's' to start recording
        recording = True
        print(f"Recording started for sequence_id {sequence_id}")

cap.release()
cv2.destroyAllWindows()
