import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
def calculate_fps(prev_time, prev_fps):
    current_time = time.time()
    fps = 0.9 * prev_fps + 0.1 * (1 / (current_time - prev_time))
    return fps, current_time

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='iris_gesture_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Define the eye and iris landmark indices, including iris centers
right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
left_iris_indices = [474, 475, 476, 477, 473]  # Include iris center point 468
right_iris_indices = [469, 470, 471, 472, 468]  # Include iris center point 473

# Combine all the indices
eye_iris_indices = left_eye_indices + left_iris_indices +right_eye_indices +  right_iris_indices

# Function to normalize landmarks
def normalize_landmarks(landmarks, indices):
    # Compute the midpoint between the two eyes
    left_eye_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in left_eye_indices], axis=0)
    right_eye_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in right_eye_indices], axis=0)
    mid_point = (left_eye_center + right_eye_center) / 2.0

    normalized = np.array([[landmarks[i].x - mid_point[0], landmarks[i].y - mid_point[1]] for i in indices])
    return normalized.flatten()


# Labels for gestures
labels = labels = ['volume up', 'volume down', 'next', 'prev', 'nothing', 'play', 'pause', 'stop']
# labels = labels = ['0', '1', '2', '3', '4', '5', '6', '7']
# labels = ['up', 'down', 'right', 'left', 'center', 'right close', 'left close', 'both close']
# Set the desired frame rate limit (in FPS)

prev_time = time.time()
prev_fps = 0
# Start capturing video from the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB as MediaPipe expects RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find faces and iris landmarks
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Draw the face landmarks on the frame
            # mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_IRISES)

            # Normalize the landmarks
            normalized_landmarks = normalize_landmarks(face_landmarks.landmark, eye_iris_indices)
            input_data = np.array(normalized_landmarks, dtype=np.float32).reshape(1, -1)

            # Set the tensor to point to the input data to be inferred
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            interpreter.invoke()

            # Get the result
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_label = np.argmax(output_data[0])
            print (labels[predicted_label])
            # Display the predicted gesture
            cv2.putText(frame, f'eye: {labels[predicted_label]}', (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)
        # Calculate and display FPS
        fps, prev_time = calculate_fps(prev_time, prev_fps)
        prev_fps = fps
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Iris Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
