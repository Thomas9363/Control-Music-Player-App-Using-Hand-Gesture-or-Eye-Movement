import cv2
import mediapipe as mp
import numpy as np
import math
import time
eye_move=""
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# function calculate FPS
def calculate_fps(prev_time, prev_fps):
    current_time = time.time()
    fps = 0.9*prev_fps+ 0.1*(1 / (current_time - prev_time))
    return fps, current_time

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

# Function to calculate moving ratio
def iris_position(iris_center, point1, point2):
    center_to_point1 = euclidean_distance(iris_center, point1)
    center_to_point2 = euclidean_distance(iris_center, point2)
    point1_to_point2 = euclidean_distance(point1, point2)
    ratio = center_to_point1 / point1_to_point2
    return ratio

# Initialize Video Capture
cap = cv2.VideoCapture(0)
prev_time = time.time()
prev_fps=0

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #        frame =cv2.flip(frame,0)
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output', 640, 480)
        cv2.moveWindow('output', 300, 100)
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark  # get the landmarks of the first face
            points = [(landmark.x, landmark.y) for landmark in landmarks]  # extract the x and y coordinates
            p = np.array(  # convert landmarks to a numpy array  and scale it using np.multiply
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in landmarks])
            if len(points) > 2:  # key landmarks around right eye. They are used to calculate movement
                ratioRH = iris_position(p[468], p[33], p[133])
                #                 print ("{:.2f}".format(ratioH))
                eye_width_R = euclidean_distance(p[33], p[133])
                eye_width_L = euclidean_distance(p[362], p[263])
                eye_height_R = euclidean_distance(p[159], p[145])
                eye_height_L = euclidean_distance(p[386], p[374])
                ratioROC = eye_height_R / eye_width_R
                ratioLOC = eye_height_L / eye_width_L

                mouth_width = euclidean_distance(p[78], p[308])
                mouth_height = euclidean_distance(p[13], p[14])
                ratioM = mouth_height / mouth_width
                # print(ratioM)

                if ratioRH > 0.7:  # eye move to left
                    eye_move = "prev"  # eye move left
                elif ratioRH <= 0.7 and ratioRH >= 0.4:  # eye at center
                    if (ratioROC < 0.2) and (ratioLOC > 0.2) and (ratioM < 0.1):  # right eye and mouth close
                        eye_move = "play"  # right eye close
                    elif (ratioROC < 0.2) and (ratioLOC > 0.2) and (ratioM >= 0.1):  # right eye close, mouth open
                        eye_move = "volume up"  # right eye close, mouth open
                    elif (ratioROC > 0.2) and (ratioLOC < 0.2) and (ratioM < 0.1):  # left eye and mouth close
                        eye_move = "pause"  # left eye and mouth close
                    elif (ratioROC > 0.2) and (ratioLOC < 0.2) and (ratioM >= 0.1):  # left eye close, mouth open
                        eye_move = "volume down"  # left eye close, mouth open
                    elif (ratioROC < 0.2) and (ratioLOC < 0.2) and (ratioM < 0.1):  # both eyes close
                        eye_move = "stop"  # both eyes close
                    else:  # eyes center
                        eye_move = "nothing"  # eyes center
                elif ratioRH < 0.4:  # eye move right
                    eye_move = "next"  # eye move right

                cv2.circle(frame, p[159], 1, (0, 255, 0), -1)
                cv2.circle(frame, p[145], 1, (0, 255, 0), -1)
                cv2.circle(frame, p[33], 1, (0, 255, 0), -1)
                cv2.circle(frame, p[133], 1, (0, 255, 0), -1)

                cv2.circle(frame, p[386], 1, (0, 255, 0), -1)
                cv2.circle(frame, p[374], 1, (0, 255, 0), -1)
                cv2.circle(frame, p[362], 1, (0, 255, 0), -1)
                cv2.circle(frame, p[263], 1, (0, 255, 0), -1)

                cv2.circle(frame, p[78], 1, (0, 255, 0), -1)
                cv2.circle(frame, p[308], 1, (0, 255, 0), -1)
                cv2.circle(frame, p[13], 1, (0, 255, 0), -1)
                cv2.circle(frame, p[14], 1, (0, 255, 0), -1)

                cv2.circle(frame, p[468], 2, (255, 255, 255), -1)

                cv2.putText(frame, f'eye: {eye_move}', (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)

        fps, prev_time = calculate_fps(prev_time, prev_fps)  # Calculate and display FPS
        prev_fps = fps
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('output', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
