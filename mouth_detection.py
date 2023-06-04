import cv2
import dlib
import numpy as np

cap = cv2.VideoCapture(0)

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# Arbitrary threshold
MOUTH_OPEN_THRESHOLD = 20

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        top_lip = (landmarks.part(62).x, landmarks.part(62).y)
        bottom_lip = (landmarks.part(66).x, landmarks.part(66).y)

        # Measure vertical distance between top and bottom lip
        lip_distance = np.linalg.norm(np.array(top_lip) - np.array(bottom_lip))

        if lip_distance > MOUTH_OPEN_THRESHOLD:
            # Apply filter effect: Turn frame into sepia
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_PINK)
            
        # Draw two red dots on the edge of your live mouth
        left_lip = (landmarks.part(48).x, landmarks.part(48).y)
        right_lip = (landmarks.part(54).x, landmarks.part(54).y)
        cv2.circle(frame, left_lip, 3, (0,0,255), -1)
        cv2.circle(frame, right_lip, 3, (0,0,255), -1)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
