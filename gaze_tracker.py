import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Eye corners (green dots)
            for idx in [33, 133, 362, 263]:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Gaze direction (using iris landmark 468)
            iris_x = int(face_landmarks.landmark[468].x * frame.shape[1])
            left_eye_left = int(face_landmarks.landmark[33].x * frame.shape[1])
            left_eye_right = int(face_landmarks.landmark[133].x * frame.shape[1])

            # Draw iris position
            cv2.circle(frame, (iris_x, 100), 5, (255, 0, 0), -1)

            # Calculate gaze ratio
            eye_width = left_eye_right - left_eye_left
            relative_x = iris_x - left_eye_left
            gaze_ratio = relative_x / (eye_width + 1e-6)

            # Estimate gaze direction
            if gaze_ratio < 0.35:
                direction = "Looking Left"
            elif gaze_ratio > 0.65:
                direction = "Looking Right"
            else:
                direction = "Looking Center"

            # Show result
            cv2.putText(frame, direction, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Gaze Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
