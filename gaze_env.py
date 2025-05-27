import gym
from gym import spaces
import numpy as np
import cv2
import mediapipe as mp

class GazeEnv(gym.Env):
    def __init__(self):
        super(GazeEnv, self).__init__()
        self.cap = cv2.VideoCapture(1)  # Changed from 0 to 1 - try 1 or 2 if needed
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)

        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)

        self.frame_count = 0
        self.max_frames = 30  # max steps per episode

    def _get_gaze_direction(self, landmarks, frame_shape):
        iris_x = int(landmarks[468].x * frame_shape[1])
        left = int(landmarks[33].x * frame_shape[1])
        right = int(landmarks[133].x * frame_shape[1])

        ratio = (iris_x - left) / (right - left + 1e-6)

        if ratio < 0.35:
            return 0  # Left
        elif ratio > 0.65:
            return 2  # Right
        else:
            return 1  # Center

    def step(self, action):
        ret, frame = self.cap.read()
        if not ret:
            return None, 0, True, {}

        self.frame_count += 1
        done = self.frame_count >= self.max_frames

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        reward = 0
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                true_gaze = self._get_gaze_direction(face_landmarks.landmark, frame.shape)
                reward = 1 if action == true_gaze else -1
                break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        obs = np.expand_dims(resized, axis=-1)

        return obs, reward, done, {}

    def reset(self):
        self.frame_count = 0
        return self.step(1)[0]  # Start with "Center" action

    def render(self, mode='human'):
        pass

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
