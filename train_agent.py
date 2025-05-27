import gym
from gym import spaces
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque
import random
import matplotlib.pyplot as plt

# --- Environment ---
class GazeEnv(gym.Env):
    def __init__(self):
        super(GazeEnv, self).__init__()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)

        # Observation: 64x64 grayscale frame
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)

        # Actions: 0 = Left, 1 = Center, 2 = Right
        self.action_space = spaces.Discrete(3)

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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        reward = 0
        done = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                true_gaze = self._get_gaze_direction(face_landmarks.landmark, frame.shape)
                reward = 1 if action == true_gaze else -1
                break

        # Show current action on frame
        cv2.putText(frame, f"Action: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Training View", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
            done = True

        # Preprocess frame to grayscale 64x64
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        obs = np.expand_dims(resized, axis=-1)

        return obs, reward, done, {}

    def reset(self):
        # Return initial observation (center gaze by default)
        obs, _, _, _ = self.step(1)
        return obs

    def render(self, mode='human'):
        pass  # Not used

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


# --- Build Model ---
def build_model(input_shape, action_size):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    return model


# --- Training Loop ---
def train():
    EPISODES = 20
    MAX_MEMORY = 5000
    BATCH_SIZE = 32
    GAMMA = 0.95
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01

    env = GazeEnv()
    input_shape = env.observation_space.shape
    action_size = env.action_space.n
    model = build_model(input_shape, action_size)

    memory = deque(maxlen=MAX_MEMORY)
    epsilon = 1.0
    reward_log = []

    for episode in range(EPISODES):
        state = env.reset() / 255.0
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() <= epsilon:
                action = random.randrange(action_size)
            else:
                q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
                action = np.argmax(q_values[0])

            next_state, reward, done, _ = env.step(action)
            next_state = next_state / 255.0

            memory.append((state, action, reward, next_state))
            state = next_state
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, targets = [], []

                for s, a, r, ns in batch:
                    target = r + GAMMA * np.amax(model.predict(np.expand_dims(ns, axis=0), verbose=0)[0])
                    target_f = model.predict(np.expand_dims(s, axis=0), verbose=0)[0]
                    target_f[a] = target
                    states.append(s)
                    targets.append(target_f)

                model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        reward_log.append(total_reward)
        print(f"Episode {episode+1}/{EPISODES} — Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    model.save("gaze_model.h5")
    print("✅ Model saved as gaze_model.h5")

    plt.plot(reward_log)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.grid()
    plt.savefig("reward_plot.png")
    plt.show()

    env.close()


if __name__ == "__main__":
    train()
