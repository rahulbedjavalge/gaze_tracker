import numpy as np
import tensorflow as tf
from gaze_env import GazeEnv

MODEL_PATH = "gaze_model.h5"

env = GazeEnv()
model = tf.keras.models.load_model(MODEL_PATH)

print("🧠 Agent running. Press ESC to quit.")
while True:
    state = env.reset() / 255.0
    for _ in range(30):
        q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
        action = np.argmax(q_values[0])

        next_state, reward, done, _ = env.step(action)
        state = next_state / 255.0
        print(f"Agent predicted: {['Left', 'Center', 'Right'][action]} | Reward: {reward}")

        if done:
            break
env.close()
