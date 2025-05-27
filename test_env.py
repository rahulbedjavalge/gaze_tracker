from gaze_env import GazeEnv
env = GazeEnv()

obs = env.reset()
for _ in range(10):
    action = env.action_space.sample()  # Random guess
    obs, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
env.close()
