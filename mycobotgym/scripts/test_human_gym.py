import numpy as np
from PIL import Image
import gymnasium as gym
import mujoco
import mycobotgym.envs


# env = gym.make("FrankaKitchen-v1", render_mode="human")
env = gym.make("PickAndPlaceEnv-v0", render_mode="human")
observation = env.reset(seed=42)
for i in range(1000):
    action = env.action_space.sample()  # User-defined policy function
    observation, reward, terminated, truncated, info = env.step(action)
    if i % 100 == 0:
        observation = env.reset()
    # frame = env.render(mode="rgb_array")
    # img = Image.fromarray(frame)
    # img.show()
env.close()
