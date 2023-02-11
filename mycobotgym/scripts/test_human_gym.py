import numpy as np
from PIL import Image
import gymnasium as gym
import mujoco
import mycobotgym.envs
from gymnasium_robotics.utils import mujoco_utils


# env = gym.make("FrankaKitchen-v1", render_mode="human")
env = gym.make("PickAndPlaceEnv-v0", render_mode="human")
observation = env.reset(seed=42)
for _ in range(10000):
    object_pos = mujoco_utils.get_site_xpos(
        env.model, env.data, "object0").copy()
    current_eef_pose = env.data.site_xpos[
        env.model_names.site_name2id["EEF"]
    ].copy()
    # object_pos = np.array([1, 1, 1])
    displacement = (object_pos - current_eef_pose) / 0.2
    action = np.zeros(7)
    action[:3] = displacement
    env.step(action)
    env.render()
# for i in range(1000):
#     action = env.action_space.sample()  # User-defined policy function
#     observation, reward, terminated, truncated, info = env.step(action)
#     if i % 100 == 0:
#         observation = env.reset()
#     # frame = env.render(mode="rgb_array")
#     # img = Image.fromarray(frame)
#     # img.show()
env.close()
