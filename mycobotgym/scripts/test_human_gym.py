import numpy as np
from PIL import Image
import gymnasium as gym
import mujoco
import mycobotgym.envs
from gymnasium_robotics.utils import mujoco_utils


def print_contacts(model, data):
    for geom1, geom2 in zip(data.contact.geom1, data.contact.geom2):
        body1_id = model.geom_bodyid[geom1]
        body2_id = model.geom_bodyid[geom2]
        body1_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
        body2_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
        print(body1_name, body2_name)


# env = gym.make("FetchPickAndPlace-v2", render_mode="human")
# env = gym.make("FrankaKitchen-v1", render_mode="human", ik_controller=False)
# env = gym.make("ReachObjectEnv-Dense-v0", render_mode="human",
#                controller_type="IK", has_object=True)
env = gym.make("PickAndPlace-v1", render_mode="human")
observation = env.reset(seed=42)
for i in range(10000):
    env.render()
    # env.render()
    # action = env.action_space.sample()  # User-defined policy function
    # observation, reward, terminated, truncated, info = env.step(action)
    # env.data.ctrl = np.array(action)
    # mujoco.mj_step(env.model, env.data, 20)
    # env.render()

    # action = np.random.choice([1, -1], size=7)

    # action *= 2
    # observation, reward, terminated, truncated, info = env.step(action)
    if i % 100 == 0:
        env.reset()

# action = env.action_space.sample()  # User-defined policy function
# observation, reward, terminated, truncated, info = env.step(action)
# if i % 100 == 0:
#         observation = env.reset()
#     # frame = env.render(mode="rgb_array")
#     # img = Image.fromarray(frame)
#     # img.show()
env.close()
