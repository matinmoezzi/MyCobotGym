import numpy as np
from PIL import Image
import gymnasium as gym
import mujoco
import mycobotgym
from gymnasium_robotics.utils import mujoco_utils


def print_contacts(model, data):
    for geom1, geom2 in zip(data.contact.geom1, data.contact.geom2):
        body1_id = model.geom_bodyid[geom1]
        body2_id = model.geom_bodyid[geom2]
        body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
        body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
        print(body1_name, body2_name)


# env = gym.make("FetchReach-v3", render_mode="human")
# env = gym.make("MyCobotFetchReach-v1", render_mode="human")
env = gym.make("MyCobotReach-Sparse-joint-v0", render_mode="human")
# env = gym.make("MyCobotReach-Sparse-IK-v0", render_mode="human")

# env = env.env.env
observation = env.reset(seed=4)
# env = env.env.env
env.render()

action = env.action_space.sample()  # User-defined policy function
for i in range(10000):
    # print(env.data.qpos)
    # print(env.data.qpos)
    # env.render()
    # mujoco.mj_step(env.model, env.data, nstep=20)
    # print(env.data.mocap_quat[0])
    # env.render()

    # Test IK
    # mujoco.mj_forward(env.model, env.data)
    # obj_pos = mujoco_utils.get_site_xpos(env.model, env.data, "target0")
    # grip_pos = mujoco_utils.get_site_xpos(env.model, env.data, "EEF")
    # print(grip_pos)
    # mpos = env.data.mocap_pos[0]
    # print(f"gripper: {grip_pos}, mocap: {mpos}")
    # delta = (obj_pos - grip_pos) / 0.2
    # action = np.zeros(7)
    # action[:3] = delta.copy()

    # if i % 50 == 0:
    #     action = env.action_space.sample()  # User-defined policy function
    #     observation, reward, terminated, truncated, info = env.step(action)
    # env.data.ctrl = np.array(action)
    # mujoco_utils.reset_mocap2body_xpos(env.model, env.data)
    # mujoco.mj_step(env.model, env.data, 10)
    env.render()

    # action = np.random.choice([1, -1], size=7)

    # action *= 2
    # observation, reward, terminated, truncated, info = env.step(action)
    if i % 100 == 0:
        env.reset()

    action = env.action_space.sample()  # User-defined policy function
    observation, reward, terminated, truncated, info = env.step(action)
    # if i % 100 == 0:
    #     observation = env.reset()
#     # frame = env.render(mode="rgb_array")
#     # img = Image.fromarray(frame)
#     # img.show()
env.close()
