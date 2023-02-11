from dm_control.utils import inverse_kinematics
from dm_control import mujoco
from dm_control.mujoco import wrapper

import gymnasium as gym
from gymnasium_robotics.utils import mujoco_utils
import mycobotgym.envs

env = gym.make("PickAndPlaceEnv-v0", render_mode="human")
observation = env.reset(seed=4)

model_wrapper = wrapper.MjModel(env.model)
physics = mujoco.Physics.from_model(model_wrapper)

for i in range(10000):
    object_pos = mujoco_utils.get_site_xpos(
        env.model, env.data, "object0").copy()

    qpos, err, steps, success = inverse_kinematics.qpos_from_site_pose(
        physics, "EEF", object_pos, [1, 0, 1, 0])

    if not success:
        print("not success")
        exit(-1)
    print(err)

    env.data.qpos = qpos
    for _ in range(10):
        mujoco.mj_step(env.model, env.data, nstep=5)

    eef_xpos = mujoco_utils.get_site_xpos(
        env.model, env.data, "EEF").copy()
    env.render()
    # if i % 100 == 0:
    #     env.reset()


env.close()
