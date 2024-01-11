import itertools
import gymnasium
from gymnasium import register

REWARD = {"dense": "Dense", "sparse": "Sparse", "reward_shaping": "RewardShaping"}
for reward_type, has_object, controller, fetch in itertools.product(
    ["dense", "sparse", "reward_shaping"],
    [True, False],
    ["mocap", "IK", "joint", "delta_joint"],
    [True, False],
):
    model_path = f"./assets/mycobot280{'_mocap' if controller == 'mocap' else ''}.xml"
    kwargs = {
        "model_path": model_path,
        "reward_type": reward_type,
        "has_object": has_object,
        "controller_type": controller,
        "fetch_env": fetch,
    }

    if fetch:
        if controller in ["joint", "delta_joint"]:
            continue

    fetch_env = "Fetch" if fetch else ""
    name = (
        f"MyCobot{fetch_env}PickAndPlace" if has_object else f"MyCobot{fetch_env}Reach"
    )
    gymnasium.register(
        f"{name}-{REWARD[reward_type]}-{controller}-v0",
        entry_point="mycobotgym.envs.mycobot:MyCobotEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
    gymnasium.register(
        f"{name}-{REWARD[reward_type]}-{controller}-v1",
        entry_point="mycobotgym.envs.mycobot:MyCobotImgEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
