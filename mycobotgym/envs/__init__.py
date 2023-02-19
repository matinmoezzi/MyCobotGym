import itertools
import gymnasium

for reward_type, has_object in itertools.product(["dense", "sparse"], [True, False]):
    reward = "Dense" if reward_type == "dense" else "Sparse"
    kwargs = {
        "reward_type": reward_type,
        "has_object": has_object
    }
    name = "PickAndPlaceEnv" if has_object else "ReachObjectEnv"
    gymnasium.register(f"{name}-{reward}-v0",
                       entry_point="mycobotgym.envs.pick_and_place:PickAndPlaceEnv", kwargs=kwargs, max_episode_steps=500)
