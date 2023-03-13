import itertools
import gymnasium

REWARD = {"dense": "Dense", "sparse": "Sparse",
          "reward_shaping": "RewardShaping"}
for reward_type, has_object in itertools.product(["dense", "sparse", "reward_shaping"], [True, False]):
    kwargs = {
        "reward_type": reward_type,
        "has_object": has_object
    }
    name = "PickAndPlaceEnv" if has_object else "ReachObjectEnv"
    gymnasium.register(f"{name}-{REWARD[reward_type]}-v0",
                       entry_point="mycobotgym.envs.pick_and_place:PickAndPlaceEnv", kwargs=kwargs, max_episode_steps=200)
