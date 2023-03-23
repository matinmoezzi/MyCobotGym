import itertools
import gymnasium

REWARD = {"dense": "Dense", "sparse": "Sparse",
          "reward_shaping": "RewardShaping"}
for reward_type, has_object, controller in itertools.product(["dense", "sparse", "reward_shaping"], [True, False], ["mocap", "IK", "joint"]):
    kwargs = {
        "reward_type": reward_type,
        "has_object": has_object,
        "controller_type": controller
    }
    name = "PickAndPlaceEnv" if has_object else "ReachObjectEnv"
    gymnasium.register(f"{name}-{REWARD[reward_type]}-{controller}-v0",
                       entry_point="mycobotgym.envs.pick_and_place:MyCobotPickAndPlace", kwargs=kwargs, max_episode_steps=100)
