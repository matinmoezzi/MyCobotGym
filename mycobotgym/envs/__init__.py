import gymnasium

gymnasium.register("PickAndPlaceEnv-v0",
                   entry_point="mycobotgym.envs.pick_and_place:PickAndPlaceEnv")
