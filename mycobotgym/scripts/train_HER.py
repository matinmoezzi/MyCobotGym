import gymnasium
import sys
sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC, HerReplayBuffer
import mycobotgym.envs

env = gymnasium.make("PickAndPlaceEnv-v0")

model = SAC("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(
    n_sampled_goal=4,
    goal_selection_strategy="future",
    online_sampling=True,
    max_episode_length=20), verbose=1,)

model.learn(1000)

env.close()
