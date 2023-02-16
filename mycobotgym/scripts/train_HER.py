import gymnasium
import sys
sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC, HerReplayBuffer
import mycobotgym.envs

env = gymnasium.make("PickAndPlaceEnv-v0",
                     render_mode="human", reward_type="dense", controller_type="joint")

model = SAC("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(
    n_sampled_goal=4,
    goal_selection_strategy="future",
    online_sampling=True,
    max_episode_length=100), verbose=2)

model.learn(10_000)

env.close()
