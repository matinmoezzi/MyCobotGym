import datetime
import gymnasium
import sys
sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC, HerReplayBuffer, TD3, PPO, DDPG
import mycobotgym.envs
import argparse

parser = argparse.ArgumentParser(
    prog="MyCobot-ReachObjectEnv-Dense-v0-SAC-HER")
parser.add_argument("-t", "--total-timesteps", type=int, default=1000)
parser.add_argument("-o", "--output-dir", type=str,
                    default="./trained_models/")
parser.add_argument("--no-her", action='store_false')
parser.add_argument("--algo", type=str, default="SAC",
                    choices=["SAC", "DDPG", "TD3", "PPO"])
parser.add_argument("-c", "--controller-type", type=str,
                    default="joint", choices=["joint", "mocap", "IK"])
parser.add_argument("--human", action="store_true")
parser.add_argument("-e", "--env", type=str, default="ReachObjectEnv-Dense-v0", choices=[
                    "ReachObjectEnv-Dense-v0", "ReachObjectEnv-Sparse-v0", "PickAndPlaceEnv-Dense-v0", "PickAndPlaceEnv-Sparse-v0",])
args = parser.parse_args()

algos = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG, "SAC": SAC}
replay_buffer_cls = None
replay_buffer_dict = None
if not args.no_her and args.algo in ["DDPG", "TD3", "SAC"]:
    replay_buffer_cls = HerReplayBuffer
    replay_buffer_dict = dict(
        n_sampled_goal=4, goal_selection_strategy="future", online_sampling=True, max_episode_length=100)

render_mode = "human" if args.human else None

env = gymnasium.make(
    args.env, controller_type=args.controller_type, render_mode=render_mode)

model = algos[args.algo]("MultiInputPolicy", env, replay_buffer_class=replay_buffer_cls,
                         replay_buffer_kwargs=replay_buffer_dict)
model.learn(args.total_timesteps, progress_bar=True)

model.save(args.output_dir +
           f"{args.env}-{args.total_timesteps:_d}-{args.algo}-{'HER' if not args.no_her else ''}-{args.controller_type}-{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

env.close()
