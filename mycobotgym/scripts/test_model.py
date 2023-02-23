from stable_baselines3 import SAC, HerReplayBuffer, TD3, PPO, DDPG, A2C
import gymnasium
import mycobotgym.envs
import sys
sys.modules["gym"] = gymnasium

if __name__ == "__main__":
    path = sys.argv[1]
    model_config = path.rsplit("/", 1)[-1]
    env_id = "-".join(model_config.split("-")[:3])
    env = gymnasium.make(env_id, render_mode="human")

    model = SAC.load(sys.argv[1], env)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        print("reward: ", rewards)
