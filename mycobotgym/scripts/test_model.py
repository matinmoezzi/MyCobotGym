from stable_baselines3 import SAC, TD3, PPO, DDPG, A2C
import gymnasium
import mycobotgym.envs
import sys
sys.modules["gym"] = gymnasium

ALGOS = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG, "TD3": TD3, "A2C": A2C}
if __name__ == "__main__":
    path = sys.argv[1]
    agent = path.rsplit("/", 1)[-1]
    agent_spec = agent.split("-")
    env_id = "-".join(agent_spec[:3])
    algorithm = agent_spec[4]
    controller_type = agent_spec[6] if len(agent_spec) > 10 else agent_spec[5]
    env = gymnasium.make(env_id, render_mode="human",
                         controller_type=controller_type)

    model = ALGOS[algorithm].load(sys.argv[1], env)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        print("reward: ", rewards)
