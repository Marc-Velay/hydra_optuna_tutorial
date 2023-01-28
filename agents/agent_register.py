from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3


AGENTS = {
    "ddpg": DDPG,
    "ppo": PPO,
}