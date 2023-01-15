from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3


AGENTS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}