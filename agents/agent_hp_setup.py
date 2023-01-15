from omegaconf import DictConfig, open_dict

import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn


def conf_ddpg(cfg: DictConfig):
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[cfg.agent.activation]

    net_arch = {
        "s": [32, 16],
        "m": [64, 32],
        "l": [64, 32, 16],
    }[cfg.agent.net_arch]

    hyperparams = {
        "policy": cfg.agent.policy,
        "gamma": cfg.agent.gamma,
        "tau": cfg.agent.tau,
        "learning_rate": cfg.agent.learning_rate,
        "batch_size": cfg.agent.batch_size,
        "buffer_size": int(cfg.agent.buffer_size),
        "train_freq": cfg.agent.train_freq,
        "gradient_steps": cfg.agent.gradient_steps,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn
        ),
    }

    if cfg.agent.noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(cfg.environment.n_actions), sigma=cfg.agent.noise_std * np.ones(cfg.environment.n_actions)
        )
    elif cfg.agent.noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(cfg.environment.n_actions), sigma=cfg.agent.noise_std * np.ones(cfg.environment.n_actions)
        )
    
    return hyperparams

AGENTS_HP = {
    "ddpg": conf_ddpg,
}