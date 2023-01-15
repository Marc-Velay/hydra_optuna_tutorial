import os
import time
import random

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import gym
import torch as th
from stable_baselines3.common.utils import set_random_seed

from agents.agent_callbacks import create_callbacks, TrialEvalCallback, make_eval_env

from exp_manager import set_config, make_env, make_agent


os.environ['HYDRA_FULL_ERROR'] = '1'


# Configure the setting around the experiment, mostly misc for reproducibility
#
# cfg:          config file from hydra, passed through command line, interpreted in @hydra.main
# root_path:    Generally where you called train_agent from, root of the project. 
#               This facilitates relative paths in different systems (user, mesocentre, ...)
def conf_exp_manager(cfg: DictConfig, root_path):
    registered_envs = set(gym.envs.registry.env_specs.keys())
    if cfg.environment.env_id not in registered_envs:
        print("Invalid environment selected")
        raise NotImplementedError
    
    #Set experiment seed: rely on SB3 to set torch, np seeds
    if cfg.experiment.seed:
        set_random_seed(cfg.experiment.seed)

    cfg = set_config(cfg, root_path)

    env = make_env(cfg)
    agent = make_agent(cfg, env)

    return agent


def model_learn(cfg, model):
    reward = None
    kwargs = {}
    if cfg.experiment.log_interval > -1:
            kwargs = {"log_interval": cfg.experiment.log_interval}

    callbacks = create_callbacks(cfg)
    if cfg.experiment.multirun:
        eval_env = make_eval_env(cfg)
        eval_callback = TrialEvalCallback(
            eval_env,
            best_model_save_path=cfg.environment.run_dir,
            log_path=cfg.experiment.log_folder,
            n_eval_episodes=cfg.experiment.n_eval_episodes,
            eval_freq=cfg.experiment.optuna_eval_freq,
            deterministic=True,
        )
        callbacks.append(eval_callback)

    if len(callbacks) > 0:
        #with open_dict(cfg):
        #    cfg.experiment.callbacks = callbacks
        kwargs["callback"] = callbacks #cfg.experiment.callbacks

    model.learn(cfg.experiment.max_timesteps, **kwargs)

    model.env.close()
    if cfg.experiment.multirun:
        reward = eval_callback.last_mean_reward
    
    return reward


# Starting point for training agents.
# Decorator captures console arguments where config paths are provided.
# Based on hydra configuration manager. Creates an outputs directory for each run.
# Example command: 
#       python train_agent.py 
#       python train_agent.py -m
@hydra.main(version_base="1.2", config_path='./configs', config_name='default')
def train(cfg: DictConfig):
    root_path = os.getcwd()
    print(OmegaConf.to_yaml(cfg))

    agent = conf_exp_manager(cfg, root_path)
    
    reward = model_learn(cfg, agent)
    
    # Return is for optuna sweeper, in hydra decorator
    return reward
    
    

if __name__ == "__main__":
  train()
