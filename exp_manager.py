import os
import time
from omegaconf import DictConfig, open_dict
from hydra.core.hydra_config import HydraConfig

from stable_baselines3.common.env_util import make_vec_env

from agents.agent_register import AGENTS
from agents.agent_hp_setup import AGENTS_HP

def set_config(cfg: DictConfig, root_path) -> DictConfig:
    # Create exp name, used to save logs and model weights
    run_name = f"{cfg.environment.env_id}__{cfg.agent.learning_algo}__{cfg.experiment.seed}__{int(time.time())}"

    # Manually exit the hydra working dir, by going up parent folders until root
    # Project root is folder name of project, here repo name
    base_dir = root_path
    while "hydra_optuna_tutorial" not in base_dir.split("/")[-1].lower():
        base_dir = os.path.dirname(base_dir)
        
    with open_dict(cfg):
        cfg.experiment.multirun = HydraConfig.get().mode.value == 2 # Enum: 1 == RUN, 2 == MULTIRUN
        cfg.environment.run_dir = HydraConfig.get().runtime.output_dir
        cfg.experiment.tensorboard_log = os.path.join(root_path, f"tboard_runs/{run_name}")
        cfg.experiment.optimization_log_path = os.path.join(root_path, cfg.experiment.optimization_log_path)
        cfg.experiment.log_folder = os.path.join(root_path, cfg.experiment.log_folder, run_name)
        cfg.experiment.study_name = f"{run_name}"
        cfg.base_dir = base_dir

    return cfg


def make_env(cfg: DictConfig):
    env = make_vec_env(cfg.environment.env_id, n_envs=cfg.experiment.n_envs)
    return env


def make_agent(cfg: DictConfig, env):
    with open_dict(cfg):
        cfg.environment.n_actions = env.action_space.shape[0]
        
    agent_hp = AGENTS_HP[cfg.agent.learning_algo](cfg)

    agent_seed = None
    if "seed" in cfg.agent:
        agent_seed = cfg.agent.seed

    agent = AGENTS[cfg.agent.learning_algo](
            env=env,
            tensorboard_log=cfg.experiment.tensorboard_log,
            seed=agent_seed,
            verbose=cfg.experiment.verbose,
            device=cfg.experiment.device,
            **agent_hp, # Each key-value pair in dict agent_hp passed as a single parameter
        )

    return agent