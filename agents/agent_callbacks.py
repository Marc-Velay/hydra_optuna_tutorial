from typing import Optional

from omegaconf import open_dict

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = False,
        render: bool = False,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            render=render,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self.eval_idx = 0
        self.mean_rewards = []
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.mean_rewards.append(self.last_mean_reward)
            #self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            #if self.trial.should_prune():
            #    self.is_pruned = True
            #    return False
        return True

def make_eval_env(cfg):
    env = make_vec_env(cfg.environment.env_id, n_envs=cfg.experiment.n_envs, seed=cfg.experiment.seed)

    return env

def create_callbacks(cfg):
    callbacks = []

    # Only log if specified in config AND not sweeping hyperparams
    if cfg.experiment.save_freq > 0 and not cfg.experiment.multirun:
        cfg.experiment.save_freq = max(cfg.experiment.save_freq // cfg.experiment.n_envs, 1)
        callbacks.append(
                CheckpointCallback(
                    save_freq=cfg.experiment.save_freq,
                    save_path=cfg.experiment.log_folder,
                    name_prefix="rl_model",
                    verbose=1,
                )
            )
    
    if cfg.experiment.eval_freq > 0 and not cfg.experiment.multirun:
        # Account for the number of parallel environments
        cfg.experiment.eval_freq = max(cfg.experiment.eval_freq // cfg.experiment.n_envs, 1)

        if cfg.experiment.verbose > 0:
            print("Creating test environment")

        eval_callback = EvalCallback(
            make_eval_env(cfg),
            best_model_save_path=cfg.environment.run_dir,
            n_eval_episodes=cfg.experiment.n_eval_episodes,
            log_path=cfg.experiment.log_folder,
            eval_freq=cfg.experiment.eval_freq,
            deterministic=True,
            render=cfg.experiment.render
        )

        callbacks.append(eval_callback)
    return callbacks