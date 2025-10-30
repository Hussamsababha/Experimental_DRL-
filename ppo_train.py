"""
training.py
Script to train a PPO agent on the cylinder environment.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Local imports
from cylinder_env import CustomEnv


def main():
    # ---------------------------
    # Configuration parameters
    # ---------------------------
    run_name = "test_name"
    total_timesteps = 50_000

    training_log_folder = "logs_training"
    model_folder = "logs_models"
    tensorboard_folder = "logs_tensorboard"

    # Create directories if they don't exist
    os.makedirs(training_log_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(tensorboard_folder, exist_ok=True)

    # TensorBoard log directory (unique per run)
    tb_log_path = os.path.join(tensorboard_folder, run_name)

    # ---------------------------
    # Initialize environment
    # ---------------------------
    env = CustomEnv()  #

    # ---------------------------
    # Define PPO policy and hyperparameters
    # ---------------------------
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        n_steps=128,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        n_epochs = 10,
        ortho_init=True,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log=tb_log_path,
    )

    # ---------------------------
    # Define checkpoint callback
    # ---------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=model_folder,
        name_prefix=run_name,
    )

    # ---------------------------
    # Train the model
    # ---------------------------
    print(f"[INFO] Starting PPO training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # ---------------------------
    # Save logs and model
    # ---------------------------
    training_log_path = os.path.join(training_log_folder, f"{run_name}.mat")
    env.save_traininglogs_to_mat(training_log_path)
    print(f"[INFO] Training logs saved to {training_log_path}")

    # Save trained model
    model_path = os.path.join(model_folder, run_name)
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")


if __name__ == "__main__":
    main()
