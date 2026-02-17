
import gymnasium as gym
import slimevolleygym
import slimevolleygym.slimevolley_mask
from sb3_contrib import RecurrentPPO # LSTM PPO
from stable_baselines3.common.env_util import make_vec_env
import os

# Create log dir
log_dir = "logs_lstm/"
os.makedirs(log_dir, exist_ok=True)

# Use the Masked environment
env_id = "SlimeVolleyMasked-v0"

# Fix for PyTorch "Float can't be cast to Char" error with MultiBinary(int8) actions
# We need to force the action space dtype to float32 so PPO buffers use float.
import numpy as np
def make_env():
    env = gym.make(env_id)
    # Hack: coerce action space to float32 so SB3 treats actions as floats
    env.action_space.dtype = np.float32
    return env

# Vectorized environment is tricky with Recurrent policies (need to handle state resets correctly),
# but SB3 handles it generally well.
# We use 4 parallel envs.
env = make_vec_env(make_env, n_envs=4, seed=0)

# Initialize the Recurrent Agent
# MlpLstmPolicy: Input -> MLP -> LSTM -> Output
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=log_dir)

# Train the agent
# Recurrent models are slower to train.
steps = 2_000_000 # Increase this for real training (e.g. 2_000_000)
print(f"Training LSTM Agent for {steps} steps...")
model.learn(total_timesteps=steps)

# Save the model
model_path = os.path.join(log_dir, "ppo_lstm_slimevolley")
model.save(model_path)
print(f"Model saved to {model_path}")
