
import gymnasium as gym
import slimevolleygym
import slimevolleygym.slimevolley_mask
from sb3_contrib import RecurrentPPO # LSTM PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import glob

# Create log dir
log_dir = "logs_lstm/"
checkpoint_dir = os.path.join(log_dir, "checkpoints")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Use the Masked environment
env_id = "SlimeVolleyMasked-v0"

# Fix for PyTorch "Float can't be cast to Char" error with MultiBinary(int8) actions
# We need to force the action space dtype to float32 so PPO buffers use float.
import numpy as np
from gymnasium import ObservationWrapper

class Float32Obs(ObservationWrapper):
    """Cast observations to float32 to satisfy SB3 and gymnasium checks."""
    def observation(self, obs):
        return np.array(obs, dtype=np.float32)

def make_env():
    env = gym.make(env_id)
    # Fix: cast observations to float32 (env returns float64 by default)
    env = Float32Obs(env)
    # Hack: coerce action space to float32 so SB3 treats actions as floats
    env.action_space.dtype = np.float32
    return env

# Vectorized environment is tricky with Recurrent policies (need to handle state resets correctly),
# but SB3 handles it generally well.
# We use 4 parallel envs.
env = make_vec_env(make_env, n_envs=4, seed=0)

# Initialize the Recurrent Agent
# MlpLstmPolicy: Input -> MLP -> LSTM -> Output

# Resume from latest checkpoint if one exists
checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.zip")))
if checkpoints:
    latest = checkpoints[-1]
    print(f"Resuming from checkpoint: {latest}")
    model = RecurrentPPO.load(latest, env=env, verbose=1, tensorboard_log=log_dir)
else:
    print("No checkpoint found, starting fresh.")
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=log_dir)

# Save a checkpoint every 50,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=checkpoint_dir,
    name_prefix="ppo_lstm",
    verbose=1
)

# Train the agent
# Recurrent models are slower to train.
steps = 2_000_000 # Increase this for real training (e.g. 2_000_000)
print(f"Training LSTM Agent for {steps} steps...")
model.learn(total_timesteps=steps, callback=checkpoint_callback, reset_num_timesteps=not bool(checkpoints))

# Save the model
model_path = os.path.join(log_dir, "ppo_lstm_slimevolley")
model.save(model_path)
print(f"Model saved to {model_path}")
