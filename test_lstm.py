
import gymnasium as gym
import slimevolleygym
import slimevolleygym.slimevolley_mask
from sb3_contrib import RecurrentPPO
import numpy as np

# Load the environment
# We need to apply the same action space patch here to suppress warnings and ensure consistency
import numpy as np
env = gym.make("SlimeVolleyMasked-v0")
env.action_space.dtype = np.float32

# Load the trained model from the latest run directory.
# Falls back to the old flat path if no timestamped run exists.
import glob as _glob
import os as _os

def _find_latest_model(log_root, name):
    # Look for final model inside any run_* subdirectory, newest first
    candidates = sorted(
        _glob.glob(_os.path.join(log_root, "run_*", name + ".zip")),
        reverse=True
    )
    if candidates:
        return candidates[0].replace(".zip", "")  # SB3 .load() doesn't want the extension
    # Fallback: old flat layout
    return _os.path.join(log_root, name)

model_path = _find_latest_model("logs_lstm", "ppo_lstm_slimevolley")
print(f"Loading model from: {model_path}")
try:
    model = RecurrentPPO.load(model_path)
except FileNotFoundError:
    print(f"Model not found at {model_path}. Please run train_lstm.py first.")
    exit()

obs, info = env.reset()
# LSTM requires tracking hidden states
lstm_states = None
num_envs = 1
# Episode start signals are required to reset the hidden states
episode_starts = np.ones((num_envs,), dtype=bool)

done = False
total_reward = 0

while not done:
    # Predict with LSTM states
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render() # Visualization
    
    episode_starts[0] = done

print(f"Total Reward (LSTM Agent): {total_reward}")
env.close()
