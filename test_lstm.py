
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

# Load the trained model
# If you haven't trained yet, this will fail.
# Train first: python train_lstm.py
model_path = "logs_lstm/ppo_lstm_slimevolley"
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
