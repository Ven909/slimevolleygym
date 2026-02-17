
import gymnasium as gym
import slimevolleygym
import slimevolleygym.slimevolley_mask
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
import os

# Create log dir
log_dir = "logs_framestack/"
os.makedirs(log_dir, exist_ok=True)

# Use the Masked environment
env_id = "SlimeVolleyMasked-v0"

# Fix for PyTorch dtype errors with MultiBinary actions
import numpy as np
def make_env():
    env = gym.make(env_id)
    # Hack: coerce action space to float32 so SB3 treats actions as floats
    env.action_space.dtype = np.float32
    return env

# 1. Create the vectorized environment
# Use the custom make_env to apply the fix
env = make_vec_env(make_env, n_envs=4, seed=0)

# 2. Wrap it in FrameStack (Stack 4 frames)
# This gives the agent access to the last 4 observations at each step,
# effectively giving it "short-term memory" as velocity/acceleration features.
env = VecFrameStack(env, n_stack=4)

# Initialize standard PPO agent
# (Frame stacking changes the input shape, but MlpPolicy handles the new flattened size internally)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Train the agent
steps = 2_000_000 # Increase for real results 
print(f"Training FrameStack (4 frames) Agent for {steps} steps...")
model.learn(total_timesteps=steps)

# Save the model
model_path = os.path.join(log_dir, "ppo_framestack_slimevolley")
model.save(model_path)
print(f"Model saved to {model_path}")
