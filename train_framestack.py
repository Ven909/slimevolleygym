
import gymnasium as gym
import slimevolleygym
import slimevolleygym.slimevolley_mask
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import glob

# Create log dir
log_dir = "logs_framestack/"
checkpoint_dir = os.path.join(log_dir, "checkpoints")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Use the Masked environment
env_id = "SlimeVolleyMasked-v0"

# Fix for PyTorch dtype errors with MultiBinary actions
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

# 1. Create the vectorized environment
# Use the custom make_env to apply the fix
env = make_vec_env(make_env, n_envs=4, seed=0)

# 2. Wrap it in FrameStack (Stack 4 frames)
# This gives the agent access to the last 4 observations at each step,
# effectively giving it "short-term memory" as velocity/acceleration features.
env = VecFrameStack(env, n_stack=4)

# Resume from latest checkpoint if one exists
checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.zip")))
if checkpoints:
    latest = checkpoints[-1]
    print(f"Resuming from checkpoint: {latest}")
    model = PPO.load(latest, env=env, verbose=1, tensorboard_log=log_dir, device="cpu")
else:
    print("No checkpoint found, starting fresh.")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cpu")

# Save a checkpoint every 50,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=checkpoint_dir,
    name_prefix="ppo_framestack",
    verbose=1
)

# Train the agent
steps = 2_000_000 # Increase for real results
print(f"Training FrameStack (4 frames) Agent for {steps} steps...")
model.learn(total_timesteps=steps, callback=checkpoint_callback, reset_num_timesteps=not bool(checkpoints))

# Save the model
model_path = os.path.join(log_dir, "ppo_framestack_slimevolley")
model.save(model_path)
print(f"Model saved to {model_path}")
