
import gymnasium as gym
import slimevolleygym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os

# Create log dir
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Create the environment
# We can use the standard one for baseline
env_id = "SlimeVolley-v0"

# Vectorized environment for faster training
env = make_vec_env(env_id, n_envs=4, seed=0)

# Initialize the agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Train the agent
# The paper says 3M steps is enough to solve it.
# We will do a short run (e.g. 100k steps) just to prove it runs.
# In real training, increase to 1_000_000 or more.
steps = 100_000 
print(f"Training for {steps} steps...")
model.learn(total_timesteps=steps)

# Save the model
model_path = os.path.join(log_dir, "ppo_slimevolley")
model.save(model_path)
print(f"Model saved to {model_path}")

# Evaluate the agent
obs = env.reset()
# We need to close and recreate single env for rendering usually, 
# but we can just print rew here.
