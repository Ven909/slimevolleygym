
import gymnasium as gym
import slimevolleygym
import slimevolleygym.slimevolley_mask
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

# 1. Recreate the EXACT same environment structure used in training
#    If the model was trained on FrameStack(4), the input must be stacked 4 frames.
#    We just wrap a single env for testing.
import numpy as np
def make_env():
    env = gym.make("SlimeVolleyMasked-v0")
    env.action_space.dtype = np.float32 # Same fix as training
    return env

env = make_vec_env(make_env, n_envs=1)
env = VecFrameStack(env, n_stack=4)

# Load the trained model
model_path = "logs_framestack/ppo_framestack_slimevolley"
try:
    model = PPO.load(model_path)
except FileNotFoundError:
    print(f"Model not found at {model_path}. Please run train_framestack.py first.")
    exit()

obs = env.reset()
done = False
total_reward = 0

while not done:
    # VecEnv returns observation as list of len(n_envs), we take [0]
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render() # Visualization via Gym's default render (might need adjustment if VecEnv hides it)
    
    # VecEnv auto-resets on done, so check 'done' array
    if done[0]:
        print(f"Completed Episode. Reward: {total_reward}")
        break

env.close()
