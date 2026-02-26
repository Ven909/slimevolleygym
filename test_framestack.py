
import argparse
import gymnasium as gym
import slimevolleygym
import slimevolleygym.slimevolley_mask
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import numpy as np
from gymnasium import ObservationWrapper

class Float32Obs(ObservationWrapper):
    def observation(self, obs):
        return np.array(obs, dtype=np.float32)

def make_env():
    env = gym.make("SlimeVolleyMasked-v0")
    env = Float32Obs(env)
    env.action_space.dtype = np.float32
    return env

# Single env wrapped in DummyVecEnv + FrameStack to match training
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

# Map friendly step labels to the final model saved at the end of each run.
# Each entry is the path to the final ppo_framestack_slimevolley.zip for that run.
STEP_MODELS = {
    "2m":  os.path.join("logs_framestack", "ppo_framestack_slimevolley"),
    "5m":  os.path.join("logs_framestack", "run_20260223_144019", "ppo_framestack_slimevolley"),
    "10m": os.path.join("logs_framestack", "run_20260223_202559", "ppo_framestack_slimevolley"),
}

import glob as _glob
import os as _os
import os

parser = argparse.ArgumentParser(description="Test a trained FrameStack model.")
parser.add_argument(
    "--steps",
    choices=["2m", "5m", "10m"],
    default="10m",
    help="Which training run to load: 2m (2M steps), 5m (5M steps), or 10m (10M steps). Default: 10m",
)
args = parser.parse_args()

model_path = STEP_MODELS[args.steps]
print(f"Loading '{args.steps}' model from: {model_path}")
try:
    model = PPO.load(model_path, device="cpu")
except FileNotFoundError:
    print(f"Model not found at {model_path}. Please run train_framestack.py first.")
    exit()

for episode in range(5):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        # Render directly from the underlying env (the one actually being played)
        env.venv.envs[0].env.render()

        if done[0]:
            print(f"Episode {episode + 1} done. Reward: {total_reward}")
            break

env.close()
