
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

model_path = _find_latest_model("logs_framestack", "ppo_framestack_slimevolley")
print(f"Loading model from: {model_path}")
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
