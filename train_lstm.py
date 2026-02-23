
import gymnasium as gym
import slimevolleygym
import slimevolleygym.slimevolley_mask
from sb3_contrib import RecurrentPPO # LSTM PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import glob
from datetime import datetime

# Each run gets its own timestamped subdirectory so old logs/checkpoints are
# never touched and TensorBoard comparisons across runs stay clean.
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("logs_lstm", f"run_{RUN_ID}")
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

class BallOnSidePenalty(gym.Wrapper):
    """
    Adds a small penalty when the ball remains on our side (right side, x > 0)
    for too many consecutive steps.  This discourages passive / defensive
    stalling without overwhelming the main scoring reward (+/-1).

    Parameters
    ----------
    threshold : int
        Number of consecutive steps the ball must stay on our side before
        the penalty kicks in (default 60 ≈ 2 seconds at 30 fps).
    penalty : float
        Reward added each step once threshold is exceeded (should be negative).
    """
    def __init__(self, env, threshold=90, penalty=-0.002):
        super().__init__(env)
        self.threshold = threshold
        self.penalty = penalty
        self._ball_on_our_side_steps = 0

    def step(self, action, **kwargs):
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)

        # ball.x > 0 means the ball is on the right side (our trained agent's side)
        if self.unwrapped.game.ball.x > 0:
            self._ball_on_our_side_steps += 1
        else:
            self._ball_on_our_side_steps = 0

        if self._ball_on_our_side_steps > self.threshold:
            reward += self.penalty

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._ball_on_our_side_steps = 0
        return self.env.reset(**kwargs)

def make_env():
    env = gym.make(env_id)
    # Increase max episode length from 3000 to 4000 so games last longer
    env.unwrapped.t_limit = 4000
    # Fix: cast observations to float32 (env returns float64 by default)
    env = Float32Obs(env)
    # Hack: coerce action space to float32 so SB3 treats actions as floats
    env.action_space.dtype = np.float32
    # Penalize passive play (ball sitting on our side too long)
    env = BallOnSidePenalty(env)
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
steps = 5_000_000 # Increased from 2M for deeper training
print(f"Training LSTM Agent for {steps} steps...")
model.learn(total_timesteps=steps, callback=checkpoint_callback, reset_num_timesteps=not bool(checkpoints))

# Save the model
model_path = os.path.join(log_dir, "ppo_lstm_slimevolley")
model.save(model_path)
print(f"Model saved to {model_path}")
