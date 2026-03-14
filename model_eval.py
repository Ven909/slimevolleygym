"""
Unified evaluation script for SlimeVolleyGym trained models.

Runs a model for N episodes (headless by default), collects per-episode
statistics, and prints a beginner-friendly metrics report.

Usage examples:
    python evaluate.py --model framestack --path logs_framestack/.../model.zip
    python evaluate.py --model lstm --path logs_lstm/.../model.zip
    python evaluate.py --model exploiter --path logs_exploiter/.../model.zip --opponent 10m
    python evaluate.py --model baseline --episodes 100
    python evaluate.py --model framestack --path model.zip --episodes 200 --save results.csv
"""

import argparse
import csv
import os
import sys
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper

import slimevolleygym
import slimevolleygym.slimevolley_mask

MAXLIVES = 5

FRAMESTACK_OPPONENT_PATHS = {
    "2m": "logs_framestack/run_20260223_144019/ppo_framestack_slimevolley.zip",
    "5m": "logs_framestack/run_20260223_202541/ppo_framestack_slimevolley.zip",
    "10m": "logs_framestack/run_20260223_202559/ppo_framestack_slimevolley.zip",
}


# ---------------------------------------------------------------------------
# Wrappers (reused from the existing test scripts)
# ---------------------------------------------------------------------------

class Float32Obs(ObservationWrapper):
    """Cast observations to float32 (base env returns float64)."""
    def observation(self, obs):
        return np.array(obs, dtype=np.float32)


class FrameStackOpponentPolicy:
    """
    Wraps a trained FrameStack PPO model so it can serve as the left-side
    opponent inside the standard SlimeVolley environment.
    """

    def __init__(self, model_path, n_stack=4, deterministic=True):
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path)
        self.n_stack = n_stack
        self.deterministic = deterministic
        self.obs_dim = 12
        self.frame_buffer = deque(maxlen=n_stack)
        self.reset()

    def reset(self):
        self.frame_buffer.clear()
        for _ in range(self.n_stack):
            self.frame_buffer.append(np.zeros(self.obs_dim, dtype=np.float32))

    def _mask_obs(self, obs):
        masked = np.array(obs, dtype=np.float32)
        masked[8:12] = 0.0
        if masked[4] < 0:
            masked[4:8] = 0.0
        return masked

    def predict(self, obs):
        masked = self._mask_obs(obs)
        self.frame_buffer.append(masked)
        stacked = np.concatenate(list(self.frame_buffer))
        action, _ = self.model.predict(stacked, deterministic=self.deterministic)
        return action


class ExploiterEnv(gym.Wrapper):
    """Installs a custom opponent policy and resets it each episode."""

    def __init__(self, env, opponent_policy):
        super().__init__(env)
        self.unwrapped.policy = opponent_policy

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if hasattr(self.unwrapped.policy, "reset"):
            self.unwrapped.policy.reset()
        return obs, info


# ---------------------------------------------------------------------------
# Environment / model loaders for each model type
# ---------------------------------------------------------------------------

def _make_masked_env():
    env = gym.make("SlimeVolleyMasked-v0")
    env = Float32Obs(env)
    env.action_space.dtype = np.float32
    return env


def _make_standard_env():
    env = gym.make("SlimeVolley-v0")
    env = Float32Obs(env)
    env.action_space.dtype = np.float32
    return env


def setup_baseline(args):
    """Built-in baseline RNN policy evaluated against itself."""
    env = _make_standard_env()
    policy = slimevolleygym.BaselinePolicy()

    def predict_fn(obs, **_kwargs):
        return policy.predict(obs), None

    return env, predict_fn, "Baseline RNN", "SlimeVolley-v0", "Baseline RNN", False


def setup_framestack(args):
    """FrameStack PPO (Agent B) on the masked environment."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    vec_env = DummyVecEnv([_make_masked_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    model = PPO.load(args.path, device="cpu")

    def predict_fn(obs, **_kwargs):
        return model.predict(obs, deterministic=True)

    return vec_env, predict_fn, "FrameStack PPO", "SlimeVolleyMasked-v0", "Baseline RNN", True


def setup_lstm(args):
    """Recurrent PPO (LSTM) on the masked environment."""
    from sb3_contrib import RecurrentPPO

    env = _make_masked_env()
    model = RecurrentPPO.load(args.path, device="cpu")

    class _LSTMPredictor:
        def __init__(self):
            self.lstm_states = None
            self.episode_starts = np.ones((1,), dtype=bool)

        def __call__(self, obs, *, episode_start=False):
            if episode_start:
                self.episode_starts[0] = True
            action, self.lstm_states = model.predict(
                obs,
                state=self.lstm_states,
                episode_start=self.episode_starts,
                deterministic=True,
            )
            self.episode_starts[0] = False
            return action, None

    return env, _LSTMPredictor(), "LSTM PPO", "SlimeVolleyMasked-v0", "Baseline RNN", False


def setup_exploiter(args):
    """Exploiter PPO (Agent C) on the standard env with Agent B as opponent."""
    from stable_baselines3 import PPO

    opponent_key = args.opponent or "10m"
    opponent_path = FRAMESTACK_OPPONENT_PATHS.get(opponent_key)
    if opponent_path is None or not os.path.exists(opponent_path):
        print(f"ERROR: Opponent model not found for '{opponent_key}' at {opponent_path}")
        sys.exit(1)

    env = _make_standard_env()
    opponent = FrameStackOpponentPolicy(opponent_path, n_stack=4, deterministic=True)
    env = ExploiterEnv(env, opponent)

    model = PPO.load(args.path, device="cpu")

    def predict_fn(obs, **_kwargs):
        return model.predict(obs, deterministic=True)

    opponent_label = f"FrameStack PPO ({opponent_key})"
    return env, predict_fn, "Exploiter PPO", "SlimeVolley-v0", opponent_label, False


MODEL_SETUP = {
    "baseline": setup_baseline,
    "framestack": setup_framestack,
    "lstm": setup_lstm,
    "exploiter": setup_exploiter,
}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(env, predict_fn, n_episodes, is_vec_env, render=False):
    """
    Play n_episodes and return per-episode records.

    Each record is a dict with keys:
        reward, length, agent_lives, opponent_lives, outcome
    """
    records = []

    for ep in range(n_episodes):
        if is_vec_env:
            obs = env.reset()
        else:
            obs, _info = env.reset()

        done = False
        total_reward = 0.0
        length = 0
        last_info = {}

        while not done:
            action, _ = predict_fn(obs, episode_start=(length == 0))

            if is_vec_env:
                obs, reward, done_arr, info_list = env.step(action)
                reward_val = float(reward[0])
                total_reward += reward_val
                length += 1
                last_info = info_list[0] if info_list else {}
                done = bool(done_arr[0])
                if render:
                    env.venv.envs[0].env.render()
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                length += 1
                last_info = info
                done = terminated or truncated
                if render:
                    env.render()

        agent_lives = last_info.get("ale.lives", MAXLIVES)
        opponent_lives = last_info.get("ale.otherLives", MAXLIVES)

        if opponent_lives <= 0 and agent_lives > 0:
            outcome = "win"
        elif agent_lives <= 0 and opponent_lives > 0:
            outcome = "loss"
        else:
            outcome = "draw"

        records.append({
            "episode": ep + 1,
            "reward": total_reward,
            "length": length,
            "agent_lives": agent_lives,
            "opponent_lives": opponent_lives,
            "outcome": outcome,
        })

        status = {"win": "W", "loss": "L", "draw": "D"}[outcome]
        print(
            f"  Episode {ep + 1:4d}/{n_episodes}  |  "
            f"Reward: {total_reward:6.2f}  |  "
            f"Length: {length:5d}  |  "
            f"Lives: {agent_lives}-{opponent_lives}  |  {status}"
        )

    return records


# ---------------------------------------------------------------------------
# Metrics computation & report
# ---------------------------------------------------------------------------

def compute_metrics(records):
    rewards = np.array([r["reward"] for r in records])
    lengths = np.array([r["length"] for r in records])
    outcomes = [r["outcome"] for r in records]
    agent_lives = np.array([r["agent_lives"] for r in records])
    opponent_lives = np.array([r["opponent_lives"] for r in records])

    n = len(records)
    wins = outcomes.count("win")
    losses = outcomes.count("loss")
    draws = outcomes.count("draw")

    points_scored = MAXLIVES - opponent_lives
    points_conceded = MAXLIVES - agent_lives

    shutouts = sum(
        1 for r in records
        if r["outcome"] == "win" and r["agent_lives"] == MAXLIVES
    )

    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": 100.0 * wins / n if n else 0,
        "loss_rate": 100.0 * losses / n if n else 0,
        "draw_rate": 100.0 * draws / n if n else 0,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "avg_scored": float(np.mean(points_scored)),
        "avg_conceded": float(np.mean(points_conceded)),
        "score_margin": float(np.mean(points_scored - points_conceded)),
        "avg_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "shutouts": shutouts,
        "shutout_rate": 100.0 * shutouts / n if n else 0,
        "avg_opp_lives": float(np.mean(opponent_lives)),
    }


def print_report(metrics, model_name, env_name, opponent_name, n_episodes):
    W = 54
    print()
    print("=" * W)
    print("  MODEL EVALUATION REPORT")
    print("=" * W)
    print(f"  Model        : {model_name}")
    print(f"  Environment  : {env_name}")
    print(f"  Opponent     : {opponent_name}")
    print(f"  Episodes     : {n_episodes}")
    print("-" * W)

    m = metrics

    print()
    print("  WIN / LOSS / DRAW")
    print(f"    Record         : W {m['wins']}  /  L {m['losses']}  /  D {m['draws']}")
    print(f"    Win rate       : {m['win_rate']:.1f}%")
    print(f"    Loss rate      : {m['loss_rate']:.1f}%")
    print(f"    Draw rate      : {m['draw_rate']:.1f}%")

    print()
    print("  SCORING")
    print(f"    Avg reward     : {m['avg_reward']:+.2f} +/- {m['std_reward']:.2f}")
    print(f"    Avg scored     : {m['avg_scored']:.2f}")
    print(f"    Avg conceded   : {m['avg_conceded']:.2f}")
    print(f"    Score margin   : {m['score_margin']:+.2f}")

    print()
    print("  EPISODE LENGTH")
    print(f"    Avg length     : {m['avg_length']:.1f} +/- {m['std_length']:.1f}")
    print(f"    Shortest       : {m['min_length']}")
    print(f"    Longest        : {m['max_length']}")

    print()
    print("  CONSISTENCY")
    print(f"    Median reward  : {m['median_reward']:+.2f}")
    print(f"    Best reward    : {m['max_reward']:+.2f}")
    print(f"    Worst reward   : {m['min_reward']:+.2f}")

    print()
    print("  DOMINANCE")
    print(f"    Shutout rate   : {m['shutout_rate']:.1f}%  ({m['shutouts']}/{m['n']} games won {MAXLIVES}-0)")
    print(f"    Avg opp lives  : {m['avg_opp_lives']:.2f}")
    print("=" * W)
    print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(records, metrics, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

        f.write("\n")
        summary_writer = csv.writer(f)
        summary_writer.writerow([])
        summary_writer.writerow(["metric", "value"])
        for key, val in metrics.items():
            summary_writer.writerow([key, val])

    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SlimeVolleyGym model and report metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_SETUP.keys()),
        help="Model type to evaluate: baseline, framestack, lstm, or exploiter",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to trained model file (.zip). Not needed for 'baseline'.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["2m", "5m", "10m"],
        default="10m",
        help="Opponent preset for exploiter mode (default: 10m)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render episodes visually (much slower)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="FILE.csv",
        help="Save per-episode results and summary to a CSV file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model != "baseline" and args.path is None:
        print("ERROR: --path is required for all model types except 'baseline'.")
        sys.exit(1)

    if args.path and not os.path.exists(args.path):
        print(f"ERROR: Model file not found at {args.path}")
        sys.exit(1)

    env, predict_fn, model_name, env_name, opponent_name, is_vec_env = MODEL_SETUP[args.model](args)

    print()
    print(f"Evaluating {model_name} on {env_name} vs {opponent_name}")
    print(f"Running {args.episodes} episodes {'(with rendering)' if args.render else '(headless)'}...")
    print()

    records = run_evaluation(env, predict_fn, args.episodes, is_vec_env, render=args.render)
    env.close()

    metrics = compute_metrics(records)
    print_report(metrics, model_name, env_name, opponent_name, args.episodes)

    if args.save:
        save_csv(records, metrics, args.save)


if __name__ == "__main__":
    main()
