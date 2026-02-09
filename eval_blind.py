
import gymnasium as gym
import slimevolleygym
import slimevolleygym.slimevolley_mask # Register blind envs
from stable_baselines3 import PPO
import numpy as np
import os

def evaluate_on_env(model_path, env_name, num_episodes=5):
    print(f"--- Evaluating model on {env_name} ---")
    
    # Load model
    if not os.path.exists(model_path + ".zip"):
        print(f"Model {model_path} not found. train first.")
        return

    model = PPO.load(model_path)
    
    env = gym.make(env_name)
    
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward}")
        
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward on {env_name}: {avg_reward}")
    return avg_reward

if __name__ == "__main__":
    model_path = "logs/ppo_slimevolley"
    
    # 1. Evaluate on Standard Environment (Should be good/decent)
    evaluate_on_env(model_path, "SlimeVolley-v0")
    
    # 2. Evaluate on Blind Environment (Should be bad)
    print("\nSanity Check: The agent should fail here because it can't see the opponent.")
    evaluate_on_env(model_path, "SlimeVolleyMasked-v0")
