
import gymnasium as gym
import slimevolleygym
import slimevolleygym.slimevolley_mask
import numpy as np

def evaluate_expert(env_name, episodes=5):
    print(f"--- Testing Expert Policy on {env_name} ---")
    env = gym.make(env_name)
    
    # The expert policy provided by the repo
    policy = slimevolleygym.BaselinePolicy()
    
    total_reward = 0
    
    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # The expert predicts action based on observation
            action = policy.predict(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
        print(f"Episode {i+1}: Score = {episode_reward}")
        total_reward += episode_reward
        
    avg = total_reward / episodes
    print(f"Average Score: {avg}")
    if avg > 0:
        print("RESULT: Expert PASSED (It can play)")
    else:
        print("RESULT: Expert FAILED (It cannot play)")
    print("-" * 30 + "\n")

if __name__ == "__main__":
    # 1. Test on Normal Game
    evaluate_expert("SlimeVolley-v0")
    
    # 2. Test on Blind Game
    evaluate_expert("SlimeVolleyMasked-v0")
