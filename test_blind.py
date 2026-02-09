
import gymnasium as gym
import slimevolleygym
import numpy as np
from time import sleep

# Import the masked environments to ensure registration
import slimevolleygym.slimevolley_mask

def manual_policy(obs):
    return [0, 0, 0] # idle

def run_env(env_name):
    print(f"Testing {env_name}...")
    env = gym.make(env_name)
    env = env.unwrapped
    
    # Render settings
    env.render()
    
    obs, _ = env.reset()
    terminated = False
    truncated = False
    steps = 0
    
    while not (terminated or truncated):
        # Add slight delay so we can see what's happening
        sleep(0.02)
        
        # Simple policy: purely random or idle
        action = [0, 0, 0] # Forward, backward, jump
        
        # We can implement a simple manual override or just let it run
        # This is just to visualize the masking if we were to print obs
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        steps += 1
        
        # Print observation every 60 frames to check masking
        if steps % 60 == 0:
            print(f"Step {steps}, BallX (obs[4]): {obs[4]:.2f}, OpponentX (obs[8]): {obs[8]:.2f}")
            
        if steps > 500: # Stop after 500 steps
            break
            
    env.close()

if __name__ == "__main__":
    # Test the standard one first
    # run_env("SlimeVolley-v0")
    
    # Test the Masked one
    run_env("SlimeVolleyMasked-v0")
