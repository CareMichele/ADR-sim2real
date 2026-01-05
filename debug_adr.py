"""
Debug script to test ADR (Automatic Domain Randomization).

Simulates the ADR curriculum:
- Start with tight range (1.0, 1.0) = no randomization
- Every N episodes, update performance and adapt ranges
- If performance > threshold → expand (more difficult)
- If performance < threshold → shrink (easier)
"""

import gymnasium as gym
import numpy as np
import sys

sys.path.insert(0, str(__file__).rsplit('/', 1)[0])

import src.envs.custom_hopper_base 

def main():
    env = gym.make("CustomHopper-source-adr-v0")
    
    print("ADR Debug: Simulating ADR curriculum\n")
    print(f"Initial ADR ranges: {env.unwrapped.adr_ranges}")
    print(f"ADR threshold: {env.unwrapped.adr_threshold}")
    print(f"ADR delta: {env.unwrapped.adr_delta}\n")
    
    # Simulate ADR: train for N episodes, update ranges every K episodes
    num_episodes = 100
    update_interval = 10
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 500
        
        while not done and steps < max_steps:
            action = env.action_space.sample()  # Random policy
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
        
        episode_rewards.append(episode_reward)
        
        # Print episode info
        masses = env.unwrapped.get_parameters()
        if (ep + 1) % update_interval == 0:
            # Calculate avg performance over last K episodes
            perf_window = episode_rewards[-update_interval:]
            perf_avg = np.mean(perf_window)
            perf_max = np.max(perf_window)
            perf_min = np.min(perf_window)
            
            print(f"\n--- Episode {ep+1}/{num_episodes} ---")
            print(f"Last {update_interval} episodes reward: avg={perf_avg:.2f}, min={perf_min:.2f}, max={perf_max:.2f}")
            print(f"Current masses (before update): {masses}")
            
            # Normalize performance (simple: divide by max possible reward)
            perf_normalized = perf_avg / 1000.0  # Rough normalization  
            #TODO cambiare 1000.0 con il valore massimo del training con PPO
            perf_normalized = np.clip(perf_normalized, 0, 1)
            
            print(f"Normalized performance: {perf_normalized:.3f}")
            
            # Update ADR ranges
            env.unwrapped.update_adr_ranges(perf_normalized)
            
            print(f"Updated ADR ranges:")
            for key, (lower, upper) in env.unwrapped.adr_ranges.items():
                print(f"  {key}: ({lower:.4f}, {upper:.4f})")
    
    env.close()
    print("\n✅ ADR debug complete!")

if __name__ == "__main__":
    main()
