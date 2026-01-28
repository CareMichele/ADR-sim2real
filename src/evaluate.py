import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from pathlib import Path
import sys
import argparse
import time
import json

sys.path.insert(0, str(Path(__file__).parent))
import envs.custom_hopper
import envs.custom_ant
from utils.env_utils import create_eval_env

# Load environment configurations
with open(Path(__file__).parent.parent / 'configs' / 'env_configs.json', 'r') as f:
    ENV_CONFIGS = json.load(f)


def evaluate_model(model_path, env_id, n_episodes=50, render=False, deterministic=True, max_steps=500):
    """
    Evaluate a model loaded from checkpoint.
    
    Args:
        model_path: Path to model checkpoint .zip file
        env_id: Gymnasium environment ID (e.g., 'CustomHopper-source-v0')
        n_episodes: Number of episodes to test
        render: If True, render the environment
        deterministic: If True, use deterministic policy
        max_steps: Maximum number of steps per episode
    
    Returns:
        dict: Evaluation statistics
    """
    print(f"\n{'='*70}")
    print(f"  MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Environment: {env_id}")
    print(f"Episodes: {n_episodes}")
    print(f"Render: {render}")
    print(f"Deterministic: {deterministic}\n")
    
    print("[INFO] Loading model...")
    model = PPO.load(model_path, device='cpu')
    print("Model loaded successfully!\n")
    
    print("[INFO] Creating environment...")
    env = create_eval_env(env_id, model_path, render=render)
    print("Environment created!\n")
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"[INFO] Starting evaluation ({n_episodes} episodes)...\n")
    
    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done_array, info = env.step(action)
            episode_reward += reward[0]
            steps += 1
            done = done_array[0]
            
            if render:
                time.sleep(0.01)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Episode {ep+1:3d}/{n_episodes}: reward={episode_reward:7.2f}, steps={steps:3d}")
    
    env.close()
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    median_reward = np.median(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    stats = {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'median_reward': median_reward,
        'mean_length': mean_length,
        'n_episodes': n_episodes,
        'all_rewards': episode_rewards,
        'all_lengths': episode_lengths,
    }
    
    print(f"\n{'='*70}")
    print("  EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Mean Reward:   {mean_reward:7.2f} ± {std_reward:.2f}")
    print(f"Median Reward: {median_reward:7.2f}")
    print(f"Min Reward:    {min_reward:7.2f}")
    print(f"Max Reward:    {max_reward:7.2f}")
    print(f"Mean Length:   {mean_length:7.2f} steps")
    print(f"{'='*70}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained ADR model')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint (.zip)')
    parser.add_argument('--env', type=str, default='CustomHopper-source-v0',
                        help='Environment ID (default: CustomHopper-source-v0)')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes (default: 50)')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy (default: deterministic)')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Max steps per episode (default: 500)')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    env_id = args.env
    if args.env in ENV_CONFIGS:
        env_config = ENV_CONFIGS[args.env]
        if isinstance(env_config, dict):
            env_id = env_config['env_id']
        else:
            env_id = env_config
        print(f"[INFO] Converted '{args.env}' -> '{env_id}'")
    
    evaluate_model(
        str(model_path),
        env_id,
        args.episodes,
        args.render,
        deterministic=not args.stochastic,
        max_steps=args.max_steps
    )


if __name__ == "__main__": 
    main()
