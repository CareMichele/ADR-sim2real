import numpy as np
import json
from pathlib import Path


def evaluate_policy(env, model, n_episodes=20, max_steps=500, deterministic=True, render=False):
    """
    Evaluate a policy on a given environment.
    Automatically handles:
    1. Switching to evaluation mode (training=False)
    2. Disabling reward normalization to get true rewards
    3. Restoring original state after evaluation
    
    Args:
        env: VecEnv environment (with or without VecNormalize wrapper)
        model: Trained model (e.g., PPO)
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        deterministic: If True, use deterministic policy
        render: If True, render during evaluation (not implemented)
    
    Returns:
        dict: Statistics with keys 'mean_reward', 'std_reward', 'min_reward', 
              'max_reward', 'median_reward', 'mean_length', 'n_episodes'
    """
    # --- 1. SAFE SETUP ---
    # Save previous state to restore it later (critical for train.py)
    was_training = env.training if hasattr(env, 'training') else None
    was_norm_reward = env.norm_reward if hasattr(env, "norm_reward") else None
    
    # Set evaluation mode
    if hasattr(env, 'training'):
        env.training = False
    if hasattr(env, "norm_reward"):
        env.norm_reward = False  # We always want true rewards in evaluation
    
    episode_rewards = []
    episode_lengths = []
    
    # --- 2. EVALUATION LOOP ---
    for _ in range(n_episodes):
        obs = env.reset()
        episode_return = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done_array, info = env.step(action)
            episode_return += reward[0]
            steps += 1
            done = done_array[0]
        
        episode_rewards.append(episode_return)
        episode_lengths.append(steps)
    
    # --- 3. RESTORE STATE ---
    if was_training is not None:
        env.training = was_training
    if was_norm_reward is not None:
        env.norm_reward = was_norm_reward
    
    # --- 4. STATISTICS ---
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'median_reward': np.median(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'n_episodes': n_episodes
    }
    
    return stats


def export_results(results, output_dir):
    """
    Export evaluation results to CSV and JSON.
    
    Args:
        results: List of result dictionaries
        output_dir: Path object or string for output directory
    
    Returns:
        DataFrame with the exported results
    """
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    output_dir = Path(output_dir)
    
    csv_path = output_dir / 'evaluation_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")
    
    json_path = output_dir / 'evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {json_path}")
    
    return df
