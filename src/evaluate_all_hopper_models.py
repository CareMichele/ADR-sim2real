import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
import sys
import json
import pickle
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import envs.custom_hopper
from utils.env_utils import create_eval_env
from utils.evaluation_utils import export_results
from utils.plotting import create_results_table, create_heatmap, create_bar_chart

# CONFIGURATION

EVALUATION_CONFIG = {
    'n_episodes': 50,
    'max_steps': 500,
    'deterministic': True,
    'base_checkpoint_dir': Path('./data/checkpoints'),
    'output_dir': Path('./data/evaluation_results'),
}

MODELS_TO_EVALUATE = [
    {
        'name': 'Upper Bound EASY',
        'checkpoint': 'hopper-target-easy_ppo/model_best.zip',
        'test_envs': ['CustomHopper-target-easy-v0'],
        'description': 'PPO trained directly on target-easy (theoretical upper bound)',
    },
    {
        'name': 'Upper Bound MEDIUM',
        'checkpoint': 'hopper-target-medium_ppo/model_best.zip',
        'test_envs': ['CustomHopper-target-medium-v0'],
        'description': 'PPO trained directly on target-medium (theoretical upper bound)',
    },
    {
        'name': 'Upper Bound HARD',
        'checkpoint': 'hopper-target-hard_ppo/model_best.zip',
        'test_envs': ['CustomHopper-target-hard-v0'],
        'description': 'PPO trained directly on target-hard (theoretical upper bound)',
    },
    {
        'name': 'ADR Easy',
        'checkpoint': 'hopper-source_adr_boundary_easy/model_best.zip',
        'test_envs': ['CustomHopper-target-easy-v0', 'CustomHopper-target-medium-v0', 'CustomHopper-target-hard-v0'],
        'description': 'ADR trained with EASY limits (masses ±30%)',
    },
    {
        'name': 'ADR Medium',
        'checkpoint': 'hopper-source_adr_boundary_medium/model_best.zip',
        'test_envs': ['CustomHopper-target-easy-v0', 'CustomHopper-target-medium-v0', 'CustomHopper-target-hard-v0'],
        'description': 'ADR trained with MEDIUM limits (masses ±40%, friction ±40%)',
    },
    {
        'name': 'ADR Hard',
        'checkpoint': 'hopper-source_adr_boundary_hard/model_best.zip',
        'test_envs': ['CustomHopper-target-easy-v0', 'CustomHopper-target-medium-v0', 'CustomHopper-target-hard-v0'],
        'description': 'ADR trained with HARD limits (masses ±60%, friction, gravity, forces)',
    },
]

# EVALUATION FUNCTIONS
def evaluate_model_on_env(model_path, env_id, n_episodes=50, max_steps=500, deterministic=True):
    """
    Evaluate a single model on a single environment.
    Returns dict with statistics.
    """
    print(f"\n{'─'*60}")
    print(f"Evaluating: {model_path.name}")
    print(f"Environment: {env_id}")
    print(f"{'─'*60}")
    
    try:
        model = PPO.load(model_path, device='cpu')
        
        # Use centralized environment creation
        env = create_eval_env(env_id, model_path, render=False)
        
        episode_rewards = []
        episode_lengths = []
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
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        env.close()
        
        results = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'median_reward': float(np.median(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'success': True,
            'error': None,
        }
        
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"   Median: {results['median_reward']:.2f}, Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return {
            'mean_reward': 0.0,
            'std_reward': 0.0,
            'median_reward': 0.0,
            'min_reward': 0.0,
            'max_reward': 0.0,
            'mean_length': 0.0,
            'success': False,
            'error': str(e),
        }


def run_comprehensive_evaluation():
    """
    Run evaluation on all configured models and environments.
    Returns list of result dictionaries.
    """
    print("\n" + "="*70)
    print("  COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    print(f"Configuration:")
    print(f"  Episodes per env: {EVALUATION_CONFIG['n_episodes']}")
    print(f"  Max steps: {EVALUATION_CONFIG['max_steps']}")
    print(f"  Deterministic: {EVALUATION_CONFIG['deterministic']}")
    print(f"  Total evaluations: {sum(len(m['test_envs']) for m in MODELS_TO_EVALUATE)}")
    print("="*70)
    
    output_dir = EVALUATION_CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for model_config in MODELS_TO_EVALUATE:
        model_name = model_config['name']
        checkpoint_path = EVALUATION_CONFIG['base_checkpoint_dir'] / model_config['checkpoint']
        
        if not checkpoint_path.exists():
            print(f"\nWARNING: Model not found: {checkpoint_path}")
            print(f"   Skipping {model_name}...")
            continue
        
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"Path: {checkpoint_path}")
        print(f"Description: {model_config['description']}")
        print(f"{'='*70}")
        
        for env_id in model_config['test_envs']:
            if 'easy' in env_id:
                difficulty = 'Easy'
            elif 'medium' in env_id:
                difficulty = 'Medium'
            elif 'hard' in env_id:
                difficulty = 'Hard'
            else:
                difficulty = 'Unknown'
            
            result = evaluate_model_on_env(
                checkpoint_path,
                env_id,
                n_episodes=EVALUATION_CONFIG['n_episodes'],
                max_steps=EVALUATION_CONFIG['max_steps'],
                deterministic=EVALUATION_CONFIG['deterministic'],
            )
            
            all_results.append({
                'Model': model_name,
                'Target Difficulty': difficulty,
                'Mean Reward': result['mean_reward'],
                'Std Reward': result['std_reward'],
                'Median Reward': result['median_reward'],
                'Min Reward': result['min_reward'],
                'Max Reward': result['max_reward'],
                'Mean Length': result['mean_length'],
                'Success': result['success'],
                'Error': result['error'],
            })
    
    return all_results


# MAIN EXECUTION

def main():
    """Main execution function."""
    results = run_comprehensive_evaluation()
    if not results:
        print("\nNo results collected. Check that models exist in checkpoints directory.")
        return
    
    results_df = export_results(results, EVALUATION_CONFIG['output_dir'])
    
    print("\n" + "="*70)
    print("  GENERATING VISUALIZATIONS")
    print("="*70)
    
    pivot_table = create_results_table(results_df)
    create_heatmap(results_df, EVALUATION_CONFIG['output_dir'])
    create_bar_chart(results_df, EVALUATION_CONFIG['output_dir'])
    
    print("\n" + "="*70)
    print("  EVALUATION COMPLETE")
    print("="*70)
    print(f"Total models evaluated: {len(results_df['Model'].unique())}")
    print(f"Total environments tested: {len(results_df)}")
    print(f"Results saved to: {EVALUATION_CONFIG['output_dir']}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

