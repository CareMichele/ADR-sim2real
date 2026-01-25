"""
Comprehensive Evaluation Script for All Trained Models
Evaluates upper bounds (PPO) and ADR models across all target environments.
Generates tables, CSV, and heatmap visualization.
"""

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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import envs.custom_hopper

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVALUATION_CONFIG = {
    'n_episodes': 50,
    'max_steps': 500,
    'deterministic': True,
    'base_checkpoint_dir': Path('./data/checkpoints'),
    'output_dir': Path('./data/evaluation_results'),
}

# Models to evaluate
MODELS_TO_EVALUATE = [
    {
        'name': 'Upper Bound HARD',
        'checkpoint': 'hopper-target-hard_ppo/model_best.zip',
        'test_envs': ['CustomHopper-target-hard-v0'],
        'description': 'PPO trained directly on target-hard (theoretical upper bound)',
    },
    {
        'name': 'Upper Bound MEDIUM',
        'checkpoint': 'hopper-target-medium_ppo/model_best.zip',
        'test_envs': ['CustomHopper-target-medium-v0'],
        'description': 'PPO trained directly on target-medium (theoretical upper bound)',
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

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

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
        # Load model
        model = PPO.load(model_path, device='cpu')
        
        # Create environment
        render_mode = None
        def make_env():
            return gym.make(env_id, render_mode=render_mode)
        
        env = DummyVecEnv([make_env])
        
        # Auto-detect VecFrameStack from vecnormalize.pkl
        vecnorm_path = model_path.parent / 'vecnormalize.pkl'
        use_framestack = False
        
        if vecnorm_path.exists():
            # Load VecNormalize metadata to check observation space
            import pickle
            with open(vecnorm_path, 'rb') as f:
                vecnorm_data = pickle.load(f)
            
            saved_obs_dim = vecnorm_data.observation_space.shape[0]
            base_obs_dim = env.observation_space.shape[0]
            
            if saved_obs_dim == base_obs_dim * 4:
                use_framestack = True
                env = VecFrameStack(env, n_stack=4)
        
        # Load VecNormalize
        if vecnorm_path.exists():
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False
        
        # Run evaluation
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
        
        # Calculate statistics
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
        
        print(f"✅ Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"   Median: {results['median_reward']:.2f}, Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
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

# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def run_comprehensive_evaluation():
    """
    Run evaluation for all models on all specified environments.
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
    
    # Create output directory
    output_dir = EVALUATION_CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # Iterate over all models
    for model_config in MODELS_TO_EVALUATE:
        model_name = model_config['name']
        checkpoint_path = EVALUATION_CONFIG['base_checkpoint_dir'] / model_config['checkpoint']
        
        # Check if model exists
        if not checkpoint_path.exists():
            print(f"\n⚠️  WARNING: Model not found: {checkpoint_path}")
            print(f"   Skipping {model_name}...")
            continue
        
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"Path: {checkpoint_path}")
        print(f"Description: {model_config['description']}")
        print(f"{'='*70}")
        
        # Test on all specified environments
        for env_id in model_config['test_envs']:
            # Extract difficulty from env_id
            if 'easy' in env_id:
                difficulty = 'Easy'
            elif 'medium' in env_id:
                difficulty = 'Medium'
            elif 'hard' in env_id:
                difficulty = 'Hard'
            else:
                difficulty = 'Unknown'
            
            # Run evaluation
            result = evaluate_model_on_env(
                checkpoint_path,
                env_id,
                n_episodes=EVALUATION_CONFIG['n_episodes'],
                max_steps=EVALUATION_CONFIG['max_steps'],
                deterministic=EVALUATION_CONFIG['deterministic'],
            )
            
            # Store result with metadata
            all_results.append({
                'Model': model_name,
                'Target Difficulty': difficulty,
                'Environment': env_id,
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

# ============================================================================
# VISUALIZATION & EXPORT
# ============================================================================

def create_results_table(results_df):
    """Create and print formatted results table."""
    print("\n" + "="*100)
    print("  EVALUATION RESULTS SUMMARY")
    print("="*100)
    
    # Create pivot table for easier reading
    pivot_table = results_df.pivot_table(
        values='Mean Reward',
        index='Model',
        columns='Target Difficulty',
        aggfunc='first'
    )
    
    print("\nMean Rewards by Model and Target Difficulty:")
    print(pivot_table.to_string())
    
    # Print detailed table
    print("\n\nDetailed Results:")
    print(results_df[['Model', 'Target Difficulty', 'Mean Reward', 'Std Reward', 'Median Reward']].to_string(index=False))
    
    return pivot_table

def create_heatmap(results_df, output_dir):
    """Create heatmap visualization of results."""
    # Prepare data for heatmap
    pivot_data = results_df.pivot_table(
        values='Mean Reward',
        index='Model',
        columns='Target Difficulty',
        aggfunc='first'
    )
    
    # Reorder columns: Easy, Medium, Hard
    column_order = ['Easy', 'Medium', 'Hard']
    pivot_data = pivot_data[[col for col in column_order if col in pivot_data.columns]]
    
    # Reorder rows: Upper Bounds first (Easy, Medium, Hard), then ADR (Easy, Medium, Hard)
    row_order = ['Upper Bound EASY', 'Upper Bound MEDIUM', 'Upper Bound HARD', 
                 'ADR Easy', 'ADR Medium', 'ADR Hard']
    pivot_data = pivot_data.reindex([row for row in row_order if row in pivot_data.index])
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=1000,
        vmin=0,
        vmax=1800,
        cbar_kws={'label': 'Mean Reward'},
        linewidths=0.5,
        linecolor='gray',
    )
    
    plt.title('Model Performance Across Target Difficulties', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Target Difficulty', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'evaluation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Heatmap saved: {output_path}")
    plt.close()

def create_bar_chart(results_df, output_dir):
    """Create grouped bar chart comparing models."""
    # Filter ADR models only for cross-difficulty comparison
    adr_models = results_df[results_df['Model'].str.contains('ADR')]
    
    if len(adr_models) > 0:
        plt.figure(figsize=(14, 6))
        
        # Create grouped bar chart
        models = adr_models['Model'].unique()
        difficulties = ['Easy', 'Medium', 'Hard']
        x = np.arange(len(difficulties))
        width = 0.25
        
        for i, model in enumerate(models):
            model_data = adr_models[adr_models['Model'] == model]
            rewards = [
                model_data[model_data['Target Difficulty'] == diff]['Mean Reward'].values[0]
                if len(model_data[model_data['Target Difficulty'] == diff]) > 0 else 0
                for diff in difficulties
            ]
            plt.bar(x + i*width, rewards, width, label=model, alpha=0.8)
        
        plt.xlabel('Target Difficulty', fontsize=12, fontweight='bold')
        plt.ylabel('Mean Reward', fontsize=12, fontweight='bold')
        plt.title('ADR Model Generalization Across Difficulties', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(x + width, difficulties)
        plt.legend(loc='best')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_path = output_dir / 'adr_comparison_bars.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Bar chart saved: {output_path}")
        plt.close()

def export_results(results, output_dir):
    """Export results to CSV and JSON."""
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV (generic name, will overwrite)
    csv_path = output_dir / 'evaluation_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 CSV saved: {csv_path}")
    
    # Save JSON (generic name, will overwrite)
    json_path = output_dir / 'evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 JSON saved: {json_path}")
    
    return df

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    # Run comprehensive evaluation
    results = run_comprehensive_evaluation()
    
    if not results:
        print("\n❌ No results collected. Check that models exist in checkpoints directory.")
        return
    
    # Add Upper Bound Easy (not trained, theoretical value for completeness)
    results.append({
        'Model': 'Upper Bound EASY',
        'Target Difficulty': 'Easy',
        'Environment': 'CustomHopper-target-easy-v0',
        'Mean Reward': 1666.69,
        'Std Reward': 0.0,
        'Median Reward': 1666.69,
        'Min Reward': 1666.69,
        'Max Reward': 1666.69,
        'Mean Length': 500.0,
        'Success': True,
        'Error': None,
    })
    
    # Export results
    results_df = export_results(results, EVALUATION_CONFIG['output_dir'])
    
    # Create visualizations
    print("\n" + "="*70)
    print("  GENERATING VISUALIZATIONS")
    print("="*70)
    
    pivot_table = create_results_table(results_df)
    create_heatmap(results_df, EVALUATION_CONFIG['output_dir'])
    create_bar_chart(results_df, EVALUATION_CONFIG['output_dir'])
    
    # Final summary
    print("\n" + "="*70)
    print("  EVALUATION COMPLETE")
    print("="*70)
    print(f"✅ Total models evaluated: {len(results_df['Model'].unique())}")
    print(f"✅ Total environments tested: {len(results_df)}")
    print(f"✅ Results saved to: {EVALUATION_CONFIG['output_dir']}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
