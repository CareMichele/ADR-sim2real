import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json


def plot_training_history(history_path, save_path=None, show=True):
    """
    Plot training metrics from JSON history file.
    Handles both ADR and vanilla PPO.
    
    Args:
        history_path: Path to adr_history.json or training_history.json
        save_path: Optional path to save the plot
        show: If True, display the plot
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    timesteps = [h['timestep'] for h in history]
    mean_rewards = [h['mean_reward'] for h in history]
    std_rewards = [h['std_reward'] for h in history]
    
    is_adr = 'threshold' in history[0]
    
    if is_adr:
        thresholds = [h['threshold'] for h in history]
        diversities = [h['diversity'] for h in history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ADR Training History', fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        ax1.plot(timesteps, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
        ax1.fill_between(timesteps, 
                          np.array(mean_rewards) - np.array(std_rewards),
                          np.array(mean_rewards) + np.array(std_rewards),
                          alpha=0.3, color='b', label='±1 Std')
        ax1.plot(timesteps, thresholds, 'r--', linewidth=2, label='Threshold')
        ax1.set_xlabel('Timesteps', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.set_title('Reward vs Threshold', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        ax2.plot(timesteps, diversities, 'g-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Timesteps', fontsize=12)
        ax2.set_ylabel('Range Diversity', fontsize=12)
        ax2.set_title('ADR Range Diversity', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        statuses = [h['status'] for h in history]
        expand_count = statuses.count('EXPAND')
        contract_count = statuses.count('CONTRACT')
        ax3.bar(['EXPAND', 'CONTRACT'], [expand_count, contract_count], 
                color=['green', 'red'], alpha=0.7)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('ADR Actions Distribution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        ax4 = axes[1, 1]
        first_mass_param = None
        for key in history[0]['ranges'].keys():
            if key not in ['friction', 'damping', 'gravity', 'force_magnitude']:
                first_mass_param = key
                break
        
        if first_mass_param:
            lower_bounds = [h['ranges'][first_mass_param][0] for h in history]
            upper_bounds = [h['ranges'][first_mass_param][1] for h in history]
            ax4.plot(timesteps, lower_bounds, 'r-', linewidth=2, label='Lower Bound')
            ax4.plot(timesteps, upper_bounds, 'b-', linewidth=2, label='Upper Bound')
            ax4.fill_between(timesteps, lower_bounds, upper_bounds, alpha=0.3, color='gray')
            ax4.axhline(y=1.0, color='k', linestyle='--', linewidth=1, label='Nominal')
            ax4.set_xlabel('Timesteps', fontsize=12)
            ax4.set_ylabel('Range', fontsize=12)
            ax4.set_title(f'Range Evolution: {first_mass_param}', fontsize=14, fontweight='bold')
            ax4.legend(loc='best')
            ax4.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('PPO Training History (no ADR)', fontsize=16, fontweight='bold')
        
        ax.plot(timesteps, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
        ax.fill_between(timesteps, 
                         np.array(mean_rewards) - np.array(std_rewards),
                         np.array(mean_rewards) + np.array(std_rewards),
                         alpha=0.3, color='b', label='±1 Std')
        ax.set_xlabel('Timesteps', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Training Reward', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_ranges(history_path, save_path=None, show=True):
    """
    Plot evolution of all ADR ranges over time.
    
    Args:
        history_path: Path to adr_history.json
        save_path: Optional path to save the plot
        show: If True, display the plot
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    timesteps = [h['timestep'] for h in history]
    
    all_params = list(history[0]['ranges'].keys())
    n_params = len(all_params)
    
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    fig.suptitle('ADR All Parameters Range Evolution', fontsize=16, fontweight='bold')
    
    if n_params == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, param in enumerate(all_params):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        lower_bounds = [h['ranges'][param][0] for h in history]
        upper_bounds = [h['ranges'][param][1] for h in history]
        
        ax.plot(timesteps, lower_bounds, 'r-', linewidth=2, label='Lower')
        ax.plot(timesteps, upper_bounds, 'b-', linewidth=2, label='Upper')
        ax.fill_between(timesteps, lower_bounds, upper_bounds, alpha=0.3, color='gray')
        
        if param == 'gravity':
            ax.axhline(y=-9.81, color='k', linestyle='--', linewidth=1, label='Nominal')
        elif param == 'force_magnitude':
            ax.axhline(y=0.0, color='k', linestyle='--', linewidth=1, label='Nominal')
        else:
            ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, label='Nominal')
        
        ax.set_xlabel('Timesteps', fontsize=10)
        ax.set_ylabel('Range', fontsize=10)
        ax.set_title(f'{param}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_params, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_quick_summary(history_path):
    """
    Create quick summary plot for fast visualization during training.
    
    Args:
        history_path: Path to adr_history.json
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    timesteps = [h['timestep'] for h in history]
    mean_rewards = [h['mean_reward'] for h in history]
    thresholds = [h['threshold'] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
    plt.plot(timesteps, thresholds, 'r--', linewidth=2, label='Threshold')
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('ADR Training Progress', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_results_table(results_df):
    """Create and print formatted results table."""
    print("\n" + "="*100)
    print("  EVALUATION RESULTS SUMMARY")
    print("="*100)
    
    pivot_table = results_df.pivot_table(
        values='Mean Reward',
        index='Model',
        columns='Target Difficulty',
        aggfunc='first'
    )
    
    print("\nMean Rewards by Model and Target Difficulty:")
    print(pivot_table.to_string())
    
    print("\n\nDetailed Results:")
    print(results_df[['Model', 'Target Difficulty', 'Mean Reward', 'Std Reward', 'Median Reward']].to_string(index=False))
    
    return pivot_table


def create_heatmap(results_df, output_dir):
    """Create heatmap visualization of results."""
    pivot_data = results_df.pivot_table(
        values='Mean Reward',
        index='Model',
        columns='Target Difficulty',
        aggfunc='first'
    )
    
    column_order = ['Easy', 'Medium', 'Hard']
    pivot_data = pivot_data[[col for col in column_order if col in pivot_data.columns]]
    
    row_order = ['Upper Bound EASY', 'Upper Bound MEDIUM', 'Upper Bound HARD', 
                 'ADR Easy', 'ADR Medium', 'ADR Hard']
    pivot_data = pivot_data.reindex([row for row in row_order if row in pivot_data.index])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
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
    
    output_path = output_dir / 'evaluation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved: {output_path}")
    plt.close()


def create_bar_chart(results_df, output_dir):
    """Create grouped bar chart comparing models."""
    adr_models = results_df[results_df['Model'].str.contains('ADR')]
    
    if len(adr_models) > 0:
        plt.figure(figsize=(14, 6))
        
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
        print(f"Bar chart saved: {output_path}")
        plt.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        history_file = Path(sys.argv[1])
        if history_file.exists():
            print(f"Plotting {history_file}...")
            plot_training_history(history_file)
            plot_all_ranges(history_file)
        else:
            print(f"File not found: {history_file}")
    else:
        print("Usage: python plotting.py <path_to_adr_history.json>")
