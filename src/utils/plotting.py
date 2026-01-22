"""
Utility functions for plotting training metrics.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


def plot_training_history(history_path, save_path=None, show=True):
    """
    Plotta le metriche di training da un file JSON history.
    
    Args:
        history_path: Path al file adr_history.json
        save_path: Path dove salvare il plot (opzionale)
        show: Se True, mostra il plot a schermo
    """
    # Carica history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Estrai dati
    timesteps = [h['timestep'] for h in history]
    mean_rewards = [h['mean_reward'] for h in history]
    std_rewards = [h['std_reward'] for h in history]
    thresholds = [h['threshold'] for h in history]
    diversities = [h['diversity'] for h in history]
    
    # Crea figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ADR Training History', fontsize=16, fontweight='bold')
    
    # 1. Reward nel tempo con threshold
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
    
    # 2. Diversity (larghezza range ADR)
    ax2 = axes[0, 1]
    ax2.plot(timesteps, diversities, 'g-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Timesteps', fontsize=12)
    ax2.set_ylabel('Range Diversity', fontsize=12)
    ax2.set_title('ADR Range Diversity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Status ADR (EXPAND vs CONTRACT)
    ax3 = axes[1, 0]
    statuses = [h['status'] for h in history]
    expand_count = statuses.count('EXPAND')
    contract_count = statuses.count('CONTRACT')
    ax3.bar(['EXPAND', 'CONTRACT'], [expand_count, contract_count], 
            color=['green', 'red'], alpha=0.7)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('ADR Actions Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Range evolution nel tempo (esempio per primo parametro)
    ax4 = axes[1, 1]
    # Prendi il primo parametro di massa
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
    
    plt.tight_layout()
    
    # Salva se richiesto
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot salvato in: {save_path}")
    
    # Mostra se richiesto
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_ranges(history_path, save_path=None, show=True):
    """
    Plotta l'evoluzione di TUTTI i range ADR nel tempo.
    
    Args:
        history_path: Path al file adr_history.json
        save_path: Path dove salvare il plot (opzionale)
        show: Se True, mostra il plot a schermo
    """
    # Carica history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    timesteps = [h['timestep'] for h in history]
    
    # Trova tutti i parametri
    all_params = list(history[0]['ranges'].keys())
    n_params = len(all_params)
    
    # Calcola dimensioni griglia
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    fig.suptitle('ADR All Parameters Range Evolution', fontsize=16, fontweight='bold')
    
    if n_rows == 1:
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
        
        # Linea nominale (diversa per gravity)
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
    
    # Rimuovi subplot vuoti
    for idx in range(n_params, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot salvato in: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_quick_summary(history_path):
    """
    Crea un plot veloce di summary per visualizzazione rapida durante training.
    
    Args:
        history_path: Path al file adr_history.json
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


if __name__ == "__main__":
    # Test: carica e plotta un history esistente
    import sys
    if len(sys.argv) > 1:
        history_file = Path(sys.argv[1])
        if history_file.exists():
            print(f"Plotting {history_file}...")
            plot_training_history(history_file)
            plot_all_ranges(history_file)
        else:
            print(f"File non trovato: {history_file}")
    else:
        print("Usage: python plotting.py <path_to_adr_history.json>")
