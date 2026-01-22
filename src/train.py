"""
ADR Training Loop - UNIVERSAL VERSION
Versione universale compatibile con qualsiasi environment Gymnasium/MuJoCo.
Supporta configurazioni per diversi environment (Hopper, Walker, Humanoid, ecc.)
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from pathlib import Path
import json
import sys
import argparse

#import ADR Manager
from adr_manager import ADRManager

#import ADR Wrapper
from adr_wrapper import ADRWrapper

# Import plotting utilities
from utils.plotting import plot_training_history, plot_all_ranges

from envs import custom_hopper
from envs import custom_ant
# ============================================================================
# ENVIRONMENT CONFIGURATIONS (Universal - può supportare qualsiasi env)
# ============================================================================

ENV_CONFIGS = {
    'hopper': {
        'env_id': 'CustomHopper-source-v0',
        'target_performance': 1666.0,
    },
    # Puoi aggiungere altri environment qui in futuro
    'ant': {
        'env_id': 'CustomAnt-source-v0',
        'target_performance': 3000.0, #TODO capire se va bene 3000 facendo training target -> target
    },
}

# ============================================================================
# ADR VARIANTS CONFIGURATION (Universal - funziona per tutti gli env)
# ============================================================================

ADR_VARIANTS = {
    'vanilla': {
        'description': 'Symmetric expansion/contraction (baseline)',
        'delta': 0.05, #Di quanto adr allarga il range 
        'threshold_pct': 0.75, #Percentuale rispetto a target performance oltre il quale adr allarga range
        'boundary_sampling': False,
        'progressive': False,
    },
    'boundary': {
        'description': 'Boundary sampling (50% at extremes)',
        'delta': 0.05,
        'threshold_pct': 0.75,
        'boundary_sampling': True, #Al posto che scegliere valore a caso dentro il range, spesso sceglie i bordi 
        'boundary_prob': 0.5, #Per il 50% delle volte dammi caso peggiore (bordo del range)
        'progressive': False,
    },
    'progressive': {
        'description': 'Progressive curriculum (increasing threshold)',
        'delta': 0.05,
        'threshold_schedule': [0.60, 0.70, 0.80], #Come prima ma la th aumenta progressivamente
        'boundary_sampling': False,
        'progressive': True,
    },
    'selective': {# in caso in cui vogliamo testare un parametro specifico 
        'description': 'Only randomize critical parameters (thigh)',
        'delta': 0.05,
        'threshold_pct': 0.75,
        'randomize_only': ['thigh'],
        'boundary_sampling': False,
        'progressive': False,
    }
}

# ============================================================================
# CONFIGURATION (Universal Settings)
# ============================================================================
# TRAINING FUNCTION (Universal)
# ============================================================================

def train_adr_variant(variant_name, environment='hopper', total_timesteps=1000000, 
                      update_freq=5000):
    """
    Allena ADR con la variante specificata su qualsiasi environment.
    
    Args:
        variant_name: Nome della variante ADR ('vanilla', 'boundary', ecc.)
        environment: Nome dell'environment ('hopper', 'walker', ecc.)
        total_timesteps: Numero totale di timesteps per il training
        update_freq: Frequenza di aggiornamento ADR (ogni quanti step)
    """
    
    # PPO hyperparameters (locali)
    ppo_learning_rate = 3e-4
    ppo_batch_size = 64
    ppo_n_epochs = 10
    ppo_verbose = 1
    
    # Setup directories
    log_dir = Path(f"./logs/{environment}_adr_{variant_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(f"./checkpoints/{environment}_adr_{variant_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    plot_dir = Path(f"./imgs/{environment}_adr_{variant_name}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    env_config = ENV_CONFIGS[environment]
    
    print(f"\n{'='*70}")
    print(f"  ADR Training - Environment: {environment.upper()}")
    print(f"  Variant: {variant_name.upper()}")
    print(f"  {ADR_VARIANTS[variant_name]['description']}")
    print(f"{'='*70}\n")
    
    # Training environment (with ADR)
    env_train = gym.make(env_config['env_id'])
    
    # Clone variant config to avoid modifying the global dictionary
    variant_config = ADR_VARIANTS[variant_name].copy()
    
    adr_manager = ADRManager(
        variant_config, 
        env_config['target_performance'],
        env_type=environment
    )
    
    print(f"Configuration:")
    print(f"  Environment: {env_config['env_id']}")
    print(f"  Delta: {adr_manager.delta}")
    print(f"  Threshold iniziale: {adr_manager.threshold:.1f}")
    print(f"  Target Performance: {env_config['target_performance']:.1f}")
    print(f"  Update Frequency: {update_freq} steps")
    print(f"  Parameters to randomize: {adr_manager.params_to_randomize}\n")
    
    env_train = ADRWrapper(env_train, adr_manager) 
    
    print("[INFO] Inizializzazione modello PPO...")
    model = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=ppo_learning_rate,
        batch_size=ppo_batch_size,
        n_epochs=ppo_n_epochs,
        verbose=ppo_verbose,
        device='cpu'  # Forza uso CPU invece di GPU
    )
    
    adr_history = []
    num_updates = total_timesteps // update_freq
    
    print(f"\n[INFO] Inizio training")
    print(f"       Timestep totali: {total_timesteps}")
    print(f"       ADR update freq: {update_freq}")
    print(f"       Numero update: {num_updates}\n")
    
    for update_idx in range(num_updates):
        # Train
        model.learn(
            total_timesteps=update_freq,
            reset_num_timesteps=False,
            progress_bar=False  # Disabilitato per evitare dipendenze tqdm/rich
        )
        
        # Test on ADR distribution (randomized masses)
        # IMPORTANT: We test on the CURRENT ADR distribution to decide if we should expand
        test_episodes = 20  # More episodes to reduce variance from randomization
        test_rewards = []
        
        print(f"\n[ADR Update {update_idx + 1}/{num_updates}]")
        print(f"Timestep totali: {model.num_timesteps}")
        
        for test_ep in range(test_episodes):
            obs, info = env_train.reset()  # ← Uses ADR! Samples random masses
            episode_return = 0
            done = False
            
            for _ in range(500):
                #durante Evaluation:
                    #deterministic = True -> azioni deterministiche (senza rumore, la migliore)
                #durante Training:
                    #azioni stocastiche (PPO esplora aggiungendo rumore)
                action, _ = model.predict(obs, deterministic=True)  
                obs, reward, terminated, truncated, info = env_train.step(action)
                episode_return += reward
                done = terminated or truncated
                if done:
                    break
            
            test_rewards.append(episode_return)
        
        mean_reward = np.mean(test_rewards)
        std_reward = np.std(test_rewards)
        min_reward = np.min(test_rewards)
        max_reward = np.max(test_rewards)
        
        print(f"Test reward (ADR dist): mean={mean_reward:.2f}, std={std_reward:.2f}")
        print(f"                        range=[{min_reward:.2f}, {max_reward:.2f}]")
        
        # Update ADR ranges based on performance on randomized distribution
        status = adr_manager.update_ranges(mean_reward)
        diversity = adr_manager.get_range_diversity()
        
        print(f"\nADR Status: {status}")
        print(f"Threshold: {adr_manager.threshold:.1f}")
        print(f"Diversity: {diversity:.3f}")
        print(f"Ranges:")
        for param, (lower, upper) in adr_manager.ranges.items():
            print(f"  {param:5s}: [{lower:.3f}, {upper:.3f}]  width={upper-lower:.3f}")
        
        adr_history.append({
            'update': update_idx + 1,
            'timestep': model.num_timesteps,
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'min_reward': float(min_reward),
            'max_reward': float(max_reward),
            'threshold': float(adr_manager.threshold),
            'status': status,
            'diversity': float(diversity),
            'ranges': {k: list(v) for k, v in adr_manager.ranges.items()},
        })
    
    # Save final model
    final_model_path = checkpoint_dir / "model_final.zip"
    model.save(str(final_model_path))
    print(f"\n✅ Training completato! Model salvato: {final_model_path}")
    
    # Save history
    history_path = log_dir / "adr_history.json"
    with open(history_path, 'w') as f:
        json.dump(adr_history, f, indent=2)
    print(f"📊 History salvato: {history_path}")
    
    # Generate plots
    print("\n[INFO] Generazione plot...")
    try:
        plot_path_main = plot_dir / "training_history.png"
        plot_training_history(history_path, save_path=plot_path_main, show=False)
        
        plot_path_ranges = plot_dir / "all_ranges.png"
        plot_all_ranges(history_path, save_path=plot_path_ranges, show=False)
        
        print("✅ Plot generati con successo!")
    except Exception as e:
        print(f"⚠️ Warning: Errore nella generazione plot: {e}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Variant: {variant_name}")
    print(f"Timesteps: {model.num_timesteps}")
    print(f"Final reward (mean last 5): {np.mean([h['mean_reward'] for h in adr_history[-5:]]):.2f}")
    print(f"Final ADR ranges: {adr_manager.ranges}")
    
# ============================================================================
# MAIN (Universal)
# ============================================================================

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Train PPO agent with ADR on MuJoCo environments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--env',
        type=str,
        default='hopper',
        choices=['hopper', 'walker', 'ant'],
        help='Environment to train on'
    )
    
    parser.add_argument(
        '--variant',
        type=str,
        default='vanilla',
        choices=list(ADR_VARIANTS.keys()),
        help='ADR variant to use'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=1000000,
        help='Total training timesteps'
    )
    
    parser.add_argument(
        '--update-freq',
        type=int,
        default=5000,
        help='ADR update frequency (every N timesteps)'
    )
    
    args = parser.parse_args()
    
    # Set fixed random seed for reproducibility
    np.random.seed(42)
    print("[INFO] Random seed set to 42")
    
    # Print configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Environment:       {args.env}")
    print(f"ADR Variant:       {args.variant}")
    print(f"Total Timesteps:   {args.timesteps:,}")
    print(f"Update Frequency:  {args.update_freq:,}")
    print(f"Target Performance: {ENV_CONFIGS[args.env]['target_performance']:.1f}")
    print(f"Random Seed:       42 (fixed)")
    print("="*60 + "\n")
    
    # Launch training
    train_adr_variant(
        args.variant, 
        environment=args.env, 
        total_timesteps=args.timesteps,
        update_freq=args.update_freq
    )