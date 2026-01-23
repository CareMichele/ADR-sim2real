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
        'target_performance': 1500.0,
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

def train_agent(variant_name, environment='hopper', total_timesteps=1000000, 
                update_freq=5000, use_adr=True):
    """
    Allena agente PPO con o senza ADR su qualsiasi environment.
    
    Args:
        variant_name: Nome della variante ADR ('vanilla', 'boundary', ecc.) - ignorato se use_adr=False
        environment: Nome dell'environment ('hopper', 'ant', ecc.)
        total_timesteps: Numero totale di timesteps per il training
        update_freq: Frequenza di valutazione (ogni quanti step)
        use_adr: Se True usa ADR, se False usa solo PPO
    """
    
    # PPO hyperparameters (locali)
    ppo_learning_rate = 3e-4
    ppo_batch_size = 64
    ppo_n_epochs = 10
    ppo_verbose = 1
    
    # Setup directories
    if use_adr:
        log_dir = Path(f"./logs/{environment}_adr_{variant_name}")
        checkpoint_dir = Path(f"./checkpoints/{environment}_adr_{variant_name}")
        plot_dir = Path(f"./imgs/{environment}_adr_{variant_name}")
    else:
        log_dir = Path(f"./logs/{environment}_ppo")
        checkpoint_dir = Path(f"./checkpoints/{environment}_ppo")
        plot_dir = Path(f"./imgs/{environment}_ppo")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    env_config = ENV_CONFIGS[environment]
    
    print(f"\n{'='*70}")
    if use_adr:
        print(f"  ADR Training - Environment: {environment.upper()}")
        print(f"  Variant: {variant_name.upper()}")
        print(f"  {ADR_VARIANTS[variant_name]['description']}")
    else:
        print(f"  PPO Training (no ADR) - Environment: {environment.upper()}")
    print(f"{'='*70}\n")
    
    # Training environment
    env_train = gym.make(env_config['env_id'])
    
    # Setup ADR if enabled
    if use_adr:
        variant_config = ADR_VARIANTS[variant_name].copy()
        adr_manager = ADRManager(
            variant_config, 
            env_config['target_performance'],
            env_type=environment
        )
        
        print(f"Configuration:")
        print(f"  Environment: {env_config['env_id']}")
        print(f"  ADR Enabled: Yes")
        print(f"  Delta: {adr_manager.delta}")
        print(f"  Threshold iniziale: {adr_manager.threshold:.1f}")
        print(f"  Target Performance: {env_config['target_performance']:.1f}")
        print(f"  Update Frequency: {update_freq} steps")
        print(f"  Parameters to randomize: {adr_manager.params_to_randomize}\n")
        
        env_train = ADRWrapper(env_train, adr_manager)
    else:
        adr_manager = None
        print(f"Configuration:")
        print(f"  Environment: {env_config['env_id']}")
        print(f"  ADR Enabled: No (vanilla PPO)")
        print(f"  Evaluation Frequency: {update_freq} steps\n") 
    
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
    
    training_history = []
    num_updates = total_timesteps // update_freq
    
    print(f"\n[INFO] Inizio training")
    print(f"       Timestep totali: {total_timesteps}")
    print(f"       Evaluation freq: {update_freq}")
    print(f"       Numero valutazioni: {num_updates}\n")
    
    for update_idx in range(num_updates):
        # Train
        model.learn(
            total_timesteps=update_freq,
            reset_num_timesteps=False,
            progress_bar=False  # Disabilitato per evitare dipendenze tqdm/rich
        )
        
        # Test on current distribution
        test_episodes = 20
        test_rewards = []
        
        eval_label = "ADR Update" if use_adr else "Evaluation"
        print(f"\n[{eval_label} {update_idx + 1}/{num_updates}]")
        print(f"Timestep totali: {model.num_timesteps}")
        
        for test_ep in range(test_episodes):
            obs, info = env_train.reset()
            episode_return = 0
            done = False
            
            for _ in range(500):
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
        
        print(f"Test reward: mean={mean_reward:.2f}, std={std_reward:.2f}")
        print(f"             range=[{min_reward:.2f}, {max_reward:.2f}]")
        
        # Update ADR if enabled
        if use_adr:
            status = adr_manager.update_ranges(mean_reward)
            diversity = adr_manager.get_range_diversity()
            
            print(f"\nADR Status: {status}")
            print(f"Threshold: {adr_manager.threshold:.1f}")
            print(f"Diversity: {diversity:.3f}")
            print(f"Ranges:")
            for param, (lower, upper) in adr_manager.ranges.items():
                print(f"  {param:5s}: [{lower:.3f}, {upper:.3f}]  width={upper-lower:.3f}")
            
            training_history.append({
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
        else:
            # Solo reward history per PPO puro
            training_history.append({
                'update': update_idx + 1,
                'timestep': model.num_timesteps,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'min_reward': float(min_reward),
                'max_reward': float(max_reward),
            })
    
    # Save final model
    final_model_path = checkpoint_dir / "model_final.zip"
    model.save(str(final_model_path))
    print(f"\n✅ Training completato! Model salvato: {final_model_path}")
    
    # Save history
    history_filename = "adr_history.json" if use_adr else "training_history.json"
    history_path = log_dir / history_filename
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"📊 History salvato: {history_path}")
    
    # Generate plots
    print("\n[INFO] Generazione plot...")
    try:
        plot_path_main = plot_dir / "training_history.png"
        plot_training_history(history_path, save_path=plot_path_main, show=False)
        
        # Plot ranges solo se ADR è abilitato
        if use_adr:
            plot_path_ranges = plot_dir / "all_ranges.png"
            plot_all_ranges(history_path, save_path=plot_path_ranges, show=False)
        
        print("✅ Plot generati con successo!")
    except Exception as e:
        print(f"⚠️ Warning: Errore nella generazione plot: {e}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    if use_adr:
        print(f"Variant: {variant_name}")
    else:
        print(f"Mode: PPO (no ADR)")
    print(f"Environment: {environment}")
    print(f"Timesteps: {model.num_timesteps}")
    print(f"Final reward (mean last 5): {np.mean([h['mean_reward'] for h in training_history[-5:]]):.2f}")
    if use_adr:
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
        help='ADR variant to use (ignored if --no-adr is set)'
    )
    
    parser.add_argument(
        '--no-adr',
        action='store_true',
        help='Train with vanilla PPO without ADR'
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
    print(f"Mode:              {'PPO (no ADR)' if args.no_adr else f'ADR ({args.variant})'}")
    if not args.no_adr:
        print(f"ADR Variant:       {args.variant}")
    print(f"Total Timesteps:   {args.timesteps:,}")
    print(f"Update Frequency:  {args.update_freq:,}")
    if not args.no_adr:
        print(f"Target Performance: {ENV_CONFIGS[args.env]['target_performance']:.1f}")
    print(f"Random Seed:       42 (fixed)")
    print("="*60 + "\n")
    
    # Launch training
    train_agent(
        args.variant, 
        environment=args.env, 
        total_timesteps=args.timesteps,
        update_freq=args.update_freq,
        use_adr=not args.no_adr
    )