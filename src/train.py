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
# ============================================================================
# ENVIRONMENT CONFIGURATIONS (Universal - può supportare qualsiasi env)
# ============================================================================

ENV_CONFIGS = {
    'hopper': {
        'env_id': 'CustomHopper-source-v0',
        'target_performance': 1666.0,
        'baseline_performance': 724.0,
        'max_episode_steps': 500,
    },
    # Puoi aggiungere altri environment qui in futuro
    'ant': {
        'env_id': 'CustomAnt-source-v0',
        'target_performance': 3000.0,
        'baseline_performance': 1200.0,
        'max_episode_steps': 1000,
    },
}

# ============================================================================
# ADR VARIANTS CONFIGURATION (Universal - funziona per tutti gli env)
# ============================================================================

ADR_VARIANTS = {
    'vanilla': {
        'description': 'Symmetric expansion/contraction (baseline)',
        'delta': 0.05,
        'threshold_pct': 0.75,
        'boundary_sampling': False,
        'progressive': False,
    },
    'boundary': {
        'description': 'Boundary sampling (50% at extremes)',
        'delta': 0.05,
        'threshold_pct': 0.75,
        'boundary_sampling': True,
        'boundary_prob': 0.5,
        'progressive': False,
    },
    'progressive': {
        'description': 'Progressive curriculum (increasing threshold)',
        'delta': 0.05,
        'threshold_schedule': [0.60, 0.70, 0.80],
        'boundary_sampling': False,
        'progressive': True,
    },
    'selective': {
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

# Seleziona quale environment usare
ENVIRONMENT = 'hopper'  # Cambia questo per usare altri environment

VARIANT = 'vanilla'    #quale variante usare
ADR_UPDATE_FREQ = 5000  #ogni quanti step aggiorna i range ADR
TOTAL_TIMESTEPS = 1000000

PPO_LEARNING_RATE = 3e-4
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_VERBOSE = 1

# Carica la config per l'environment selezionato
ENV_CONFIG = ENV_CONFIGS[ENVIRONMENT]
TARGET_TARGET_PERFORMANCE = ENV_CONFIG['target_performance']
SOURCE_TARGET_BASELINE = ENV_CONFIG['baseline_performance']

# Directory structure: logs for history, checkpoints for models, imgs for plots
LOG_DIR = Path(f"./logs/{ENVIRONMENT}_adr_{VARIANT}")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = Path(f"./checkpoints/{ENVIRONMENT}_adr_{VARIANT}")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = Path(f"./imgs/{ENVIRONMENT}_adr_{VARIANT}")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TRAINING FUNCTION (Universal)
# ============================================================================

def train_adr_variant(variant_name, environment='hopper', delta_override=None):
    """
    Allena ADR con la variante specificata su qualsiasi environment.
    
    Args:
        variant_name: Nome della variante ADR ('vanilla', 'boundary', ecc.)
        environment: Nome dell'environment ('hopper', 'walker', ecc.)
        delta_override: Se specificato, sovrascrive il valore delta della variante
    """
    
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
    
    # Override delta if specified
    if delta_override is not None:
        original_delta = variant_config.get('delta', 'N/A')
        variant_config['delta'] = delta_override
        print(f"[INFO] Overriding delta: {original_delta} → {delta_override}")
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
    print(f"  Update Frequency: {ADR_UPDATE_FREQ} steps")
    print(f"  Parameters to randomize: {adr_manager.params_to_randomize}\n")
    
    env_train = ADRWrapper(env_train, adr_manager) 
    
    print("[INFO] Inizializzazione modello PPO...")
    model = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=PPO_LEARNING_RATE,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        verbose=PPO_VERBOSE,
        device='cpu'  # Forza uso CPU invece di GPU
    )
    
    adr_history = []
    num_updates = TOTAL_TIMESTEPS // ADR_UPDATE_FREQ
    
    print(f"\n[INFO] Inizio training")
    print(f"       Timestep totali: {TOTAL_TIMESTEPS}")
    print(f"       ADR update freq: {ADR_UPDATE_FREQ}")
    print(f"       Numero update: {num_updates}\n")
    
    for update_idx in range(num_updates):
        # Train
        model.learn(
            total_timesteps=ADR_UPDATE_FREQ,
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
    final_model_path = CHECKPOINT_DIR / "model_final.zip"
    model.save(str(final_model_path))
    print(f"\n✅ Training completato! Model salvato: {final_model_path}")
    
    # Save history
    history_path = LOG_DIR / "adr_history.json"
    with open(history_path, 'w') as f:
        json.dump(adr_history, f, indent=2)
    print(f"📊 History salvato: {history_path}")
    
    # Generate plots
    print("\n[INFO] Generazione plot...")
    try:
        plot_path_main = PLOT_DIR / "training_history.png"
        plot_training_history(history_path, save_path=plot_path_main, show=False)
        
        plot_path_ranges = PLOT_DIR / "all_ranges.png"
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
    
    
    """
    # Final evaluation on target domain
    print(f"\n{'='*70}")
    print("  FINAL EVALUATION ON TARGET")
    print(f"{'='*70}")
    
    env_target = gym.make("CustomHopper-target-v0")
    target_rewards = []
    
    for ep in range(50):
        obs, _ = env_target.reset()
        episode_return = 0
        done = False
        
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env_target.step(action)
            episode_return += reward
            done = terminated or truncated
            if done:
                break
        
        target_rewards.append(episode_return)
    
    mean_target = np.mean(target_rewards)
    std_target = np.std(target_rewards)
    
    print(f"\nTarget domain performance:")
    print(f"  Mean: {mean_target:.1f} ± {std_target:.1f}")
    print(f"  Range: [{np.min(target_rewards):.1f}, {np.max(target_rewards):.1f}]")
    print(f"\nBaselines:")
    print(f"  Source→Target (naive): {SOURCE_TARGET_BASELINE:.1f}")
    print(f"  Target→Target (upper): {TARGET_TARGET_PERFORMANCE:.1f}")
    print(f"  ADR improvement: {mean_target - SOURCE_TARGET_BASELINE:+.1f}")
    
    # Save final results
    results = {
        'variant': variant_name,
        'config': variant_config,
        'target_performance': TARGET_TARGET_PERFORMANCE,
        'source_target_baseline': SOURCE_TARGET_BASELINE,
        'final_target_mean': float(mean_target),
        'final_target_std': float(std_target),
        'improvement': float(mean_target - SOURCE_TARGET_BASELINE),
        'adr_history': adr_history,
    }
    
    results_path = LOG_DIR / "results_complete.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 All results saved in: {LOG_DIR}")
    
    env_train.close()
    env_target.close()
    
    """
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
        '--target-perf',
        type=float,
        default=None,
        help='Override target performance threshold (if not set, uses default from ENV_CONFIGS)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--delta',
        type=float,
        default=None,
        help='Override ADR delta parameter (range expansion/contraction rate)'
    )
    
    args = parser.parse_args()
    
    # Update global variables
    ENVIRONMENT = args.env
    VARIANT = args.variant
    TOTAL_TIMESTEPS = args.timesteps
    
    # Override target performance if specified
    if args.target_perf is not None:
        print(f"[INFO] Overriding target performance: {ENV_CONFIGS[ENVIRONMENT]['target_performance']:.1f} → {args.target_perf:.1f}")
        ENV_CONFIGS[ENVIRONMENT]['target_performance'] = args.target_perf
    
    # Set seed if specified
    if args.seed is not None:
        print(f"[INFO] Setting random seed: {args.seed}")
        np.random.seed(args.seed)
    
    # Print configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Environment:       {ENVIRONMENT}")
    print(f"ADR Variant:       {VARIANT}")
    print(f"Total Timesteps:   {TOTAL_TIMESTEPS:,}")
    print(f"Target Performance: {ENV_CONFIGS[ENVIRONMENT]['target_performance']:.1f}")
    print(f"Delta Override:    {args.delta if args.delta is not None else 'None (use variant default)'}")
    print(f"Seed:              {args.seed if args.seed is not None else 'None (random)'}")
    print("="*60 + "\n")
    
    # Launch training
    train_adr_variant(VARIANT, environment=ENVIRONMENT, delta_override=args.delta)