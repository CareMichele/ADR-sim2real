"""
ADR Training Loop for Ant - ADAPTED FROM HOPPER
Fixes: separate test environment, proper evaluation, better logging
Adapted for Ant with different performance targets and complexity
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from pathlib import Path
import json

# Importa il modulo con gli environment registrati
import envs.custom_ant

#import ADR Manager
from adr_manager import ADRManager

#import ADR Wrapper
from adr_wrapper import ADRWrapper
# ============================================================================
# ADR VARIANTS CONFIGURATION FOR ANT
# ============================================================================

ADR_VARIANTS = {
    'vanilla': {
        'description': 'Symmetric expansion/contraction (baseline) - all 8 legs',
        'delta': 0.05,
        'threshold_pct': 0.75,
        'boundary_sampling': False,
        'progressive': False,
        'parameters': ['front_left_hip', 'front_left_foot', 'front_right_hip', 'front_right_foot',
                       'back_left_hip', 'back_left_foot', 'back_right_hip', 'back_right_foot'],
    },
    'boundary': {
        'description': 'Boundary sampling (50% at extremes) - all 8 legs',
        'delta': 0.05,
        'threshold_pct': 0.75,
        'boundary_sampling': True,
        'boundary_prob': 0.5,
        'progressive': False,
        'parameters': ['front_left_hip', 'front_left_foot', 'front_right_hip', 'front_right_foot',
                       'back_left_hip', 'back_left_foot', 'back_right_hip', 'back_right_foot'],
    },
    'progressive': {
        'description': 'Progressive curriculum (increasing threshold) - all 8 legs',
        'delta': 0.05,
        'threshold_schedule': [0.60, 0.70, 0.80],
        'boundary_sampling': False,
        'progressive': True,
        'parameters': ['front_left_hip', 'front_left_foot', 'front_right_hip', 'front_right_foot',
                       'back_left_hip', 'back_left_foot', 'back_right_hip', 'back_right_foot'],
    },
    'selective': {
        'description': 'Only randomize critical parameters (front legs)',
        'delta': 0.05,
        'threshold_pct': 0.75,
        'randomize_only': ['front_left_hip', 'front_left_foot', 'front_right_hip', 'front_right_foot'],
        'boundary_sampling': False,
        'progressive': False,
        'parameters': ['front_left_hip', 'front_left_foot', 'front_right_hip', 'front_right_foot'],
    }
}

# ============================================================================
# CONFIGURATION - ADAPTED FOR ANT
# ============================================================================

VARIANT = 'vanilla'    # quale variante usare
ADR_UPDATE_FREQ = 5000  # ogni quanti step aggiorna i range ADR
TOTAL_TIMESTEPS = 2000000  # Ant richiede più timestep (più complesso)

PPO_LEARNING_RATE = 3e-4
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_VERBOSE = 1

# Ant ha target performance diverso da Hopper
TARGET_TARGET_PERFORMANCE = 3000.0
SOURCE_TARGET_BASELINE = 800.0

LOG_DIR = Path(f"./logs/adr_ant_{VARIANT}")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ADR MANAGER CLASS
# ============================================================================

# Nota: Usiamo la stessa ADRManager importata da adr_manager.py
# Il file train.py lo fa già, quindi è riutilizzabile


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_adr_variant(variant_name):
    """Allena ADR con la variante specificata su Ant"""
    
    print(f"\n{'='*70}")
    print(f"  ADR Training - Ant - Variant: {variant_name.upper()}")
    print(f"  {ADR_VARIANTS[variant_name]['description']}")
    print(f"{'='*70}\n")
    
    print("[INFO] Creazione environment...")
    # Training environment (with ADR)
    env_train = gym.make("CustomAnt-source-v0")
    
    variant_config = ADR_VARIANTS[variant_name]
    parameters = variant_config.get('parameters', None)
    
    # Mapping esplicito nome_parametro -> indice_massa per Ant
    # original_masses = [front_left_hip, front_left_foot, front_right_hip, front_right_foot,
    #                    back_left_hip, back_left_foot, back_right_hip, back_right_foot]
    ant_param_to_idx = {
        'front_left_hip': 0,
        'front_left_foot': 1,
        'front_right_hip': 2,
        'front_right_foot': 3,
        'back_left_hip': 4,
        'back_left_foot': 5,
        'back_right_hip': 6,
        'back_right_foot': 7,
    }
    
    adr_manager = ADRManager(
        variant_config, 
        TARGET_TARGET_PERFORMANCE,
        parameters=parameters,
        param_to_idx=ant_param_to_idx,
        limits=[0.85, 1.15]  # ±15% per Ant (più conservativo)
    )
    
    print(f"Configuration:")
    print(f"  Delta: {adr_manager.delta}")
    print(f"  Threshold iniziale: {adr_manager.threshold:.1f}")
    print(f"  Target Performance: {TARGET_TARGET_PERFORMANCE:.1f}")
    print(f"  Update Frequency: {ADR_UPDATE_FREQ} steps\n")
    
    class ADRWrapper(gym.Wrapper):
        def __init__(self, env, adr_manager):
            super().__init__(env)
            self.adr_manager = adr_manager
        
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            original_masses = self.env.unwrapped.original_masses
            new_masses = self.adr_manager.sample_parameters(original_masses)
            self.env.unwrapped.set_parameters(new_masses)
            return obs, info
    
    env_train = ADRWrapper(env_train, adr_manager)
    
    print("[INFO] Inizializzazione modello PPO...")
    model = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=PPO_LEARNING_RATE,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        verbose=PPO_VERBOSE
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
            progress_bar=True
        )
        
        # Test on ADR distribution (randomized masses)
        test_episodes = 20
        test_rewards = []
        
        print(f"\n[ADR Update {update_idx + 1}/{num_updates}]")
        print(f"Timestep totali: {model.num_timesteps}")
        
        for test_ep in range(test_episodes):
            obs, info = env_train.reset()
            episode_return = 0
            done = False
            
            for _ in range(1000):  # Ant ha max_episode_steps = 1000
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
            print(f"  {param:20s}: [{lower:.3f}, {upper:.3f}]  width={upper-lower:.3f}")
        
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
    final_model_path = LOG_DIR / "model_final.zip"
    model.save(str(final_model_path))
    print(f"\n✅ Training completato! Model salvato: {final_model_path}")
    
    # Save history
    history_path = LOG_DIR / "adr_history.json"
    with open(history_path, 'w') as f:
        json.dump(adr_history, f, indent=2)
    print(f"📊 History salvato: {history_path}")
    
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
    
    env_target = gym.make("CustomAnt-target-v0")
    target_rewards = []
    
    for ep in range(50):
        obs, _ = env_target.reset()
        episode_return = 0
        done = False
        
        for _ in range(1000):  # Ant max_episode_steps = 1000
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
# MAIN
# ============================================================================

if __name__ == "__main__":
    train_adr_variant(VARIANT)
