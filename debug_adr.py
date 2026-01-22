"""
Debug script to test ADR (Automatic Domain Randomization) - UNIVERSAL VERSION.

Simulates the ADR curriculum with expanded physical parameters:
- Masses (thigh, leg, foot)
- Friction (attrito pavimento)
- Damping (smorzamento giunti)
- Gravity (gravità)
- Force magnitude (perturbazioni esterne)

Start with tight range (1.0, 1.0) = no randomization
Every N episodes, update performance and adapt ranges
If performance > threshold → expand (more difficult)
If performance < threshold → shrink (easier)
"""

import gymnasium as gym
import numpy as np
import sys
from pathlib import Path

# Aggiungi le directory necessarie al path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir / "env"))
sys.path.insert(0, str(root_dir / "src"))

# Import corretto delle classi
from custom_hopper import CustomHopper
from adr_manager import ADRManager
from adr_wrapper import ADRWrapper

def main():
    print("="*70)
    print("  ADR Debug Script - EXPANDED PHYSICS")
    print("="*70)
    
    # 1. Crea l'ambiente base (senza ADR)
    print("\n[INFO] Creazione ambiente base CustomHopper-source-v0...")
    env = gym.make("CustomHopper-source-v0")
    
    # 2. Crea l'ADRManager con una config di test (TUTTI i parametri fisici)
    print("[INFO] Inizializzazione ADRManager...")
    test_config = {
        'delta': 0.05,
        'threshold_pct': 0.75,
        'boundary_sampling': False,
        'progressive': False,
        # Randomizza TUTTI i parametri fisici
        'randomize_only': None,  # None = tutti
    }
    target_performance = 1000.0  # Valore target di riferimento
    adr_manager = ADRManager(test_config, target_performance, env_type='hopper')
    
    print(f"  Delta: {adr_manager.delta}")
    print(f"  Threshold: {adr_manager.threshold:.1f}")
    print(f"  Target Performance: {target_performance:.1f}")
    print(f"  Initial ranges:")
    for param, (lower, upper) in adr_manager.ranges.items():
        print(f"    {param:15s}: [{lower:6.2f}, {upper:6.2f}]")
    
    # 3. Applica manualmente l'ADRWrapper
    print("\n[INFO] Applicazione ADRWrapper...")
    env = ADRWrapper(env, adr_manager)
    print("✅ ADRWrapper applicato correttamente")
    
    # 4. Simula il curriculum ADR
    print("\n" + "="*70)
    print("  Simulazione ADR Curriculum")
    print("="*70)
    
    num_episodes = 50
    update_interval = 10
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 500
        
        while not done and steps < max_steps:
            action = env.action_space.sample()  # Policy random per test
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
        
        episode_rewards.append(episode_reward)
        
        # Stampa info episodio
        if (ep + 1) % 5 == 0:
            recent_rewards = episode_rewards[-5:]
            print(f"Episode {ep+1:3d}: reward={episode_reward:6.1f}  "
                  f"(avg last 5: {np.mean(recent_rewards):6.1f})")
        
        # Update ADR ranges ogni update_interval episodi
        if (ep + 1) % update_interval == 0:
            perf_window = episode_rewards[-update_interval:]
            perf_avg = np.mean(perf_window)
            perf_max = np.max(perf_window)
            perf_min = np.min(perf_window)
            
            print(f"\n{'─'*70}")
            print(f"ADR Update at Episode {ep+1}/{num_episodes}")
            print(f"{'─'*70}")
            print(f"Performance window ({update_interval} eps):")
            print(f"  Mean: {perf_avg:.2f}")
            print(f"  Range: [{perf_min:.2f}, {perf_max:.2f}]")
            
            # Mostra i parametri fisici applicati nell'ultimo episodio
            print(f"\nCurrent physical parameters:")
            masses = env.env.unwrapped.get_parameters()
            print(f"  Masses: {masses}")
            print(f"  Max Push Force: {env.env.unwrapped.current_max_push:.1f} N")
            print(f"  Gravity Z: {env.env.unwrapped.model.opt.gravity[2]:.2f} m/s²")
            
            # Update ADR ranges
            status = env.adr_manager.update_ranges(perf_avg)
            diversity = env.adr_manager.get_range_diversity()
            
            print(f"\nADR Status: {status}")
            print(f"Threshold: {env.adr_manager.threshold:.1f}")
            print(f"Diversity: {diversity:.3f}")
            print(f"\nUpdated ADR ranges:")
            for param, (lower, upper) in env.adr_manager.ranges.items():
                width = upper - lower
                print(f"  {param:15s}: [{lower:6.2f}, {upper:6.2f}]  width={width:.3f}")
            print()
    
    # Summary finale
    print("\n" + "="*70)
    print("  Final Summary - EXPANDED ADR")
    print("="*70)
    print(f"Total episodes: {num_episodes}")
    print(f"Mean reward (all): {np.mean(episode_rewards):.2f}")
    print(f"Mean reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"\nFinal ADR ranges (all physical parameters):")
    for param, (lower, upper) in env.adr_manager.ranges.items():
        print(f"  {param:15s}: [{lower:6.2f}, {upper:6.2f}]")
    print(f"\nFinal diversity: {env.adr_manager.get_range_diversity():.3f}")
    
    env.close()
    print("\n✅ ADR debug completato con successo!")
    print("\n💡 Nota: Durante l'esecuzione, quando il robot diventa ROSSO")
    print("   significa che è stata applicata una perturbazione esterna (push)!")

if __name__ == "__main__":
    main()
