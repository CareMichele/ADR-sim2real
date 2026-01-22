"""
Evaluation script for trained ADR models.
Carica un checkpoint salvato e valuta il modello su vari environment.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from pathlib import Path
import sys
import argparse
import time

# Importa le utility per gli environment
# Gli envs sono in src/envs/, quindi basta aggiungere src al path
sys.path.insert(0, str(Path(__file__).parent))
import envs.custom_hopper


def evaluate_model(model_path, env_id, n_episodes=50, render=False, deterministic=True, max_steps=500):
    """
    Valuta un modello caricato da checkpoint.
    
    Args:
        model_path: Path al checkpoint .zip del modello
        env_id: ID dell'environment Gymnasium (es. 'CustomHopper-source-v0')
        n_episodes: Numero di episodi da testare
        render: Se True, renderizza l'environment
        deterministic: Se True, usa policy deterministica
        max_steps: Massimo numero di step per episodio
    
    Returns:
        dict: Statistiche della valutazione
    """
    print(f"\n{'='*70}")
    print(f"  MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Environment: {env_id}")
    print(f"Episodes: {n_episodes}")
    print(f"Render: {render}")
    print(f"Deterministic: {deterministic}\n")
    
    # Carica il modello
    print("[INFO] Caricamento modello...")
    model = PPO.load(model_path)
    print("✅ Modello caricato con successo!\n")
    
    # Crea l'environment
    print("[INFO] Creazione environment...")
    if render:
        env = gym.make(env_id, render_mode="human")
    else:
        env = gym.make(env_id)
    print("✅ Environment creato!\n")
    
    # Statistiche
    episode_rewards = []
    episode_lengths = []
    
    print(f"[INFO] Inizio valutazione ({n_episodes} episodi)...\n")
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Predizione
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Rendering
            if render:
                env.render()
                time.sleep(0.01)  # Rallenta per visualizzazione
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Stampa progress ogni 10 episodi
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Episode {ep+1:3d}/{n_episodes}: reward={episode_reward:7.2f}, steps={steps:3d}")
    
    env.close()
    
    # Calcola statistiche
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    median_reward = np.median(episode_rewards)
    
    mean_length = np.mean(episode_lengths)
    
    stats = {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'median_reward': median_reward,
        'mean_length': mean_length,
        'n_episodes': n_episodes,
        'all_rewards': episode_rewards,
        'all_lengths': episode_lengths,
    }
    
    # Stampa summary
    print(f"\n{'='*70}")
    print("  EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Mean Reward:   {mean_reward:7.2f} ± {std_reward:.2f}")
    print(f"Median Reward: {median_reward:7.2f}")
    print(f"Min Reward:    {min_reward:7.2f}")
    print(f"Max Reward:    {max_reward:7.2f}")
    print(f"Mean Length:   {mean_length:7.2f} steps")
    print(f"{'='*70}\n")
    
    return stats


def compare_environments(model_path, env_ids, n_episodes=50):
    """
    Confronta le performance del modello su diversi environment.
    
    Args:
        model_path: Path al checkpoint
        env_ids: Lista di environment ID da testare
        n_episodes: Episodi per environment
    
    Returns:
        dict: Risultati per ogni environment
    """
    print(f"\n{'='*70}")
    print(f"  MULTI-ENVIRONMENT COMPARISON")
    print(f"{'='*70}\n")
    
    results = {}
    
    for env_id in env_ids:
        print(f"\n>>> Testing on: {env_id}")
        stats = evaluate_model(model_path, env_id, n_episodes, render=False, deterministic=True)
        results[env_id] = stats
        print()
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Environment':<30s} {'Mean Reward':<15s} {'Std':>10s}")
    print(f"{'-'*70}")
    for env_id, stats in results.items():
        print(f"{env_id:<30s} {stats['mean_reward']:>7.2f} ± {stats['std_reward']:>6.2f}")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained ADR model')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint (.zip)')
    parser.add_argument('--env', type=str, default='CustomHopper-source-v0',
                        help='Environment ID (default: CustomHopper-source-v0)')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes (default: 50)')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy (default: deterministic)')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Max steps per episode (default: 500)')
    parser.add_argument('--compare', nargs='+', type=str,
                        help='Compare on multiple environments (e.g., --compare CustomHopper-source-v0 CustomHopper-target-v0)')
    
    args = parser.parse_args()
    
    # Verifica che il modello esista
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Errore: Model file non trovato: {model_path}")
        sys.exit(1)
    
    # Modalità comparison
    if args.compare:
        compare_environments(str(model_path), args.compare, args.episodes)
    else:
        # Valutazione singola
        evaluate_model(
            str(model_path),
            args.env,
            args.episodes,
            args.render,
            deterministic=not args.stochastic,
            max_steps=args.max_steps
        )


if __name__ == "__main__":
    # Esempi di utilizzo
    """
    # Valutazione base
    python evaluate.py logs/hopper_adr_vanilla/model_final.zip
    
    # Con rendering
    python evaluate.py logs/hopper_adr_vanilla/model_final.zip --render
    
    # Su environment specifico
    python evaluate.py logs/hopper_adr_vanilla/model_final.zip --env CustomHopper-target-v0
    
    # Confronto su più environment
    python evaluate.py logs/hopper_adr_vanilla/model_final.zip --compare CustomHopper-source-v0 CustomHopper-target-v0
    
    # Policy stocastica (invece di deterministica)
    python evaluate.py logs/hopper_adr_vanilla/model_final.zip --stochastic
    
    # Solo 10 episodi
    python evaluate.py logs/hopper_adr_vanilla/model_final.zip --episodes 10
    """
    
    main()
