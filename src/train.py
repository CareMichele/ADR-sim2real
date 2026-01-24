"""
ADR Training Loop - UNIVERSAL VERSION
Versione universale compatibile con qualsiasi environment Gymnasium/MuJoCo.
Supporta configurazioni per diversi environment (Hopper, Walker, Humanoid, ecc.)
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
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
    # Source per ADR training (target_performance viene presa dal target-X quando si usa --difficulty)
    'hopper-source': {
        'env_id': 'CustomHopper-source-v0',
    },
    # Target per testing/upper bound
    'hopper-target-easy': {
        'env_id': 'CustomHopper-target-easy-v0',
        'target_performance': 1666.0,
        'description': 'EASY: Only mass difference (source -1kg torso, target standard)',
    },
    'hopper-target-medium': {
        'env_id': 'CustomHopper-target-medium-v0',
        'target_performance': 1500.0, #TODO da fare training su target per determinare upper bound
        'description': 'MEDIUM: Moderate mass changes (+/-20%) + friction (0.7x)',
    },
    'hopper-target-hard': {
        'env_id': 'CustomHopper-target-hard-v0',
        'target_performance': 1358.0,
        'description': 'HARD: Full hostile config (masses +/-50%, friction 0.5x, gravity -11, pushes)',
    },
    'ant-source': {
        'env_id': 'CustomAnt-source-v0',
        'target_performance': 3000.0,
    },
    'ant-target': {
        'env_id': 'CustomAnt-target-v0',
        #'target_performance': 3000.0,
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
        'delta': 0.03,
        'threshold_pct': 0.7,
        'boundary_sampling': True, #Al posto che scegliere valore a caso dentro il range, spesso sceglie i bordi 
        'boundary_prob': 0.10, #Per il 50% delle volte dammi caso peggiore (bordo del range)
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
                update_freq=32768, use_adr=True, checkpoint_path=None, difficulty=None):
    """
    Allena agente PPO con o senza ADR su qualsiasi environment.
    
    Args:
        variant_name: Nome della variante ADR ('vanilla', 'boundary', ecc.) - ignorato se use_adr=False
        environment: Nome dell'environment ('hopper', 'ant', ecc.)
        total_timesteps: Numero totale di timesteps per il training
        update_freq: Frequenza di valutazione (ogni quanti step)
        use_adr: Se True usa ADR, se False usa solo PPO
        checkpoint_path: Path a checkpoint esistente per riprendere training (opzionale)
        difficulty: Livello di difficoltà ('easy', 'medium', 'hard') - solo per source training con ADR
    """
    
    # PPO hyperparameters (locali)
    initial_lr = 5e-4
    final_lr = 1e-4
    ppo_learning_rate = initial_lr
    ppo_batch_size = 64  # CRITICO: non cambiare, stabilizza il training
    ppo_n_epochs = 10
    ppo_ent_coef = 0.02
    ppo_verbose = 1
    
    # Calibra hyperparameters in base a difficulty (opzionale ma raccomandato)
    if difficulty == 'easy':
        ppo_ent_coef = 0.01  # Meno esplorazione casuale
        final_lr = 1e-4
    elif difficulty == 'medium':
        ppo_ent_coef = 0.02  # Configurazione standard
        final_lr = 1e-4
    elif difficulty == 'hard':
        ppo_ent_coef = 0.03  # Più esplorazione
        final_lr = 5e-5      # LR finale più basso per stabilità
    
    if difficulty:
        print(f"[INFO] PPO hyperparameters calibrated for {difficulty.upper()} difficulty:")
        print(f"       ent_coef={ppo_ent_coef}, final_lr={final_lr:.0e}")
    
    # Setup directories
    # Se difficulty è specificato, aggiungi al nome della directory
    difficulty_suffix = f"_{difficulty}" if difficulty else ""
    
    if use_adr:
        log_dir = Path(f"./data/logs/{environment}_adr_{variant_name}{difficulty_suffix}")
        checkpoint_dir = Path(f"./data/checkpoints/{environment}_adr_{variant_name}{difficulty_suffix}")
        plot_dir = Path(f"./data/imgs/{environment}_adr_{variant_name}{difficulty_suffix}")
    else:
        log_dir = Path(f"./data/logs/{environment}_ppo{difficulty_suffix}")
        checkpoint_dir = Path(f"./data/checkpoints/{environment}_ppo{difficulty_suffix}")
        plot_dir = Path(f"./data/imgs/{environment}_ppo{difficulty_suffix}")
    
    # Verifica che le directory base esistano (non crearle automaticamente)
    base_dirs = [Path("./data/logs"), Path("./data/checkpoints"), Path("./data/imgs")]
    for base_dir in base_dirs:
        if not base_dir.exists():
            print(f"\n❌ ERROR: directory '{base_dir}' does not exist!")
            print(f"   Base folders must be created manually to avoid errors.")
            print(f"   Make sure you are in the project root and that the following exist:")
            print(f"   - data/logs/")
            print(f"   - data/checkpoints/")
            print(f"   - data/imgs/")
            sys.exit(1)
    
    # Crea solo le sottocartelle specifiche per questo run
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    
    env_config = ENV_CONFIGS[environment]
    
    def _persist_adr_state(target_dir: Path):
        if not use_adr or not adr_manager:
            return
        adr_state_path = target_dir / "adr_state.json"
        serializable_ranges = {k: list(v) for k, v in adr_manager.ranges.items()}
        with open(adr_state_path, 'w') as f:
            json.dump(serializable_ranges, f, indent=2)
        print(f"[INFO] ADR Ranges salvati in {adr_state_path}")

    print(f"\n{'='*70}")
    if use_adr:
        print(f"  ADR Training - Environment: {environment.upper()}")
        if difficulty:
            print(f"  Target Difficulty: {difficulty.upper()}")
        print(f"  Variant: {variant_name.upper()}")
        print(f"  {ADR_VARIANTS[variant_name]['description']}")
    else:
        print(f"  PPO Training (no ADR) - Environment: {environment.upper()}")
        if difficulty:
            print(f"  Target Difficulty: {difficulty.upper()}")
    print(f"{'='*70}\n")
    
    # Setup ADR if enabled (prima del VecNormalize)
    if use_adr:
        variant_config = ADR_VARIANTS[variant_name].copy()
        
        # Calibra parametri ADR in base a difficulty
        if difficulty == 'easy':
            variant_config['delta'] = 0.03
            variant_config['threshold_pct'] = 0.7
            variant_config['boundary_prob'] = 0.10
        elif difficulty == 'medium':
            variant_config['delta'] = 0.05
            variant_config['threshold_pct'] = 0.6
            variant_config['boundary_prob'] = 0.15
        elif difficulty == 'hard':
            variant_config['delta'] = 0.07
            variant_config['threshold_pct'] = 0.5
            variant_config['boundary_prob'] = 0.25
        
        if difficulty:
            print(f"[INFO] ADR parameters calibrated for {difficulty.upper()} difficulty:")
            print(f"       delta={variant_config['delta']}, threshold_pct={variant_config['threshold_pct']}, boundary_prob={variant_config.get('boundary_prob', 'N/A')}")
        
        # Estrai il tipo base dell'environment per ADRManager
        if 'hopper' in environment:
            domain_type = 'hopper'
        elif 'ant' in environment:
            domain_type = 'ant'
        else:
            domain_type = environment
        
        # Determina target_performance in base a difficulty
        if difficulty and environment == 'hopper-source':
            target_env_key = f'hopper-target-{difficulty}'
            target_perf = ENV_CONFIGS[target_env_key]['target_performance']
            print(f"[INFO] Using target_performance from {target_env_key}: {target_perf:.1f}")
        else:
            target_perf = env_config.get('target_performance', 1358.0)  # Fallback
        
        adr_manager = ADRManager(
            variant_config, 
            target_perf,
            env_type=domain_type,
            difficulty=difficulty  # Pass difficulty for calibrated limits
        )
        
        print(f"Configuration:")
        print(f"  Environment: {env_config['env_id']}")
        print(f"  ADR Enabled: Yes")
        print(f"  Delta: {adr_manager.delta}")
        print(f"  Threshold iniziale: {adr_manager.threshold:.1f}")
        print(f"  Target Performance: {target_perf:.1f}")
        print(f"  Update Frequency: {update_freq} steps")
        print(f"  Parameters to randomize: {adr_manager.params_to_randomize}\n")
    else:
        adr_manager = None
        print(f"Configuration:")
        print(f"  Environment: {env_config['env_id']}")
        print(f"  ADR Enabled: No (vanilla PPO)")
        print(f"  Evaluation Frequency: {update_freq} steps\n") 

    # Training environment (wrappa l'env Gymnasium PRIMA del VecEnv)
    # ADRWrapper è un gymnasium.Wrapper e non può wrappare direttamente DummyVecEnv.
    if use_adr:
        env_fn = lambda: ADRWrapper(gym.make(env_config['env_id']), adr_manager)
    else:
        env_fn = lambda: gym.make(env_config['env_id'])
    env_train = DummyVecEnv([env_fn])
    env_train = VecFrameStack(env_train, n_stack=4)
    
    # Carica o crea modello (VecNormalize viene gestito qui)
    if checkpoint_path:
        print(f"[INFO] Caricamento checkpoint da: {checkpoint_path}")
        
        # Carica VecNormalize stats PRIMA di creare VecNormalize
        vecnorm_path = checkpoint_path.parent / "vecnormalize.pkl"
        if vecnorm_path.exists():
            print(f"[INFO] Caricamento VecNormalize stats da: {vecnorm_path}")
            env_train = VecNormalize.load(str(vecnorm_path), env_train)
            env_train.training = True
            env_train.norm_obs = True
            env_train.norm_reward = False
            print("✅ VecNormalize stats caricati!")
        else:
            print("⚠️  Warning: VecNormalize stats non trovati, creando nuovo VecNormalize")
            env_train = VecNormalize(env_train, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        if use_adr and adr_manager:
            adr_state_path = checkpoint_path.parent / "adr_state.json"
            if adr_state_path.exists():
                with open(adr_state_path, 'r') as f:
                    saved_ranges = json.load(f)
                adr_manager.ranges = {k: tuple(v) for k, v in saved_ranges.items()}
                print("[INFO] ADR Ranges caricati da file!")

        model = PPO.load(checkpoint_path, env=env_train, device='cpu')
        use_lr_schedule = False
        
        # Sovrascrivi parametri per fine-tuning conservativo
        ft_lr = initial_lr      # Rifinitura stabile al learning rate di partenza
        ft_clip = 0.1          # Conservativo: vieta cambiamenti bruschi
        ft_batch = ppo_batch_size  # Mantieni l'ampio batch per stabilità (512)
        ft_ent_coef = 0.0      # Zero esplorazione, solo performance
        
        model.learning_rate = ft_lr
        model.clip_range = lambda _: ft_clip  # DEVE essere callable
        model.batch_size = ft_batch
        model.ent_coef = ft_ent_coef
        model.lr_schedule = lambda _: ft_lr  # Blocca LR fisso
        
        print("✅ Checkpoint caricato! Parametri fine-tuning applicati:")
        print(f"   LR={ft_lr:.0e}, Clip={ft_clip}, Batch={ft_batch}, Entropy={ft_ent_coef}")
        
        # Valuta il modello caricato per inizializzare best_reward
        print("\n[INFO] Valutazione modello caricato per inizializzare best_reward...")
        initial_test_episodes = 20
        initial_test_rewards = []
        env_train.training = False
        env_train.norm_reward = False
        
        for test_ep in range(initial_test_episodes):
            obs = env_train.reset()
            episode_return = 0
            done = False
            
            for _ in range(500):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env_train.step(action)
                episode_return += reward[0]
                if done[0]:
                    break
            
            initial_test_rewards.append(episode_return)
        
        env_train.training = True
        env_train.norm_reward = False
        
        initial_mean_reward = np.mean(initial_test_rewards)
        initial_std_reward = np.std(initial_test_rewards)
        print(f"Reward modello caricato: mean={initial_mean_reward:.2f}, std={initial_std_reward:.2f}")
        print(f"                         range=[{np.min(initial_test_rewards):.2f}, {np.max(initial_test_rewards):.2f}]")
        best_reward = initial_mean_reward
        print(f"🏆 Best reward inizializzato a: {best_reward:.2f}\n")
    else:
        print("[INFO] Inizializzazione nuovo modello PPO...")
        
        # Crea VecNormalize per nuovo training
        env_train = VecNormalize(env_train, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        model = PPO(
            "MlpPolicy",
            env_train,
            learning_rate=ppo_learning_rate,
            batch_size=ppo_batch_size,
            n_epochs=ppo_n_epochs,
            ent_coef=ppo_ent_coef,
            verbose=ppo_verbose,
            device='cpu',
        )
        best_reward = -np.inf
        use_lr_schedule = True
    
    training_history = []
    num_updates = total_timesteps // update_freq
    
    print(f"\n[INFO] Inizio training")
    print(f"       Timestep totali: {total_timesteps}")
    print(f"       Evaluation freq: {update_freq}")
    print(f"       Numero valutazioni: {num_updates}\n")
    
    for update_idx in range(num_updates):
        # Aggiorna LR con schedule lineare globale (solo quando non si sta facendo fine-tuning da checkpoint)
        if use_lr_schedule:
            progress = min(1.0, model.num_timesteps / total_timesteps)
            current_lr = initial_lr + (final_lr - initial_lr) * progress
            model.learning_rate = current_lr
            model.lr_schedule = lambda _: current_lr
            # Aggiorna direttamente l'optimizer (evita _update_learning_rate prima del logger)
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = current_lr
        else:
            current_lr = model.lr_schedule(1.0) if callable(getattr(model, "lr_schedule", None)) else model.learning_rate

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
        print(f"Current LR: {current_lr:.2e}")
        
        # Set evaluation mode (reward reale, non normalizzato)
        env_train.training = False
        if hasattr(env_train, "norm_reward"):
            env_train.norm_reward = False
        
        for test_ep in range(test_episodes):
            obs = env_train.reset()
            episode_return = 0
            done = False
            
            for _ in range(500):
                action, _ = model.predict(obs, deterministic=True)  
                obs, reward, done, info = env_train.step(action)
                episode_return += reward[0]
                if done[0]:
                    break
            
            test_rewards.append(episode_return)
        
        # Re-enable training mode
        env_train.training = True
        if hasattr(env_train, "norm_reward"):
            env_train.norm_reward = False
        
        mean_reward = np.mean(test_rewards)
        std_reward = np.std(test_rewards)
        min_reward = np.min(test_rewards)
        max_reward = np.max(test_rewards)
        
        print(f"Test reward: mean={mean_reward:.2f}, std={std_reward:.2f}")
        print(f"             range=[{min_reward:.2f}, {max_reward:.2f}]")
        
        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_model_path = checkpoint_dir / "model_best.zip"
            model.save(str(best_model_path))
            env_train.save(str(best_model_path.parent / "vecnormalize.pkl"))
            _persist_adr_state(best_model_path.parent)
            print(f"🏆 New best model! Reward={best_reward:.2f} -> Salvato in {best_model_path}")
        
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
    
    # Training completato - salva final model
    final_model_path = checkpoint_dir / "model_final.zip"
    model.save(str(final_model_path))
    # Salva sempre VecNormalize stats nella stessa cartella del modello (utile per resume da model_final)
    env_train.save(str(final_model_path.parent / "vecnormalize.pkl"))
    _persist_adr_state(final_model_path.parent)
    
    print(f"\n✅ Training completato!")
    print(f"💾 Final model salvato: {final_model_path}")
    print(f"🏆 Best model salvato: {checkpoint_dir / 'model_best.zip'} (reward={best_reward:.2f})")
    
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
        default='hopper-source',
        choices=['hopper-source', 'hopper-target', 'hopper-target-easy', 'hopper-target-medium', 'hopper-target-hard', 'ant-source', 'ant-target'],
        help='Environment to train on (source or target domain with difficulty level)'
    )
    
    parser.add_argument(
        '--difficulty',
        type=str,
        default=None,
        choices=['easy', 'medium', 'hard'],
        help='Target difficulty level for ADR training on source domain (easy/medium/hard). Ignored for target envs.'
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
        default=16384,
        help='ADR update frequency (every N timesteps)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from (e.g., ./data/checkpoints/hopper-source_ppo/model_best.zip)'
    )
    
    args = parser.parse_args()
    
    # Auto-disable ADR for target environments (non ha senso fare ADR su target)
    use_adr = not args.no_adr
    if '-target' in args.env:
        if use_adr:
            print("[INFO] Target environment detected: automatically disabling ADR (using vanilla PPO)")
        use_adr = False
        # Per target env, difficulty non ha senso
        if args.difficulty:
            print("[WARNING] --difficulty parameter ignored for target environments")
            args.difficulty = None
    
    # Valida che difficulty sia usato solo con source + ADR
    if args.difficulty and not use_adr:
        print("[WARNING] --difficulty parameter ignored when ADR is disabled")
        args.difficulty = None
    
    if args.difficulty and '-source' not in args.env:
        print("[WARNING] --difficulty should be used with source environment for ADR training")
    
    # Set fixed random seed for reproducibility
    np.random.seed(42)
    print("[INFO] Random seed set to 42")
    
    # Print configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Environment:       {args.env}")
    if args.difficulty:
        print(f"Target Difficulty: {args.difficulty.upper()}")
        if args.env == 'hopper-source':
            target_env_name = f"hopper-target-{args.difficulty}"
            if target_env_name in ENV_CONFIGS:
                print(f"Target Env:        {ENV_CONFIGS[target_env_name]['env_id']}")
                print(f"Description:       {ENV_CONFIGS[target_env_name]['description']}")
    print(f"Mode:              {'PPO (no ADR)' if not use_adr else f'ADR ({args.variant})'}")
    if not use_adr:
        print(f"ADR Variant:       N/A (disabled)")
    else:
        print(f"ADR Variant:       {args.variant}")
    print(f"Total Timesteps:   {args.timesteps:,}")
    print(f"Update Frequency:  {args.update_freq:,}")
    if use_adr:
        # Determina target_performance come nel codice ADR
        if args.difficulty and args.env == 'hopper-source':
            target_env_key = f'hopper-target-{args.difficulty}'
            target_perf = ENV_CONFIGS[target_env_key]['target_performance']
        else:
            target_perf = ENV_CONFIGS[args.env].get('target_performance', 1358.0)
        print(f"Target Performance: {target_perf:.1f}")
    print(f"Random Seed:       42 (fixed)")
    print("="*60 + "\n")
    
    # Verifica checkpoint se specificato
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"❌ Errore: Checkpoint non trovato: {checkpoint_path}")
            sys.exit(1)
        print(f"[INFO] Checkpoint specificato: {checkpoint_path}")
    
    # Launch training
    train_agent(
        args.variant, 
        environment=args.env, 
        total_timesteps=args.timesteps,
        update_freq=args.update_freq,
        use_adr=use_adr,
        checkpoint_path=checkpoint_path,
        difficulty=args.difficulty
    )