import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from pathlib import Path
import json
import sys
import argparse

from adr_manager import ADRManager
from adr_wrapper import ADRWrapper
from utils.plotting import plot_training_history, plot_all_ranges
from utils.evaluation_utils import evaluate_policy
from envs import custom_hopper
from envs import custom_ant

# Load environment configurations
with open(Path(__file__).parent.parent / 'configs' / 'env_configs.json', 'r') as f:
    ENV_CONFIGS = json.load(f)

# Load ADR configurations
with open(Path(__file__).parent.parent / 'configs' / 'adr_configs.json', 'r') as f:
    ADR_CONFIG = json.load(f)
    ADR_VARIANTS = ADR_CONFIG['variants']


def train_agent(variant_name, environment='hopper', total_timesteps=1000000, 
                update_freq=32768, use_adr=True, checkpoint_path=None, difficulty=None):
    """
    Train PPO agent with or without ADR on any environment.
    
    Args:
        variant_name: ADR variant name ('vanilla', 'boundary', etc.) - ignored if use_adr=False
        environment: Environment name ('hopper', 'ant', etc.)
        total_timesteps: Total training timesteps
        update_freq: Evaluation frequency (every N steps)
        use_adr: If True uses ADR, if False uses vanilla PPO
        checkpoint_path: Path to existing checkpoint to resume training (optional)
        difficulty: Difficulty level ('easy', 'medium', 'hard') - only for source training with ADR
    """
    
    # PPO hyperparameters
    ppo_learning_rate = 3e-4  
    ppo_clip_range = 0.2 
    ppo_ent_coef = 0.01  
    ppo_gamma = 0.995
    batch_size = 64
    
    # Calibrate entropy coefficient for ADR difficulty (--difficulty works only for hopper)
    if difficulty and environment in ADR_CONFIG['difficulty_calibration']:
        calibration = ADR_CONFIG['difficulty_calibration'][environment].get(difficulty, {})
        ppo_ent_coef = calibration.get('ppo_ent_coef', ppo_ent_coef)
        print(f"[INFO] PPO ent_coef calibrated for {difficulty.upper()}: {ppo_ent_coef}")
    
    difficulty_suffix = f"_{difficulty}" if difficulty else ""
    
    if use_adr:
        log_dir = Path(f"./data/logs/{environment}_adr_{variant_name}{difficulty_suffix}")
        checkpoint_dir = Path(f"./data/checkpoints/{environment}_adr_{variant_name}{difficulty_suffix}")
        plot_dir = Path(f"./data/imgs/{environment}_adr_{variant_name}{difficulty_suffix}")
    else:
        log_dir = Path(f"./data/logs/{environment}_ppo{difficulty_suffix}")
        checkpoint_dir = Path(f"./data/checkpoints/{environment}_ppo{difficulty_suffix}")
        plot_dir = Path(f"./data/imgs/{environment}_ppo{difficulty_suffix}")
    
    base_dirs = [Path("./data/logs"), Path("./data/checkpoints"), Path("./data/imgs")]
    for base_dir in base_dirs:
        if not base_dir.exists():
            print(f"\nERROR: directory '{base_dir}' does not exist!")
            print(f"   Base folders must be created manually to avoid errors.")
            print(f"   Make sure you are in the project root and that the following exist:")
            print(f"   - data/logs/")
            print(f"   - data/checkpoints/")
            print(f"   - data/imgs/")
            sys.exit(1)
    
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    
    env_config = ENV_CONFIGS[environment]
    
    def _persist_adr_state(target_dir: Path):
        """Save ADR ranges to JSON file."""
        if not use_adr or not adr_manager:
            return
        adr_state_path = target_dir / "adr_state.json"
        serializable_ranges = {k: list(v) for k, v in adr_manager.ranges.items()}
        with open(adr_state_path, 'w') as f:
            json.dump(serializable_ranges, f, indent=2)
        print(f"[INFO] ADR ranges saved to {adr_state_path}")

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
    
    if use_adr:
        variant_config = ADR_VARIANTS[variant_name].copy()
        
        # Apply difficulty-specific ADR calibration if available
        if difficulty and environment in ADR_CONFIG['difficulty_calibration']:
            calibration = ADR_CONFIG['difficulty_calibration'][environment].get(difficulty, {})
            variant_config['delta'] = calibration.get('delta', variant_config['delta'])
            variant_config['threshold_pct'] = calibration.get('threshold_pct', variant_config.get('threshold_pct', 0.75))
            if 'boundary_prob' in calibration:
                variant_config['boundary_prob'] = calibration['boundary_prob']
        
        if difficulty:
            print(f"[INFO] ADR parameters calibrated for {difficulty.upper()} difficulty:")
            print(f"       delta={variant_config['delta']}, threshold_pct={variant_config['threshold_pct']}, boundary_prob={variant_config.get('boundary_prob', 'N/A')}")
        
        if 'hopper' in environment:
            domain_type = 'hopper'
        elif 'ant' in environment:
            domain_type = 'ant'
        else:
            domain_type = environment
        
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
            difficulty=difficulty
        )
        
        print(f"Configuration:")
        print(f"  Environment: {env_config['env_id']}")
        print(f"  ADR Enabled: Yes")
        print(f"  Delta: {adr_manager.delta}")
        print(f"  Initial Threshold: {adr_manager.threshold:.1f}")
        print(f"  Target Performance: {target_perf:.1f}")
        print(f"  Update Frequency: {update_freq} steps")
        print(f"  Parameters to randomize: {adr_manager.params_to_randomize}\n")
    else:
        adr_manager = None
        print(f"Configuration:")
        print(f"  Environment: {env_config['env_id']}")
        print(f"  ADR Enabled: No (vanilla PPO)")
        print(f"  Evaluation Frequency: {update_freq} steps\n") 

    # PARALLEL ENVIRONMENTS FOR FASTER TRAINING (4x-8x speedup)
    n_envs = 8
    
    # Create a list of functions, each creates an independent environment
    if use_adr:
        env_fns = [lambda: ADRWrapper(gym.make(env_config['env_id']), adr_manager) for _ in range(n_envs)]
    else:
        env_fns = [lambda: gym.make(env_config['env_id']) for _ in range(n_envs)]
    
    # DummyVecEnv runs envs sequentially but batches data for the network
    env_train = DummyVecEnv(env_fns)
    env_train = VecFrameStack(env_train, n_stack=4)
    
    if checkpoint_path:
        print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
        
        vecnorm_path = checkpoint_path.parent / "vecnormalize.pkl"
        if vecnorm_path.exists():
            print(f"[INFO] Loading VecNormalize stats from: {vecnorm_path}")
            env_train = VecNormalize.load(str(vecnorm_path), env_train)
            env_train.training = True
            env_train.norm_obs = True
            env_train.norm_reward = True
            print("VecNormalize stats loaded!")
        else:
            print("Warning: VecNormalize stats not found, creating new VecNormalize")
            env_train = VecNormalize(env_train, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        if use_adr and adr_manager:
            adr_state_path = checkpoint_path.parent / "adr_state.json"
            if adr_state_path.exists():
                with open(adr_state_path, 'r') as f:
                    saved_ranges = json.load(f)
                adr_manager.ranges = {k: tuple(v) for k, v in saved_ranges.items()}
                print("[INFO] ADR ranges loaded from file!")

        model = PPO.load(checkpoint_path, env=env_train, device='cpu')
        model.learning_rate = ppo_learning_rate
        
        # Evaluate loaded model to initialize best_reward
        print("\n[INFO] Evaluating loaded model to initialize best_reward...")
        initial_stats = evaluate_policy(env_train, model, n_episodes=20, max_steps=500, deterministic=True)
        
        initial_mean_reward = initial_stats['mean_reward']
        initial_std_reward = initial_stats['std_reward']
        print(f"Loaded model reward: mean={initial_mean_reward:.2f}, std={initial_std_reward:.2f}")
        print(f"                     range=[{initial_stats['min_reward']:.2f}, {initial_stats['max_reward']:.2f}]")
        best_reward = initial_mean_reward
        print(f"Best reward initialized to: {best_reward:.2f}\n")
    else:
        print("[INFO] Initializing new PPO model...")
        
        # Create VecNormalize for new training
        env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        model = PPO(
            "MlpPolicy",
            env_train,
            learning_rate=ppo_learning_rate,
            ent_coef=ppo_ent_coef,
            clip_range=ppo_clip_range,
            gamma=ppo_gamma,
            batch_size=batch_size,
            verbose=1,
            device='cpu',
        )
        best_reward = -np.inf
    
    training_history = []
    num_updates = total_timesteps // update_freq
    
    print(f"\n[INFO] Starting training")
    print(f"       Total timesteps: {total_timesteps}")
    print(f"       Evaluation freq: {update_freq}")
    print(f"       Number of evaluations: {num_updates}\n")
    
    for update_idx in range(num_updates):
        # Train for update_freq steps
        model.learn(
            total_timesteps=update_freq,
            reset_num_timesteps=False,
            progress_bar=False
        )
        
        # Evaluate on current distribution
        eval_label = "ADR Update" if use_adr else "Evaluation"
        print(f"\n[{eval_label} {update_idx + 1}/{num_updates}]")
        print(f"Total timesteps: {model.num_timesteps}")
        
        # Use centralized evaluation function
        stats = evaluate_policy(env_train, model, n_episodes=20, max_steps=500, deterministic=True)
        
        mean_reward = stats['mean_reward']
        std_reward = stats['std_reward']
        min_reward = stats['min_reward']
        max_reward = stats['max_reward']
        
        print(f"Test reward: mean={mean_reward:.2f}, std={std_reward:.2f}")
        print(f"             range=[{min_reward:.2f}, {max_reward:.2f}]")
        
        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_model_path = checkpoint_dir / "model_best.zip"
            model.save(str(best_model_path))
            env_train.save(str(best_model_path.parent / "vecnormalize.pkl"))
            _persist_adr_state(best_model_path.parent)
            print(f"New best model! Reward={best_reward:.2f} -> Saved to {best_model_path}")
        
        # Update ADR ranges if enabled
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
    env_train.save(str(final_model_path.parent / "vecnormalize.pkl"))
    _persist_adr_state(final_model_path.parent)
    
    print(f"\nTraining completed!")
    print(f"Final model saved: {final_model_path}")
    print(f"Best model saved: {checkpoint_dir / 'model_best.zip'} (reward={best_reward:.2f})")
    
    # Save training history
    history_filename = "adr_history.json" if use_adr else "training_history.json"
    history_path = log_dir / history_filename
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"History saved: {history_path}")
    
    # Generate plots
    print("\n[INFO] Generating plots...")
    try:
        plot_path_main = plot_dir / "training_history.png"
        plot_training_history(history_path, save_path=plot_path_main, show=False)
        
        if use_adr:
            plot_path_ranges = plot_dir / "all_ranges.png"
            plot_all_ranges(history_path, save_path=plot_path_ranges, show=False)
        
        print("Plots generated successfully!")
    except Exception as e:
        print(f"Warning: Error generating plots: {e}")
    
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

# MAIN
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
        choices=['hopper-source', 'hopper-target-easy', 'hopper-target-medium', 'hopper-target-hard', 'ant-source'],
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
    
    # Auto-disable ADR for target environments
    use_adr = not args.no_adr
    if '-target' in args.env:
        if use_adr:
            print("[INFO] Target environment detected: automatically disabling ADR (using vanilla PPO)")
        use_adr = False
        if args.difficulty:
            print("[WARNING] --difficulty parameter ignored for target environments")
            args.difficulty = None
    
    # Validate difficulty usage
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
        if args.difficulty and args.env == 'hopper-source':
            target_env_key = f'hopper-target-{args.difficulty}'
            target_perf = ENV_CONFIGS[target_env_key]['target_performance']
        else:
            target_perf = ENV_CONFIGS[args.env].get('target_performance', 1358.0)
        print(f"Target Performance: {target_perf:.1f}")
    print(f"Random Seed:       42 (fixed)")
    print("="*60 + "\n")
    
    # Verify checkpoint if specified
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        print(f"[INFO] Checkpoint specified: {checkpoint_path}")
    
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