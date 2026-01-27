import gymnasium as gym
import pickle
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize


def create_eval_env(env_id, checkpoint_path, render=False):
    """
    Create evaluation environment configured automatically based on 
    how the model was trained (VecNormalize, FrameStack).
    
    This function:
    1. Creates base DummyVecEnv
    2. Auto-detects if VecFrameStack was used during training
    3. Applies VecFrameStack if needed
    4. Loads VecNormalize statistics if available
    5. Sets proper evaluation mode (training=False, norm_reward=False)
    
    Args:
        env_id: Gymnasium environment ID (e.g., 'CustomHopper-source-v0')
        checkpoint_path: Path to model checkpoint (used to find vecnormalize.pkl)
        render: If True, create environment with human rendering
    
    Returns:
        Configured VecEnv ready for evaluation
    """
    # 1. Create base environment
    if render:
        env = DummyVecEnv([lambda: gym.make(env_id, render_mode="human", width=2000, height=1080)])
    else:
        env = DummyVecEnv([lambda: gym.make(env_id)])
    
    # 2. Path to normalization statistics
    model_path = Path(checkpoint_path)
    vecnorm_path = model_path.parent / "vecnormalize.pkl"
    
    use_frame_stack = False
    
    # 3. Auto-detect FrameStack usage
    if vecnorm_path.exists():
        with open(vecnorm_path, 'rb') as f:
            vecnorm_data = pickle.load(f)
        
        saved_obs_dim = vecnorm_data.observation_space.shape[0]
        base_obs_dim = env.observation_space.shape[0]
        
        # If saved observation is 4x base, FrameStack was used
        if saved_obs_dim == base_obs_dim * 4:
            use_frame_stack = True
            print(f"[INFO] Detected VecFrameStack (obs: {base_obs_dim} → {saved_obs_dim})")
        else:
            print(f"[INFO] No VecFrameStack detected (obs: {saved_obs_dim})")
    
    # 4. Apply wrappers
    if use_frame_stack:
        env = VecFrameStack(env, n_stack=4)
    
    if vecnorm_path.exists():
        print(f"[INFO] Loading VecNormalize stats from {vecnorm_path.name}")
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False  # Important for evaluation
        env.norm_reward = False  # Important for evaluation
        print("VecNormalize stats loaded!")
    
    return env
