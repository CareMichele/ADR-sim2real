import gymnasium as gym
import numpy as np

class ADRWrapper(gym.Wrapper):
    """
    Universal wrapper for ADR (Automatic Domain Randomization).
    
    Compatible with any environment that implements:
    - get_parameters() [optional, for masses]
    - set_parameters(dict) to apply randomized parameters
    """
    
    def __init__(self, env, adr_manager):
        """
        Args:
            env: Gymnasium environment (e.g., CustomHopper, CustomWalker, etc.)
            adr_manager: ADRManager instance with ADR configuration
        """
        super().__init__(env)
        self.adr_manager = adr_manager
        self.supports_masses = hasattr(self.env.unwrapped, 'original_masses')
    
    def reset(self, **kwargs):
        """Reset environment with randomized parameter application"""
        obs, info = self.env.reset(**kwargs)
        
        if self.supports_masses:
            original_masses = self.env.unwrapped.original_masses
            sampled_params = self.adr_manager.sample_parameters(original_masses)
        else:
            sampled_params = self.adr_manager.sample_parameters(original_masses=None)
        
        if hasattr(self.env.unwrapped, 'set_parameters'):
            self.env.unwrapped.set_parameters(sampled_params)
        else:
            print("[WARNING] Environment does not support set_parameters()!")
        
        return obs, info