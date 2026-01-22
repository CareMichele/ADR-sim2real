"""
ADR Wrapper - UNIVERSAL VERSION
Wrapper universale per applicare ADR a qualsiasi environment Gymnasium/MuJoCo.
Supporta randomizzazione di: masse, friction, damping, gravity, perturbazioni.
"""
import gymnasium as gym
import numpy as np

class ADRWrapper(gym.Wrapper):
    """
    Wrapper universale per ADR (Automatic Domain Randomization).
    
    Compatibile con qualsiasi environment che implementi:
    - get_parameters() [opzionale, per masse]
    - set_parameters(dict) per applicare i parametri randomizzati
    """
    
    def __init__(self, env, adr_manager):
        """
        Args:
            env: Gymnasium environment (es. CustomHopper, CustomWalker, ecc.)
            adr_manager: Istanza di ADRManager con la configurazione ADR
        """
        super().__init__(env)
        self.adr_manager = adr_manager
        
        # Detecta automaticamente se l'env supporta masse (per backward compatibility)
        self.supports_masses = hasattr(self.env.unwrapped, 'original_masses')
    
    def reset(self, **kwargs):
        """Reset dell'environment con applicazione dei parametri randomizzati"""
        obs, info = self.env.reset(**kwargs)
        
        # Campiona TUTTI i parametri fisici dall'ADR Manager
        if self.supports_masses:
            original_masses = self.env.unwrapped.original_masses
            sampled_params = self.adr_manager.sample_parameters(original_masses)
        else:
            # Environment senza masse (es. environments diversi da Hopper)
            sampled_params = self.adr_manager.sample_parameters(original_masses=None)
        
        # Applica i parametri all'environment
        if hasattr(self.env.unwrapped, 'set_parameters'):
            self.env.unwrapped.set_parameters(sampled_params)
        else:
            print("[WARNING] Environment non supporta set_parameters()!")
        
        return obs, info