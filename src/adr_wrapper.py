import gymnasium as gym

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