
# Universal ADR manager for both Hopper and Ant

import numpy as np
import json
from pathlib import Path

class ADRManager:
    """
    Manages ADR range updates for different variants.
    Universal version supporting randomization of:
    - Masses (thigh, leg, foot)
    - Friction
    - Gravity
    - Force magnitude (perturbations)
    """
    
    def __init__(self, variant_config, target_performance, env_type='hopper', difficulty=None):
        self.config = variant_config
        self.target_performance = target_performance
        self.env_type = env_type
        self.difficulty = difficulty
        
        self.ranges, self.limits = self._init_ranges_for_env(env_type, difficulty)
        self.mass_params = self._get_mass_params_for_env(env_type)
        
        self.delta = variant_config['delta']
        
        self.params_to_randomize = variant_config.get('randomize_only', None)
        if self.params_to_randomize is None:
            self.params_to_randomize = list(self.ranges.keys())
        
        # Progressive curriculum
        if variant_config.get('progressive', False):
            self.threshold_schedule = variant_config['threshold_schedule']
            self.current_phase = 0
            self.threshold = self.threshold_schedule[0] * target_performance
            self.stable_count = 0
            print(f"[INFO] Progressive ADR initialized")
            print(f"       Schedule: {self.threshold_schedule}")
            print(f"       Initial threshold: {self.threshold:.1f}")
        else:
            threshold_pct = variant_config.get('threshold_pct', 0.75)
            self.threshold = threshold_pct * target_performance
        
        self.performance_history = []
    
    def _init_ranges_for_env(self, env_type, difficulty=None):
        """Initialize ranges and limits based on environment type and difficulty"""
        # Load ADR config from JSON
        config_path = Path(__file__).parent.parent / 'configs' / 'adr_configs.json'
        with open(config_path, 'r') as f:
            adr_config = json.load(f)
        
        # Get initial ranges
        ranges_template = adr_config['initial_ranges'].get(env_type, adr_config['initial_ranges']['default'])
        ranges = {k: list(v) for k, v in ranges_template.items()}
        
        # Get limits based on difficulty
        env_limits = adr_config['difficulty_limits'].get(env_type, {})
        if difficulty and difficulty in env_limits:
            limits = {k: list(v) for k, v in env_limits[difficulty].items()}
            difficulty_label = difficulty.upper()
        else:
            limits = {k: list(v) for k, v in env_limits.get('default', {}).items()}
            difficulty_label = "DEFAULT"
        
        print(f"[ADR] {difficulty_label} limits loaded from config")
        
        return ranges, limits
    
    def _get_mass_params_for_env(self, env_type):
        """Returns the list of parameters representing masses"""
        config_path = Path(__file__).parent.parent / 'configs' / 'adr_configs.json'
        with open(config_path, 'r') as f:
            adr_config = json.load(f)
        
        return adr_config['mass_params'].get(env_type, [])
    
    def update_ranges(self, mean_reward):
        """Update ADR ranges based on performance"""
        self.performance_history.append(mean_reward)
        
        # Check progressive phase advancement
        if self.config.get('progressive', False):
            self._check_phase_advancement(mean_reward)
        
        if mean_reward >= self.threshold:
            self._expand_ranges()
            status = "EXPAND"
        else:
            self._contract_ranges()
            status = "CONTRACT"
        
        return status
    
    def _check_phase_advancement(self, mean_reward):
        """Advance to next phase if performance is stable"""
        if mean_reward >= self.threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0
        
        if self.stable_count >= 3 and self.current_phase < len(self.threshold_schedule) - 1:
            self.current_phase += 1
            self.threshold = self.threshold_schedule[self.current_phase] * self.target_performance
            self.stable_count = 0
            print(f"\nPROGRESSIVE: Advanced to phase {self.current_phase + 1}")
            print(f"   New threshold: {self.threshold:.1f} ({self.threshold_schedule[self.current_phase]*100:.0f}%)")
    
    def _expand_ranges(self):
        """Expand ranges symmetrically"""
        for param in self.params_to_randomize:
            if param not in self.ranges:
                continue
            
            lower, upper = self.ranges[param]
            limit_lower, limit_upper = self.limits[param]
            
            if param == 'gravity':
                center = -9.81
                current_width = (upper - lower) / 2
                new_width = current_width + self.delta
                new_lower = max(center - new_width, limit_lower)
                new_upper = min(center + new_width, limit_upper)
            else:
                new_lower = max(lower - self.delta, limit_lower)
                new_upper = min(upper + self.delta, limit_upper)
            
            self.ranges[param] = [new_lower, new_upper]
    
    def _contract_ranges(self):
        """Contract ranges toward nominal center"""
        for param in self.params_to_randomize:
            if param not in self.ranges:
                continue
            
            lower, upper = self.ranges[param]
            center = self._get_nominal_value(param)
            
            new_lower = min(lower + self.delta, center)
            new_upper = max(upper - self.delta, center)
            
            if new_lower > new_upper:
                new_lower = new_upper = center
            
            self.ranges[param] = [new_lower, new_upper]
    
    def sample_parameters(self, original_masses=None):
        """
        Sample all physical parameters according to ADR strategy.
        Fully dynamic and environment-agnostic version.
        
        Args:
            original_masses: Array of original masses (optional)
        
        Returns:
            dict: Complete dictionary with all sampled parameters
        """
        sampled = {}
        
        for param_name in self.ranges.keys():
            if param_name in self.params_to_randomize:
                sampled[param_name] = self._sample_value(param_name)
            else:
                sampled[param_name] = self._get_nominal_value(param_name)
        
        if original_masses is not None and len(self.mass_params) > 0:
            required_len = len(self.mass_params) + 1
            if len(original_masses) >= required_len:
                new_masses = np.copy(original_masses)
                for i, mass_param in enumerate(self.mass_params, start=1):
                    if mass_param in sampled:
                        new_masses[i] = original_masses[i] * sampled[mass_param]
                sampled['masses'] = new_masses
            else:
                sampled['masses'] = None
        else:
            sampled['masses'] = None
        
        return sampled
    
    def _get_nominal_value(self, param_name):
        """
        Returns the nominal (default) value for a parameter.
        Dynamic method that doesn't hardcode specific names.
        """
        if param_name == 'gravity':
            return -9.81
        elif param_name == 'force_magnitude':
            return 0.0
        elif param_name in self.mass_params:
            return 1.0
        elif param_name in ['friction', 'damping']:
            return 1.0 
        else:
            return 1.0
    
    def _sample_value(self, param):
        """Sample a single value according to variant strategy"""
        if param not in self.ranges:
            return self._get_nominal_value(param)
        
        lower, upper = self.ranges[param]
        
        if self.config.get('boundary_sampling', False):
            boundary_prob = self.config.get('boundary_prob', 0.5)
            if np.random.rand() < boundary_prob:
                return np.random.choice([lower, upper])
        
        return np.random.uniform(lower, upper)
    
    def get_range_diversity(self):
        """Calculate average range width (diversity measure)"""
        widths = [upper - lower for lower, upper in self.ranges.values()]
        return np.mean(widths)

