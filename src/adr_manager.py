
# ============================================================================
# ADR MANAGER CLASS
# ============================================================================
import numpy as np 

class ADRManager:
    """Gestisce gli aggiornamenti dei range ADR per diverse varianti"""
    
    def __init__(self, variant_config, target_performance):
        self.config = variant_config
        self.target_performance = target_performance
        
        self.ranges = {
            'thigh': [1.0, 1.0],
            'leg': [1.0, 1.0],
            'foot': [1.0, 1.0],
        }
        
        self.limits = {
            'thigh': [0.5, 1.5],
            'leg': [0.5, 1.5],
            'foot': [0.5, 1.5],
        }
        
        self.delta = variant_config['delta']
        
        # Progressive curriculum
        if variant_config.get('progressive', False):
            self.threshold_schedule = variant_config['threshold_schedule']
            self.current_phase = 0
            self.threshold = self.threshold_schedule[0] * target_performance
            self.stable_count = 0
            print(f"[INFO] Progressive ADR inizializzato")
            print(f"       Schedule: {self.threshold_schedule}")
            print(f"       Threshold iniziale: {self.threshold:.1f}")
        else:
            threshold_pct = variant_config.get('threshold_pct', 0.75)
            self.threshold = threshold_pct * target_performance
        
        self.performance_history = []
    
    def update_ranges(self, mean_reward):
        """Aggiorna i range ADR in base alla performance"""
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
        """Avanza alla fase successiva se performance è stabile"""
        if mean_reward >= self.threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0
        
        # Avanza se stabile per 3 update consecutivi
        if self.stable_count >= 3 and self.current_phase < len(self.threshold_schedule) - 1:
            self.current_phase += 1
            self.threshold = self.threshold_schedule[self.current_phase] * self.target_performance
            self.stable_count = 0
            print(f"\n🎯 PROGRESSIVE: Avanzato a fase {self.current_phase + 1}")
            print(f"   Nuovo threshold: {self.threshold:.1f} ({self.threshold_schedule[self.current_phase]*100:.0f}%)")
    
    def _expand_ranges(self):
        """Espandi i range simmetricamente"""
        for param in self.ranges:
            if self.config.get('randomize_only') and param not in self.config['randomize_only']:
                continue
            
            lower, upper = self.ranges[param]
            limit_lower, limit_upper = self.limits[param]
            
            new_lower = max(lower - self.delta, limit_lower)
            new_upper = min(upper + self.delta, limit_upper)
            
            self.ranges[param] = [new_lower, new_upper]
    
    def _contract_ranges(self):
        """Contrai i range verso il centro (nominale = 1.0)"""
        for param in self.ranges:
            if self.config.get('randomize_only') and param not in self.config['randomize_only']:
                continue
            
            lower, upper = self.ranges[param]
            center = 1.0
            
            new_lower = min(lower + self.delta, center)
            new_upper = max(upper - self.delta, center)
            
            if new_lower > new_upper:
                new_lower = new_upper = center
            
            self.ranges[param] = [new_lower, new_upper]
    
    def sample_parameters(self, original_masses):
        """Campiona i moltiplicatori di massa secondo la strategia della variante"""
        new_masses = np.copy(original_masses)
        
        params_to_randomize = self.config.get('randomize_only', ['thigh', 'leg', 'foot'])
        
        for i, param in enumerate(['thigh', 'leg', 'foot'], start=1):
            if param not in params_to_randomize:
                continue
            
            lower, upper = self.ranges[param]
            
            # Boundary sampling
            if self.config.get('boundary_sampling', False):
                boundary_prob = self.config.get('boundary_prob', 0.5)
                if np.random.rand() < boundary_prob:
                    # Sample at boundaries
                    multiplier = np.random.choice([lower, upper])
                else:
                    # Sample uniformly
                    multiplier = np.random.uniform(lower, upper)
            else:
                # Vanilla: uniform sampling
                multiplier = np.random.uniform(lower, upper)
            
            new_masses[i] = original_masses[i] * multiplier
        
        return new_masses
    
    def get_range_diversity(self):
        """Calcola la larghezza media dei range (misura di diversity)"""
        widths = [upper - lower for lower, upper in self.ranges.values()]
        return np.mean(widths)

