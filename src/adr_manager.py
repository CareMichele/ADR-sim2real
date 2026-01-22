
# ============================================================================
# ADR MANAGER CLASS
# ============================================================================
import numpy as np 

class ADRManager:
    """Gestisce gli aggiornamenti dei range ADR per diverse varianti"""
    
    def __init__(self, variant_config, target_performance, parameters=None, param_to_idx=None, limits=None):
        self.config = variant_config
        self.target_performance = target_performance
        
        # Se 'parameters' non è specificato, usa default (Hopper: thigh, leg, foot)
        if parameters is None:
            parameters = ['thigh', 'leg', 'foot']
        
        # Mapping esplicito nome_parametro -> indice_massa in MuJoCo
        # Fondamentale per randomizzare il parametro corretto!
        if param_to_idx is None:
            # Default per Hopper
            param_to_idx = {
                'thigh': 0,
                'leg': 1,
                'foot': 2,
            }
        
        self.param_to_idx = param_to_idx
        
        # Inizializza ranges
        self.ranges = {param: [1.0, 1.0] for param in parameters}
        
        # Inizializza limits (default: [0.5, 1.5], ma può essere customizzato)
        if limits is None:
            limits = [0.5, 1.5]
        
        self.limits = {param: limits for param in parameters}
        
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
        """Campiona i moltiplicatori di massa secondo la strategia della variante
        
        Generico per qualsiasi ambiente (Hopper, Ant, ecc.)
        Usa il mapping param_to_idx per applicare il moltiplicatore all'indice CORRETTO.
        """
        new_masses = np.copy(original_masses)
        
        # Parametri da randomizzare (default: tutti quelli nei ranges)
        params_to_randomize = self.config.get('randomize_only', list(self.ranges.keys()))
        
        # Itera sui parametri in ranges e applica moltiplicatori
        for param in self.ranges.keys():
            if param not in params_to_randomize:
                continue
            
            # Usa il mapping per trovare l'indice corretto in MuJoCo
            idx = self.param_to_idx[param]
            
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
            
            # Applica il moltiplicatore all'INDICE CORRETTO
            new_masses[idx] = original_masses[idx] * multiplier
        
        return new_masses
    
    def get_range_diversity(self):
        """Calcola la larghezza media dei range (misura di diversity)"""
        widths = [upper - lower for lower, upper in self.ranges.values()]
        return np.mean(widths)

