
# ============================================================================
# ADR MANAGER CLASS - UNIVERSAL VERSION
# ============================================================================
import numpy as np 

class ADRManager:
    """
    Gestisce gli aggiornamenti dei range ADR per diverse varianti.
    Versione universale che supporta randomizzazione di:
    - Masse (thigh, leg, foot)
    - Friction (attrito)
    - Damping (smorzamento)
    - Gravity (gravità)
    - Force magnitude (perturbazioni)
    """
    
    def __init__(self, variant_config, target_performance, env_type='hopper', difficulty=None):
        self.config = variant_config
        self.target_performance = target_performance
        self.env_type = env_type
        self.difficulty = difficulty
        
        # ===== CONFIGURAZIONE RANGES E LIMITS DIPENDENTE DA ENV_TYPE E DIFFICULTY =====
        self.ranges, self.limits = self._init_ranges_for_env(env_type, difficulty)
        
        # Identificazione parametri che rappresentano masse (per logica speciale)
        self.mass_params = self._get_mass_params_for_env(env_type)
        
        self.delta = variant_config['delta']
        
        # Configurazione parametri da randomizzare (per env-specific o selective ADR)
        # Se non specificato, randomizza tutto
        self.params_to_randomize = variant_config.get('randomize_only', None)
        if self.params_to_randomize is None:
            # Default: randomizza tutti i parametri disponibili
            self.params_to_randomize = list(self.ranges.keys())
        
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
    
    def _init_ranges_for_env(self, env_type, difficulty=None):
        """Inizializza ranges e limits in base all'environment type e difficulty"""
        if env_type == 'hopper':
            ranges = {
                # Masse specifiche Hopper
                'thigh': [1.0, 1.0],
                'leg': [1.0, 1.0],
                'foot': [1.0, 1.0],
                # Parametri fisici comuni
                'friction': [1.0, 1.0],
                'gravity': [-9.81, -9.81],
                'force_magnitude': [0.0, 0.0],
            }
            
            # ===== LIMITI CALIBRATI PER DIFFICOLTÀ =====
            if difficulty == 'easy':
                # EASY: solo masse ±30% (target ha solo torso -1kg)
                limits = {
                    'thigh': [0.7, 1.3],
                    'leg': [0.7, 1.3],
                    'foot': [0.7, 1.3],
                    'friction': [1.0, 1.0],      # NO friction randomization
                    'gravity': [-9.81, -9.81],   # NO gravity randomization
                    'force_magnitude': [0.0, 0.0],  # NO forces
                }
                print(f"[ADR] Limiti EASY: masse ±30% only")
            elif difficulty == 'medium':
                # MEDIUM: masse ±40%, friction ±40%
                limits = {
                    'thigh': [0.6, 1.4],
                    'leg': [0.6, 1.4],
                    'foot': [0.6, 1.4],
                    'friction': [0.6, 1.4],
                    'gravity': [-9.81, -9.81],   # NO gravity randomization
                    'force_magnitude': [0.0, 0.0],  # NO forces
                }
                print(f"[ADR] Limiti MEDIUM: masse ±40%, friction ±40%")
            elif difficulty == 'hard':
                # HARD: masse ±60%, friction ±80%, gravity ±30%, forces 0-3N
                limits = {
                    'thigh': [0.4, 1.6],
                    'leg': [0.4, 1.6],
                    'foot': [0.4, 1.6],
                    'friction': [0.2, 1.8],
                    'gravity': [-12.0, -7.0],
                    'force_magnitude': [0.0, 3.0],
                }
                print(f"[ADR] Limiti HARD: masse ±60%, friction ±80%, gravity ±30%, forces 0-3N")
            else:
                # Default: limiti HARD (backward compatibility)
                limits = {
                    'thigh': [0.4, 1.6],
                    'leg': [0.4, 1.6],
                    'foot': [0.4, 1.6],
                    'friction': [0.1, 2.4],
                    'gravity': [-14.0, -6.0],
                    'force_magnitude': [0.0, 3.0],
                }
                print(f"[ADR] Limiti DEFAULT (hard): full randomization")
        elif env_type == 'ant':
            # Placeholder per Ant (4 gambe, 8 giunti)
            ranges = {
                'hip_1': [1.0, 1.0], 'ankle_1': [1.0, 1.0],
                'hip_2': [1.0, 1.0], 'ankle_2': [1.0, 1.0],
                'hip_3': [1.0, 1.0], 'ankle_3': [1.0, 1.0],
                'hip_4': [1.0, 1.0], 'ankle_4': [1.0, 1.0],
                'friction': [1.0, 1.0],
                'damping': [1.0, 1.0],
                'gravity': [-9.81, -9.81],
                'force_magnitude': [0.0, 0.0],
            }
            limits = {
                'hip_1': [0.5, 1.5], 'ankle_1': [0.5, 1.5],
                'hip_2': [0.5, 1.5], 'ankle_2': [0.5, 1.5],
                'hip_3': [0.5, 1.5], 'ankle_3': [0.5, 1.5],
                'hip_4': [0.5, 1.5], 'ankle_4': [0.5, 1.5],
                'friction': [0.2, 2.0],
                'damping': [0.5, 2.0],
                'gravity': [-12.0, -7.0],
                'force_magnitude': [0.0, 50.0],
            }
        else:
            # Default generico: solo parametri fisici comuni
            ranges = {
                'friction': [1.0, 1.0],
                'damping': [1.0, 1.0],
                'gravity': [-9.81, -9.81],
                'force_magnitude': [0.0, 0.0],
            }
            limits = {
                'friction': [0.2, 2.0],
                'damping': [0.5, 2.0],
                'gravity': [-12.0, -7.0],
                'force_magnitude': [0.0, 50.0],
            }
        
        return ranges, limits
    
    def _get_mass_params_for_env(self, env_type):
        """Restituisce la lista dei parametri che rappresentano masse"""
        if env_type == 'hopper':
            return ['thigh', 'leg', 'foot']
        elif env_type == 'ant':
            return ['hip_1', 'ankle_1', 'hip_2', 'ankle_2', 
                    'hip_3', 'ankle_3', 'hip_4', 'ankle_4']
        else:
            return []  # Nessun parametro di massa
    
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
        for param in self.params_to_randomize:
            if param not in self.ranges:
                continue  # Skip parametri non supportati
            
            lower, upper = self.ranges[param]
            limit_lower, limit_upper = self.limits[param]
            
            # Gestione speciale per gravity (range assoluto, non relativo)
            if param == 'gravity':
                # Espandi simmetricamente attorno a -9.81
                center = -9.81
                current_width = (upper - lower) / 2
                new_width = current_width + self.delta
                new_lower = max(center - new_width, limit_lower)
                new_upper = min(center + new_width, limit_upper)
            else:
                # Espansione normale per parametri relativi
                new_lower = max(lower - self.delta, limit_lower)
                new_upper = min(upper + self.delta, limit_upper)
            
            self.ranges[param] = [new_lower, new_upper]
    
    def _contract_ranges(self):
        """Contrai i range verso il centro (nominale)"""
        for param in self.params_to_randomize:
            if param not in self.ranges:
                continue
            
            lower, upper = self.ranges[param]
            
            # Centro nominale dipende dal tipo di parametro (dinamico)
            center = self._get_nominal_value(param)
            
            # Contrai verso il centro
            new_lower = min(lower + self.delta, center)
            new_upper = max(upper - self.delta, center)
            
            if new_lower > new_upper:
                new_lower = new_upper = center
            
            self.ranges[param] = [new_lower, new_upper]
    
    def sample_parameters(self, original_masses=None):
        """
        Campiona TUTTI i parametri fisici secondo la strategia ADR.
        Versione completamente dinamica e agnostica.
        
        Args:
            original_masses: Array delle masse originali (opzionale)
        
        Returns:
            dict: Dizionario completo con tutti i parametri campionati
        """
        sampled = {}
        
        # Itera su TUTTI i parametri definiti nei ranges
        for param_name in self.ranges.keys():
            # Se il parametro è da randomizzare, campiona un valore
            if param_name in self.params_to_randomize:
                sampled[param_name] = self._sample_value(param_name)
            else:
                # Altrimenti usa il valore nominale
                sampled[param_name] = self._get_nominal_value(param_name)
        
        # ===== GESTIONE SPECIALE PER MASSE (se fornite) =====
        # Se abbiamo original_masses, applichiamo i moltiplicatori campionati
        if original_masses is not None and len(self.mass_params) > 0:
            # Verifica che ci siano abbastanza masse
            required_len = len(self.mass_params) + 1  # +1 per torso/world
            if len(original_masses) >= required_len:
                new_masses = np.copy(original_masses)
                # Applica i moltiplicatori alle masse corrispondenti
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
        Restituisce il valore nominale (default) per un parametro.
        Metodo dinamico che non hardcoda nomi specifici.
        """
        if param_name == 'gravity':
            return -9.81  # Gravità terrestre
        elif param_name == 'force_magnitude':
            return 0.0  # Nessuna perturbazione
        elif param_name in self.mass_params:
            return 1.0  # Moltiplicatore neutro per masse
        elif param_name in ['friction', 'damping']:
            return 1.0  # Moltiplicatore neutro
        else:
            # Default generico per parametri sconosciuti
            return 1.0
    
    def _sample_value(self, param):
        """Campiona un singolo valore secondo la strategia della variante"""
        if param not in self.ranges:
            return self._get_nominal_value(param)
        
        lower, upper = self.ranges[param]
        
        # Boundary sampling (se abilitato)
        if self.config.get('boundary_sampling', False):
            boundary_prob = self.config.get('boundary_prob', 0.5)
            if np.random.rand() < boundary_prob:
                # Sample at boundaries
                return np.random.choice([lower, upper])
        
        # Uniform sampling (vanilla)
        return np.random.uniform(lower, upper)
    
    def get_range_diversity(self):
        """Calcola la larghezza media dei range (misura di diversity)"""
        widths = [upper - lower for lower, upper in self.ranges.values()]
        return np.mean(widths)

