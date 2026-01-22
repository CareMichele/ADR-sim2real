"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.stats import truncnorm

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 9.5,  # Aumentato da 3.0 a 5.5 (più lontano)
    "lookat": np.array((4.0, 0.0, 1.3)),  # Aumentato da 1.15 a 1.3 (guarda leggermente più in alto)
    "elevation": -20.0,  # Aumentato da -20.0 a -15.0 (angolo meno ripido, vista più dall'alto)
}

class CustomHopper(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
        "render_fps": 125,
    }

    def __init__(
        self,
        xml_file: str = "hopper.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),
        healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        domain: Optional[str] = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            domain,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if xml_file == "hopper.xml":
            # Il file hopper.xml è nella directory data/ alla root del progetto
            xml_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hopper.xml")

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

        # ===== STORE ORIGINAL PHYSICAL PARAMETERS =====
        # Salviamo i valori originali per poterli ripristinare/modificare con ADR
        self.original_masses = np.copy(self.model.body_mass[1:])    # Default link masses
        self.original_friction = np.copy(self.model.geom_friction)  # Friction coefficients [sliding, torsional, rolling]
        self.original_damping = np.copy(self.model.dof_damping)     # Joint damping
        self.original_gravity = np.copy(self.model.opt.gravity)     # Gravity vector [x, y, z]
        
        # Variabili per gestire le perturbazioni (forze esterne)
        self.current_max_push = 0.0  # Magnitudine massima della spinta (settata da ADR)
        self.push_probability = 0.1  # Probabilità di applicare spinta ad ogni step (10%)
        self.push_active = False     # Flag per visualizzazione
        self.original_colors = {}    # Salvataggio colori originali per visualizzazione

        # ===== DOMAIN-SPECIFIC CONFIGURATIONS =====
        if domain == 'source':
            # Source environment: imprecise torso mass (1kg shift)
            self.model.body_mass[1] -= 1.0
        
        elif domain == 'target':
            # Target environment: HOSTILE/DIFFICULT configuration for testing robustness
            print("[INFO] Initializing TARGET domain (hostile environment)")
            
            # 1. MASSES: Unbalanced configuration
            # Thigh heavier (+50%), Leg lighter (-50%)
            self.model.body_mass[2] *= 1.5   # Thigh: +50%
            self.model.body_mass[3] *= 0.5   # Leg: -50%
            print(f"  - Masses modified: thigh={self.model.body_mass[2]:.2f}kg, leg={self.model.body_mass[3]:.2f}kg")
            
            # 2. FRICTION: Slippery floor (low friction)
            for geom_id in range(self.model.ngeom):
                self.model.geom_friction[geom_id, 0] *= 0.5  # Sliding friction reduced to 50%
            print(f"  - Friction reduced to 0.5x (slippery)")
            
            # 3. GRAVITY: Heavier gravity
            self.model.opt.gravity[2] = -11.0  # Stronger than Earth
            print(f"  - Gravity: {self.model.opt.gravity[2]} m/s²")
            
            # 4. DISTURBANCES: Enable random pushes natively
            self.current_max_push = 1.0         # 1 Newton lateral pushes
            self.push_probability = 0.05        # 5% chance per step
            print(f"  - Random pushes enabled: max={self.current_max_push}N, prob={self.push_probability*100:.1f}%")
            print("[INFO] Target domain initialized (HARD MODE)")



    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = np.clip(self.data.qvel.flatten(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]
        
        # ===== APPLY RANDOM EXTERNAL FORCES (ADR Perturbations) =====
        # Applica spinte laterali casuali con probabilità definita
        if self.current_max_push > 0 and np.random.rand() < self.push_probability:
            # Genera forza casuale: direzione (x o y) e magnitudine
            force_direction = np.random.choice(['x', 'y'])
            force_magnitude = np.random.uniform(-self.current_max_push, self.current_max_push)
            
            # Applica la forza al torso (body index 1)
            # xfrc_applied[body_id] = [force_x, force_y, force_z, torque_x, torque_y, torque_z]
            if force_direction == 'x':
                self.data.xfrc_applied[1, 0] = force_magnitude  # Forza lungo X
            else:
                self.data.xfrc_applied[1, 1] = force_magnitude  # Forza lungo Y
            
            self.push_active = True
            self._set_robot_color([1.0, 0.0, 0.0, 1.0])  # Rosso quando viene spinto
        else:
            # Nessuna forza esterna
            self.data.xfrc_applied[1, :] = 0.0
            if self.push_active:
                self._restore_robot_color()  # Ripristina colore originale
                self.push_active = False
        
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def reset_model(self):
        #print(self.model.body_mass)
        #Questo serve per non far partire il robot mai nella stessa identica posizione perfetta al millimetro. 
        #Se partisse sempre identico, la rete neurale imparerebbe a memoria la sequenza di partenza invece di imparare a reagire.
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        
        #calcola posizione e velocità iniziale del robot con l'aggiunta di un piccolo rumores
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        #"Teletrasporta il robot istantaneamente in questa posizione qpos con questa velocità qvel"
        self.set_state(qpos, qvel)

        # Custom domain randomization
        # DISABLED: La randomizzazione è gestita esternamente da ADRWrapper
        # self.set_random_parameters() # TODO: May be useful insert a parameter to control if to sample new mass params
        
        """
        original_masses = np.array([4.05789051, 2.7813567, 5.31557477])
        
        sigma = 0.6
        
        lower_bounds = original_masses - (original_masses*sigma)
        upper_bounds = original_masses + (original_masses*sigma)
        new_masses = self.np_random.uniform(low = lower_bounds, high = upper_bounds)
        
        self.model.body_mass[2:] = new_masses
        
        #print(f"Masse episodio: {self.model.body_mass[2:]}")
        """
        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }

    def set_random_parameters(self):
        """Set random masses
        TODO
        """
        self.set_parameters(*self.sample_parameters())

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution
        N.B. You can't change the mass of the torso (first link)
        TODO
        """
        return

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.model.body_mass[1:] )
        return masses

    def set_parameters(self, parameters):
        """
        Applica TUTTI i parametri fisici dall'ADR Manager.
        Versione agnostica con mass_mapping locale.
        
        Args:
            parameters: dict o array (per backward compatibility)
                Se dict: {'thigh': float, 'leg': float, 'foot': float, 
                         'friction': float, 'damping': float, 'gravity': float, 
                         'force_magnitude': float, 'masses': array [optional]}
                Se array: solo masse (vecchio comportamento)
        """
        # ===== BACKWARD COMPATIBILITY: se riceve un array, trattalo come masse =====
        if isinstance(parameters, (np.ndarray, list)):
            self.model.body_mass[1:] = parameters
            return
        
        # ===== NUOVO COMPORTAMENTO: dizionario completo =====
        if not isinstance(parameters, dict):
            print(f"[WARNING] set_parameters ricevuto tipo inatteso: {type(parameters)}")
            return
        
        # === 1. MASSES (gestione con mapping locale) ===
        # Mappa locale: nome parametro -> indice in body_mass
        mass_mapping = {
            'thigh': 2,  # body_mass[2] = thigh
            'leg': 3,    # body_mass[3] = leg  
            'foot': 4,   # body_mass[4] = foot
        }
        
        # Applica le masse usando la mappa locale
        for param_name, body_idx in mass_mapping.items():
            if param_name in parameters:
                # Se il parametro è un moltiplicatore, applicalo alla massa originale
                self.model.body_mass[body_idx] = self.original_masses[body_idx-1] * parameters[param_name]
        
        # Backward compatibility: se c'è 'masses' come array completo, usalo
        if 'masses' in parameters and parameters['masses'] is not None:
            self.model.body_mass[1:] = parameters['masses']
        
        # === 2. FRICTION ===
        if 'friction' in parameters:
            multiplier = parameters['friction']
            # MuJoCo friction: array di shape (n_geoms, 3) = [sliding, torsional, rolling]
            # Moltiplichiamo solo sliding friction (colonna 0), più importante per locomozione
            for geom_id in range(self.model.ngeom):
                self.model.geom_friction[geom_id, 0] = self.original_friction[geom_id, 0] * multiplier
        
        # === 3. DAMPING ===
        if 'damping' in parameters:
            multiplier = parameters['damping']
            # Applica lo smorzamento ai giunti attivi (dof = degrees of freedom)
            # Nel Hopper: 6 DOF (3 per root free joint, 3 per i giunti delle gambe)
            # Ci interessa modificare solo i giunti attivi (da index 3 in poi)
            for dof_id in range(3, self.model.nv):  # Skip root joint (primi 3 DOF)
                self.model.dof_damping[dof_id] = self.original_damping[dof_id] * multiplier
        
        # === 4. GRAVITY ===
        if 'gravity' in parameters:
            gravity_value = parameters['gravity']
            # MuJoCo gravity: vettore [gx, gy, gz]. Normalmente [0, 0, -9.81]
            # Modifichiamo solo la componente Z (verticale)
            self.model.opt.gravity[2] = gravity_value
        
        # === 5. FORCE MAGNITUDE ===
        if 'force_magnitude' in parameters:
            # Salva il valore massimo della spinta per usarlo durante step()
            self.current_max_push = parameters['force_magnitude']
    
    def _set_robot_color(self, rgba):
        """Cambia il colore del robot (per visualizzare perturbazioni)"""
        # Salva i colori originali solo la prima volta
        if not self.original_colors:
            for geom_id in range(self.model.ngeom):
                self.original_colors[geom_id] = np.copy(self.model.geom_rgba[geom_id])
        
        # Applica il nuovo colore a tutti i geom del robot (escluso il pavimento)
        # Assumiamo che il pavimento sia il primo geom (index 0)
        for geom_id in range(1, self.model.ngeom):
            self.model.geom_rgba[geom_id] = rgba
    
    def _restore_robot_color(self):
        """Ripristina i colori originali del robot"""
        if self.original_colors:
            for geom_id, color in self.original_colors.items():
                self.model.geom_rgba[geom_id] = color


"""
    Registered environments
"""
gym.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)


