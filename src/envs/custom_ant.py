"""Implementation of the Ant environment supporting
domain randomization optimization."""
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 12.0,
    "lookat": np.array((0.0, 0.0, 0.5)),
    "elevation": -20.0,
}

class CustomAnt(MujocoEnv, utils.EzPickle):
    """
    Ant environment for Universal ADR (Automatic Domain Randomization).
    
    BODY STRUCTURE (9 bodies total):
    - Body 0: torso (main body)
    - Body 1: front_left_hip
    - Body 2: front_left_ankle  
    - Body 3: front_right_hip
    - Body 4: front_right_ankle
    - Body 5: back_left_hip
    - Body 6: back_left_ankle
    - Body 7: back_right_hip
    - Body 8: back_right_ankle
    
    ADR PARAMETERS MAPPING (8 link masses):
    - hip_1 (front_left_hip): body_mass[1]
    - ankle_1 (front_left_ankle): body_mass[2]
    - hip_2 (front_right_hip): body_mass[3]
    - ankle_2 (front_right_ankle): body_mass[4]
    - hip_3 (back_left_hip): body_mass[5]
    - ankle_3 (back_left_ankle): body_mass[6]
    - hip_4 (back_right_hip): body_mass[7]
    - ankle_4 (back_right_ankle): body_mass[8]
    """
    
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        xml_file: str = "ant.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.5,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.2, 1.0),
        reset_noise_scale: float = 0.1,
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
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            domain,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if xml_file == "ant.xml":
            xml_file = os.path.join(os.path.dirname(__file__), "../../data/ant.xml")

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
        # original_masses[0] = torso, [1-8] = 8 limbs (total 9 elements)
        self.original_masses = np.copy(self.model.body_mass[1:])    # Skip world, include torso + 8 limbs
        self.original_friction = np.copy(self.model.geom_friction)  # Friction coefficients
        self.original_damping = np.copy(self.model.dof_damping)     # Joint damping
        self.original_gravity = np.copy(self.model.opt.gravity)     # Gravity vector
        
        # Variabili per gestire le perturbazioni (forze esterne)
        self.current_max_push = 0.0  # Magnitudine massima della spinta (settata da ADR)
        self.push_probability = 0.1  # Probabilità di applicare spinta ad ogni step (10%)
        self.push_active = False     # Flag per visualizzazione
        self.original_colors = {}    # Salvataggio colori originali per visualizzazione

        # ===== DOMAIN-SPECIFIC CONFIGURATIONS =====
        if domain == 'source':
            # Source environment: standard configuration
            pass
        
        elif domain == 'target':
            # Target environment: HELL MODE - extremely difficult configuration
            print("[INFO] Initializing TARGET domain for Ant (HELL MODE)")
            
            # 1. MASSES: Unbalanced configuration
            # Front legs heavier (+50%), Back legs lighter (-50%)
            # Body 0 = world, Body 1 = torso, Bodies 2-9 = 8 limbs
            self.model.body_mass[2] *= 1.5   # hip_1 (front_left_hip): +50%
            self.model.body_mass[3] *= 1.5   # ankle_1 (front_left_ankle): +50%
            self.model.body_mass[4] *= 1.5   # hip_2 (front_right_hip): +50%
            self.model.body_mass[5] *= 1.5   # ankle_2 (front_right_ankle): +50%
            
            self.model.body_mass[6] *= 0.5   # hip_3 (back_left_hip): -50%
            self.model.body_mass[7] *= 0.5   # ankle_3 (back_left_ankle): -50%
            self.model.body_mass[8] *= 0.5   # hip_4 (back_right_hip): -50%
            self.model.body_mass[9] *= 0.5   # ankle_4 (back_right_ankle): -50%
            print(f"  - Masses unbalanced: front legs +50%, back legs -50%")
            
            # 2. FRICTION: Slippery floor (low friction)
            for geom_id in range(self.model.ngeom):
                self.model.geom_friction[geom_id, 0] *= 0.5  # Sliding friction reduced to 50%
            print(f"  - Friction reduced to 0.5x (slippery)")
            
            # 3. GRAVITY: Heavier gravity
            self.model.opt.gravity[2] = -12.0  # Stronger than Earth
            print(f"  - Gravity: {self.model.opt.gravity[2]} m/s²")
            
            # 4. DISTURBANCES: Enable random pushes natively
            self.current_max_push = 1.0         # 1 Newton lateral pushes
            self.push_probability = 0.05        # 5% chance per step
            print(f"  - Random pushes enabled: max={self.current_max_push}N, prob={self.push_probability*100:.1f}%")
            print("[INFO] Target domain initialized (HELL MODE)")

    def _get_obs(self) -> np.ndarray:
        """Get observation from the environment"""
        position = self.data.qpos.flat[:]
        velocity = self.data.qvel.flat[:]

        if self._exclude_current_positions_from_observation:
            position = position[2:]  # Skip x, y position for Ant

        observations = np.concatenate((position, velocity))
        return observations

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
        }

    @property
    def is_healthy(self):
        """Check if the ant is in a healthy state"""
        z_pos = self.data.qpos[2]
        return self._healthy_z_range[0] < z_pos < self._healthy_z_range[1]

    def step(self, action):
        """Execute one step of environment dynamics"""
        if np.array(action).shape != self.action_space.shape:
            raise ValueError(f"Action shape mismatch: {action.shape} vs {self.action_space.shape}")
        
        if not np.isfinite(action).all():
            raise ValueError(f"Action contains NaN or Inf: {action}")

        # ===== APPLY RANDOM EXTERNAL FORCES (ADR Perturbations) =====
        # Applica spinte laterali casuali con probabilità definita
        if self.current_max_push > 0 and np.random.rand() < self.push_probability:
            # Genera forza casuale: direzione (x o y) e magnitudine
            force_direction = np.random.choice(['x', 'y'])
            force_magnitude = np.random.uniform(-self.current_max_push, self.current_max_push)
            
            # Applica la forza al torso (body index 0 per Ant)
            # xfrc_applied[body_id] = [force_x, force_y, force_z, torque_x, torque_y, torque_z]
            if force_direction == 'x':
                self.data.xfrc_applied[0, 0] = force_magnitude  # Forza lungo X
            else:
                self.data.xfrc_applied[0, 1] = force_magnitude  # Forza lungo Y
            
            self.push_active = True
            self._set_robot_color([1.0, 0.0, 0.0, 1.0])  # Rosso quando viene spinto
        else:
            # Nessuna forza esterna
            self.data.xfrc_applied[0, :] = 0.0
            if self.push_active:
                self._restore_robot_color()  # Ripristina colore originale
                self.push_active = False

        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        x_velocity = self.data.qvel[0]
        forward_reward = self._forward_reward_weight * x_velocity

        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1))
        )

        reward = forward_reward - ctrl_cost - contact_cost + self._healthy_reward
        
        terminated = False
        if self._terminate_when_unhealthy:
            terminated = not self.is_healthy

        truncated = False

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, self._get_reset_info()

    def reset_model(self):
        """Reset the environment to initial state"""
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        obs = self._get_obs()
        return obs

    def get_parameters(self):
        """Get value of mass for each link (excluding torso)"""
        masses = np.array(self.model.body_mass[1:])
        return masses

    def set_parameters(self, parameters):
        """
        Applica TUTTI i parametri fisici dall'ADR Manager (Universal ADR).
        Versione compatibile con Ant a 8 link.
        
        Args:
            parameters: dict con parametri ADR
                {'hip_1': float, 'ankle_1': float, 'hip_2': float, 'ankle_2': float,
                 'hip_3': float, 'ankle_3': float, 'hip_4': float, 'ankle_4': float,
                 'friction': float, 'damping': float, 'gravity': float, 
                 'force_magnitude': float}
        """
        # ===== BACKWARD COMPATIBILITY: se riceve un array, trattalo come masse =====
        if isinstance(parameters, (np.ndarray, list)):
            self.model.body_mass[1:] = parameters
            return
        
        # ===== NUOVO COMPORTAMENTO: dizionario completo =====
        if not isinstance(parameters, dict):
            print(f"[WARNING] set_parameters ricevuto tipo inatteso: {type(parameters)}")
            return
        
        # === 1. MASSES (gestione con mapping per Ant) ===
        # Mappa locale: nome parametro -> indice in body_mass
        # Body 0 = world, Body 1 = torso, Bodies 2-9 = 8 limbs
        mass_mapping = {
            'hip_1': 2,     # front_left_hip
            'ankle_1': 3,   # front_left_ankle
            'hip_2': 4,     # front_right_hip
            'ankle_2': 5,   # front_right_ankle
            'hip_3': 6,     # back_left_hip
            'ankle_3': 7,   # back_left_ankle
            'hip_4': 8,     # back_right_hip
            'ankle_4': 9,   # back_right_ankle
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
            # Nel Ant: 14 DOF totali (6 per root free joint, 8 per i giunti delle gambe)
            # Ci interessa modificare solo i giunti attivi (da index 6 in poi)
            for dof_id in range(6, self.model.nv):  # Skip root joint (primi 6 DOF)
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
    Registered environments for Ant
"""
gym.register(
        id="CustomAnt-v0",
        entry_point="%s:CustomAnt" % __name__,
        max_episode_steps=1000,
)

gym.register(
        id="CustomAnt-source-v0",
        entry_point="%s:CustomAnt" % __name__,
        max_episode_steps=1000,
        kwargs={"domain": "source"}
)

gym.register(
        id="CustomAnt-target-v0",
        entry_point="%s:CustomAnt" % __name__,
        max_episode_steps=1000,
        kwargs={"domain": "target"}
)
