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
    "distance": 9.5,
    "lookat": np.array((4.0, 0.0, 1.3)),
    "elevation": -20.0,
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
            xml_file = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "hopper.xml")

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

        self.original_masses = np.copy(self.model.body_mass[1:])
        self.original_friction = np.copy(self.model.geom_friction)
        self.original_damping = np.copy(self.model.dof_damping)
        self.original_gravity = np.copy(self.model.opt.gravity)
        
        self.current_max_push = 0.0
        self.push_probability = 0.1
        self.push_active = False
        self.original_colors = {}

        if domain == 'source':
            self.model.body_mass[1] -= 1.0
        
        elif domain == 'target-easy':
            print("[INFO] Initializing TARGET-EASY domain")
            print(f"  - Standard masses (source has -1kg torso difference)")
            print("[INFO] Target-EASY initialized")
        
        elif domain == 'target-medium':
            print("[INFO] Initializing TARGET-MEDIUM domain")
            
            self.model.body_mass[2] *= 1.2
            self.model.body_mass[3] *= 0.8
            print(f"  - Masses modified: thigh={self.model.body_mass[2]:.2f}kg, leg={self.model.body_mass[3]:.2f}kg")
            
            for geom_id in range(self.model.ngeom):
                self.model.geom_friction[geom_id, 0] *= 0.7
            print(f"  - Friction reduced to 0.7x (moderately slippery)")
            print("[INFO] Target-MEDIUM initialized")
        
        elif domain == 'target' or domain == 'target-hard':
            print("[INFO] Initializing TARGET-HARD domain (hostile environment)")
            
            self.model.body_mass[2] *= 1.5
            self.model.body_mass[3] *= 0.5
            print(f"  - Masses modified: thigh={self.model.body_mass[2]:.2f}kg, leg={self.model.body_mass[3]:.2f}kg")
            
            for geom_id in range(self.model.ngeom):
                self.model.geom_friction[geom_id, 0] *= 0.5
            print(f"  - Friction reduced to 0.5x (slippery)")
            
            self.model.opt.gravity[2] = -11.0
            print(f"  - Gravity: {self.model.opt.gravity[2]} m/s²")
            
            self.current_max_push = 10.0
            self.push_probability = 0.05
            print(f"  - Random pushes enabled: max={self.current_max_push}N, prob={self.push_probability*100:.1f}%")
            print("[INFO] Target-HARD initialized")



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
        
        if self.current_max_push > 0 and np.random.rand() < self.push_probability:
            force_direction = np.random.choice(['x', 'y'])
            force_magnitude = np.random.uniform(-self.current_max_push, self.current_max_push)
            
            if force_direction == 'x':
                self.data.xfrc_applied[1, 0] = force_magnitude
            else:
                self.data.xfrc_applied[1, 1] = force_magnitude
            
            self.push_active = True
            self._set_robot_color([1.0, 0.0, 0.0, 1.0])
        else:
            self.data.xfrc_applied[1, :] = 0.0
            if self.push_active:
                self._restore_robot_color()
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
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)
        
        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.model.body_mass[1:] )
        return masses

    def set_parameters(self, parameters):
        """
        Apply all physical parameters from ADR Manager.
        Environment-agnostic version with local mass_mapping.
        
        Args:
            parameters: dict or array (for backward compatibility)
                If dict: {'thigh': float, 'leg': float, 'foot': float, 
                         'friction': float, 'damping': float, 'gravity': float, 
                         'force_magnitude': float, 'masses': array [optional]}
                If array: masses only (old behavior)
        """
        if isinstance(parameters, (np.ndarray, list)):
            self.model.body_mass[1:] = parameters
            return
        
        if not isinstance(parameters, dict):
            print(f"[WARNING] set_parameters received unexpected type: {type(parameters)}")
            return
        
        mass_mapping = {
            'thigh': 2,
            'leg': 3,
            'foot': 4,
        }
        
        for param_name, body_idx in mass_mapping.items():
            if param_name in parameters:
                self.model.body_mass[body_idx] = self.original_masses[body_idx-1] * parameters[param_name]
        
        if 'masses' in parameters and parameters['masses'] is not None:
            self.model.body_mass[1:] = parameters['masses']
        
        if 'friction' in parameters:
            multiplier = parameters['friction']
            for geom_id in range(self.model.ngeom):
                self.model.geom_friction[geom_id, 0] = self.original_friction[geom_id, 0] * multiplier
        if 'gravity' in parameters:
            gravity_value = parameters['gravity']
            self.model.opt.gravity[2] = gravity_value
        if 'force_magnitude' in parameters:
            self.current_max_push = parameters['force_magnitude']
    
    def _set_robot_color(self, rgba):
        """Change robot color (to visualize perturbations)"""
        if not self.original_colors:
            for geom_id in range(self.model.ngeom):
                self.original_colors[geom_id] = np.copy(self.model.geom_rgba[geom_id])
        
        for geom_id in range(1, self.model.ngeom):
            self.model.geom_rgba[geom_id] = rgba
    
    def _restore_robot_color(self):
        """Restore robot's original colors"""
        if self.original_colors:
            for geom_id, color in self.original_colors.items():
                self.model.geom_rgba[geom_id] = color

# Register envs

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
        id="CustomHopper-target-easy-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target-easy"}
)

gym.register(
        id="CustomHopper-target-medium-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target-medium"}
)

gym.register(
        id="CustomHopper-target-hard-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target-hard"}
)


