"""Implementation of the Ant environment supporting
domain randomization optimization."""
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.stats import truncnorm

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 12.0,
    "lookat": np.array((0.0, 0.0, 0.5)),
    "elevation": -20.0,
}

class CustomAnt(MujocoEnv, utils.EzPickle):
    """
    Ant environment for domain randomization.
    
    Ant has 8 bodies (1 torso + 4 legs with 2 segments each):
    0: torso
    1: front_left_hip
    2: front_left_foot
    3: front_right_hip
    4: front_right_foot
    5: back_left_hip
    6: back_left_foot
    7: back_right_hip
    8: back_right_foot
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
            dr_type: str = None,
            dr_params: dict = None,
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
            dr_type,
            dr_params,
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
        self.dr_type = dr_type or 'none'
        self.dr_params = dr_params or {}
        self.domain = domain

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

        # Store original masses before any domain randomization
        # Skip torso (index 0), keep all limbs [1:9]
        self.original_masses = np.copy(self.model.body_mass[1:])
        
        self._init_domain_randomization()

    def _init_domain_randomization(self):
        """Initialize domain randomization parameters based on dr_type"""
        if self.dr_type == 'none':
            pass
        else:
            raise ValueError(f"Unknown domain randomization type: {self.dr_type}")
    
    def _get_obs(self) -> np.ndarray:
        """Get observation from the environment"""
        position = self.data.qpos.flat[:]
        velocity = self.data.qvel.flat[:]

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observations = np.concatenate((position, velocity))
        return observations

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
        }

    def set_random_parameters(self):
        """Set random masses for domain randomization"""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses according to the domain randomization distribution
        
        Ant has 8 body parts (excluding torso):
        1: front_left_hip
        2: front_left_foot
        3: front_right_hip
        4: front_right_foot
        5: back_left_hip
        6: back_left_foot
        7: back_right_hip
        8: back_right_foot
        
        Returns:
            masses: array of randomized masses
        """
        # No randomization by default
        return np.copy(self.original_masses)
    
    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array(self.model.body_mass[1:])
        return masses

    def set_parameters(self, task):
        """Set each ant link's mass to a new value"""
        self.model.body_mass[1:] = task

    def step(self, action):
        """Execute one step of environment dynamics"""
        if np.array(action).shape != self.action_space.shape:
            raise ValueError(f"Action shape mismatch: {action.shape} vs {self.action_space.shape}")
        
        if not np.isfinite(action).all():
            raise ValueError(f"Action contains NaN or Inf: {action}")

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
            z_pos = self.data.qpos[2]
            terminated = z_pos < self._healthy_z_range[0] or z_pos > self._healthy_z_range[1]

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
