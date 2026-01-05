"""Implementation of the Hopper environment supporting
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
    "trackbodyid": 2,
    "distance": 10.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
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
        "render_fps": 24,
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
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            domain,
            dr_type,
            dr_params,
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
        self.dr_type = dr_type or 'none'
        self.dr_params = dr_params or {}
        self.domain = domain

        if xml_file == "hopper.xml":
             xml_file = os.path.join(os.path.dirname(__file__), "./data/hopper.xml")

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

        if self.domain == 'source':
            self.model.body_mass[1] -= 1.0
        
        self.original_masses = np.copy(self.model.body_mass[1:]) 
        
        self._init_domain_randomization()

    def _init_domain_randomization(self):
        """Initialize domain randomization parameters based on dr_type"""
        if self.dr_type == 'none':
            pass
        
        elif self.dr_type == 'udr':
            self._init_udr()
        
        elif self.dr_type == 'adr':
            self._init_adr()
        
        elif self.dr_type == 'simopt':
            self._init_simopt()
        
        elif self.dr_type == 'doraemon':
            self._init_doraemon()
        
        else:
            raise ValueError(f"Unknown domain randomization type: {self.dr_type}")
    
    def _init_udr(self):
        """Initialize Uniform Domain Randomization"""
        # Get ranges from dr_params or use defaults
        if 'ranges' in self.dr_params:
            self.udr_ranges = self.dr_params['ranges']
        else:
            # Default UDR ranges: ±50% of original mass (heavy configuration)
            self.udr_ranges = {
                'thigh': (0.5, 1.5),
                'leg': (0.5, 1.5),
                'foot': (0.5, 1.5),
            }
    
    def _init_adr(self):
        """Initialize Automatic Domain Randomization"""
        # TODO: Implement ADR initialization
        # ADR adapts ranges based on performance
        self.adr_ranges = {}
        self.adr_performance_buffer = []
        pass
    
    def _init_simopt(self):
        """Initialize Simulation Optimization"""
        # TODO: Implement SimOpt initialization
        # SimOpt optimizes simulation parameters
        self.simopt_params = {}
        pass
    
    def _init_doraemon(self):
        """Initialize Doraemon method"""
        # TODO: Implement Doraemon initialization
        self.doraemon_params = {}
        pass

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

        # Apply domain randomization if enabled
        if self.dr_type != 'none':
            self.set_random_parameters()

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }

    def set_random_parameters(self):
        """Set random masses for domain randomization"""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses according to the domain randomization distribution
        N.B. You can't change the mass of the torso (first link)
        
        Returns:
            masses: array of randomized masses [torso, thigh, leg, foot]
        """
        if self.dr_type == 'udr':
            return self._sample_udr()
        elif self.dr_type == 'adr':
            return self._sample_adr()
        elif self.dr_type == 'simopt':
            return self._sample_simopt()
        elif self.dr_type == 'doraemon':
            return self._sample_doraemon()
        else:
            # No randomization, return original masses
            return np.copy(self.original_masses)
    
    def _sample_udr(self):
        """Sample parameters using Uniform Domain Randomization"""
        # Start with original masses
        new_masses = np.copy(self.original_masses)
        
        # Don't randomize torso mass (index 0), keep it as is
        # Randomize thigh mass (index 1)
        thigh_multiplier = np.random.uniform(*self.udr_ranges['thigh'])
        new_masses[1] = self.original_masses[1] * thigh_multiplier
        
        # Randomize leg mass (index 2)
        leg_multiplier = np.random.uniform(*self.udr_ranges['leg'])
        new_masses[2] = self.original_masses[2] * leg_multiplier
        
        # Randomize foot mass (index 3)
        foot_multiplier = np.random.uniform(*self.udr_ranges['foot'])
        new_masses[3] = self.original_masses[3] * foot_multiplier
        
        return new_masses
    
    def _sample_adr(self):
        """Sample parameters using Automatic Domain Randomization"""
        # TODO: Implement ADR sampling logic
        # ADR adjusts ranges based on agent performance
        new_masses = np.copy(self.original_masses)
        # Placeholder: return original masses for now
        return new_masses
    
    def _sample_simopt(self):
        """Sample parameters using Simulation Optimization"""
        # TODO: Implement SimOpt sampling logic
        new_masses = np.copy(self.original_masses)
        # Placeholder: return original masses for now
        return new_masses
    
    def _sample_doraemon(self):
        """Sample parameters using Doraemon method"""
        # TODO: Implement Doraemon sampling logic
        new_masses = np.copy(self.original_masses)
        # Placeholder: return original masses for now
        return new_masses

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.model.body_mass[1:] )
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.model.body_mass[1:] = task


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

gym.register(
        id="CustomHopper-source-udr-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source", "dr_type": "udr"}
)

gym.register(
        id="CustomHopper-target-udr-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target", "dr_type": "udr"}
)

gym.register(
        id="CustomHopper-source-adr-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source", "dr_type": "adr"}
)
gym.register(
        id="CustomHopper-target-adr-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target", "dr_type": "adr"}
)

gym.register(
        id="CustomHopper-source-simopt-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source", "dr_type": "simopt"}
)
gym.register(
        id="CustomHopper-target-simopt-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target", "dr_type": "simopt"}
)

gym.register(
        id="CustomHopper-source-doraemon-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source", "dr_type": "doraemon"}
)
gym.register(
        id="CustomHopper-target-doraemon-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target", "dr_type": "doraemon"}
)


