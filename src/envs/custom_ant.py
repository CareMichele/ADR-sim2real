"""Implementation of the Ant environment supporting
domain randomization optimization."""

from typing import Optional, Tuple

import numpy as np
import gymnasium as gym

# IMPORTANTE: Importiamo la classe specifica AntEnv da Gymnasium
try:
    from gymnasium.envs.mujoco.ant_v5 import AntEnv
except ImportError:
    from gymnasium.envs.mujoco.ant_v4 import AntEnv


class CustomAnt(AntEnv):
    """
    Custom Ant environment for Universal ADR.
    Inherits directly from Gymnasium's AntEnv to use the built-in XML.
    """

    def __init__(
        self,
        domain: Optional[str] = None,
        ctrl_cost_weight: float = 0.5,
        use_contact_forces: bool = False,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.2, 1.0),
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        # AntEnv signature differs between gymnasium versions (v4 vs v5).
        # Try v4-style args first, then fall back to v5-style args.
        try:
            super().__init__(
                ctrl_cost_weight=ctrl_cost_weight,
                use_contact_forces=use_contact_forces,
                contact_cost_weight=contact_cost_weight,
                healthy_reward=healthy_reward,
                terminate_when_unhealthy=terminate_when_unhealthy,
                healthy_z_range=healthy_z_range,
                contact_force_range=contact_force_range,
                reset_noise_scale=reset_noise_scale,
                exclude_current_positions_from_observation=exclude_current_positions_from_observation,
                **kwargs,
            )
        except TypeError:
            super().__init__(
                ctrl_cost_weight=ctrl_cost_weight,
                contact_cost_weight=contact_cost_weight,
                healthy_reward=healthy_reward,
                terminate_when_unhealthy=terminate_when_unhealthy,
                healthy_z_range=healthy_z_range,
                reset_noise_scale=reset_noise_scale,
                exclude_current_positions_from_observation=exclude_current_positions_from_observation,
                **kwargs,
            )

        self.domain = domain

        # Store original physical parameters
        self.original_masses = np.copy(self.model.body_mass[1:])
        self.original_friction = np.copy(self.model.geom_friction)
        self.original_damping = np.copy(self.model.dof_damping)
        self.original_gravity = np.copy(self.model.opt.gravity)

        # ADR perturbations
        self.current_max_push = 0.0
        self.push_probability = 0.1
        self.push_active = False
        self.original_colors = {}

        if domain == "target":
            self._apply_hell_mode()

    def _apply_hell_mode(self):
        print("[INFO] Initializing TARGET domain for Ant (HELL MODE)")

        # Mass unbalance
        if len(self.model.body_mass) > 9:
            self.model.body_mass[2] *= 1.5
            self.model.body_mass[3] *= 1.5
            self.model.body_mass[4] *= 1.5
            self.model.body_mass[5] *= 1.5

            self.model.body_mass[6] *= 0.5
            self.model.body_mass[7] *= 0.5
            self.model.body_mass[8] *= 0.5
            self.model.body_mass[9] *= 0.5
            print("  - Masses unbalanced: front legs +50%, back legs -50%")

        # Slippery floor
        self.model.geom_friction[:, 0] *= 0.5
        print("  - Friction reduced to 0.5x (slippery)")

        # Heavier gravity
        self.model.opt.gravity[2] = -12.0
        print(f"  - Gravity: {self.model.opt.gravity[2]} m/s²")

        # Random pushes
        self.current_max_push = 1.0
        self.push_probability = 0.05
        print(
            f"  - Random pushes enabled: max={self.current_max_push}N, prob={self.push_probability*100:.1f}%"
        )
        print("[INFO] Target domain initialized (HELL MODE)")

    def step(self, action):
        if self.current_max_push > 0 and np.random.rand() < self.push_probability:
            force_direction = np.random.choice(["x", "y"])
            force_magnitude = np.random.uniform(-self.current_max_push, self.current_max_push)
            torso_id = 1  # body 0 = world, body 1 = torso
            if force_direction == "x":
                self.data.xfrc_applied[torso_id, 0] = force_magnitude
            else:
                self.data.xfrc_applied[torso_id, 1] = force_magnitude
            self.push_active = True
            self._set_robot_color([1.0, 0.0, 0.0, 1.0])
        else:
            self.data.xfrc_applied[1, :] = 0.0
            if self.push_active:
                self._restore_robot_color()
            self.push_active = False

        return super().step(action)

    def get_parameters(self):
        return np.copy(self.model.body_mass[1:])

    def set_parameters(self, parameters):
        if isinstance(parameters, (np.ndarray, list)):
            self.model.body_mass[1:] = parameters
            return

        if not isinstance(parameters, dict):
            return

        mass_mapping = {
            "hip_1": 2,
            "ankle_1": 3,
            "hip_2": 4,
            "ankle_2": 5,
            "hip_3": 6,
            "ankle_3": 7,
            "hip_4": 8,
            "ankle_4": 9,
        }

        for param_name, body_idx in mass_mapping.items():
            if param_name in parameters and body_idx < len(self.original_masses) + 1:
                self.model.body_mass[body_idx] = self.original_masses[body_idx - 1] * parameters[param_name]

        if "friction" in parameters:
            self.model.geom_friction[:, 0] = self.original_friction[:, 0] * parameters["friction"]

        if "damping" in parameters:
            for dof_id in range(6, self.model.nv):
                self.model.dof_damping[dof_id] = self.original_damping[dof_id] * parameters["damping"]

        if "gravity" in parameters:
            self.model.opt.gravity[2] = parameters["gravity"]

        if "force_magnitude" in parameters:
            self.current_max_push = parameters["force_magnitude"]

    def _set_robot_color(self, rgba):
        """Cambia il colore del robot (per visualizzare perturbazioni)"""
        if not self.original_colors:
            for geom_id in range(self.model.ngeom):
                self.original_colors[geom_id] = np.copy(self.model.geom_rgba[geom_id])

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
