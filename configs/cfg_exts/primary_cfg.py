"""
Primary Configuration

This module defines PrimaryConfig which contains shared parameters used across
multiple configuration objects. These are parameters that don't belong to a specific
Isaac Lab or SKRL configuration class but are used for computing derived values.
"""

from typing import List, Union, Optional
from dataclasses import dataclass

from .version_compat import get_isaac_lab_ctrl_imports

# Get configclass decorator with version compatibility
configclass, _ = get_isaac_lab_ctrl_imports()


@configclass
@dataclass
class PrimaryConfig:
    """
    Primary configuration containing shared parameters.

    These parameters are used across multiple configuration objects and for
    computing derived values like total_agents, total_num_envs, rollout_steps.
    Parameters that belong to specific Isaac Lab or SKRL configs should not
    be duplicated here - reference the single source of truth instead.
    """

    # Multi-agent setup parameters
    agents_per_break_force: int = 2
    """Number of agents per break force condition"""

    num_envs_per_agent: int = 256
    """Environments per individual agent"""

    break_forces: Union[int, List[int]] = -1
    """Break force conditions (-1 = unbreakable)"""

    # Simulation timing parameters
    decimation: int = 8
    """Physics decimation factor"""

    policy_hz: int = 15
    """Policy frequency in Hz"""

    # Training parameters
    max_steps: int = 10240000
    """Maximum training steps"""

    debug_mode: bool = False
    """Enable debug mode with simplified settings"""

    seed: int = -1
    """Random seed for reproducibility (-1 for random)"""

    # Experiment tracking
    ckpt_tracker_path: str = "/nfs/stak/users/brownhun/ckpt_tracker2.txt"
    """Path to checkpoint tracker file"""

    # Control mode
    ctrl_torque: bool = False
    """Use torque control instead of position control"""

    def __post_init__(self):
        """Post-initialization validation with clear error messages."""
        self._validate_break_forces()
        self._validate_positive_integers()
        self._validate_timing_parameters()

    def _validate_break_forces(self):
        """Validate break forces parameter."""
        if isinstance(self.break_forces, list):
            if len(self.break_forces) == 0:
                raise ValueError("break_forces list cannot be empty")
            for force in self.break_forces:
                if not isinstance(force, (int, float)):
                    raise ValueError(f"All break forces must be numeric, got {type(force)}")
        elif not isinstance(self.break_forces, (int, float)):
            raise ValueError(f"break_forces must be a number or list of numbers, got {type(self.break_forces)}")

    def _validate_positive_integers(self):
        """Validate that certain parameters are positive integers."""
        positive_int_params = {
            'agents_per_break_force': self.agents_per_break_force,
            'num_envs_per_agent': self.num_envs_per_agent,
            'decimation': self.decimation,
            'policy_hz': self.policy_hz,
            'max_steps': self.max_steps
        }

        for param_name, value in positive_int_params.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{param_name} must be a positive integer, got {value}")

    def _validate_timing_parameters(self):
        """Validate timing-related parameters."""
        if self.policy_hz > 1000:
            raise ValueError(f"policy_hz seems unreasonably high: {self.policy_hz} Hz")
        if self.decimation > 100:
            raise ValueError(f"decimation seems unreasonably high: {self.decimation}")

    @property
    def total_agents(self) -> int:
        """Total number of agents across all break force conditions."""
        num_break_forces = len(self.break_forces) if isinstance(self.break_forces, list) else 1
        return num_break_forces * self.agents_per_break_force

    @property
    def total_num_envs(self) -> int:
        """Total number of environments across all agents."""
        return self.total_agents * self.num_envs_per_agent

    @property
    def sim_dt(self) -> float:
        """Simulation time step in seconds."""
        return (1.0 / self.policy_hz) / self.decimation

    def rollout_steps(self, episode_length_s: float) -> int:
        """
        Calculate rollout steps based on episode length.

        Args:
            episode_length_s: Episode length in seconds (from Isaac Lab task config)

        Returns:
            Number of rollout steps for this episode length
        """
        if episode_length_s <= 0:
            raise ValueError(f"episode_length_s must be positive, got {episode_length_s}")

        return int((1.0 / self.sim_dt) / self.decimation * episode_length_s)

    def get_num_break_forces(self) -> int:
        """Get the number of different break force conditions."""
        return len(self.break_forces) if isinstance(self.break_forces, list) else 1

    def is_multi_agent(self) -> bool:
        """Check if this is a multi-agent configuration."""
        return self.total_agents > 1

    def is_multi_break_force(self) -> bool:
        """Check if multiple break force conditions are configured."""
        return self.get_num_break_forces() > 1