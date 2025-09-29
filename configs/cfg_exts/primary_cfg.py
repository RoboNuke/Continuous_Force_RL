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
        return int((1.0 / self.sim_dt) / self.decimation * episode_length_s)

