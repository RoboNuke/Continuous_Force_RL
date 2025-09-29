"""
Hybrid Control Configuration

This module defines configuration classes for hybrid force-position control
that inherit from Isaac Lab's control configuration structure.
"""

from dataclasses import dataclass
from typing import List, Literal

try:
    # Try Isaac Lab v2+ import (isaaclab.utils.configclass)
    from isaaclab.utils.configclass import configclass
except ImportError:
    try:
        # Try Isaac Lab v1.4.1 import (omni.isaac.lab.utils.configclass)
        from omni.isaac.lab.utils.configclass import configclass
    except ImportError:
        try:
            # Try Isaac Sim import (isaacsim.core.utils.configclass)
            from isaacsim.core.utils.configclass import configclass
        except ImportError:
            # Fallback to basic dataclass if Isaac Lab not available
            configclass = dataclass


@configclass
class HybridTaskCfg:
    """
    Configuration for hybrid control task-specific parameters.

    These parameters control the reward structure and behavior specific
    to hybrid force-position control tasks.
    """

    # Force activity thresholds
    force_active_threshold: float = 0.1
    """Minimum force magnitude to consider a direction as having active force"""

    torque_active_threshold: float = 0.01
    """Minimum torque magnitude to consider a direction as having active torque"""

    # Reward parameters
    good_force_cmd_rew: float = 0.1
    """Reward for using force control when force is active"""

    bad_force_cmd_rew: float = -0.1
    """Penalty for using force control when no force is active"""

    wrench_norm_scale: float = 0.01
    """Scale factor for wrench norm penalty (used in wrench_norm reward type)"""


# Default configuration instances
DEFAULT_HYBRID_TASK_CFG = HybridTaskCfg()