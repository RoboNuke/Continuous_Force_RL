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
class HybridCtrlCfg:
    """
    Configuration for hybrid force-position control parameters.

    This extends Isaac Lab's control configuration with parameters specific
    to hybrid force-position control including EMA filtering and target initialization.
    """

    # EMA (Exponential Moving Average) parameters
    ema_factor: float = 0.2
    """Factor for EMA filtering of targets. target = ema_factor * goal + (1-ema_factor) * prev_target"""

    no_sel_ema: bool = True
    """If True, selection matrix is not filtered with EMA (set directly from action)"""

    # Target initialization strategy
    target_init_mode: Literal["zero", "first_goal"] = "zero"
    """How to initialize targets: 'zero' for zeros, 'first_goal' for first goal after reset"""

    # Force control parameters
    default_task_force_gains: List[float] = None
    """Force task gains [fx, fy, fz, tx, ty, tz]. If None, uses environment defaults"""

    force_action_bounds: List[float] = None
    """Force action bounds [fx, fy, fz]. If None, uses environment defaults"""

    torque_action_bounds: List[float] = None
    """Torque action bounds [tx, ty, tz]. If None, uses environment defaults"""

    force_action_threshold: List[float] = None
    """Force action threshold for goal calculation. If None, uses environment defaults"""

    torque_action_threshold: List[float] = None
    """Torque action threshold for goal calculation. If None, uses environment defaults"""

    # Position control parameters
    pos_action_bounds: List[float] = None
    """Position action bounds [x, y, z]. If None, uses environment defaults"""

    rot_action_bounds: List[float] = None
    """Rotation action bounds [rx, ry, rz]. If None, uses environment defaults"""

    def __post_init__(self):
        """Post-initialization validation and default setting."""
        if self.ema_factor < 0.0 or self.ema_factor > 1.0:
            raise ValueError(f"ema_factor must be between 0 and 1, got {self.ema_factor}")

        if self.target_init_mode not in ["zero", "first_goal"]:
            raise ValueError(f"target_init_mode must be 'zero' or 'first_goal', got {self.target_init_mode}")


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
DEFAULT_HYBRID_CTRL_CFG = HybridCtrlCfg()
DEFAULT_HYBRID_TASK_CFG = HybridTaskCfg()