"""
Extended Control Configuration

This module defines ExtendedCtrlCfg which inherits from Isaac Lab's CtrlCfg
and adds our custom force control parameters for hybrid control.
"""

from typing import List, Literal
from .version_compat import get_isaac_lab_ctrl_imports

# Get Isaac Lab imports with version compatibility
configclass, CtrlCfg = get_isaac_lab_ctrl_imports()


@configclass
class ExtendedCtrlCfg(CtrlCfg):
    """
    Extended control configuration with hybrid force-position parameters.

    This class inherits from Isaac Lab's CtrlCfg and adds our custom parameters
    for hybrid force-position control. All attributes are properly defined as
    class members, ensuring correct serialization with wandb and other systems.
    """
    # EMA (Exponential Moving Average) parameters
    ema_factor: float = 0.2
    """Factor for EMA filtering of targets. target = ema_factor * goal + (1-ema_factor) * prev_target"""

    no_sel_ema: bool = True
    """If True, selection matrix is not filtered with EMA (set directly from action)"""

    # Target initialization strategy
    target_init_mode: Literal["zero", "first_goal"] = "zero"
    """How to initialize targets: 'zero' for zeros, 'first_goal' for first goal after reset"""

    # Force control parameters (our extensions)
    force_action_bounds: List[float] = None
    """Force action bounds [fx, fy, fz] in Newtons"""

    torque_action_bounds: List[float] = None
    """Torque action bounds [tx, ty, tz] in Newton-meters"""

    force_action_threshold: List[float] = None
    """Force action threshold for goal calculation [fx, fy, fz] (used in delta mode)"""

    torque_action_threshold: List[float] = None
    """Torque action threshold for goal calculation [tx, ty, tz] (used in delta mode)"""

    default_task_force_gains: List[float] = None
    """Force task gains [fx, fy, fz, tx, ty, tz] for hybrid control"""

    use_delta_force: bool = False
    """If True, force target = action * threshold + current_force (delta mode).
    If False, force target = action * bounds (absolute mode). Default: False (absolute)."""
    
    async_z_force_bounds: bool = False
    """If True, target force will be applied only on the range [0,-force_action_bounds], 
    preventing positive z-force commands"""

    apply_ema_force: bool = True
    """When true ema is applied to pose and force commands, when false only to pose"""

    ema_mode: Literal["action", "wrench"] = "action"
    """EMA mode: 'action' applies EMA to actions before control targets,
    'wrench' applies EMA to final task wrench after combining pose/force wrenches"""

    # Force PID control flags
    enable_force_derivative: bool = False
    """Enable derivative (D) term in force control. D gains derived from Kp (2*sqrt(Kp))."""

    enable_force_integral: bool = False
    """Enable integral (I) term in force control."""

    # Integral-specific parameters
    default_task_force_integ_gains: List[float] = None
    """Force integral gains [fx_ki, fy_ki, fz_ki, tx_ki, ty_ki, tz_ki]. Required if enable_force_integral=True."""

    force_integral_clamp: float = 50.0
    """Anti-windup clamp for integral term (symmetric bounds)."""

    def __post_init__(self):
        """Post-initialization to set defaults for our custom parameters."""
        # Call parent post_init if it exists
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
        
        if self.ema_factor < 0.0 or self.ema_factor > 1.0:
            raise ValueError(f"ema_factor must be between 0 and 1, got {self.ema_factor}")

        if self.target_init_mode not in ["zero", "first_goal"]:
            raise ValueError(f"target_init_mode must be 'zero' or 'first_goal', got {self.target_init_mode}")
        # Set defaults for our custom parameters if not provided
        if self.force_action_bounds is None:
            self.force_action_bounds = [50.0, 50.0, 50.0]

        if self.torque_action_bounds is None:
            self.torque_action_bounds = [0.5, 0.5, 0.5]

        if self.force_action_threshold is None:
            self.force_action_threshold = [10.0, 10.0, 10.0]

        if self.torque_action_threshold is None:
            self.torque_action_threshold = [0.1, 0.1, 0.1]

        if self.default_task_force_gains is None:
            self.default_task_force_gains = [0.1, 0.1, 0.1, 0.001, 0.001, 0.001]

        # Validate integral config
        if self.enable_force_integral and self.default_task_force_integ_gains is None:
            raise ValueError(
                "enable_force_integral=True but default_task_force_integ_gains not set. "
                "Please provide integral gains, e.g., [0.01, 0.01, 0.01, 0.001, 0.001, 0.001]"
            )