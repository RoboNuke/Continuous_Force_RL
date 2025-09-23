"""
Extended Control Configuration

This module defines ExtendedCtrlCfg which inherits from Isaac Lab's CtrlCfg
and adds our custom force control parameters for hybrid control.
"""

from typing import List
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

    # Force control parameters (our extensions)
    force_action_bounds: List[float] = None
    """Force action bounds [fx, fy, fz] in Newtons"""

    torque_action_bounds: List[float] = None
    """Torque action bounds [tx, ty, tz] in Newton-meters"""

    force_action_threshold: List[float] = None
    """Force action threshold for goal calculation [fx, fy, fz]"""

    torque_action_threshold: List[float] = None
    """Torque action threshold for goal calculation [tx, ty, tz]"""

    default_task_force_gains: List[float] = None
    """Force task gains [fx, fy, fz, tx, ty, tz] for hybrid control"""

    def __post_init__(self):
        """Post-initialization to set defaults for our custom parameters."""
        # Call parent post_init if it exists
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

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