"""
Control Mode Configuration

This module defines the CtrlMode enum and helper functions for hybrid force-position control.
"""

from enum import Enum
from typing import Literal


class CtrlMode(str, Enum):
    """Control mode for hybrid force-position control."""
    FORCE_ONLY = "force_only"      # 3DOF: Fx, Fy, Fz
    FORCE_TZ = "force_tz"          # 4DOF: Fx, Fy, Fz, Tz
    FORCE_TORQUE = "force_torque"  # 6DOF: Fx, Fy, Fz, Tx, Ty, Tz


# Valid control mode strings for validation
VALID_CTRL_MODES = ["force_only", "force_tz", "force_torque"]


def get_force_size(ctrl_mode: str) -> int:
    """
    Return the force/torque dimension count for a given control mode.

    Args:
        ctrl_mode: One of 'force_only', 'force_tz', or 'force_torque'

    Returns:
        Number of force/torque dimensions (3, 4, or 6)

    Raises:
        ValueError: If ctrl_mode is not valid
    """
    if ctrl_mode == "force_only":
        return 3
    elif ctrl_mode == "force_tz":
        return 4
    elif ctrl_mode == "force_torque":
        return 6
    else:
        raise ValueError(
            f"Invalid ctrl_mode: {ctrl_mode}. Must be one of: {VALID_CTRL_MODES}"
        )


def get_action_space_size(ctrl_mode: str) -> int:
    """
    Return total action space size for a given control mode.

    Action space = force_size (selection) + 6 (pose) + force_size (force/torque)
                 = 2 * force_size + 6

    Args:
        ctrl_mode: One of 'force_only', 'force_tz', or 'force_torque'

    Returns:
        Total action space dimensions (12, 14, or 18)

    Raises:
        ValueError: If ctrl_mode is not valid
    """
    force_size = get_force_size(ctrl_mode)
    return 2 * force_size + 6


def validate_ctrl_mode(ctrl_mode: str) -> None:
    """
    Validate that ctrl_mode is a valid value.

    Args:
        ctrl_mode: The control mode to validate

    Raises:
        ValueError: If ctrl_mode is None or not a valid value
    """
    if ctrl_mode is None:
        raise ValueError(
            f"ctrl_mode is required. Must be one of: {VALID_CTRL_MODES}"
        )
    if ctrl_mode not in VALID_CTRL_MODES:
        raise ValueError(
            f"Invalid ctrl_mode: '{ctrl_mode}'. Must be one of: {VALID_CTRL_MODES}"
        )
