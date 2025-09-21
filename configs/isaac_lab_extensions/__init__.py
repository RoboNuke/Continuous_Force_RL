"""
Isaac Lab Configuration Extensions

This module provides extended configuration classes that inherit from Isaac Lab's
base configurations and add our custom parameters for hybrid force-position control.
"""

from .ctrl_cfg import ExtendedCtrlCfg

__all__ = ["ExtendedCtrlCfg"]