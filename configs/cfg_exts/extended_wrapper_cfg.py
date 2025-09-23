"""
Extended Wrapper Configuration

This module defines ExtendedWrapperConfig which contains configuration
for all environment wrappers. This is our own organizational structure
for wrapper configuration.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .version_compat import get_isaac_lab_ctrl_imports

# Get configclass decorator with version compatibility
configclass, _ = get_isaac_lab_ctrl_imports()


@configclass
@dataclass
class ExtendedWrapperConfig:
    """
    Extended wrapper configuration for environment wrappers.

    Contains enable/disable flags and parameters for all wrapper types
    used in the factory environment.
    """

    # Fragile objects wrapper
    fragile_objects_enabled: bool = True
    """Enable fragile objects wrapper"""

    # Efficient reset wrapper
    efficient_reset_enabled: bool = True
    """Enable efficient environment resetting"""

    # Force-torque sensor wrapper
    force_torque_sensor_enabled: bool = False
    """Enable force-torque sensor wrapper"""

    force_torque_use_tanh_scaling: bool = False
    """Use tanh scaling for force-torque values"""

    force_torque_tanh_scale: float = 0.03
    """Tanh scaling factor for force-torque sensor"""

    # Observation manager wrapper
    observation_manager_enabled: bool = True
    """Enable observation format conversion wrapper"""

    observation_manager_merge_strategy: str = "concatenate"
    """Strategy for merging observations ('concatenate', etc.)"""

    # Observation noise wrapper
    observation_noise_enabled: bool = False
    """Enable domain randomization noise"""

    observation_noise_global_scale: float = 1.0
    """Global scaling factor for all observation noise"""

    observation_noise_apply_to_critic: bool = True
    """Apply observation noise to critic observations"""

    observation_noise_seed: Optional[int] = None
    """Seed for observation noise (None for random)"""

    # Hybrid control wrapper
    hybrid_control_enabled: bool = False
    """Enable hybrid force-position control wrapper"""

    hybrid_control_reward_type: str = "simp"
    """Reward type for hybrid control ('simp', 'complex', etc.)"""

    # Factory metrics wrapper
    factory_metrics_enabled: bool = True
    """Enable factory-specific metrics tracking"""

    # Wandb logging wrapper
    wandb_logging_enabled: bool = True
    """Enable Wandb logging wrapper"""

    wandb_project: str = "Continuous_Force_RL"
    """Wandb project name"""

    wandb_entity: str = "hur"
    """Wandb entity name"""

    wandb_name: Optional[str] = None
    """Wandb run name (None for auto-generated)"""

    wandb_group: Optional[str] = None
    """Wandb group name"""

    wandb_tags: Optional[List[str]] = None
    """Wandb tags"""

    # Action logging wrapper
    action_logging_enabled: bool = False
    """Enable enhanced action logging"""

    action_logging_track_selection: bool = True
    """Track action selection in logging"""

    action_logging_track_pos: bool = True
    """Track position actions in logging"""

    action_logging_track_rot: bool = True
    """Track rotation actions in logging"""

    action_logging_track_force: bool = True
    """Track force actions in logging"""

    action_logging_track_torque: bool = True
    """Track torque actions in logging"""

    action_logging_force_size: int = 6
    """Size of force vector for logging"""

    action_logging_frequency: int = 100
    """Logging frequency for actions"""

    action_logging_track_histograms: bool = True
    """Track action histograms"""

    action_logging_track_observation_histograms: bool = False
    """Track observation histograms"""

    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_wrapper_params()
        self._setup_defaults()

    def _validate_wrapper_params(self):
        """Validate wrapper configuration parameters."""
        # Validate scaling factors
        if self.force_torque_tanh_scale <= 0:
            raise ValueError(f"force_torque_tanh_scale must be positive, got {self.force_torque_tanh_scale}")

        if self.observation_noise_global_scale < 0:
            raise ValueError(f"observation_noise_global_scale must be non-negative, got {self.observation_noise_global_scale}")

        # Validate string parameters
        valid_merge_strategies = ["concatenate"]
        if self.observation_manager_merge_strategy not in valid_merge_strategies:
            raise ValueError(f"observation_manager_merge_strategy must be one of {valid_merge_strategies}, got {self.observation_manager_merge_strategy}")

        valid_reward_types = ["simp", "complex"]
        if self.hybrid_control_reward_type not in valid_reward_types:
            raise ValueError(f"hybrid_control_reward_type must be one of {valid_reward_types}, got {self.hybrid_control_reward_type}")

        # Validate logging parameters
        if self.action_logging_frequency <= 0:
            raise ValueError(f"action_logging_frequency must be positive, got {self.action_logging_frequency}")

        if self.action_logging_force_size <= 0:
            raise ValueError(f"action_logging_force_size must be positive, got {self.action_logging_force_size}")

    def _setup_defaults(self):
        """Set up default values for optional parameters."""
        if self.wandb_tags is None:
            self.wandb_tags = []

    def apply_primary_cfg(self, primary_cfg) -> None:
        """
        Apply primary configuration values to wrapper config.

        Args:
            primary_cfg: PrimaryConfig instance containing shared parameters
        """
        # Store reference to primary config
        self._primary_cfg = primary_cfg

        # Apply primary config values that affect wrappers
        # (Most wrapper configs are independent, but some may use primary values)

    def get_fragile_objects_config(self) -> Dict[str, Any]:
        """Get fragile objects wrapper configuration."""
        if not hasattr(self, '_primary_cfg'):
            raise ValueError("Primary config not applied. Call apply_primary_cfg() first.")

        return {
            'enabled': self.fragile_objects_enabled,
            'break_force': self._primary_cfg.break_forces,
            'num_agents': self._primary_cfg.total_agents
        }

    def get_force_torque_sensor_config(self) -> Dict[str, Any]:
        """Get force-torque sensor wrapper configuration."""
        return {
            'enabled': self.force_torque_sensor_enabled,
            'use_tanh_scaling': self.force_torque_use_tanh_scaling,
            'tanh_scale': self.force_torque_tanh_scale
        }

    def get_observation_noise_config(self) -> Dict[str, Any]:
        """Get observation noise wrapper configuration."""
        return {
            'enabled': self.observation_noise_enabled,
            'global_noise_scale': self.observation_noise_global_scale,
            'apply_to_critic': self.observation_noise_apply_to_critic,
            'seed': self.observation_noise_seed
        }

    def get_hybrid_control_config(self) -> Dict[str, Any]:
        """Get hybrid control wrapper configuration."""
        if not hasattr(self, '_primary_cfg'):
            raise ValueError("Primary config not applied. Call apply_primary_cfg() first.")

        return {
            'enabled': self.hybrid_control_enabled,
            'ctrl_torque': self._primary_cfg.ctrl_torque,
            'reward_type': self.hybrid_control_reward_type
        }

    def get_wandb_config(self) -> Dict[str, Any]:
        """Get Wandb logging wrapper configuration."""
        return {
            'enabled': self.wandb_logging_enabled,
            'wandb_project': self.wandb_project,
            'wandb_entity': self.wandb_entity,
            'wandb_name': self.wandb_name,
            'wandb_group': self.wandb_group,
            'wandb_tags': self.wandb_tags
        }

    def get_action_logging_config(self) -> Dict[str, Any]:
        """Get action logging wrapper configuration."""
        return {
            'enabled': self.action_logging_enabled,
            'track_selection': self.action_logging_track_selection,
            'track_pos': self.action_logging_track_pos,
            'track_rot': self.action_logging_track_rot,
            'track_force': self.action_logging_track_force,
            'track_torque': self.action_logging_track_torque,
            'force_size': self.action_logging_force_size,
            'logging_frequency': self.action_logging_frequency,
            'track_action_histograms': self.action_logging_track_histograms,
            'track_observation_histograms': self.action_logging_track_observation_histograms
        }

    def get_factory_metrics_config(self) -> Dict[str, Any]:
        """Get factory metrics wrapper configuration."""
        if not hasattr(self, '_primary_cfg'):
            raise ValueError("Primary config not applied. Call apply_primary_cfg() first.")

        return {
            'enabled': self.factory_metrics_enabled,
            'num_agents': self._primary_cfg.total_agents
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'fragile_objects': self.get_fragile_objects_config() if hasattr(self, '_primary_cfg') else {'enabled': self.fragile_objects_enabled},
            'efficient_reset': {'enabled': self.efficient_reset_enabled},
            'force_torque_sensor': self.get_force_torque_sensor_config(),
            'observation_manager': {
                'enabled': self.observation_manager_enabled,
                'merge_strategy': self.observation_manager_merge_strategy
            },
            'observation_noise': self.get_observation_noise_config(),
            'hybrid_control': self.get_hybrid_control_config() if hasattr(self, '_primary_cfg') else {'enabled': self.hybrid_control_enabled},
            'factory_metrics': self.get_factory_metrics_config() if hasattr(self, '_primary_cfg') else {'enabled': self.factory_metrics_enabled},
            'wandb_logging': self.get_wandb_config(),
            'action_logging': self.get_action_logging_config()
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        enabled_wrappers = []
        if self.fragile_objects_enabled:
            enabled_wrappers.append("fragile_objects")
        if self.force_torque_sensor_enabled:
            enabled_wrappers.append("force_torque")
        if self.hybrid_control_enabled:
            enabled_wrappers.append("hybrid_control")
        if self.observation_noise_enabled:
            enabled_wrappers.append("obs_noise")
        if self.action_logging_enabled:
            enabled_wrappers.append("action_logging")

        return f"ExtendedWrapperConfig(enabled: {', '.join(enabled_wrappers) if enabled_wrappers else 'none'})"