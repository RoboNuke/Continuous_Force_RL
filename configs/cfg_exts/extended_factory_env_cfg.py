"""
Extended Factory Environment Configuration

This module defines ExtendedFactoryEnvCfg which extends Isaac Lab's FactoryEnvCfg
with our custom parameters and computed properties.
"""

from typing import Optional, Any
from .version_compat import get_isaac_lab_factory_imports
from .ctrl_cfg import ExtendedCtrlCfg

# Get Isaac Lab imports with version compatibility
configclass, FactoryEnvCfg = get_isaac_lab_factory_imports()


@configclass
class ExtendedFactoryEnvCfg(FactoryEnvCfg):
    """
    Extended factory environment configuration.

    Inherits from Isaac Lab's FactoryEnvCfg and adds our custom parameters
    and computed properties. Uses our ExtendedCtrlCfg by default.
    """

    def __post_init__(self):
        """Post-initialization to set up extended configurations."""
        self.sim.render_interval=self.decimation
        self.sim.physx.gpu_collision_stack_size = 2**29
        # Replace ctrl with our extended version if it's not already extended
        if not isinstance(self.ctrl, ExtendedCtrlCfg):
            # Preserve any existing ctrl settings while upgrading to extended version
            if hasattr(self, 'ctrl') and self.ctrl is not None:
                # Copy existing settings to extended config
                extended_ctrl = ExtendedCtrlCfg()
                for attr_name in dir(self.ctrl):
                    if not attr_name.startswith('_') and hasattr(extended_ctrl, attr_name):
                        setattr(extended_ctrl, attr_name, getattr(self.ctrl, attr_name))
                self.ctrl = extended_ctrl
            else:
                # Create new extended ctrl config
                self.ctrl = ExtendedCtrlCfg()

    def apply_primary_cfg(self, primary_cfg) -> None:
        """
        Apply primary configuration values to this environment config.

        Args:
            primary_cfg: PrimaryConfig instance containing shared parameters
        """
        # Apply primary config values that belong to environment
        self.decimation = primary_cfg.decimation

        # Update scene configuration for multi-agent setup
        if hasattr(self, 'scene') and self.scene is not None:
            self.scene.num_envs = primary_cfg.total_num_envs
            # Isaac Lab scene configs typically have replicate_physics
            if hasattr(self.scene, 'replicate_physics'):
                self.scene.replicate_physics = True

        # Store reference to primary config for computed properties
        self._primary_cfg = primary_cfg

    @property
    def num_agents(self) -> int:
        """Number of agents (computed from primary config if available)."""
        if hasattr(self, '_primary_cfg'):
            return self._primary_cfg.total_agents
        # Fallback to single agent if no primary config
        return 1

    @property
    def total_envs(self) -> int:
        """Total number of environments (computed from primary config if available)."""
        if hasattr(self, '_primary_cfg'):
            return self._primary_cfg.total_num_envs
        # Fallback to scene.num_envs if available
        if hasattr(self, 'scene') and hasattr(self.scene, 'num_envs'):
            return self.scene.num_envs
        return 1

    @property
    def sim_dt(self) -> float:
        """Simulation time step (computed from primary config if available)."""
        if hasattr(self, '_primary_cfg'):
            return self._primary_cfg.sim_dt
        # Fallback calculation using decimation and policy_hz
        policy_hz = getattr(self, 'policy_hz', 15)  # Default to 15 Hz
        return (1.0 / policy_hz) / self.decimation

    def get_rollout_steps(self) -> int:
        """
        Get rollout steps for this environment's episode length.

        Returns:
            Number of rollout steps based on episode_length_s and primary config timing
        """
        if not hasattr(self, '_primary_cfg'):
            raise ValueError("Primary config not applied. Call apply_primary_cfg() first.")

        return self._primary_cfg.rollout_steps(self.episode_length_s)

    def validate_configuration(self) -> None:
        """
        Validate that the configuration is complete and consistent.

        Raises:
            ValueError: If configuration is invalid or incomplete
        """
        # Check that essential attributes exist
        required_attrs = ['decimation', 'episode_length_s']
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise ValueError(f"Missing required attribute: {attr}")

        # Validate decimation
        if self.decimation <= 0:
            raise ValueError(f"decimation must be positive, got {self.decimation}")

        # Validate episode length
        if self.episode_length_s <= 0:
            raise ValueError(f"episode_length_s must be positive, got {self.episode_length_s}")

        # Validate scene configuration if present
        if hasattr(self, 'scene') and self.scene is not None:
            if hasattr(self.scene, 'num_envs') and self.scene.num_envs <= 0:
                raise ValueError(f"scene.num_envs must be positive, got {self.scene.num_envs}")

        # Validate ctrl configuration
        if not hasattr(self, 'ctrl') or self.ctrl is None:
            raise ValueError("ctrl configuration is required")

        # Validate that ctrl is our extended version
        if not isinstance(self.ctrl, ExtendedCtrlCfg):
            raise ValueError(f"ctrl must be ExtendedCtrlCfg instance, got {type(self.ctrl)}")

    def __repr__(self) -> str:
        """String representation for debugging."""
        env_info = f"decimation={self.decimation}, episode_length_s={self.episode_length_s}"
        if hasattr(self, '_primary_cfg'):
            env_info += f", total_agents={self.num_agents}, total_envs={self.total_envs}"
        return f"{self.__class__.__name__}({env_info})"