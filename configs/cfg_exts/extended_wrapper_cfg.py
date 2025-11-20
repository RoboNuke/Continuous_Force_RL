"""
Extended Wrapper Configuration

This module defines ExtendedWrapperConfig which contains configuration
for all environment wrappers. This is our own organizational structure
for wrapper configuration.
"""

from dataclasses import dataclass, field

from .version_compat import get_isaac_lab_ctrl_imports
from .wrapper_sub_configs import (
    ForceTorqueSensorConfig, HybridControlConfig, ObservationNoiseConfig,
    EEPoseNoiseConfig, WandbLoggingConfig, ActionLoggingConfig, ForceRewardConfig,
    FragileObjectConfig, ObsManagerConfig, FactoryMetricsConfig,
    PoseContactLoggingConfig, MisalignmentPenaltyConfig,
    TwoStageKeypointRewardConfig, CurriculumConfig, ManualControlConfig,
    DynamicsRandomizationConfig, EfficientResetConfig
)

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

    # Nested wrapper configurations
    force_torque_sensor: ForceTorqueSensorConfig = field(default_factory=ForceTorqueSensorConfig)
    """Force-torque sensor wrapper configuration"""

    hybrid_control: HybridControlConfig = field(default_factory=HybridControlConfig)
    """Hybrid control wrapper configuration"""

    observation_noise: ObservationNoiseConfig = field(default_factory=ObservationNoiseConfig)
    """Observation noise wrapper configuration"""

    ee_pose_noise: EEPoseNoiseConfig = field(default_factory=EEPoseNoiseConfig)
    """End-effector pose noise wrapper configuration"""

    wandb_logging: WandbLoggingConfig = field(default_factory=WandbLoggingConfig)
    """Wandb logging wrapper configuration"""

    action_logging: ActionLoggingConfig = field(default_factory=ActionLoggingConfig)
    """Action logging wrapper configuration"""

    force_reward: ForceRewardConfig = field(default_factory=ForceRewardConfig)
    """Force reward wrapper configuration"""

    # Simple wrapper configurations (flat)
    fragile_objects: FragileObjectConfig = field(default_factory=FragileObjectConfig)
    """Enable fragile objects wrapper"""

    observation_manager: ObsManagerConfig = field(default_factory=ObsManagerConfig)

    factory_metrics: FactoryMetricsConfig = field(default_factory=FactoryMetricsConfig)
    """Enable factory-specific metrics tracking"""

    pose_contact_logging: PoseContactLoggingConfig = field(default_factory=PoseContactLoggingConfig)
    """Pose contact logging wrapper configuration"""

    misalignment_penalty: MisalignmentPenaltyConfig = field(default_factory=MisalignmentPenaltyConfig)
    """Misalignment penalty wrapper configuration"""

    two_stage_keypoint_reward: TwoStageKeypointRewardConfig = field(default_factory=TwoStageKeypointRewardConfig)
    """Two-stage keypoint reward wrapper configuration"""

    spawn_height_curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    """Spawn height curriculum wrapper configuration"""

    manual_control: ManualControlConfig = field(default_factory=ManualControlConfig)
    """Manual control wrapper configuration"""

    dynamics_randomization: DynamicsRandomizationConfig = field(default_factory=DynamicsRandomizationConfig)
    """Dynamics randomization wrapper configuration"""

    efficient_reset: EfficientResetConfig = field(default_factory=EfficientResetConfig)
    """Efficient reset wrapper configuration"""

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