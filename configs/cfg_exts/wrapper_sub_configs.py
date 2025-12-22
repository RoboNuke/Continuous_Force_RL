from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FactoryMetricsConfig:
    enabled: bool = False
    publish_to_wandb: bool = True
    engagement_reward_scale: float = 1.0
    """Scale factor for engagement rewards. Default 1.0 preserves original behavior."""
    success_reward_scale: float = 1.0
    """Scale factor for success rewards. Default 1.0 preserves original behavior."""

@dataclass
class ObsManagerConfig:
    enabled: bool = False
    observation_manager_merge_strategy: str = "concatenate"
    
@dataclass
class FragileObjectConfig:
    enabled: bool = False
    peg_break_rew: float = -10.0


@dataclass
class ForceTorqueSensorConfig:
    """Force-torque sensor wrapper configuration."""
    enabled: bool = False
    use_tanh_scaling: bool = False
    tanh_scale: float = 0.03
    add_force_obs: bool = False
    add_contact_obs: bool = False
    add_contact_state: bool = True

    # Contact detection parameters (moved from HybridTaskCfg)
    contact_force_threshold: float = 0.1
    contact_torque_threshold: float = 0.01
    log_contact_state: bool = True
    use_contact_sensor: bool = True  # True = use ContactSensor, False = use force-torque thresholds


@dataclass
class HybridControlConfig:
    """Hybrid control wrapper configuration."""
    enabled: bool = False
    reward_type: str = "simp"
    use_ground_truth_selection: bool = False


@dataclass
class ObservationNoiseConfig:
    """Observation noise wrapper configuration."""
    enabled: bool = False
    global_scale: float = 1.0
    apply_to_critic: bool = True
    seed: Optional[int] = None


@dataclass
class EEPoseNoiseConfig:
    """End-effector pose noise wrapper configuration."""
    enabled: bool = False


@dataclass
class WandbLoggingConfig:
    """Wandb logging wrapper configuration."""
    enabled: bool = True
    disable_logging: bool = False  # If True, keeps wrapper interface but skips actual wandb logging
    wandb_project: str = "Continuous_Force_RL"
    wandb_entity: str = "hur"
    wandb_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    track_rewards_by_outcome: bool = False  # If True, split component rewards by success/failure


@dataclass
class ActionLoggingConfig:
    """Action logging wrapper configuration."""
    enabled: bool = False
    track_selection: bool = True
    track_pos: bool = True
    track_rot: bool = True
    track_force: bool = True
    track_torque: bool = True
    force_size: int = 6
    logging_frequency: int = 100


@dataclass
class ForceRewardConfig:
    """Force reward wrapper configuration with 8 reward functions."""
    # Global wrapper settings
    enabled: bool = False
    contact_force_threshold: float = 1.0
    contact_window_size: int = 10

    # Force Magnitude Reward
    enable_force_magnitude_reward: bool = False
    force_magnitude_reward_weight: float = 1.0
    force_magnitude_base_force: float = 5.0
    force_magnitude_keep_sign: bool = True

    # Alignment Award (contact only)
    enable_alignment_award: bool = False
    alignment_award_reward_weight: float = 1.0
    alignment_goal_orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])

    # Force Action Error
    enable_force_action_error: bool = False
    force_action_error_reward_weight: float = 1.0

    # Contact Consistency (contact only)
    enable_contact_consistency: bool = False
    contact_consistency_reward_weight: float = 1.0
    contact_consistency_beta: float = 1.0
    contact_consistency_use_ema: bool = True
    contact_consistency_ema_alpha: float = 0.1

    # Oscillation Penalty
    enable_oscillation_penalty: bool = False
    oscillation_penalty_reward_weight: float = 1.0
    oscillation_penalty_window_size: int = 5

    # Contact Transition Reward
    enable_contact_transition_reward: bool = False
    contact_transition_reward_weight: float = 1.0

    # Efficiency
    enable_efficiency: bool = False
    efficiency_reward_weight: float = 1.0

    # Force Ratio (contact only)
    enable_force_ratio: bool = False
    force_ratio_reward_weight: float = 1.0

    enable_contact_reward: bool = False
    contact_reward_weight: float = 1.0

    # Square Velocity Reward
    enable_square_vel: bool = False
    square_vel_weight: float = 1.0




@dataclass
class PoseContactLoggingConfig:
    """Pose contact logging wrapper configuration."""
    enabled: bool = False


@dataclass
class MisalignmentPenaltyConfig:
    """Misalignment penalty wrapper configuration."""
    enabled: bool = False
    xy_threshold: float = 0.0025  # XY distance threshold for alignment (default from factory)
    height_threshold_fraction: float = 0.9  # Fraction of fixed asset height (default engage_threshold)


@dataclass
class TwoStageKeypointRewardConfig:
    """Two-stage keypoint reward wrapper configuration."""
    enabled: bool = False
    xy_threshold: float = 0.0025  # XY distance threshold for stage transition (same as is_centered)


@dataclass
class CurriculumConfig:
    """Spawn height curriculum wrapper configuration for bidirectional difficulty adjustment."""
    enabled: bool = False

    # Progression thresholds
    progress_threshold: float = 0.80        # Increase min_height when success rate >= this
    regress_threshold: float = 0.50         # Decrease min_height when success rate < this

    # Height deltas (in meters)
    progress_height_delta: float = 0.01     # Increase min by 1cm when succeeding
    regression_height_delta: float = 0.02   # Decrease min by 2cm when struggling

    # Bounds
    min_height: float = 0.0                 # Starting/minimum height (never go below this)

    # Evaluation
    min_episodes_for_evaluation: int = 10   # Minimum episodes before adjusting height


@dataclass
class ManualControlConfig:
    """Manual control wrapper configuration for keyboard-based robot control."""
    enabled: bool = False
    checkpoint_path: Optional[str] = None   # Path to RL checkpoint for toggle mode
    action_scale: float = 0.1               # Scale for incremental position/rotation actions
    force_scale: float = 0.1                # Scale for force commands (hybrid control only)
    start_in_manual_mode: bool = True       # Start in manual mode (True) or RL mode (False)


@dataclass
class DynamicsRandomizationConfig:
    """Dynamics randomization wrapper configuration for domain randomization."""
    enabled: bool = False

    # Friction randomization (held asset only)
    randomize_friction: bool = False
    friction_range: List[float] = field(default_factory=lambda: [0.5, 1.0])

    # Controller gains randomization (separate for pos/rot, same across all dims within type)
    randomize_gains: bool = False
    pos_gains_range: List[float] = field(default_factory=lambda: [400.0, 800.0])   # Absolute range for position gains
    rot_gains_range: List[float] = field(default_factory=lambda: [20.0, 40.0])    # Absolute range for rotation gains

    # Force/torque gains randomization (hybrid control only)
    randomize_force_gains: bool = False
    force_gains_range: List[float] = field(default_factory=lambda: [0.08, 0.12])     # Absolute range for force gains
    torque_gains_range: List[float] = field(default_factory=lambda: [0.0008, 0.0012]) # Absolute range for torque gains

    # Action thresholds randomization (sampled once per env, replicated to all dims)
    randomize_pos_threshold: bool = False
    pos_threshold_range: List[float] = field(default_factory=lambda: [0.016, 0.025])

    randomize_rot_threshold: bool = False
    rot_threshold_range: List[float] = field(default_factory=lambda: [0.08, 0.12])

    randomize_force_threshold: bool = False
    force_threshold_range: List[float] = field(default_factory=lambda: [8.0, 12.0])

    # Mass randomization (held asset only, scale factors)
    randomize_held_mass: bool = False
    held_mass_range: List[float] = field(default_factory=lambda: [0.5, 2.0])  # Scale factors for mass randomization


@dataclass
class EfficientResetConfig:
    """Efficient reset wrapper configuration."""
    enabled: bool = True
    terminate_on_success: bool = False  # Terminate episodes immediately upon success
    success_bonus: float = 0.0  # Total reward to give on success (base env gives +1, wrapper adjusts to this total)
    use_remaining_steps_bonus: bool = False  # If true, bonus = max_episode_length - steps_taken (overrides success_bonus)