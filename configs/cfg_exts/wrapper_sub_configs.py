from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FactoryMetricsConfig:
    enabled: bool = False
    publish_to_wandb: bool = True

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


@dataclass
class ObservationNoiseConfig:
    """Observation noise wrapper configuration."""
    enabled: bool = False
    global_scale: float = 1.0
    apply_to_critic: bool = True
    seed: Optional[int] = None


@dataclass
class WandbLoggingConfig:
    """Wandb logging wrapper configuration."""
    enabled: bool = True
    wandb_project: str = "Continuous_Force_RL"
    wandb_entity: str = "hur"
    wandb_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)


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


@dataclass
class PoseContactLoggingConfig:
    """Pose contact logging wrapper configuration."""
    enabled: bool = False