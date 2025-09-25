from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ForceTorqueSensorConfig:
    """Force-torque sensor wrapper configuration."""
    enabled: bool = False
    use_tanh_scaling: bool = False
    tanh_scale: float = 0.03
    add_force_obs: bool = False


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