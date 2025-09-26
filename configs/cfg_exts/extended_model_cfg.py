"""
Extended Model Configuration

This module defines ExtendedModelConfig which contains model architecture
parameters for SimBa and hybrid agents. This is not an Isaac Lab config
but our own organizational structure for model configuration.
"""

from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field

from .version_compat import get_isaac_lab_ctrl_imports
from .actor_cfg import ActorConfig
from .critic_cfg import CriticConfig

# Get configclass decorator with version compatibility
configclass, _ = get_isaac_lab_ctrl_imports()

def _default_selection_adjustment_types():
    """Default factory for selection_adjustment_types field."""
    return ['l2_norm', 'none']


@configclass
@dataclass
class ExtendedModelConfig:
    """
    Extended model configuration for neural network architectures.

    Contains parameters for both standard and hybrid agents, including
    SimBa architecture settings and initialization parameters.
    """

    # Nested configurations (maintain natural YAML structure)
    actor: ActorConfig = field(default_factory=ActorConfig)
    """Actor network configuration"""

    critic: CriticConfig = field(default_factory=CriticConfig)
    """Critic network configuration"""

    hybrid_agent: Optional['ExtendedHybridAgentConfig'] = None
    """Hybrid agent configuration (when use_hybrid_agent=True)"""

    # General model parameters
    force_encoding: Optional[str] = None
    """Force encoding method (None for default)"""

    last_layer_scale: float = 1.0
    """Scale factor for last layer initialization"""

    act_init_std: float = 1.0
    """Initial standard deviation for action outputs"""

    critic_output_init_mean: float = 50.0
    """Initial mean value for critic output"""

    # Hybrid agent configuration
    use_hybrid_agent: bool = False
    """Whether to use hybrid force-position agent"""

    def __post_init__(self):
        """Post-initialization validation."""
        self.validate()

    def validate(self) -> None:
        """Validate model configuration."""
        self.actor.validate()
        self.critic.validate()

        if self.use_hybrid_agent and self.hybrid_agent is None:
            raise ValueError("use_hybrid_agent=True but no hybrid_agent config provided")

        if not self.use_hybrid_agent and self.hybrid_agent is not None:
            print(f"\033[93m[CONFIG]: Warning: hybrid_agent config provided but use_hybrid_agent=False\033[0m")

        # Validate scale factors
        if self.last_layer_scale <= 0:
            raise ValueError(f"last_layer_scale must be positive, got {self.last_layer_scale}")

        if self.act_init_std <= 0:
            raise ValueError(f"act_init_std must be positive, got {self.act_init_std}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary structure for launch_utils."""
        result = {
            'actor': {
                'n': self.actor.n,
                'latent_size': self.actor.latent_size
            },
            'critic': {
                'n': self.critic.n,
                'latent_size': self.critic.latent_size
            },
            'force_encoding': self.force_encoding,
            'last_layer_scale': self.last_layer_scale,
            'act_init_std': self.act_init_std,
            'critic_output_init_mean': self.critic_output_init_mean,
            'use_hybrid_agent': self.use_hybrid_agent
        }

        if self.hybrid_agent is not None:
            result['hybrid_agent'] = self.hybrid_agent.to_dict()

        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExtendedModelConfig':
        """Create from dictionary."""
        # Handle nested actor/critic configuration
        actor_config = config_dict.get('actor', {})
        critic_config = config_dict.get('critic', {})
        hybrid_config = config_dict.get('hybrid_agent')

        return cls(
            actor=ActorConfig(
                n=actor_config.get('n', 1),
                latent_size=actor_config.get('latent_size', 256)
            ),
            critic=CriticConfig(
                n=critic_config.get('n', 3),
                latent_size=critic_config.get('latent_size', 1024)
            ),
            hybrid_agent=ExtendedHybridAgentConfig(**hybrid_config) if hybrid_config else None,
            force_encoding=config_dict.get('force_encoding'),
            last_layer_scale=config_dict.get('last_layer_scale', 1.0),
            act_init_std=config_dict.get('act_init_std', 1.0),
            critic_output_init_mean=config_dict.get('critic_output_init_mean', 50.0),
            use_hybrid_agent=config_dict.get('use_hybrid_agent', False)
        )

    def is_hybrid_agent(self) -> bool:
        """Check if this is configured for hybrid agent."""
        return self.use_hybrid_agent

    def get_actor_config(self) -> Dict[str, Any]:
        """Get actor-specific configuration."""
        return {
            'n': self.actor.n,
            'latent_size': self.actor.latent_size
        }

    def get_critic_config(self) -> Dict[str, Any]:
        """Get critic-specific configuration."""
        return {
            'n': self.critic.n,
            'latent_size': self.critic.latent_size
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        hybrid_info = f", hybrid={self.use_hybrid_agent}" if self.use_hybrid_agent else ""
        return f"ExtendedModelConfig(actor={self.actor.n}x{self.actor.latent_size}, critic={self.critic.n}x{self.critic.latent_size}{hybrid_info})"


@configclass
@dataclass
class ExtendedHybridAgentConfig:
    """
    Configuration for hybrid force-position agent specific parameters.

    This is separate from the main model config to keep hybrid-specific
    parameters organized.
    """

    # Selection network parameters (field with factory must come first)
    selection_adjustment_types: List[str] = field(default_factory=_default_selection_adjustment_types)
    """List of selection adjustment types ('none', 'l2_norm', 'bias', 'scale', etc.)"""

    # Control mode
    ctrl_torque: bool = False
    """Use torque control instead of position control"""

    # Initialization parameters
    unit_std_init: bool = True
    """Use unit standard deviation initialization"""

    pos_init_std: float = 1.0
    """Initial standard deviation for position outputs"""

    rot_init_std: float = 1.0
    """Initial standard deviation for rotation outputs"""

    force_init_std: float = 1.0
    """Initial standard deviation for force outputs"""

    # Output scaling
    pos_scale: float = 1.0
    """Position output scaling factor"""

    rot_scale: float = 1.0
    """Rotation output scaling factor"""

    force_scale: float = 1.0
    """Force output scaling factor"""

    torque_scale: float = 1.0
    """Torque output scaling factor"""

    init_scale_weights_factor: float = 0.1
    """Factor for initializing scale weights"""

    init_bias: float = -2.5
    """Initial bias value for selection network"""

    pre_layer_scale_factor: float = 0.1
    """Pre-layer scaling factor"""

    init_scale_last_layer: bool = True
    """Whether to scale the last layer initialization"""

    init_layer_scale: float = 1.0
    """Scale factor for layer initialization"""

    uniform_sampling_rate: float = 0.0
    """Rate for uniform sampling (0.0 = no uniform sampling)"""

    def __post_init__(self):
        """Post-initialization validation."""
        self._validate_hybrid_params()

    def _validate_hybrid_params(self):
        """Validate hybrid agent parameters."""
        # Validate standard deviations
        std_params = {
            'pos_init_std': self.pos_init_std,
            'rot_init_std': self.rot_init_std,
            'force_init_std': self.force_init_std
        }

        for param_name, value in std_params.items():
            if value <= 0:
                raise ValueError(f"{param_name} must be positive, got {value}")

        # Validate scale factors
        scale_params = {
            'pos_scale': self.pos_scale,
            'rot_scale': self.rot_scale,
            'force_scale': self.force_scale,
            'torque_scale': self.torque_scale
        }

        for param_name, value in scale_params.items():
            if value <= 0:
                raise ValueError(f"{param_name} must be positive, got {value}")

        # Validate sampling rate
        if not (0.0 <= self.uniform_sampling_rate <= 1.0):
            raise ValueError(f"uniform_sampling_rate must be in [0, 1], got {self.uniform_sampling_rate}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ctrl_torque': self.ctrl_torque,
            'unit_std_init': self.unit_std_init,
            'pos_init_std': self.pos_init_std,
            'rot_init_std': self.rot_init_std,
            'force_init_std': self.force_init_std,
            'pos_scale': self.pos_scale,
            'rot_scale': self.rot_scale,
            'force_scale': self.force_scale,
            'torque_scale': self.torque_scale,
            'selection_adjustment_types': self.selection_adjustment_types,
            'init_scale_weights_factor': self.init_scale_weights_factor,
            'init_bias': self.init_bias,
            'pre_layer_scale_factor': self.pre_layer_scale_factor,
            'init_scale_last_layer': self.init_scale_last_layer,
            'init_layer_scale': self.init_layer_scale,
            'uniform_sampling_rate': self.uniform_sampling_rate
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        ctrl_info = f"torque={self.ctrl_torque}"
        return f"ExtendedHybridAgentConfig({ctrl_info}, pos_scale={self.pos_scale}, force_scale={self.force_scale})"