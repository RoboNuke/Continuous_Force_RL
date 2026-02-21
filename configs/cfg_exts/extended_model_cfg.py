"""
Extended Model Configuration

This module defines ExtendedModelConfig which contains model architecture
parameters for SimBa and hybrid agents. This is not an Isaac Lab config
but our own organizational structure for model configuration.
"""

from typing import Optional, List
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

    # Squashed Gaussian action bounding
    squash_actions: bool = False
    """When True, apply tanh after sampling (squashed Gaussian) instead of on the network mean"""

    correct_squash_log_prob: bool = False
    """When True (and squash_actions=True), apply the Jacobian correction to log_prob"""



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
    ctrl_mode: str = None
    """Control mode: 'force_only' (3DOF), 'force_tz' (4DOF), 'force_torque' (6DOF). Set from primary config."""

    # Initialization parameters
    unit_std_init: bool = True
    """Use unit standard deviation initialization"""

    unit_factor_std_init: float = 1.0
    """The scale after using inverse control gains for initial std devs"""
    
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

    use_separate_heads: bool = False
    """Use separate 2-layer MLP heads for each selection variable and component groups"""

    selection_head_hidden_dim: int = 64
    """Hidden dimension for separate selection head MLPs"""

    component_head_hidden_dim: int = 128
    """Hidden dimension for separate component head MLPs (pos/rot and force/torque)"""

    full_clop: bool = False
    """Use full CLoP (soft GMM mixture) instead of LCLoP (hard selection) for log_prob/entropy"""