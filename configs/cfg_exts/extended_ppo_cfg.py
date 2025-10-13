"""
Extended PPO Configuration

This module defines ExtendedPPOConfig which extends SKRL's PPO_DEFAULT_CONFIG
with our custom parameters and computed properties. This follows SKRL's pattern
of keeping agent-specific configurations in the agents/ folder.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from configs.cfg_exts.experiment_cfg import ExperimentConfig
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
#print(PPO_DEFAULT_CONFIG)

@dataclass
class ExtendedPPOConfig:
    """
    Extended PPO configuration with SKRL defaults and our custom parameters.

    This configuration includes all SKRL PPO defaults plus our custom extensions
    for Block PPO and computed properties for rollout calculations.
    """
    
    # CUSTOM EXTENSIONS FOR BLOCK PPO
    
    disable_progressbar: bool = True
    """Should we display a CLI progress bar?"""
    # Multi-agent parameters
    num_agents: int = 1
    """Number of agents in multi-agent setup"""

    # Custom learning parameters
    policy_learning_rate: Optional[float] = None
    """Separate learning rate for policy (if None, uses learning_rate)"""

    critic_learning_rate: Optional[float] = None
    """Separate learning rate for critic (if None, uses learning_rate)"""

    value_update_ratio: int = 1
    """Number of value updates per policy update"""

    # Optimizer parameters
    optimizer_betas: List[float] = field(default_factory=lambda: [0.999, 0.999])
    """Adam optimizer beta parameters"""

    optimizer_eps: float = 1.0e-8
    """Adam optimizer epsilon"""

    optimizer_weight_decay: float = 0
    """Adam optimizer weight decay"""

    # Loss function parameters
    use_huber_value_loss: bool = False
    """Use Huber loss for value function"""

    # Reward shaping
    reward_shaper_type: str = 'const_scale'
    """Type of reward shaper ('const_scale', etc.)"""

    rewards_shaper_scale: float = 0.1
    """Reward shaping scale factor"""

    # Preprocessors
    state_preprocessor: bool = True
    """Enable state preprocessing"""

    value_preprocessor: bool = True
    """Enable value preprocessing"""

    # Block PPO specific parameters
    break_force: Optional[float] = None
    """Break force for this agent (set during multi-agent setup)"""

    easy_mode: bool = False
    """Enable easy mode for debugging"""

    track_ckpts: bool = True
    """Enable checkpoint tracking"""

    ckpt_tracker_path: str = "/nfs/stak/users/brownhun/ckpt_tracker2.txt"
    """Path to checkpoint tracker file"""

    # Training control
    random_value_timesteps: int = 150
    """Number of random timesteps for value function warmup"""

    write_interval: int = 150
    """Logging write interval"""

    checkpoint_interval: int = 1500
    """Checkpoint save interval"""

    # Experiment tracking
    wandb_experiment: ExperimentConfig = None

    state_preprocessor_kwargs: dict = None

    value_preprocessor_kwargs: dict = None

    agent_exp_cfgs: list = None

    def __post_init__(self):
        """Post-initialization validation and setup."""

        # adds default ppo configs if not overwritten here
        #print(PPO_DEFAULT_CONFIG)
        #print(PPO_DEFAULT_CONFIG.items())
        for key, value in PPO_DEFAULT_CONFIG.items():
            #print("Key:", key)
            if not hasattr(self, key):
                if key == 'lambda':
                    #print("got lambda")
                    setattr(self, key + "_", value)
                else:
                    setattr(self, key, value)

        self._validate_ppo_params()
        self._setup_computed_defaults()

    def _validate_ppo_params(self):
        """Validate PPO configuration parameters."""
        # Validate core parameters
        if self.rollouts <= 0:
            raise ValueError(f"rollouts must be positive, got {self.rollouts}")

        if self.learning_epochs <= 0:
            raise ValueError(f"learning_epochs must be positive, got {self.learning_epochs}")

        if self.mini_batches <= 0:
            raise ValueError(f"mini_batches must be positive, got {self.mini_batches}")

        # Validate probability parameters
        prob_params = {
            'discount_factor': self.discount_factor,
            'lambda_': self.lambda_,
            'ratio_clip': self.ratio_clip,
            'value_clip': self.value_clip
        }

        for param_name, value in prob_params.items():
            if not (0 <= value <= 1):
                raise ValueError(f"{param_name} must be in [0, 1], got {value}")

        # Validate learning rates
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if self.policy_learning_rate is not None and self.policy_learning_rate <= 0:
            raise ValueError(f"policy_learning_rate must be positive, got {self.policy_learning_rate}")

        if self.critic_learning_rate is not None and self.critic_learning_rate <= 0:
            raise ValueError(f"critic_learning_rate must be positive, got {self.critic_learning_rate}")

        # Validate optimizer parameters
        if len(self.optimizer_betas) != 2:
            raise ValueError(f"optimizer_betas must have 2 elements, got {len(self.optimizer_betas)}")

        for beta in self.optimizer_betas:
            if not (0 <= beta <= 1):
                raise ValueError(f"optimizer beta values must be in [0, 1], got {beta}")

    def _setup_computed_defaults(self):
        """Set up computed default values."""
        # Set separate learning rates to main learning rate if not specified
        if self.policy_learning_rate is None:
            self.policy_learning_rate = self.learning_rate

        if self.critic_learning_rate is None:
            self.critic_learning_rate = self.learning_rate

    def apply_primary_cfg(self, primary_cfg) -> None:
        """
        Apply primary configuration values to PPO config.

        Args:
            primary_cfg: PrimaryConfig instance containing shared parameters
        """
        # Store reference to primary config for computed properties
        self._primary_cfg = primary_cfg

        # Apply primary config values that affect training
        self.num_agents = primary_cfg.total_agents
        self.ckpt_tracker_path = primary_cfg.ckpt_tracker_path

    def get_rollout_steps(self, episode_length_s: float) -> int:
        """
        Calculate rollout steps based on episode length and primary config.

        Args:
            episode_length_s: Episode length in seconds

        Returns:
            Number of rollout steps for this episode length
        """
        if not hasattr(self, '_primary_cfg'):
            raise ValueError("Primary config not applied. Call apply_primary_cfg() first.")

        return self._primary_cfg.rollout_steps(episode_length_s)

    def get_computed_rollouts(self, episode_length_s: float) -> int:
        """
        Get computed rollouts value based on episode timing.

        This replaces the hardcoded rollouts value with one computed from
        episode length and simulation parameters.

        Args:
            episode_length_s: Episode length in seconds

        Returns:
            Computed rollouts value
        """
        rollout_steps = self.get_rollout_steps(episode_length_s)
        # Ensure rollouts is reasonable for training
        return max(rollout_steps, 16)  # Minimum of 16 rollouts

    def get_computed_mini_batches(self, episode_length_s: float) -> int:
        """
        Get computed mini_batches value based on rollouts.

        Args:
            episode_length_s: Episode length in seconds

        Returns:
            Computed mini_batches value
        """
        computed_rollouts = self.get_computed_rollouts(episode_length_s)
        # Mini batches should divide rollouts evenly, with reasonable range
        return self.mini_batches #computed_rollouts #max(min(computed_rollouts // 8, 32), 2)

    def to_skrl_dict(self, episode_length_s: float) -> Dict[str, Any]:
        """
        Convert to SKRL-compatible configuration dictionary.

        Args:
            episode_length_s: Episode length in seconds for computed values

        Returns:
            Dictionary compatible with SKRL PPO agent
        """
        # Use computed values for rollouts and mini_batches
        computed_rollouts = self.get_computed_rollouts(episode_length_s)
        computed_mini_batches = self.get_computed_mini_batches(episode_length_s)

        return {
            # Core PPO parameters with computed values
            'rollouts': computed_rollouts,
            'learning_epochs': self.learning_epochs,
            'mini_batches': computed_mini_batches,
            'discount_factor': self.discount_factor,
            'lambda': self.lambda_,  # Note: no underscore in SKRL
            'learning_rate': self.learning_rate,

            # PPO-specific parameters
            'ratio_clip': self.ratio_clip,
            'value_clip': self.value_clip,
            'entropy_loss_scale': self.entropy_loss_scale,
            'value_loss_scale': self.value_loss_scale,

            # Training parameters
            'random_timesteps': self.random_timesteps,
            'learning_starts': self.learning_starts,
            'grad_norm_clip': self.grad_norm_clip,
            'kl_threshold': self.kl_threshold,

            # Advanced parameters
            'mixed_precision': self.mixed_precision,
            'time_limit_bootstrap': self.time_limit_bootstrap,

            # Custom extensions
            'policy_learning_rate': self.policy_learning_rate,
            'critic_learning_rate': self.critic_learning_rate,
            'value_update_ratio': self.value_update_ratio,
            'use_huber_value_loss': self.use_huber_value_loss,
            'clip_predicted_values': self.clip_predicted_values,
            'reward_shaper_type': self.reward_shaper_type,
            'rewards_shaper_scale': self.rewards_shaper_scale,
            'rewards_shaper': self.rewards_shaper if hasattr(self, 'rewards_shaper') else None,
            'state_preprocessor': self.state_preprocessor,
            'state_preprocessor_kwargs': self.state_preprocessor_kwargs,
            'value_preprocessor': self.value_preprocessor,
            'value_preprocessor_kwargs': self.value_preprocessor_kwargs,
            'learning_rate_scheduler': self.learning_rate_scheduler,
            'learning_rate_scheduler_kwargs': self.learning_rate_scheduler_kwargs,
            'agent_exp_cfgs': self.agent_exp_cfgs,
            # Optimizer parameters
            'optimizer': {
                'betas': self.optimizer_betas,
                'eps': self.optimizer_eps,
                'weight_decay': self.optimizer_weight_decay
            },

            # Agent-specific parameters
            'disable_progressbar': self.disable_progressbar,
            'break_force': self.break_force,
            'easy_mode': self.easy_mode,
            'track_ckpts': self.track_ckpts,
            'ckpt_tracker_path': self.ckpt_tracker_path,
            'random_value_timesteps': self.random_value_timesteps,

            # Experiment tracking
            'write_interval': self.write_interval,
            'checkpoint_interval': self.checkpoint_interval,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ExtendedPPOConfig(rollouts={self.rollouts}, lr={self.learning_rate}, agents={self.num_agents})"