from dataclasses import dataclass


@dataclass
class ActorConfig:
    """Actor network configuration with Isaac Lab defaults."""
    n: int = 1                    # Number of SimBa layers (Isaac Lab default)
    latent_size: int = 256        # Hidden dimension (Isaac Lab default)
    use_state_dependent_std: bool = False  # If True, std is predicted from state instead of learned parameter