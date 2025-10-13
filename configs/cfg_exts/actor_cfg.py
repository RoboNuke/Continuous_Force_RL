from dataclasses import dataclass


@dataclass
class ActorConfig:
    """Actor network configuration with Isaac Lab defaults."""
    n: int = 1                    # Number of SimBa layers (Isaac Lab default)
    latent_size: int = 256        # Hidden dimension (Isaac Lab default)