from dataclasses import dataclass


@dataclass
class CriticConfig:
    """Critic network configuration with Isaac Lab defaults."""
    n: int = 3                    # Number of SimBa layers (Isaac Lab default)
    latent_size: int = 1024       # Hidden dimension (Isaac Lab default)