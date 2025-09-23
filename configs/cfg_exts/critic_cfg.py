from dataclasses import dataclass


@dataclass
class CriticConfig:
    """Critic network configuration with Isaac Lab defaults."""
    n: int = 3                    # Number of SimBa layers (Isaac Lab default)
    latent_size: int = 1024       # Hidden dimension (Isaac Lab default)

    def validate(self) -> None:
        """Validate critic configuration parameters."""
        if self.n <= 0:
            raise ValueError(f"Critic layers (n) must be positive, got: {self.n}")
        if self.latent_size <= 0:
            raise ValueError(f"Critic latent_size must be positive, got: {self.latent_size}")