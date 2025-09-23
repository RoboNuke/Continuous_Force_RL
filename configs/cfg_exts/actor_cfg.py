from dataclasses import dataclass


@dataclass
class ActorConfig:
    """Actor network configuration with Isaac Lab defaults."""
    n: int = 1                    # Number of SimBa layers (Isaac Lab default)
    latent_size: int = 256        # Hidden dimension (Isaac Lab default)

    def validate(self) -> None:
        """Validate actor configuration parameters."""
        if self.n <= 0:
            raise ValueError(f"Actor layers (n) must be positive, got: {self.n}")
        if self.latent_size <= 0:
            raise ValueError(f"Actor latent_size must be positive, got: {self.latent_size}")