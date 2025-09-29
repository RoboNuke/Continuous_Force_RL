"""
Experiment Configuration

Simple dataclass for experiment metadata including tags, name, group, etc.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for experiment metadata."""

    name: str = "default_experiment"
    tags: List[str] = field(default_factory=list)
    group: str = "default_group"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    description: Optional[str] = None