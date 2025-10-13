#!/usr/bin/env python3
"""
Enhanced Action Logging Wrapper

This wrapper provides comprehensive action statistics tracking with a focus
on action components. This is a simplified implementation that integrates
with the unified wandb logging system.

Key Features:
- Configurable action component tracking (Selection, Position, Rotation, Force, Torque)
- Boolean-based configuration (no complex config classes needed)
- Automatic action space layout based on force_size parameter
- Component-wise action statistics (mean, std)
- Environment-agnostic design
- Integration with unified wandb logging system
"""

import torch
import gymnasium as gym
import wandb
import numpy as np
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict


class EnhancedActionLoggingWrapper(gym.Wrapper):
    """
    Enhanced wrapper for action statistics tracking.

    This wrapper tracks detailed action statistics by component (Selection, Position,
    Rotation, Force, Torque) using a simplified boolean-based configuration.
    It's designed to be environment-agnostic and integrate with the unified
    wandb logging system.
    """

    def __init__(
        self,
        env,
        track_selection: bool = True,
        track_pos: bool = True,
        track_rot: bool = True,
        track_force: bool = True,
        track_torque: bool = True,
        force_size: int = 6,
        logging_frequency: int = 100
    ):
        """
        Initialize enhanced action logging wrapper.

        Args:
            env: Environment to wrap
            track_selection: Whether to track selection components
            track_pos: Whether to track position components
            track_rot: Whether to track rotation components
            track_force: Whether to track force components
            track_torque: Whether to track torque components
            force_size: 0 (no force/torque), 3 (force only), or 6 (force + torque)
            logging_frequency: How often to collect metrics (steps)
        """
        super().__init__(env)

        # Store configuration
        self.track_selection = track_selection
        self.track_pos = track_pos
        self.track_rot = track_rot
        self.track_force = track_force
        self.track_torque = track_torque
        self.force_size = force_size
        self.logging_frequency = logging_frequency

        # Generate action components based on boolean flags
        self.action_components = self._generate_action_components()

        self.step_count = 0

        # Action statistics storage
        self._action_stats = defaultdict(list)

        # Validate that the wandb wrapper is present somewhere in the chain
        # Walk up the wrapper chain to find add_metrics method
        current_env = self.env
        has_add_metrics = False
        while current_env and not has_add_metrics:
            if hasattr(current_env, 'add_metrics'):
                has_add_metrics = True
                break
            # Check unwrapped
            if hasattr(current_env, 'unwrapped') and hasattr(current_env.unwrapped, 'add_metrics'):
                has_add_metrics = True
                break
            # Move to next wrapper in chain
            current_env = getattr(current_env, 'env', None)

        if not has_add_metrics:
            raise ValueError(
                "Enhanced Action Logging Wrapper requires GenericWandbLoggingWrapper to be applied first. "
                "The environment must have an 'add_metrics' method. "
                "Make sure wandb_logging wrapper is enabled and applied before this wrapper."
            )

        print(f"[INFO]: Enhanced Action Logging Wrapper initialized")
        print(f"  - Components: {list(self.action_components.keys())}")
        print(f"  - Force size: {self.force_size}")
        print(f"  - Wandb integration: ✓ Validated")

    def _generate_action_components(self) -> Dict[str, Dict]:
        """Generate action components based on boolean flags and force_size."""
        components = {}

        # Selection: 0 to force_size (if force_size > 0)
        if self.track_selection and self.force_size > 0:
            names = ['Sel_X','Sel_Y', 'Sel_Z']
            if self.force_size == 6:
                names += ['Sel_TX', 'Sel_TY', 'Sel_TZ']
            components['Selection'] = {
                'start_idx': 0, 'end_idx': self.force_size,
                'names': names, 'enabled': True
            }

        # Position: force_size to force_size+3
        if self.track_pos:
            components['Position'] = {
                'start_idx': self.force_size, 'end_idx': self.force_size + 3,
                'names': ['X', 'Y', 'Z'], 'enabled': True
            }

        # Rotation: force_size+3 to force_size+6
        if self.track_rot:
            components['Rotation'] = {
                'start_idx': self.force_size + 3, 'end_idx': self.force_size + 6,
                'names': ['RX', 'RY', 'RZ'], 'enabled': True
            }

        # Force: force_size+6 to force_size+9 (if force_size > 0)
        if self.track_force and self.force_size > 0:
            components['Force'] = {
                'start_idx': self.force_size + 6, 'end_idx': self.force_size + 9,
                'names': ['FX', 'FY', 'FZ'], 'enabled': True
            }

        # Torque: force_size+9 to force_size+12 (if force_size == 6)
        if self.track_torque and self.force_size == 6:
            components['Torque'] = {
                'start_idx': self.force_size + 9, 'end_idx': self.force_size + 12,
                'names': ['TX', 'TY', 'TZ'], 'enabled': True
            }

        return components

    def step(self, action):
        """Step environment and log action data."""
        print("[DEBUG] EnhancedActionLoggingWrapper.step() CALLED")
        # Log action data
        self._log_action_data(action)

        # Execute environment step
        obs, reward, terminated, truncated, info = super().step(action)

        self.step_count += 1

        # Send metrics to wandb wrapper only when any environment truncates
        if torch.any(truncated):
            # Find the add_metrics method by walking the wrapper chain
            add_metrics_target = None
            current_env = self.env
            while current_env and not add_metrics_target:
                if hasattr(current_env, 'add_metrics'):
                    add_metrics_target = current_env
                    break
                # Check unwrapped
                if hasattr(current_env, 'unwrapped') and hasattr(current_env.unwrapped, 'add_metrics'):
                    add_metrics_target = current_env.unwrapped
                    break
                # Move to next wrapper in chain
                current_env = getattr(current_env, 'env', None)

            if add_metrics_target:
                action_metrics = self._collect_action_metrics()
                if action_metrics:
                    add_metrics_target.add_metrics(action_metrics)

        return obs, reward, terminated, truncated, info

    def _log_action_data(self, actions: torch.Tensor):
        """Log action statistics."""
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)

        actions = actions.detach().cpu()

        # Overall action statistics
        self._action_stats['mean'].append(actions.mean().item())
        self._action_stats['std'].append(actions.std().item())
        self._action_stats['max'].append(actions.max().item())
        self._action_stats['min'].append(actions.min().item())

        # Component-wise analysis
        for component_name, component_config in self.action_components.items():
            if not component_config.get('enabled', True):
                continue

            start_idx = component_config['start_idx']
            end_idx = component_config['end_idx']
            names = component_config['names']

            # Check if action space is large enough
            if actions.shape[-1] < end_idx:
                continue

            # Analyze each dimension in this component
            for i, dim_name in enumerate(names):
                if start_idx + i >= actions.shape[-1]:
                    break

                action_dim = actions[..., start_idx + i]

                # Component statistics
                comp_key = f"{component_name}_{dim_name}"
                self._action_stats[f"{comp_key}_mean"].append(action_dim.mean().item())
                self._action_stats[f"{comp_key}_std"].append(action_dim.std().item())

    def _collect_action_metrics(self):
        """Collect action metrics for wandb logging."""
        metrics = {}

        # Only collect metrics if we have recent data
        if not self._action_stats:
            return metrics

        # Get action statistics - convert lists to tensors
        num_envs = getattr(self.env.unwrapped, 'num_envs', 1)

        # Overall action statistics
        for stat_name in ['mean', 'std', 'max', 'min']:
            if stat_name in self._action_stats and self._action_stats[stat_name]:
                # Take the most recent value and broadcast to all envs
                recent_value = self._action_stats[stat_name][-1]
                metrics[f'Action_Stats/{stat_name}'] = torch.full((num_envs,), recent_value, dtype=torch.float32)

        # Component-wise statistics
        for component_name, component_config in self.action_components.items():
            if not component_config.get('enabled', True):
                continue

            names = component_config['names']
            for dim_name in names:
                key = f"{component_name}_{dim_name}"

                # Component statistics
                for stat_type in ['mean', 'std']:
                    stat_key = f"{key}_{stat_type}"
                    if stat_key in self._action_stats and self._action_stats[stat_key]:
                        recent_value = self._action_stats[stat_key][-1]
                        metrics[f'Action_Components/{component_name}_{dim_name}_{stat_type}'] = torch.full((num_envs,), recent_value, dtype=torch.float32)

        return metrics

    def get_action_statistics(self) -> Dict[str, float]:
        """Get current action statistics."""
        stats = {}
        for key, values in self._action_stats.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        return stats

    def get_component_analysis(self) -> Dict[str, Dict]:
        """Get detailed component-wise action analysis."""
        analysis = {}

        for component_name, component_config in self.action_components.items():
            if not component_config.get('enabled', True):
                continue

            names = component_config['names']
            component_stats = {}

            for dim_name in names:
                key = f"{component_name}_{dim_name}"
                if f"{key}_mean" in self._action_stats:
                    component_stats[dim_name] = {
                        'mean': np.mean(self._action_stats[f"{key}_mean"]),
                        'std': np.mean(self._action_stats[f"{key}_std"])
                    }

            if component_stats:
                analysis[component_name] = component_stats

        return analysis

    def reset(self, **kwargs):
        """Reset environment and clear temporary data."""
        print("[DEBUG] EnhancedActionLoggingWrapper.reset() CALLED")
        return super().reset(**kwargs)

    def close(self):
        """Close wrapper."""
        super().close()


# Factory function for easy creation
def create_factory_action_logger(
    env,
    track_selection: bool = True,
    track_pos: bool = True,
    track_rot: bool = True,
    track_force: bool = True,
    track_torque: bool = True,
    force_size: int = 6,
    logging_frequency: int = 100
) -> EnhancedActionLoggingWrapper:
    """
    Create an action logging wrapper configured for factory tasks.

    Args:
        env: Environment to wrap
        track_selection: Whether to track selection components
        track_pos: Whether to track position components
        track_rot: Whether to track rotation components
        track_force: Whether to track force components
        track_torque: Whether to track torque components
        force_size: 0 (no force/torque), 3 (force only), or 6 (force + torque)
        logging_frequency: How often to collect metrics (steps)

    Returns:
        Configured action logging wrapper
    """
    return EnhancedActionLoggingWrapper(
        env,
        track_selection=track_selection,
        track_pos=track_pos,
        track_rot=track_rot,
        track_force=track_force,
        track_torque=track_torque,
        force_size=force_size,
        logging_frequency=logging_frequency
    )


def create_research_action_logger(
    env,
    track_selection: bool = True,
    track_pos: bool = True,
    track_rot: bool = True,
    track_force: bool = True,
    track_torque: bool = True,
    force_size: int = 6,
    logging_frequency: int = 50
) -> EnhancedActionLoggingWrapper:
    """
    Create an action logging wrapper configured for research purposes.

    Args:
        env: Environment to wrap
        track_selection: Whether to track selection components
        track_pos: Whether to track position components
        track_rot: Whether to track rotation components
        track_force: Whether to track force components
        track_torque: Whether to track torque components
        force_size: 0 (no force/torque), 3 (force only), or 6 (force + torque)
        logging_frequency: How often to collect metrics (steps)

    Returns:
        Configured action logging wrapper for research
    """
    return EnhancedActionLoggingWrapper(
        env,
        track_selection=track_selection,
        track_pos=track_pos,
        track_rot=track_rot,
        track_force=track_force,
        track_torque=track_torque,
        force_size=force_size,
        logging_frequency=logging_frequency
    )


# Example usage
if __name__ == "__main__":
    """Example usage of enhanced action logging wrapper."""

    print("Enhanced Action Logging Wrapper - Example Usage")
    print("=" * 50)

    # This would typically be your actual environment
    # env = gym.make("YourEnvironment")

    print("✓ Simplified action logging with boolean parameters")
    print("  - No configuration classes needed")
    print("  - Direct boolean parameters for component tracking")
    print("  - Automatic action space layout based on force_size")

    print("\nUsage patterns:")
    print("  # Factory environment (with force/torque)")
    print("  factory_env = create_factory_action_logger(env)")
    print("  ")
    print("  # Research with custom settings")
    print("  research_env = create_research_action_logger(env, force_size=3, logging_frequency=25)")
    print("  ")
    print("  # Direct instantiation")
    print("  custom_env = EnhancedActionLoggingWrapper(env, track_selection=True, force_size=6)")