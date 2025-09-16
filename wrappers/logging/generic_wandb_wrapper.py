"""
Generic Wandb Logging Wrapper

This wrapper provides truly environment-agnostic Wandb logging functionality.
It uses configuration to determine which metrics to track, making no assumptions
about the environment type.
"""

import torch
import gymnasium as gym
import wandb
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Optional, List, Union

from .logging_config import LoggingConfig, MetricConfig


class GenericEpisodeTracker:
    """
    Generic episode tracking and metrics computation for Wandb logging.

    This tracker is completely configurable and makes no assumptions about
    the environment type or available metrics.
    """

    def __init__(self, logging_config: LoggingConfig, num_envs: int, device: torch.device):
        """
        Initialize generic episode tracker.

        Args:
            logging_config: Configuration for what metrics to track
            num_envs: Number of environments to track
            device: Torch device
        """
        self.config = logging_config
        self.num_envs = num_envs
        self.device = device
        self.reset_all()

        # Initialize Wandb
        wandb_config = logging_config.to_wandb_config()
        self.run = wandb.init(
            entity=wandb_config.get('entity'),
            project=wandb_config.get('project'),
            name=wandb_config.get('name'),
            reinit=True,
            config=wandb_config,
            group=wandb_config.get('group'),
            tags=wandb_config.get('tags')
        )

        self.wandb_cfg = wandb_config.copy()
        self.wandb_cfg['run_id'] = self.run.id
        self.metrics = {}
        self.learning_metrics = []

    def reset_all(self):
        """Reset all episode tracking variables."""
        # Core episode tracking
        if self.config.track_episodes:
            self.env_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.env_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        if self.config.track_terminations:
            self.env_terminations = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Configurable metric tracking
        self.metric_accumulators = {}
        self.episode_metrics = {}
        self.finished_metrics = []

        # Initialize accumulators for each tracked metric
        for metric_name, metric_config in self.config.tracked_metrics.items():
            if metric_config.enabled:
                if metric_config.metric_type == "scalar":
                    self.metric_accumulators[metric_name] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                elif metric_config.metric_type == "boolean":
                    self.metric_accumulators[metric_name] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                elif metric_config.metric_type == "tensor":
                    # Initialize with default shape - will be resized when first value arrives
                    self.metric_accumulators[metric_name] = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

    def reset_envs(self, done_mask: torch.Tensor, infos: Dict[str, Any]):
        """Reset specified environments and gather metrics."""
        if done_mask.any():
            # Gather metrics for finished episodes
            metrics = self._gather_metrics(done_mask, infos)
            self.finished_metrics.append(metrics)

        # Reset core tracking
        idxs = torch.nonzero(done_mask, as_tuple=False).flatten()

        if self.config.track_episodes:
            self.env_returns[idxs] = 0.0
            self.env_steps[idxs] = 0

        if self.config.track_terminations:
            self.env_terminations[idxs] = 0

        # Reset metric accumulators
        for metric_name, accumulator in self.metric_accumulators.items():
            if isinstance(accumulator, torch.Tensor):
                if accumulator.dim() == 1:
                    accumulator[idxs] = 0
                else:
                    accumulator[idxs] = 0

    def step(self, reward: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, infos: Dict[str, Any]):
        """
        Update episode tracking for one step.

        Args:
            reward: Step rewards for all environments
            terminated: Boolean tensor indicating termination
            truncated: Boolean tensor indicating truncation
            infos: Environment info dictionary
        """
        reward = reward.detach().squeeze()
        terminated = terminated.detach().bool()
        truncated = truncated.detach().bool()

        # Update core tracking
        if self.config.track_episodes:
            self.env_returns += reward
            self.env_steps += 1

        if self.config.track_terminations:
            self.env_terminations += terminated.long()

        # Update configurable metrics
        self._update_tracked_metrics(infos)

        # Handle completed episodes
        done = torch.logical_or(terminated, truncated)
        if done.any():
            self.reset_envs(done, infos)

    def _update_tracked_metrics(self, infos: Dict[str, Any]):
        """Update tracked metrics based on configuration."""
        for metric_name, metric_config in self.config.tracked_metrics.items():
            if not metric_config.enabled:
                continue

            # Get metric value from infos
            metric_value = self._extract_metric_value(infos, metric_config)
            if metric_value is None:
                continue

            # Update accumulator based on metric type
            if metric_config.metric_type == "scalar":
                self.metric_accumulators[metric_name] += metric_value.detach()
            elif metric_config.metric_type == "boolean":
                self.metric_accumulators[metric_name] = torch.logical_or(
                    self.metric_accumulators[metric_name],
                    metric_value.detach().bool()
                )
            elif metric_config.metric_type == "tensor":
                # For tensors, we store the most recent value (could be extended for other aggregations)
                if metric_value.shape[0] == self.num_envs:
                    self.metric_accumulators[metric_name] = metric_value.detach()

    def _extract_metric_value(self, infos: Dict[str, Any], metric_config: MetricConfig) -> Optional[torch.Tensor]:
        """Extract a metric value from the infos dictionary."""
        # Try to get the metric directly
        if metric_config.name in infos:
            value = infos[metric_config.name]
            return self._convert_to_tensor(value, metric_config)

        # Try to get from nested dictionaries (e.g., "Reward / component_name", "smoothness / Smoothness / Avg Force")
        if " / " in metric_config.name:
            parts = metric_config.name.split(" / ")
            current = infos
            try:
                for part in parts:
                    current = current[part]
                return self._convert_to_tensor(current, metric_config)
            except (KeyError, TypeError):
                pass

        # Use default value
        if isinstance(metric_config.default_value, torch.Tensor):
            if metric_config.default_value.numel() == 1:
                return torch.full((self.num_envs,), metric_config.default_value.item(), device=self.device)
            else:
                return metric_config.default_value.expand(self.num_envs, -1).clone()
        else:
            return torch.full((self.num_envs,), float(metric_config.default_value), device=self.device)

    def _convert_to_tensor(self, value: Any, metric_config: MetricConfig) -> torch.Tensor:
        """Convert a value to a tensor appropriate for the metric type."""
        if isinstance(value, torch.Tensor):
            if value.device != self.device:
                value = value.to(self.device)
            return value
        elif isinstance(value, (list, np.ndarray)):
            return torch.tensor(value, device=self.device, dtype=torch.float32)
        elif isinstance(value, (int, float, bool)):
            return torch.full((self.num_envs,), float(value), device=self.device)
        else:
            # Try to convert to float
            try:
                return torch.full((self.num_envs,), float(value), device=self.device)
            except (ValueError, TypeError):
                return torch.full((self.num_envs,), float(metric_config.default_value), device=self.device)

    def _gather_metrics(self, mask: torch.Tensor, infos: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Gather metrics for completed episodes."""
        idxs = torch.nonzero(mask, as_tuple=False).flatten()
        metrics = {}

        # Core episode metrics
        if self.config.track_episodes:
            metrics["Episode / Return (Avg)"] = self.env_returns[idxs]
            metrics["Episode / Return (Median)"] = self.env_returns[idxs]
            metrics["Episode / Return (std)"] = self.env_returns[idxs]

        if self.config.track_episode_length:
            metrics["Episode / Episode Length"] = self.env_steps[idxs].float()

        if self.config.track_terminations:
            metrics["Episode / Terminations"] = self.env_terminations[idxs].float()

        # Configurable metrics
        for metric_name, metric_config in self.config.tracked_metrics.items():
            if not metric_config.enabled or metric_name not in self.metric_accumulators:
                continue

            accumulator = self.metric_accumulators[metric_name]
            wandb_name = metric_config.wandb_name or metric_name

            if metric_config.metric_type == "scalar":
                if metric_config.normalize_by_episode_length and self.config.track_episode_length:
                    # Normalize by episode length
                    normalized_values = accumulator[idxs] / torch.clamp(self.env_steps[idxs], min=1).float()
                    metrics[wandb_name] = normalized_values
                else:
                    metrics[wandb_name] = accumulator[idxs]
            elif metric_config.metric_type == "boolean":
                metrics[wandb_name] = accumulator[idxs].float()
            elif metric_config.metric_type == "tensor":
                if accumulator.dim() > 1:
                    metrics[wandb_name] = accumulator[idxs]

        # Add extra metrics from infos
        for key, val in infos.items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    if hasattr(sub_val, '__getitem__') and len(sub_val) >= len(idxs):
                        try:
                            metrics[f"{key} / {sub_key}"] = sub_val[idxs]
                        except (IndexError, TypeError):
                            pass

        return metrics

    def pop_finished(self) -> Dict[str, torch.Tensor]:
        """Get and clear finished episode metrics."""
        if not self.finished_metrics:
            return {}

        merged = {}
        for m in self.finished_metrics:
            for k, v in m.items():
                merged.setdefault(k, [])
                merged[k].append(v)

        for k, v in merged.items():
            if v:  # Check if list is not empty
                merged[k] = torch.cat(v, 0)

        self.finished_metrics = []
        return merged

    def log_minibatch_update(self, returns=None, values=None, advantages=None, old_log_probs=None,
                           new_log_probs=None, entropies=None, policy_losses=None, value_losses=None,
                           policy_state=None, critic_state=None, optimizer=None):
        """Log learning metrics for a minibatch update."""
        if not self.config.track_learning_metrics:
            return

        stats = {}

        with torch.no_grad():
            if new_log_probs is not None and old_log_probs is not None:
                # Policy stats
                ratio = (new_log_probs - old_log_probs).exp()
                clip_mask = (ratio < 1 - self.config.clip_eps) | (ratio > 1 + self.config.clip_eps)
                kl = old_log_probs - new_log_probs
                stats["Policy / KL-Divergence (Avg)"] = kl.mean().item()
                stats["Policy / KL-Divergence (0.95 Quantile)"] = kl.quantile(0.95).item()
                stats["Policy / Clip Fraction"] = clip_mask.float().mean().item()

            if entropies is not None:
                stats["Policy / Entropy (Avg)"] = entropies.mean().item()
            if policy_losses is not None:
                stats["Policy / Loss (Avg)"] = policy_losses.mean().item()

            if value_losses is not None and values is not None and returns is not None:
                # Value stats
                stats["Critic / Loss (Avg)"] = value_losses.mean().item()
                stats["Critic / Loss (Median)"] = value_losses.median().item()
                stats["Critic / Loss (0.95 Quantile)"] = value_losses.quantile(0.95).item()
                stats["Critic / Predicted Values (Avg)"] = values.mean().item()
                stats["Critic / Predicted Values (Std)"] = values.std().item()

                stats["Critic / Explained Variance"] = (
                    1 - ((returns - values).var(unbiased=False) / returns.var(unbiased=False).clamp(min=1e-8))
                ).item()

            if advantages is not None:
                # Advantage diagnostics
                stats["Advantage / Mean"] = advantages.mean().item()
                stats["Advantage / Std Dev"] = advantages.std().item()
                stats["Advantage / Skew"] = (
                    (((advantages - advantages.mean()) ** 3).mean() / (advantages.std() ** 3 + 1e-8))
                ).item()

            # Gradient norms
            def grad_norm(model_grads):
                if len(model_grads) == 0:
                    return torch.tensor(0.0)
                return torch.norm(torch.stack(model_grads), 2)

            if policy_state is not None:
                stats['Policy / Gradient Norm'] = grad_norm(policy_state.get('gradients', []))

            if critic_state is not None:
                stats['Critic / Gradient Norm'] = grad_norm(critic_state.get('gradients', []))

        self.learning_metrics.append(stats)

    def aggregate_learning_metrics(self) -> Dict[str, torch.Tensor]:
        """Aggregate learning metrics across minibatches."""
        if not self.learning_metrics:
            return {}

        out = {}
        all_keys = set()
        for m in self.learning_metrics:
            all_keys.update(m.keys())

        for k in all_keys:
            agg_met = []
            for m in self.learning_metrics:
                if k in m:
                    agg_met.append(m[k])
            if agg_met:
                vals = torch.tensor(agg_met, device=self.device, dtype=torch.float32)
                out[k] = vals

        self.learning_metrics = []
        return out

    def one_time_learning_metrics(self, actions: torch.Tensor, global_step: int = -1):
        """Log one-time learning metrics like action statistics."""
        if not self.config.track_action_metrics:
            return

        with torch.no_grad():
            # Action statistics
            self.metrics['Action / Mean'] = actions.mean().item()
            self.metrics['Action / Std'] = actions.std().item()
            self.metrics['Action / Max'] = actions.max().item()
            self.metrics['Action / Min'] = actions.min().item()

            # Action saturation
            saturation = torch.logical_or(actions <= -0.98, actions >= 0.98).float().mean()
            self.metrics['Action / Saturation'] = saturation.item()

            if global_step > 0:
                self.metrics['Total Steps'] = global_step

    def _finalize_metrics(self, mets: Dict[str, torch.Tensor], mean_only: bool = False):
        """Finalize metrics for logging."""
        for key, val in mets.items():
            tag = key.lower()
            if mean_only:
                self.metrics[key] = val.mean().item()
            elif "median" in tag:
                self.metrics[key] = val.median().item()
            elif "std" in tag:
                self.metrics[key] = val.std().item()
            elif "total" in tag:
                self.metrics[key] = torch.sum(val).item()
            else:
                self.metrics[key] = val.mean().item()

    def add_metric(self, tag: str, val: Union[float, int, torch.Tensor]):
        """Add a custom metric."""
        self.metrics[tag] = val

    def publish(self):
        """Publish all metrics to Wandb."""
        # Get episode data
        episode_dict = self.pop_finished()
        self._finalize_metrics(episode_dict)

        # Get learning metrics
        agged_lms = self.aggregate_learning_metrics()
        self._finalize_metrics(agged_lms, mean_only=True)

        # Log everything
        self.run.log(self.metrics)
        self.metrics = {}


class GenericWandbLoggingWrapper(gym.Wrapper):
    """
    Truly generic Wandb logging wrapper with configurable metric tracking.

    This wrapper makes no assumptions about the environment type and tracks
    only the metrics specified in the configuration.
    """

    def __init__(self, env: gym.Env, logging_config: LoggingConfig, num_agents: int = 1):
        """
        Initialize generic Wandb logging wrapper.

        Args:
            env: Base environment to wrap
            logging_config: Configuration specifying which metrics to track
            num_agents: Number of agents for static assignment
        """
        super().__init__(env)

        self.config = logging_config
        self.num_agents = num_agents
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # Validate agent assignment
        if self.num_envs % self.num_agents != 0:
            raise ValueError(f"Number of environments ({self.num_envs}) must be divisible by number of agents ({self.num_agents})")

        self.envs_per_agent = self.num_envs // self.num_agents

        # Create episode trackers for each agent
        self.trackers = []
        for i in range(self.num_agents):
            # Create agent-specific config
            agent_config = LoggingConfig()
            agent_config.__dict__.update(logging_config.__dict__)
            agent_config.wandb_name = f"{logging_config.wandb_name or 'agent'}_{i}"

            tracker = GenericEpisodeTracker(agent_config, self.envs_per_agent, self.device)
            self.trackers.append(tracker)

    def get_agent_assignment(self) -> Dict[int, List[int]]:
        """Get environment indices assigned to each agent."""
        assignments = {}
        for i in range(self.num_agents):
            start_idx = i * self.envs_per_agent
            end_idx = (i + 1) * self.envs_per_agent
            assignments[i] = list(range(start_idx, end_idx))
        return assignments

    def step(self, action):
        """Step environment and log metrics."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Log for each agent
        for i, tracker in enumerate(self.trackers):
            start_idx = i * self.envs_per_agent
            end_idx = (i + 1) * self.envs_per_agent

            agent_reward = reward[start_idx:end_idx]
            agent_terminated = terminated[start_idx:end_idx]
            agent_truncated = truncated[start_idx:end_idx]

            # Extract agent-specific info
            agent_info = self._extract_agent_info(info, start_idx, end_idx)

            tracker.step(
                reward=agent_reward,
                terminated=agent_terminated,
                truncated=agent_truncated,
                infos=agent_info
            )

        return obs, reward, terminated, truncated, info

    def _extract_agent_info(self, info: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Extract agent-specific information from the full info dictionary."""
        agent_info = {}

        for key, val in info.items():
            if isinstance(val, dict):
                agent_info[key] = {}
                for sub_key, sub_val in val.items():
                    if hasattr(sub_val, '__getitem__') and hasattr(sub_val, '__len__'):
                        try:
                            if len(sub_val) >= end_idx:
                                agent_info[key][sub_key] = sub_val[start_idx:end_idx]
                        except (TypeError, IndexError):
                            pass
            elif hasattr(val, '__getitem__') and hasattr(val, '__len__'):
                try:
                    if len(val) >= end_idx:
                        agent_info[key] = val[start_idx:end_idx]
                except (TypeError, IndexError):
                    pass
            else:
                # Scalar value - same for all agents
                agent_info[key] = val

        return agent_info

    def add_metric(self, tag: str, value: Union[float, int, torch.Tensor]):
        """Add a metric to all agent trackers."""
        if isinstance(value, torch.Tensor) and len(value) == self.num_envs:
            # Split metric by agent
            for i, tracker in enumerate(self.trackers):
                start_idx = i * self.envs_per_agent
                end_idx = (i + 1) * self.envs_per_agent
                agent_value = value[start_idx:end_idx].mean().item()
                tracker.add_metric(tag, agent_value)
        else:
            # Same value for all agents
            for tracker in self.trackers:
                tracker.add_metric(tag, value)

    def log_learning_metrics(self, **kwargs):
        """Log learning metrics for all agents."""
        for tracker in self.trackers:
            tracker.log_minibatch_update(**kwargs)

    def log_action_metrics(self, actions: torch.Tensor, global_step: int = -1):
        """Log action metrics for all agents."""
        for i, tracker in enumerate(self.trackers):
            start_idx = i * self.envs_per_agent
            end_idx = (i + 1) * self.envs_per_agent
            agent_actions = actions[start_idx:end_idx]
            tracker.one_time_learning_metrics(agent_actions, global_step)

    def publish_metrics(self):
        """Publish all metrics to Wandb."""
        for tracker in self.trackers:
            tracker.publish()

    def close(self):
        """Close Wandb runs."""
        for tracker in self.trackers:
            if hasattr(tracker, 'run') and tracker.run:
                tracker.run.finish()
        super().close()