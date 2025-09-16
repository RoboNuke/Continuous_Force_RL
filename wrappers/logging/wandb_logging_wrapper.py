"""
Generic Wandb Logging Wrapper

This wrapper provides environment-agnostic Wandb logging functionality for any environment,
with support for multi-agent scenarios using static environment assignment.

Extracted from: MultiWandbLoggerPPO.py and block_wandb_logger_PPO.py
Key features:
- Episode tracking and metrics
- Learning metrics computation
- Histogram tracking
- Multi-agent support
- Termination tracking
"""

import torch
import gymnasium as gym
import wandb
import numpy as np
from collections import defaultdict


class EpisodeTracker:
    """
    Episode tracking and metrics computation for Wandb logging.

    Extracted from: MultiWandbLoggerPPO.py lines 45-424
    """

    def __init__(self, wandb_config, num_envs, device, clip_eps=0.2):
        """
        Initialize episode tracker.

        Args:
            wandb_config: Wandb configuration dictionary
            num_envs: Number of environments to track
            device: Torch device
            clip_eps: Clipping epsilon for PPO diagnostics
        """
        self.num_envs = num_envs
        self.device = device
        self.clip_eps = clip_eps
        self.reset_all()

        # Initialize Wandb
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
        self.env_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.env_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env_terminations = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Engagement and success tracking
        self.engaged_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.succeeded_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Current state tracking
        self.current_engaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.engagement_start = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.engagement_lengths = defaultdict(list)

        # Success tracking
        self.steps_to_first_success = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        # Metrics storage
        self.comp_sums = {}
        self.finished_metrics = []

    def reset_envs(self, done_mask, infos):
        """Reset specified environments and gather metrics."""
        if done_mask.any():
            # Calculate final engagement lengths for environments ending while engaged
            idxs = torch.nonzero(done_mask, as_tuple=False).flatten()
            for i in idxs.tolist():
                if self.current_engaged[i] and self.engagement_start[i] >= 0:
                    length = self.env_steps[i].item() - self.engagement_start[i].item()
                    self.engagement_lengths[i].append(length)

            # Gather metrics for finished episodes
            metrics = self._gather_metrics(done_mask, infos)
            self.finished_metrics.append(metrics)

        # Reset selected environments
        idxs = torch.nonzero(done_mask, as_tuple=False).flatten()
        self.env_returns[idxs] = 0.0
        self.env_steps[idxs] = 0
        self.env_terminations[idxs] = 0
        self.engaged_any[idxs] = False
        self.succeeded_any[idxs] = False
        self.current_engaged[idxs] = False
        self.engagement_start[idxs] = -1
        self.steps_to_first_success[idxs] = -1

        # Reset engagement lengths
        for i in idxs.tolist():
            self.engagement_lengths[i] = []

        # Reset component sums
        for k in self.comp_sums.keys():
            self.comp_sums[k][idxs] = 0.0

    def step(self, reward, reward_components, engaged, success, terminated, truncated, infos):
        """
        Update episode tracking for one step.

        Args:
            reward: Step rewards for all environments
            reward_components: Dictionary of reward component tensors
            engaged: Boolean tensor indicating engagement
            success: Boolean tensor indicating success
            terminated: Boolean tensor indicating termination
            truncated: Boolean tensor indicating truncation
            infos: Environment info dictionary
        """
        reward = reward.detach().squeeze()
        engaged = engaged.detach().bool()
        success = success.detach().bool()
        terminated = terminated.detach().bool()
        truncated = truncated.detach().bool()

        # Update returns and steps
        self.env_returns += reward
        self.env_steps += 1

        # Track terminations
        self.env_terminations += terminated.long()

        # Update component rewards
        for k, v in reward_components.items():
            v = v.detach()
            if k not in self.comp_sums:
                self.comp_sums[k] = torch.zeros_like(v, dtype=torch.float32, device=self.device)
            self.comp_sums[k] += v

        # Track engagement transitions
        just_started = engaged & ~self.current_engaged
        just_ended = ~engaged & self.current_engaged

        self.engagement_start[just_started] = self.env_steps[just_started]
        ended_idxs = torch.nonzero(just_ended, as_tuple=False).flatten()
        for i in ended_idxs.tolist():
            start = self.engagement_start[i].item()
            if start >= 0:
                length = self.env_steps[i].item() - start
                self.engagement_lengths[i].append(length)

        self.current_engaged = engaged
        self.engaged_any |= engaged

        # Track success
        new_success = success & ~self.succeeded_any
        self.steps_to_first_success[new_success] = self.env_steps[new_success]
        self.succeeded_any |= success

        # Handle completed episodes
        done = torch.logical_or(terminated, truncated)
        if done.any():
            self.reset_envs(done, infos)

    def _gather_metrics(self, mask, infos):
        """Gather metrics for completed episodes."""
        idxs = torch.nonzero(mask, as_tuple=False).flatten()
        metrics = {
            "Episode / Return (Avg)": self.env_returns[idxs],
            "Episode / Return (Median)": self.env_returns[idxs],
            "Episode / Return (std)": self.env_returns[idxs],
            "Episode / Episode Length": self.env_steps[idxs].float(),
            "Episode / Terminations": self.env_terminations[idxs].float(),
            "Engagement / Engaged Rate": self.engaged_any[idxs].float(),
            "Success / Success Rate": self.succeeded_any[idxs].float(),
        }

        # Component rewards (averaged per step)
        for k, total in self.comp_sums.items():
            avg = total[idxs] / torch.clamp(self.env_steps[idxs], min=1)
            tag = k
            if "kp_" in k:
                tag = "".join([word.capitalize() for word in k.replace("kp_", "").split()]).replace("/", " / ") + " Keypoint Reward"
            metrics[tag] = avg

        # Engagement statistics
        avg_lengths, counts = [], []
        for i in idxs.tolist():
            if self.engagement_lengths[i]:
                avg_lengths.append(sum(self.engagement_lengths[i]) / len(self.engagement_lengths[i]))
                counts.append(len(self.engagement_lengths[i]))
            else:
                avg_lengths.append(None)
                counts.append(0)

        if any(x is not None for x in avg_lengths):
            metrics["Engagement / Engagement Length (Avg)"] = torch.tensor(
                [x for x in avg_lengths if x is not None], device=self.device, dtype=torch.float32
            )
        if any(c > 0 for c in counts):
            metrics["Engagement / Total Engagements"] = torch.tensor(
                [c for c in counts if c > 0], device=self.device, dtype=torch.float32
            )

        # Steps to first success
        success_mask = self.succeeded_any[idxs]
        if success_mask.any():
            metrics["Success / Steps to Success (Avg)"] = self.steps_to_first_success[idxs][success_mask].float()

        # Add extra metrics from infos
        for key, val in infos.items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    if hasattr(sub_val, '__getitem__'):  # Check if indexable
                        metrics[f"{key} / {sub_key}"] = sub_val[idxs]

        return metrics

    def pop_finished(self):
        """Get and clear finished episode metrics."""
        if not self.finished_metrics:
            return {}

        merged = {}
        for m in self.finished_metrics:
            for k, v in m.items():
                merged.setdefault(k, [])
                merged[k].append(v)

        for k, v in merged.items():
            merged[k] = torch.cat(v, 0)

        self.finished_metrics = []
        return merged

    def log_minibatch_update(self, returns=None, values=None, advantages=None, old_log_probs=None,
                           new_log_probs=None, entropies=None, policy_losses=None, value_losses=None,
                           policy_state=None, critic_state=None, optimizer=None):
        """Log learning metrics for a minibatch update."""
        stats = {}

        with torch.no_grad():
            if new_log_probs is not None and old_log_probs is not None:
                # Policy stats
                ratio = (new_log_probs - old_log_probs).exp()
                clip_mask = (ratio < 1 - self.clip_eps) | (ratio > 1 + self.clip_eps)
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

    def aggregate_learning_metrics(self):
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

    def one_time_learning_metrics(self, actions, global_step=-1):
        """Log one-time learning metrics like action statistics."""
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

    def _finalize_metrics(self, mets, mean_only=False):
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

    def add_metric(self, tag, val):
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


class WandbLoggingWrapper(gym.Wrapper):
    """
    Generic Wandb logging wrapper with multi-agent support.

    Features:
    - Environment-agnostic episode tracking
    - Multi-agent support with static environment assignment
    - Learning metrics computation
    - Histogram tracking
    - Termination tracking
    """

    def __init__(self, env, wandb_config, num_agents=1, clip_eps=0.2):
        """
        Initialize Wandb logging wrapper.

        Args:
            env: Base environment to wrap
            wandb_config: Wandb configuration dictionary
            num_agents: Number of agents for static assignment
            clip_eps: Clipping epsilon for PPO diagnostics
        """
        super().__init__(env)

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
            agent_config = wandb_config.copy()
            agent_config['name'] = f"{wandb_config.get('name', 'agent')}_{i}"
            agent_config['num_envs'] = self.envs_per_agent

            tracker = EpisodeTracker(agent_config, self.envs_per_agent, self.device, clip_eps)
            self.trackers.append(tracker)

    def get_agent_assignment(self):
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

        # Extract engagement and success from info if available
        engaged = info.get('current_engagements', torch.zeros(self.num_envs, device=self.device))
        success = info.get('current_successes', torch.zeros(self.num_envs, device=self.device))

        # Extract reward components
        reward_components = {}
        for key, val in info.items():
            if 'Reward /' in key and hasattr(val, '__getitem__'):
                reward_components[key] = val

        # Log for each agent
        for i, tracker in enumerate(self.trackers):
            start_idx = i * self.envs_per_agent
            end_idx = (i + 1) * self.envs_per_agent

            agent_reward = reward[start_idx:end_idx]
            agent_engaged = engaged[start_idx:end_idx]
            agent_success = success[start_idx:end_idx]
            agent_terminated = terminated[start_idx:end_idx]
            agent_truncated = truncated[start_idx:end_idx]

            # Extract agent-specific reward components
            agent_reward_components = {}
            for key, val in reward_components.items():
                if hasattr(val, '__getitem__'):
                    agent_reward_components[key] = val[start_idx:end_idx]

            # Extract agent-specific info
            agent_info = {}
            for key, val in info.items():
                if isinstance(val, dict):
                    agent_info[key] = {}
                    for sub_key, sub_val in val.items():
                        if hasattr(sub_val, '__getitem__'):
                            agent_info[key][sub_key] = sub_val[start_idx:end_idx]

            tracker.step(
                reward=agent_reward,
                reward_components=agent_reward_components,
                engaged=agent_engaged,
                success=agent_success,
                terminated=agent_terminated,
                truncated=agent_truncated,
                infos=agent_info
            )

        return obs, reward, terminated, truncated, info

    def add_metric(self, tag, value):
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

    def log_action_metrics(self, actions, global_step=-1):
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