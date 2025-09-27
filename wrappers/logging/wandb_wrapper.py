"""
Simple Wandb Logging Wrapper

This wrapper provides simple, function-based Wandb logging.
Other wrappers call its functions to add metrics, and it handles all the complexity.
"""

import torch
import gymnasium as gym
import wandb
import copy
from typing import Dict, Any, Optional, List, Union


class SimpleEpisodeTracker:
    """Simple episode tracker that accumulates metrics and publishes to wandb."""

    def __init__(self, num_envs: int, device: torch.device, agent_config: Dict[str, Any], env_config: Any):
        """Initialize simple episode tracker."""
        self.num_envs = num_envs
        self.device = device

        # Combine env_config with agent-specific config for wandb
        combined_config = copy.deepcopy(env_config)
        combined_config.__dict__.update(agent_config)

        self.run = wandb.init(
            entity=agent_config.get('wandb_entity'),
            project=agent_config.get('wandb_project'),
            name=agent_config.get('wandb_name'),
            reinit="create_new",
            config=combined_config,
            group=agent_config.get('wandb_group'),
            tags=agent_config.get('wandb_tags'),
            #settings=wandb.Settings(
            #    _disable_stats=True,  # Reduce wandb overhead
            #    _disable_meta=True,   # Reduce metadata collection
            #    console="off",        # Reduce console spam
            #    _service_wait=300     # Longer service timeout
            #)

        )

        # Metric storage
        self.accumulated_metrics = {}
        self.episode_count = 0

        # Step tracking for x-axis
        self.env_steps = 0  # Number of environment steps taken
        self.total_steps = 0  # env_steps * num_envs

    def increment_steps(self):
        """Increment step counters for this agent."""
        self.env_steps += 1
        self.total_steps = self.env_steps * self.num_envs

    def add_metrics(self, metrics: Dict[str, torch.Tensor]):
        """Add metrics for this agent's environments."""
        for name, values in metrics.items():
            if not isinstance(values, torch.Tensor):
                raise TypeError(
                    f"Metric '{name}' must be a torch.Tensor, got {type(values)}. "
                    f"Convert lists to tensors using: torch.tensor(your_list)"
                )
            if name not in self.accumulated_metrics:
                self.accumulated_metrics[name] = []
            self.accumulated_metrics[name].append(values)

    def publish(self, onetime_metrics: Dict[str, torch.Tensor] = {}):
        """Add onetime metrics and publish everything to wandb."""
        # Validate onetime_metrics are tensors
        for name, value in onetime_metrics.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Onetime metric '{name}' must be a torch.Tensor, got {type(value)}. "
                    f"Convert to tensor using: torch.tensor(your_value)"
                )

        # Aggregate accumulated metrics
        final_metrics = {}

        # Process regular metrics (take mean across all accumulated values)
        for name, value_list in self.accumulated_metrics.items():
            if value_list:
                stacked = torch.stack(value_list)
                result = stacked.mean().item()
                final_metrics[name] = result

        # Add onetime metrics (now guaranteed to be tensors)
        for name, value in onetime_metrics.items():
            result = value.mean().item()
            final_metrics[f"learning/{name}"] = result

        # Add required step metrics for x-axis
        final_metrics["total_steps"] = self.total_steps
        final_metrics["env_steps"] = self.env_steps

        # Publish to wandb
        self.run.log(final_metrics)

        # Clear accumulated data
        self.accumulated_metrics.clear()

    def close(self):
        """Close wandb run."""
        if hasattr(self, 'run'):
            self.run.finish()


class GenericWandbLoggingWrapper(gym.Wrapper):
    """
    Simple Wandb logging wrapper that provides metric collection functions.
    Other wrappers call these functions to add their metrics.
    """

    def __init__(self, env: gym.Env, num_agents: int = 1, env_cfg: Any = None):
        """
        Initialize simple Wandb logging wrapper.

        Args:
            env: Base environment to wrap
            num_agents: Number of agents for static assignment
            env_cfg: Environment configuration - REQUIRED, contains agent_configs
        """
        super().__init__(env)

        if env_cfg is None:
            raise ValueError("env_cfg is required and cannot be None")

        # Extract agent configs from env_cfg and remove them
        if hasattr(env_cfg, 'agent_configs'):
            agent_configs = copy.deepcopy(env_cfg.agent_configs)
            # Create clean env_cfg without agent configs
            clean_env_cfg = copy.deepcopy(env_cfg)
            delattr(clean_env_cfg, 'agent_configs')
        else:
            raise ValueError("env_cfg must contain agent_configs")

        self.num_agents = num_agents
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.clean_env_cfg = clean_env_cfg

        # Validate agent assignment
        if self.num_envs % self.num_agents != 0:
            raise ValueError(f"Number of environments ({self.num_envs}) must be divisible by number of agents ({self.num_agents})")

        self.envs_per_agent = self.num_envs // self.num_agents

        # Create episode trackers for each agent
        self.trackers = []
        for i in range(self.num_agents):
            agent_config = agent_configs.get(f'agent_{i}', {})
            tracker = SimpleEpisodeTracker(self.envs_per_agent, self.device, agent_config, self.clean_env_cfg)
            self.trackers.append(tracker)

        # Episode tracking for basic metrics
        self.max_episode_length = getattr(env.unwrapped, 'max_episode_length', 1000)

        # Current episode tracking (per environment)
        self.current_episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self.current_episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Per-agent completed episode data storage
        self.agent_episode_data = {}
        for i in range(self.num_agents):
            self.agent_episode_data[i] = {
                'episode_lengths': [],
                'episode_rewards': [],
                'episode_avg_rewards': [],
                'completed_count': [],
                'component_rewards': {}  # Will store component reward lists dynamically
            }

        # Component reward tracking
        self.component_reward_keys = set()  # Discovered component reward keys
        self.current_component_rewards = torch.zeros((self.num_envs, 0), device=self.device)  # Will expand as needed

    def _store_completed_episodes(self, completed_mask):
        """Store completed episode data in agent lists."""
        if not torch.any(completed_mask):
            return

        completed_indices = completed_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(completed_indices.shape) == 0:  # Handle single element case
            completed_indices = completed_indices.unsqueeze(0)

        # Store completed episodes in agent-specific lists
        for env_idx in completed_indices:
            env_idx = env_idx.item()

            # Determine which agent this environment belongs to
            agent_id = env_idx // self.envs_per_agent

            # Get episode data
            episode_length = self.current_episode_lengths[env_idx].item()
            episode_reward = self.current_episode_rewards[env_idx].item()
            episode_avg_reward = episode_reward / episode_length if episode_length > 0 else 0.0

            # Store in agent's episode lists
            self.agent_episode_data[agent_id]['episode_lengths'].append(episode_length)
            self.agent_episode_data[agent_id]['episode_rewards'].append(episode_reward)
            self.agent_episode_data[agent_id]['episode_avg_rewards'].append(episode_avg_reward)
            self.agent_episode_data[agent_id]['completed_count'].append(1.0)

    def _send_aggregated_episode_metrics(self):
        """Send aggregated episode metrics for all agents."""
        # Create agent-level aggregated metrics
        for agent_id in range(self.num_agents):
            agent_data = self.agent_episode_data[agent_id]

            if not agent_data['episode_lengths']:  # No completed episodes for this agent
                continue

            # Calculate means across completed episodes for this agent
            mean_length = sum(agent_data['episode_lengths']) / len(agent_data['episode_lengths'])
            mean_reward = sum(agent_data['episode_rewards']) / len(agent_data['episode_rewards'])
            mean_avg_reward = sum(agent_data['episode_avg_rewards']) / len(agent_data['episode_avg_rewards'])
            total_completed = sum(agent_data['completed_count'])

            # Create metrics for this agent (as single-element tensors)
            agent_metrics = {
                'Episode/length': torch.tensor([mean_length], dtype=torch.float32),
                'Episode/total_reward': torch.tensor([mean_reward], dtype=torch.float32),
                'Episode/avg_reward_per_step': torch.tensor([mean_avg_reward], dtype=torch.float32),
                'Episode/completed_episodes': torch.tensor([total_completed], dtype=torch.float32)
            }

            # Send directly to the specific agent's tracker
            self.trackers[agent_id].add_metrics(agent_metrics)

            # Clear the agent's episode data after sending
            agent_data['episode_lengths'].clear()
            agent_data['episode_rewards'].clear()
            agent_data['episode_avg_rewards'].clear()
            agent_data['completed_count'].clear()
            # Clear component rewards for this agent
            for component_key in agent_data['component_rewards']:
                agent_data['component_rewards'][component_key].clear()

    def _split_by_agent(self, metrics: Dict[str, torch.Tensor], tracker_method_name: str):
        """Split metrics by agent based on vector length and call specified tracker method."""
        if metrics == {}:
            for tracker in self.trackers:
                getattr(tracker,tracker_method_name)()
            return
        
        for name, values in metrics.items():
            if isinstance(values, torch.Tensor):
                # Handle 0-dimensional tensors (scalars)
                if values.dim() == 0:
                    # Scalar tensor - broadcast to all agents
                    for tracker in self.trackers:
                        getattr(tracker, tracker_method_name)({name: values})
                elif len(values) == self.num_envs:
                    # Split by environment assignment to agents
                    for i, tracker in enumerate(self.trackers):
                        start_idx = i * self.envs_per_agent
                        end_idx = (i + 1) * self.envs_per_agent
                        getattr(tracker, tracker_method_name)({name: values[start_idx:end_idx]})
                elif len(values) == self.num_agents:
                    # Direct agent assignment
                    for i, tracker in enumerate(self.trackers):
                        getattr(tracker, tracker_method_name)({name: values[i:i+1]})  # Keep as tensor
                else:
                    # Other size - broadcast to all
                    for tracker in self.trackers:
                        getattr(tracker, tracker_method_name)({name: values})
            else:
                # Non-tensor scalar - broadcast to all agents
                for tracker in self.trackers:
                    getattr(tracker, tracker_method_name)({name: values})

    def _extract_component_rewards(self) -> Dict[str, torch.Tensor]:
        """
        Extract component rewards from environment extras.

        Looks for keys starting with 'logs_rew_' in self.unwrapped.extras
        and transforms them into component reward metrics.

        Returns:
            Dictionary of component reward metrics with format 'Rewards/{component_name}' -> tensor
        """
        component_rewards = {}

        if not hasattr(self.unwrapped, 'extras') or not self.unwrapped.extras:
            return component_rewards

        # Look for reward component keys starting with 'logs_rew_'
        for key, value in self.unwrapped.extras.items():
            if key.startswith('logs_rew_'):
                # Extract component name (remove 'logs_rew_' prefix)
                component_name = key[9:]  # Remove 'logs_rew_' prefix

                # Convert value to tensor if needed
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, device=self.device)

                # Ensure tensor is on correct device
                if value.device != self.device:
                    value = value.to(self.device)

                # If scalar, broadcast to all environments
                if value.dim() == 0:
                    value = value.expand(self.num_envs)
                elif len(value) == 1:
                    value = value.expand(self.num_envs)

                # Store with standardized naming
                component_rewards[f"Rewards/{component_name}"] = value

        return component_rewards

    def add_metrics(self, metrics: Dict[str, torch.Tensor]):
        """
        Add metrics from other wrappers. Automatically splits by agent.

        Args:
            metrics: Dictionary of metric_name -> tensor with length num_envs, num_agents, or scalar
        """
        # Validate all metrics are tensors
        for name, values in metrics.items():
            if not isinstance(values, torch.Tensor):
                raise TypeError(
                    f"Metric '{name}' must be a torch.Tensor, got {type(values)}. "
                    f"Convert lists to tensors using: torch.tensor(your_list). "
                    f"This error occurred in GenericWandbLoggingWrapper.add_metrics()"
                )

        self._split_by_agent(metrics, 'add_metrics')

    def publish(self, onetime_metrics: Dict[str, torch.Tensor] = {}):
        """
        Add onetime metrics and publish everything to wandb.

        Args:
            onetime_metrics: Final onetime metrics to add before publishing
        """
        # Validate onetime_metrics are tensors
        for name, value in onetime_metrics.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Onetime metric '{name}' must be a torch.Tensor, got {type(value)}. "
                    f"Convert to tensor using: torch.tensor(your_value). "
                    f"This error occurred in GenericWandbLoggingWrapper.publish()"
                )

        self._split_by_agent(onetime_metrics, 'publish')

    def step(self, action):
        """Step environment and track episode metrics."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Check for pending factory metrics from FactoryMetricsWrapper
        if hasattr(self.env, '_pending_factory_metrics'):
            factory_metrics = self.env._pending_factory_metrics
            self.add_metrics(factory_metrics)
            # Clear the pending metrics to avoid double-processing
            delattr(self.env, '_pending_factory_metrics')

        # Extract and collect component rewards
        component_rewards = self._extract_component_rewards()
        if component_rewards:
            # Send component rewards to trackers immediately
            self.add_metrics(component_rewards)

        # Track current episode metrics
        self.current_episode_rewards += reward
        self.current_episode_lengths += 1

        # Increment step counters for all trackers
        for tracker in self.trackers:
            tracker.increment_steps()

        # Check for episode endings
        any_ended = torch.any(terminated | truncated)
        if any_ended:
            # Store completed episodes (terminated OR full-length truncated)
            completed_mask = terminated | (truncated & (self.current_episode_lengths >= self.max_episode_length))
            self._store_completed_episodes(completed_mask)

            # Reset tracking for ALL ended episodes (terminated or truncated)
            reset_mask = terminated | truncated
            self.current_episode_rewards[reset_mask] = 0
            self.current_episode_lengths[reset_mask] = 0

            # Send aggregated metrics when ANY environment truncates
            any_truncated = torch.any(truncated)
            if any_truncated:
                self._send_aggregated_episode_metrics()

        return obs, reward, terminated, truncated, info

    def close(self):
        """Close all wandb runs."""
        for tracker in self.trackers:
            tracker.close()