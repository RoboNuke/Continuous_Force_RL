"""
Simple Wandb Logging Wrapper

This wrapper provides simple, function-based Wandb logging.
Other wrappers call its functions to add metrics, and it handles all the complexity.
"""

import torch
import gymnasium as gym
import wandb
import copy
import yaml
import os
from typing import Dict, Any, Optional, List, Union


# Global tracking to ensure artifacts are uploaded only once per config path
_uploaded_artifacts = set()


def _extract_base_config_path(experiment_config_path: str) -> Optional[str]:
    """
    Extract base_config path from experiment YAML file.

    Args:
        experiment_config_path: Path to the experiment configuration YAML file

    Returns:
        Path to base configuration file if it exists, None otherwise
    """
    try:
        if not os.path.exists(experiment_config_path):
            return None

        with open(experiment_config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        base_config_path = config.get('base_config')
        if base_config_path:
            # Handle relative paths - if relative, treat as relative to current working directory
            if not os.path.isabs(base_config_path):
                # Don't modify relative paths - they should be relative to the project root
                base_config_path = os.path.normpath(base_config_path)
            return base_config_path

        return None
    except Exception as e:
        print(f"Warning: Could not extract base_config path from {experiment_config_path}: {e}")
        return None


class SimpleEpisodeTracker:
    """Simple episode tracker that accumulates metrics and publishes to wandb."""

    def __init__(self, num_envs: int, device: torch.device, agent_config: Dict[str, Any], env_config: Any, config_path: Optional[str] = None):
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

        # Upload YAML configuration artifacts
        self._upload_config_artifacts(config_path)

        # Metric storage
        self.accumulated_metrics = {}
        self.episode_count = 0

        # Step tracking for x-axis
        self.env_steps = 0  # Number of environment steps taken
        self.total_steps = 0  # env_steps * num_envs

    def _upload_config_artifacts(self, config_path: Optional[str]) -> None:
        """
        Upload YAML configuration files as wandb artifacts.
        Only uploads once per unique config path to avoid duplicates.

        Args:
            config_path: Path to the experiment configuration file
        """
        if not config_path or not os.path.exists(config_path):
            print(f"Warning: Config path not provided or doesn't exist: {config_path}")
            return

        # Check if artifacts for this config path have already been uploaded
        config_key = os.path.abspath(config_path)
        if config_key in _uploaded_artifacts:
            print(f"Config artifacts already uploaded for: {config_path}")
            return

        try:
            # Create configuration artifact
            config_artifact = wandb.Artifact(
                name="configuration",
                type="config",
                description="Experiment configuration files"
            )

            # Upload experiment configuration
            config_artifact.add_file(
                local_path=config_path,
                name="experiment_config.yaml"
            )
            print(f"Added experiment config: {config_path}")

            # Upload base configuration if it exists
            base_config_path = _extract_base_config_path(config_path)
            if base_config_path and os.path.exists(base_config_path):
                config_artifact.add_file(
                    local_path=base_config_path,
                    name="base_config.yaml"
                )
                print(f"Added base config: {base_config_path}")

            # Log the artifact to wandb
            self.run.log_artifact(config_artifact)
            print("Successfully uploaded configuration artifacts to wandb")

            # Mark this config path as uploaded to prevent duplicates
            _uploaded_artifacts.add(config_key)

        except Exception as e:
            print(f"Warning: Failed to upload configuration artifacts: {e}")

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

    def __init__(self, env: gym.Env, num_agents: int = 1, env_cfg: Any = None, config_path: Optional[str] = None):
        """
        Initialize simple Wandb logging wrapper.

        Args:
            env: Base environment to wrap
            num_agents: Number of agents for static assignment
            env_cfg: Environment configuration - REQUIRED, contains agent_configs
            config_path: Path to the experiment configuration file (for artifact upload)
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
            tracker = SimpleEpisodeTracker(self.envs_per_agent, self.device, agent_config, self.clean_env_cfg, config_path)
            self.trackers.append(tracker)

        # Episode tracking for basic metrics
        self.max_episode_length = getattr(env.unwrapped, 'max_episode_length', 1000)

        # Current episode tracking (per environment)
        self.current_episode_rewards = torch.zeros(self.num_envs, device=self.device)

        # Store original _reset_idx method for wrapper chaining
        self._original_reset_idx = None
        if hasattr(self.unwrapped, '_reset_idx'):
            self._original_reset_idx = self.unwrapped._reset_idx
            self.unwrapped._reset_idx = self._wrapped_reset_idx

        # Per-agent completed episode data storage
        self.agent_episode_data = {}
        for i in range(self.num_agents):
            self.agent_episode_data[i] = {
                'episode_lengths': [],
                'episode_rewards': [],
                'episode_avg_rewards': [],
                'completed_count': [],
                'terminations': [],  
                'component_rewards': {}  # Will store component reward lists dynamically
            }

        # Component reward tracking
        self.component_reward_keys = set()  # Discovered component reward keys
        self.current_component_rewards = torch.zeros((self.num_envs, 0), device=self.device)  # Will expand as needed

        # Flag to prevent publishing on first reset (all zeros)
        self._first_reset = True

    def _store_completed_episodes(self, completed_mask, is_termination=False):
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
            episode_length = getattr(self.unwrapped, 'episode_length_buf', torch.zeros(self.num_envs, device=self.device, dtype=torch.long))[env_idx].item()
            episode_reward = self.current_episode_rewards[env_idx].item()
            episode_avg_reward = episode_reward / episode_length if episode_length > 0 else 0.0

            # Store in agent's episode lists
            self.agent_episode_data[agent_id]['episode_lengths'].append(episode_length)
            self.agent_episode_data[agent_id]['episode_rewards'].append(episode_reward)
            self.agent_episode_data[agent_id]['episode_avg_rewards'].append(episode_avg_reward)
            self.agent_episode_data[agent_id]['completed_count'].append(1.0)

            # Add termination tracking - 1.0 if termination, 0.0 if timeout
            self.agent_episode_data[agent_id]['terminations'].append(1.0 if is_termination else 0.0)

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
            terms = sum(agent_data['terminations'])

            # Create metrics for this agent (as single-element tensors)
            agent_metrics = {
                'Episode/length': torch.tensor([mean_length], dtype=torch.float32),
                'Episode/total_reward': torch.tensor([mean_reward], dtype=torch.float32),
                'Episode/avg_reward_per_step': torch.tensor([mean_avg_reward], dtype=torch.float32),
                'Episode/completed_episodes': torch.tensor([total_completed], dtype=torch.float32),
                'Episode/terminations': torch.tensor([terms],dtype=torch.float32)
            }

            # Send directly to the specific agent's tracker
            self.trackers[agent_id].add_metrics(agent_metrics)

            # Clear the agent's episode data after sending
            agent_data['episode_lengths'].clear()
            agent_data['episode_rewards'].clear()
            agent_data['episode_avg_rewards'].clear()
            agent_data['completed_count'].clear()
            agent_data['terminations'].clear()
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

        # Increment step counters for all trackers
        for tracker in self.trackers:
            tracker.increment_steps()

        return obs, reward, terminated, truncated, info

    def _wrapped_reset_idx(self, env_ids):
        """Handle environment resets via _reset_idx calls."""

        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        # Skip processing on first reset (all data would be zeros)
        if self._first_reset:
            self._first_reset = False
        else:
            if len(env_ids) == self.num_envs:
                # Full reset - all environments hit max_steps
                # All environments that hit max_steps are completed episodes
                episode_lengths = getattr(self.unwrapped, 'episode_length_buf', torch.zeros(self.num_envs, device=self.device, dtype=torch.long))
                max_step_mask = episode_lengths >= self.max_episode_length
                max_step_env_ids = max_step_mask.nonzero(as_tuple=False).squeeze(-1).tolist()


                if torch.any(max_step_mask):
                    self._store_completed_episodes(max_step_mask, is_termination=False)

                # Publish all accumulated metrics and reset all tracking
                self._send_aggregated_episode_metrics()
                self._reset_all_tracking_variables()
            else:
                # Partial reset - these environments terminated
                env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids

                # Store all environments in env_ids as completed episodes (these are terminations)
                completed_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                completed_mask[env_ids] = True
                self._store_completed_episodes(completed_mask, is_termination=True)

                # Reset env-specific tracking only for completed environments
                self.current_episode_rewards[env_ids] = 0

        # Call original _reset_idx to maintain wrapper chain
        if self._original_reset_idx is not None:
            self._original_reset_idx(env_ids)

    def reset(self, **kwargs):
        """Reset environment - now just calls super().reset()."""
        return super().reset(**kwargs)

    def _reset_all_tracking_variables(self):
        """Reset all tracking variables to initial state."""
        # Reset current episode tracking
        self.current_episode_rewards.fill_(0)

        # Clear all agent episode data
        for agent_id in range(self.num_agents):
            agent_data = self.agent_episode_data[agent_id]
            agent_data['episode_lengths'].clear()
            agent_data['episode_rewards'].clear()
            agent_data['episode_avg_rewards'].clear()
            agent_data['completed_count'].clear()
            agent_data['terminations'].clear()
            # Clear component rewards for this agent
            for component_key in agent_data['component_rewards']:
                agent_data['component_rewards'][component_key].clear()

    def close(self):
        """Close all wandb runs."""
        for tracker in self.trackers:
            tracker.close()