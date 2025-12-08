"""
Simple Wandb Logging Wrapper

This wrapper provides simple, function-based Wandb logging.
Other wrappers call its functions to add metrics, and it handles all the complexity.
"""

import torch
import gymnasium as gym
import wandb
from typing import Dict, Any


class SimpleEpisodeTracker:
    """Simple episode tracker that accumulates metrics and publishes to wandb."""

    def __init__(
            self,
            num_envs: int,
            device: torch.device,
            agent_config: Dict[str, Any],
            all_configs: Dict[str, Any],
            config_paths: Dict[str, str],
            disable_logging: bool = False
        ):
        """Initialize simple episode tracker."""
        self.num_envs = num_envs
        self.device = device
        self.disable_logging = disable_logging
        self.config_paths = config_paths

        # Build complete config dict from all sections
        complete_config = {}
        for key in ['primary', 'environment', 'model', 'wrappers', 'experiment', 'agent']:
            if key in all_configs:
                complete_config[key] = self._convert_to_dict(all_configs[key])

        # Remove agent_exp_cfgs from agent dict to avoid uploading all agents' configs
        if 'agent' in complete_config and 'agent_exp_cfgs' in complete_config['agent']:
            del complete_config['agent']['agent_exp_cfgs']

        # Add agent-specific config
        complete_config['agent_specific'] = self._convert_to_dict(agent_config)

        # Only initialize wandb if logging is enabled
        if not self.disable_logging:
            wandb_kwargs = agent_config['experiment']['wandb_kwargs']
            self.run = wandb.init(
                entity=wandb_kwargs.get('entity'),
                project=wandb_kwargs.get('project'),
                name=wandb_kwargs.get('run_name'),
                reinit="create_new",
                config=complete_config,
                group=wandb_kwargs.get('group'),
                tags=wandb_kwargs.get('tags'),
                #settings=wandb.Settings(
                #    _disable_stats=True,  # Reduce wandb overhead
                #    _disable_meta=True,   # Reduce metadata collection
                #    console="off",        # Reduce console spam
                #    _service_wait=300     # Longer service timeout
                #)

            )

            # Store run directory path for cleanup
            self.run_dir = self.run.dir

            # Upload config YAML files to WandB
            print(f"  Uploading config files to WandB...")
            self.upload_config_files()
        else:
            self.run = None

        # Metric storage
        self.accumulated_metrics = {}
        self.episode_count = 0

        # Step tracking for x-axis
        self.env_steps = 0  # Number of environment steps taken
        self.total_steps = 0  # env_steps * num_envs

    def _convert_to_dict(self, obj, max_depth=10, _current_depth=0):
        """
        Recursively convert Python objects to plain dictionaries for wandb.

        Args:
            obj: Object to convert
            max_depth: Maximum recursion depth to prevent infinite loops
            _current_depth: Internal parameter for tracking recursion depth

        Returns:
            Plain dictionary, list, or primitive type suitable for wandb
        """
        # Prevent infinite recursion
        if _current_depth > max_depth:
            return str(obj)

        # Handle None
        if obj is None:
            return None

        # Handle primitive types
        if isinstance(obj, (int, float, str, bool)):
            return obj

        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [self._convert_to_dict(item, max_depth, _current_depth + 1) for item in obj]

        # Handle dictionaries
        if isinstance(obj, dict):
            return {
                key: self._convert_to_dict(value, max_depth, _current_depth + 1)
                for key, value in obj.items()
            }

        # Handle special types that need string conversion
        # torch.device, torch.dtype, pathlib.Path, etc.
        if type(obj).__name__ in ['device', 'dtype', 'PosixPath', 'WindowsPath', 'Path']:
            return str(obj)

        # Handle objects with __dict__ (dataclasses, custom classes, etc.)
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                # Skip callables (methods, functions, classes)
                if callable(value):
                    continue
                # Skip class types
                if isinstance(value, type):
                    continue
                # Convert the value recursively
                result[key] = self._convert_to_dict(value, max_depth, _current_depth + 1)
            return result

        # Fallback: convert to string representation
        return str(obj)

    def increment_steps(self):
        """Increment step counters for this agent."""
        self.env_steps += 1
        self.total_steps = self.env_steps * self.num_envs

    def upload_checkpoint(self, checkpoint_path: str, checkpoint_type: str) -> bool:
        """Upload checkpoint file to WandB with standardized naming and directory structure.

        Uploads checkpoints to:
        - ckpts/policies/{step}.pt
        - ckpts/critics/{step}.pt

        Note: Agent index is removed from filenames since each WandB run corresponds to a single agent.

        Args:
            checkpoint_path: Full path to checkpoint file (e.g., agent_{i}_{step}.pt or critic_{i}_{step}.pt)
            checkpoint_type: Type identifier ('policy' or 'critic')

        Returns:
            bool: True if upload succeeded, False otherwise

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint filename format is invalid
            RuntimeError: If upload fails after retries
        """
        # Skip upload if logging is disabled
        if self.disable_logging:
            return True

        import os
        import re
        import shutil
        import time
        from requests.exceptions import ReadTimeout, ConnectionError, Timeout

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Extract step number from filename
        # Expected format: agent_{agent_idx}_{step}.pt or critic_{agent_idx}_{step}.pt
        filename = os.path.basename(checkpoint_path)
        match = re.search(r'(?:agent|critic)_\d+_(\d+)\.pt$', filename)
        if not match:
            raise ValueError(
                f"Invalid checkpoint filename format: {filename}. "
                f"Expected format: agent_{{agent_idx}}_{{step}}.pt or critic_{{agent_idx}}_{{step}}.pt"
            )

        step_number = int(match.group(1))

        # Determine subdirectory based on checkpoint type
        if checkpoint_type == 'policy':
            subdir = 'policies'
        elif checkpoint_type == 'critic':
            subdir = 'critics'
        else:
            raise ValueError(f"Invalid checkpoint_type: {checkpoint_type}. Must be 'policy' or 'critic'")

        # Create directory structure in wandb.run.dir
        ckpts_base_dir = os.path.join(self.run.dir, "ckpts")
        target_dir = os.path.join(ckpts_base_dir, subdir)
        os.makedirs(target_dir, exist_ok=True)

        # Create filename without agent index (since each run is a single agent)
        new_filename = f"{step_number}.pt"
        target_path = os.path.join(target_dir, new_filename)

        # Copy file to wandb run directory
        shutil.copy2(checkpoint_path, target_path)

        # Upload with retry logic for timeout handling
        max_upload_retries = 3
        upload_retry_delay = 5.0

        for attempt in range(max_upload_retries):
            try:
                # Use self.run.save() with base_path to preserve ckpts/ directory structure
                self.run.save(target_path, base_path=self.run.dir, policy="now")
                return True
            except (ReadTimeout, ConnectionError, Timeout, TimeoutError) as e:
                if attempt < max_upload_retries - 1:
                    print(f"    Upload timeout for {checkpoint_type}, retrying in {upload_retry_delay:.1f}s (attempt {attempt+1}/{max_upload_retries-1})...")
                    time.sleep(upload_retry_delay)
                    continue
                else:
                    raise RuntimeError(f"Failed to upload {checkpoint_type} checkpoint after {max_upload_retries} attempts: {e}")

        return False  # Should not reach here

    def upload_config_files(self) -> bool:
        """Upload config YAML files to WandB run.

        Uses self.config_paths dict with 'base' and optionally 'exp' keys.

        Returns:
            bool: True if successful

        Raises:
            RuntimeError: If config files don't exist or upload fails
        """
        import os
        import shutil

        # Validate config_paths structure
        if 'base' not in self.config_paths:
            raise RuntimeError("config_paths missing required 'base' key")

        # Upload base config
        base_path = self.config_paths['base']
        if not os.path.exists(base_path):
            raise RuntimeError(f"Base config file not found: {base_path}")

        target_base = os.path.join(self.run.dir, 'config_base.yaml')
        shutil.copy(base_path, target_base)
        self.run.save(target_base, base_path=self.run.dir, policy="now")
        print(f"    Uploaded base config: {base_path} -> config_base.yaml")

        # Upload experiment config if present
        if 'exp' in self.config_paths:
            exp_path = self.config_paths['exp']
            if not os.path.exists(exp_path):
                raise RuntimeError(f"Experiment config file not found: {exp_path}")

            target_exp = os.path.join(self.run.dir, 'config_experiment.yaml')
            shutil.copy(exp_path, target_exp)
            self.run.save(target_exp, base_path=self.run.dir, policy="now")
            print(f"    Uploaded experiment config: {exp_path} -> config_experiment.yaml")

        return True

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

        # Process regular metrics (take mean or max across all accumulated values)
        for name, value_list in self.accumulated_metrics.items():
            if value_list:
                stacked = torch.stack(value_list)
                # Filter out NaN values (from blocked agents in KL masking)
                valid_mask = ~torch.isnan(stacked)
                if valid_mask.any():
                    valid_values = stacked[valid_mask]
                    # Use max aggregation for metrics with 'max' in name, mean for others
                    if 'max' in name:
                        result = valid_values.max().item()
                    elif 'summed_total' in name:
                        result = valid_values.sum().item()
                    else:
                        result = valid_values.mean().item()
                    final_metrics[name] = result
                # If all values are NaN, skip this metric entirely

        # Add onetime metrics (now guaranteed to be tensors)
        for name, value in onetime_metrics.items():
            result = value.mean().item()
            final_metrics[f"learning/{name}"] = result

        # Add required step metrics for x-axis
        final_metrics["total_steps"] = self.total_steps
        final_metrics["env_steps"] = self.env_steps

        # Publish to wandb only if logging is enabled
        if not self.disable_logging:
            self.run.log(final_metrics)

        # Clear accumulated data
        self.accumulated_metrics.clear()

    def close(self, delete_local_files=True):
        """Close wandb run and optionally delete local files.

        Args:
            delete_local_files: If True, delete local run directory after syncing (default: True)
        """
        if hasattr(self, 'run') and self.run is not None:
            try:
                # Finish run (syncs data to cloud)
                if not self.run._is_finished:
                    self.run.finish()
            except Exception as e:
                print(f"[WARNING]: Error finishing wandb run: {e}")
            finally:
                self.run = None  # Prevent double-finish

        # Delete local files after finishing
        if delete_local_files and hasattr(self, 'run_dir'):
            import shutil
            import os
            try:
                if os.path.exists(self.run_dir):
                    shutil.rmtree(self.run_dir, ignore_errors=True)
            except Exception as e:
                print(f"[WARNING]: Could not delete {self.run_dir}: {e}")


class GenericWandbLoggingWrapper(gym.Wrapper):
    """
    Simple Wandb logging wrapper that provides metric collection functions.
    Other wrappers call these functions to add their metrics.
    """

    def __init__(self, env: gym.Env, num_agents: int = 1, all_configs: Dict[str, Any] = None):
        """
        Initialize simple Wandb logging wrapper.

        Args:
            env: Base environment to wrap
            num_agents: Number of agents for static assignment
            all_configs: Complete configuration dict - REQUIRED, contains all config sections
        """
        super().__init__(env)

        if all_configs is None:
            raise ValueError("all_configs is required and cannot be None")

        # Extract agent configs from all_configs
        if 'agent' not in all_configs or not hasattr(all_configs['agent'], 'agent_exp_cfgs'):
            raise ValueError("all_configs must contain 'agent' section with agent_exp_cfgs")

        agent_configs = all_configs['agent'].agent_exp_cfgs

        # Extract config flags from wrappers config
        disable_logging = False
        track_rewards_by_outcome = False
        if 'wrappers' in all_configs and hasattr(all_configs['wrappers'], 'wandb_logging'):
            disable_logging = getattr(all_configs['wrappers'].wandb_logging, 'disable_logging', False)
            track_rewards_by_outcome = getattr(all_configs['wrappers'].wandb_logging, 'track_rewards_by_outcome', False)

        self.num_agents = num_agents
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.track_rewards_by_outcome = track_rewards_by_outcome

        # Validate agent assignment
        if self.num_envs % self.num_agents != 0:
            raise ValueError(f"Number of environments ({self.num_envs}) must be divisible by number of agents ({self.num_agents})")

        self.envs_per_agent = self.num_envs // self.num_agents

        # Create episode trackers for each agent
        self.trackers = []
        for i in range(self.num_agents):
            agent_config = agent_configs[i]
            tracker = SimpleEpisodeTracker(
                self.envs_per_agent,
                self.device,
                agent_config,
                all_configs,
                all_configs.get('config_paths', {}),  # Pass config_paths explicitly
                disable_logging
            )
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
                'component_rewards': {},  # Will store component reward lists dynamically
                'component_rewards_success': {},  # Component rewards for successful episodes
                'component_rewards_failure': {},  # Component rewards for failed episodes
                'component_rewards_timeout': {}  # Component rewards for timed-out episodes
            }

        # Component reward tracking
        self.component_reward_keys = set()  # Discovered component reward keys
        self.current_component_rewards = torch.zeros((self.num_envs, 0), device=self.device)  # Will expand as needed

        # Flag to prevent publishing on first reset (all zeros)
        self._first_reset = True

        # Initialize to_log dictionary for wrappers to publish arbitrary metrics
        self.unwrapped.extras['to_log'] = {}

        # Component reward tracking - accumulate per environment during episode
        self.current_component_rewards = {}  # Will be populated dynamically: {component_name: torch.zeros(num_envs)}

    def _calculate_std(self, values_list):
        """Calculate sample standard deviation from list."""
        if len(values_list) <= 1:
            return 0.0
        mean = sum(values_list) / len(values_list)
        variance = sum((x - mean) ** 2 for x in values_list) / (len(values_list) - 1)
        return variance ** 0.5

    def _find_factory_metrics_wrapper(self):
        """Find FactoryMetricsWrapper in the wrapper chain."""
        current = self.env
        while hasattr(current, 'env'):
            if hasattr(current, 'ep_succeeded'):  # FactoryMetricsWrapper has this attribute
                return current
            current = current.env
        return None

    def _store_completed_episodes(self, completed_mask, is_termination=False):
        """Store completed episode data in agent lists."""
        if not torch.any(completed_mask):
            return

        completed_indices = completed_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(completed_indices.shape) == 0:  # Handle single element case
            completed_indices = completed_indices.unsqueeze(0)

        # Find factory_metrics_wrapper for success tracking
        factory_metrics_wrapper = None
        if self.track_rewards_by_outcome:
            factory_metrics_wrapper = self._find_factory_metrics_wrapper()

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

            # Store component reward episode sums
            for component_name, accumulated_values in self.current_component_rewards.items():
                reward_value = accumulated_values[env_idx].item()

                # Overall tracking (existing)
                if component_name not in self.agent_episode_data[agent_id]['component_rewards']:
                    self.agent_episode_data[agent_id]['component_rewards'][component_name] = []
                self.agent_episode_data[agent_id]['component_rewards'][component_name].append(reward_value)

                # Split by outcome (success/failure/timeout)
                if self.track_rewards_by_outcome:
                    if is_termination:
                        # Episode terminated early - check if success or failure
                        is_success = False
                        if factory_metrics_wrapper is not None:
                            is_success = factory_metrics_wrapper.ep_succeeded[env_idx].item()

                        if is_success:
                            # Successful termination
                            if component_name not in self.agent_episode_data[agent_id]['component_rewards_success']:
                                self.agent_episode_data[agent_id]['component_rewards_success'][component_name] = []
                            self.agent_episode_data[agent_id]['component_rewards_success'][component_name].append(reward_value)
                        else:
                            # Failed termination (peg break, etc.)
                            if component_name not in self.agent_episode_data[agent_id]['component_rewards_failure']:
                                self.agent_episode_data[agent_id]['component_rewards_failure'][component_name] = []
                            self.agent_episode_data[agent_id]['component_rewards_failure'][component_name].append(reward_value)
                    else:
                        # Episode timed out (hit max_episode_length)
                        if component_name not in self.agent_episode_data[agent_id]['component_rewards_timeout']:
                            self.agent_episode_data[agent_id]['component_rewards_timeout'][component_name] = []
                        self.agent_episode_data[agent_id]['component_rewards_timeout'][component_name].append(reward_value)

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
            std_reward = self._calculate_std(agent_data['episode_rewards'])
            mean_avg_reward = sum(agent_data['episode_avg_rewards']) / len(agent_data['episode_avg_rewards'])
            total_completed = sum(agent_data['completed_count'])
            terms = sum(agent_data['terminations'])

            # Create metrics for this agent (as single-element tensors)
            agent_metrics = {
                'Episode/length': torch.tensor([mean_length], dtype=torch.float32),
                'Episode/total_reward': torch.tensor([mean_reward], dtype=torch.float32),
                'Episode/total_reward_std': torch.tensor([std_reward], dtype=torch.float32),
                'Episode/avg_reward_per_step': torch.tensor([mean_avg_reward], dtype=torch.float32),
                'Episode/completed_episodes': torch.tensor([total_completed], dtype=torch.float32),
                'Episode/terminations': torch.tensor([terms],dtype=torch.float32)
            }

            # Add component reward statistics
            for component_name, component_values in agent_data['component_rewards'].items():
                if component_values:  # Only if we have data
                    mean_component = sum(component_values) / len(component_values)
                    std_component = self._calculate_std(component_values)

                    agent_metrics[f'Rewards/{component_name}'] = torch.tensor([mean_component], dtype=torch.float32)
                    agent_metrics[f'Rewards/{component_name}_std'] = torch.tensor([std_component], dtype=torch.float32)

            # Add success/failure/timeout split metrics
            if self.track_rewards_by_outcome:
                # Success rewards
                for component_name, component_values in agent_data['component_rewards_success'].items():
                    if component_values:
                        mean_success = sum(component_values) / len(component_values)
                        agent_metrics[f'Reward(success)/{component_name}'] = torch.tensor([mean_success], dtype=torch.float32)

                # Failure rewards
                for component_name, component_values in agent_data['component_rewards_failure'].items():
                    if component_values:
                        mean_failure = sum(component_values) / len(component_values)
                        agent_metrics[f'Reward(failure)/{component_name}'] = torch.tensor([mean_failure], dtype=torch.float32)

                # Timeout rewards
                for component_name, component_values in agent_data['component_rewards_timeout'].items():
                    if component_values:
                        mean_timeout = sum(component_values) / len(component_values)
                        agent_metrics[f'Reward(timeout)/{component_name}'] = torch.tensor([mean_timeout], dtype=torch.float32)

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
            # Clear success/failure/timeout tracking
            if self.track_rewards_by_outcome:
                for component_key in agent_data['component_rewards_success']:
                    agent_data['component_rewards_success'][component_key].clear()
                for component_key in agent_data['component_rewards_failure']:
                    agent_data['component_rewards_failure'][component_key].clear()
                for component_key in agent_data['component_rewards_timeout']:
                    agent_data['component_rewards_timeout'][component_key].clear()

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

    def _extract_to_log_metrics(self) -> Dict[str, torch.Tensor]:
        """
        Extract metrics from environment extras['to_log'] dictionary.

        Looks for the 'to_log' key in self.unwrapped.extras and extracts
        all key-value pairs as metrics.

        Returns:
            Dictionary of metrics with format '{metric_name}' -> tensor
        """
        to_log_metrics = {}

        if not hasattr(self.unwrapped, 'extras') or not self.unwrapped.extras:
            return to_log_metrics

        # Check if 'to_log' key exists
        if 'to_log' not in self.unwrapped.extras:
            return to_log_metrics

        # Extract all metrics from to_log dictionary
        for key, value in self.unwrapped.extras['to_log'].items():
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

            # Store with metric name as provided
            to_log_metrics[key] = value

        return to_log_metrics

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

    def upload_checkpoint(self, agent_id: int, checkpoint_path: str, checkpoint_type: str) -> bool:
        """Delegate checkpoint upload to specific agent's tracker.

        Args:
            agent_id: Index of agent (0 to num_agents-1)
            checkpoint_path: Full path to checkpoint file
            checkpoint_type: Type identifier ('policy' or 'critic')

        Returns:
            bool: True if upload succeeded, False otherwise
        """
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(f"Invalid agent_id {agent_id}, must be in range [0, {self.num_agents})")
        return self.trackers[agent_id].upload_checkpoint(checkpoint_path, checkpoint_type)

    def step(self, action):
        """Step environment and track episode metrics."""
        obs, reward, terminated, truncated, info = super().step(action)
        # Check for pending factory metrics from FactoryMetricsWrapper
        if hasattr(self.env, '_pending_factory_metrics'):
            factory_metrics = self.env._pending_factory_metrics
            self.add_metrics(factory_metrics)
            # Clear the pending metrics to avoid double-processing
            delattr(self.env, '_pending_factory_metrics')

        # Extract and accumulate component rewards per environment
        component_rewards = self._extract_component_rewards()
        if component_rewards:
            # Accumulate per environment instead of sending immediately
            for key, value in component_rewards.items():
                # Extract clean component name (remove 'Rewards/' prefix)
                component_name = key.replace('Rewards/', '')

                # Initialize accumulator if first time seeing this component
                if component_name not in self.current_component_rewards:
                    self.current_component_rewards[component_name] = torch.zeros(self.num_envs, device=self.device)

                # Accumulate step reward
                self.current_component_rewards[component_name] += value

        # Extract and collect to_log metrics
        to_log_metrics = self._extract_to_log_metrics()
        if to_log_metrics:
            # Send to_log metrics to trackers immediately
            self.add_metrics(to_log_metrics)
            # Clear the to_log dictionary after extraction
            self.unwrapped.extras['to_log'].clear()

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
                # Use >= (max - 1) to handle 0-indexed episode counting (episodes complete at step 149 for max_length=150)
                max_step_mask = episode_lengths >= (self.max_episode_length - 1)
                #max_step_env_ids = max_step_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

                if torch.any(max_step_mask):
                    self._store_completed_episodes(max_step_mask, is_termination=False)

                # Publish all accumulated metrics and reset all tracking
                self._send_aggregated_episode_metrics()
                self._reset_all_tracking_variables()
            else:
                # Partial reset - these environments terminated
                #env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids

                # Store all environments in env_ids as completed episodes (these are terminations)
                completed_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                completed_mask[env_ids] = True
                self._store_completed_episodes(completed_mask, is_termination=True)

                # Reset env-specific tracking only for completed environments
                self.current_episode_rewards[env_ids] = 0

                # Reset component reward accumulators for completed environments
                for component_name in self.current_component_rewards:
                    self.current_component_rewards[component_name][env_ids] = 0

        # Call original _reset_idx to maintain wrapper chain
        if self._original_reset_idx is not None:
            self._original_reset_idx(env_ids)

        # Re-initialize to_log dictionary (defensive, in case extras was cleared)
        #self.unwrapped.extras['to_log'] = {}

    def reset(self, **kwargs):
        """Reset environment - now just calls super().reset()."""
        return super().reset(**kwargs)

    def _reset_all_tracking_variables(self):
        """Reset all tracking variables to initial state."""
        # Reset current episode tracking
        self.current_episode_rewards.fill_(0)

        # Reset component reward accumulators
        for component_name in self.current_component_rewards:
            self.current_component_rewards[component_name].fill_(0)

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
            # Clear success/failure/timeout tracking
            if self.track_rewards_by_outcome:
                for component_key in agent_data.get('component_rewards_success', {}):
                    agent_data['component_rewards_success'][component_key].clear()
                for component_key in agent_data.get('component_rewards_failure', {}):
                    agent_data['component_rewards_failure'][component_key].clear()
                for component_key in agent_data.get('component_rewards_timeout', {}):
                    agent_data['component_rewards_timeout'][component_key].clear()

    def close(self, delete_local_files=True):
        """Close all wandb runs and optionally delete local files.

        Args:
            delete_local_files: If True, delete local run directories after syncing (default: True)
        """
        import traceback

        print(f"[INFO]: Closing {len(self.trackers)} wandb tracker(s)...")

        for i, tracker in enumerate(self.trackers):
            try:
                print(f"[INFO]:   - Closing tracker {i+1}/{len(self.trackers)}... ", end="", flush=True)
                tracker.close(delete_local_files=delete_local_files)
                print("✓")
            except Exception as e:
                print(f"✗ Failed: {e}")
                traceback.print_exc()
                # Continue to next tracker even if this one fails

        print("[INFO]: All wandb trackers closed")