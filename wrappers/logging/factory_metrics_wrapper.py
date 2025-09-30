"""
Factory-Specific Metrics Wrapper

This wrapper adds factory task-specific metrics tracking including success rates,
engagement tracking, smoothness metrics, and force/torque statistics.

Simplified implementation following the same pattern as GenericWandbLoggingWrapper:
- Collect metrics step-by-step
- Aggregate per environment
- Send aggregated metrics when truncations occur
"""

import torch
import gymnasium as gym


class FactoryMetricsWrapper(gym.Wrapper):
    """
    Wrapper that adds factory-specific metrics tracking to environments.

    Features:
    - Success and engagement rate tracking
    - Smoothness metrics (sum squared velocity, force/torque statistics)
    - Multi-agent support with static environment assignment
    - Simple step-by-step collection following WandbWrapper pattern
    """

    def __init__(self, env, num_agents=1):
        """
        Initialize factory metrics wrapper.

        Args:
            env: Base environment to wrap (must have GenericWandbLoggingWrapper in chain)
            num_agents: Number of agents for static assignment
        """
        super().__init__(env)

        # Validate that we have access to add_metrics (WandbWrapper in chain)
        if not self._find_wandb_wrapper():
            raise ValueError(
                "Factory Metrics Wrapper requires GenericWandbLoggingWrapper to be applied first. "
                "Apply wrappers in order: base_env -> WandbWrapper -> FactoryMetricsWrapper"
            )

        self.num_agents = num_agents
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # Validate agent assignment
        if self.num_envs % self.num_agents != 0:
            raise ValueError(f"Number of environments ({self.num_envs}) must be divisible by number of agents ({self.num_agents})")

        self.envs_per_agent = self.num_envs // self.num_agents

        # Initialize tracking variables
        self._init_tracking_variables()

        # Check for force/torque capability
        # The ForceTorqueWrapper creates this attribute, but we need to check dynamically
        # during the first step since the wrapper may initialize lazily
        self.has_force_data = False  # Will be set properly in first step
        self._force_data_checked = False

        # Per-agent completed episode data storage (like WandbWrapper)
        self.agent_episode_data = {}
        for i in range(self.num_agents):
            self.agent_episode_data[i] = {
                'success_rates': [],
                'success_times': [],
                'engagement_rates': [],
                'engagement_times': [],
                'engagement_lengths': [],
                'ssv_values': [],
                'ssjv_values': [],
                # Force metrics will be added dynamically when force data is detected
                'max_forces': None,
                'max_torques': None,
                'avg_forces': None,
                'avg_torques': None,
            }

        # Episode tracking for episode length calculation
        self.max_episode_length = getattr(env.unwrapped, 'max_episode_length', 1000)

        # Wrapper initialization flag for compatibility with tests
        self._wrapper_initialized = hasattr(self.unwrapped, '_robot')

        # Flag to prevent publishing on first reset (all zeros)
        self._first_reset = True

        # Override the base environment's _update_rew_buf to use our version
        # This ensures per-environment reward components instead of averaged scalars
        if hasattr(self.unwrapped, '_update_rew_buf'):
            # Store reference to original method in case we need it
            self._original_update_rew_buf = self.unwrapped._update_rew_buf
            # Create a wrapper function that calls our method with the right context
            def patched_update_rew_buf(curr_successes):
                return self._update_rew_buf_for_env(self.unwrapped, curr_successes)
            # Replace the method
            self.unwrapped._update_rew_buf = patched_update_rew_buf

    @property
    def cfg(self):
        return self.unwrapped.cfg
    def _find_wandb_wrapper(self):
        """Find GenericWandbLoggingWrapper in the wrapper chain."""
        current = self.env
        while hasattr(current, 'env'):
            if hasattr(current, 'add_metrics'):
                return True
            current = current.env
        # Check the unwrapped environment too
        return hasattr(current, 'add_metrics')

    def _init_tracking_variables(self):
        """Initialize all tracking variables."""
        # Success tracking
        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # Engagement tracking
        self.ep_engaged = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.ep_engaged_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_engaged_length = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # Smoothness metrics
        self.ep_ssv = torch.zeros((self.num_envs,), device=self.device)  # Sum squared velocity
        self.ep_ssjv = torch.zeros((self.num_envs,), device=self.device)  # Sum squared joint velocity

        # Store original _reset_idx method for wrapper chaining
        self._original_reset_idx = None
        if hasattr(self.unwrapped, '_reset_idx'):
            self._original_reset_idx = self.unwrapped._reset_idx
            self.unwrapped._reset_idx = self._wrapped_reset_idx

    def _collect_step_metrics(self):
        """Collect metrics for this step."""
        # Check for force data on first step (lazy initialization)
        if not self._force_data_checked:
            self._check_force_data_availability()
            self._force_data_checked = True

        # Update smoothness metrics
        if hasattr(self.unwrapped, 'ee_linvel_fd'):
            self.ep_ssv += torch.linalg.norm(self.unwrapped.ee_linvel_fd, axis=1)

        # Update sum squared joint velocity from robot data
        if hasattr(self.unwrapped, '_robot') and hasattr(self.unwrapped._robot, 'data') and hasattr(self.unwrapped._robot.data, 'joint_vel'):
            joint_vel = self.unwrapped._robot.data.joint_vel
            self.ep_ssjv += torch.linalg.norm(joint_vel * joint_vel, axis=1)

        # Update force/torque metrics if available
        if self.has_force_data and hasattr(self.unwrapped, 'robot_force_torque'):
            force_magnitude = torch.linalg.norm(self.unwrapped.robot_force_torque[:, :3], axis=1)
            torque_magnitude = torch.linalg.norm(self.unwrapped.robot_force_torque[:, 3:], axis=1)

            self.ep_sum_force += force_magnitude
            self.ep_sum_torque += torque_magnitude
            self.ep_max_force = torch.max(self.ep_max_force, force_magnitude)
            self.ep_max_torque = torch.max(self.ep_max_torque, torque_magnitude)

        # Update success and engagement tracking
        self._update_success_engagement_tracking()

    def _check_force_data_availability(self):
        """Check if force data is available and initialize force tracking if needed."""
        if hasattr(self.unwrapped, 'robot_force_torque'):
            self.has_force_data = True
            # Initialize force tracking variables if not already done
            if not hasattr(self, 'ep_max_force'):
                self.ep_max_force = torch.zeros((self.num_envs,), device=self.device)
                self.ep_max_torque = torch.zeros((self.num_envs,), device=self.device)
                self.ep_sum_force = torch.zeros((self.num_envs,), device=self.device)
                self.ep_sum_torque = torch.zeros((self.num_envs,), device=self.device)

                # Update agent episode data to include force metrics
                for i in range(self.num_agents):
                    self.agent_episode_data[i]['max_forces'] = []
                    self.agent_episode_data[i]['max_torques'] = []
                    self.agent_episode_data[i]['avg_forces'] = []
                    self.agent_episode_data[i]['avg_torques'] = []

        else:
            self.has_force_data = False

    def _update_rew_buf_for_env(self, env, curr_successes):
        """Compute reward at current timestep."""
        # First, get base rewards from previous wrappers (e.g., HybridForcePositionWrapper)
        # This ensures we don't overwrite rewards added by other wrappers in the chain
        if self._original_update_rew_buf is not None:
            rew_buf = self._original_update_rew_buf(curr_successes)
        else:
            # Fallback: start with zero rewards if no original method exists
            rew_buf = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

        rew_dict = {}

        # Keypoint rewards.
        def squashing_fn(x, a, b):
            return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))

        a0, b0 = env.cfg_task.keypoint_coef_baseline
        rew_dict["kp_baseline"] = squashing_fn(env.keypoint_dist, a0, b0)
        # a1, b1 = 25, 2
        a1, b1 = env.cfg_task.keypoint_coef_coarse
        rew_dict["kp_coarse"] = squashing_fn(env.keypoint_dist, a1, b1)
        a2, b2 = env.cfg_task.keypoint_coef_fine
        # a2, b2 = 300, 0
        rew_dict["kp_fine"] = squashing_fn(env.keypoint_dist, a2, b2)

        # Action penalties.
        rew_dict["action_penalty"] = torch.norm(env.actions, p=2, dim=-1)
        rew_dict["action_grad_penalty"] = torch.norm(env.actions - env.prev_actions, p=2, dim=-1)
        rew_dict["curr_engaged"] = (
            env._get_curr_successes(success_threshold=env.cfg_task.engage_threshold, check_rot=False).clone().float()
        )
        rew_dict["curr_successes"] = curr_successes.clone().float()

        # Add factory-specific reward components to the base rewards
        rew_buf += (
            rew_dict["kp_coarse"]
            + rew_dict["kp_baseline"]
            + rew_dict["kp_fine"]
            - rew_dict["action_penalty"] * env.cfg_task.action_penalty_scale
            - rew_dict["action_grad_penalty"] * env.cfg_task.action_grad_penalty_scale
            + rew_dict["curr_engaged"]
            + rew_dict["curr_successes"]
        )

        for rew_name, rew in rew_dict.items():
            env.extras[f"logs_rew_{rew_name}"] = rew

        return rew_buf
    
    def _update_success_engagement_tracking(self):
        """Update success and engagement tracking."""
        # Get current successes and engagements if available
        curr_successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        curr_engaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Get from environment methods if available
        if hasattr(self.unwrapped, '_get_curr_successes') and hasattr(self.unwrapped, 'cfg_task'):
            if hasattr(self.unwrapped.cfg_task, 'success_threshold'):
                success_threshold = self.unwrapped.cfg_task.success_threshold
                check_rot = getattr(self.unwrapped.cfg_task, 'name', '') == "nut_thread"
                curr_successes = self.unwrapped._get_curr_successes(success_threshold, check_rot)

            if hasattr(self.unwrapped.cfg_task, 'engage_threshold'):
                engage_threshold = self.unwrapped.cfg_task.engage_threshold
                curr_engaged = self.unwrapped._get_curr_successes(engage_threshold, False)

        # Update success tracking
        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = True
        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        if len(first_success_ids) > 0 and hasattr(self.unwrapped, 'episode_length_buf'):
            self.ep_success_times[first_success_ids] = self.unwrapped.episode_length_buf[first_success_ids].clone()

        # Update engagement tracking
        first_engaged = torch.logical_and(curr_engaged, torch.logical_not(self.ep_engaged))
        first_engaged_ids = first_engaged.nonzero(as_tuple=False).squeeze(-1)
        if len(first_engaged_ids) > 0 and hasattr(self.unwrapped, 'episode_length_buf'):
            self.ep_engaged_times[first_engaged_ids] = self.unwrapped.episode_length_buf[first_engaged_ids].clone()

        self.ep_engaged[curr_engaged] = True
        self.ep_engaged_length[curr_engaged] += 1

    def _store_completed_episodes(self, completed_mask):
        """Store completed episode data in agent lists (like WandbWrapper)."""
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

            # Calculate episode metrics for this completed episode
            episode_length = getattr(self.unwrapped, 'episode_length_buf', torch.zeros(self.num_envs, device=self.device))[env_idx].item()

            # Normalize forces/torques by episode length
            decimation = getattr(self.unwrapped.cfg, 'decimation', 1)
            norm_factor = decimation * episode_length if episode_length > 0 else 1

            # Store factory metrics in agent's episode lists
            self.agent_episode_data[agent_id]['success_rates'].append(float(self.ep_succeeded[env_idx]))
            self.agent_episode_data[agent_id]['success_times'].append(float(self.ep_success_times[env_idx]))
            self.agent_episode_data[agent_id]['engagement_rates'].append(float(self.ep_engaged[env_idx]))
            self.agent_episode_data[agent_id]['engagement_times'].append(float(self.ep_engaged_times[env_idx]))
            self.agent_episode_data[agent_id]['engagement_lengths'].append(float(self.ep_engaged_length[env_idx]))
            self.agent_episode_data[agent_id]['ssv_values'].append(float(self.ep_ssv[env_idx]))
            self.agent_episode_data[agent_id]['ssjv_values'].append(float(self.ep_ssjv[env_idx]))

            if self.has_force_data and hasattr(self, 'ep_max_force'):
                max_force_val = float(self.ep_max_force[env_idx])
                avg_force_val = float(self.ep_sum_force[env_idx] / norm_factor)

                self.agent_episode_data[agent_id]['max_forces'].append(max_force_val)
                self.agent_episode_data[agent_id]['max_torques'].append(float(self.ep_max_torque[env_idx]))
                self.agent_episode_data[agent_id]['avg_forces'].append(avg_force_val)
                self.agent_episode_data[agent_id]['avg_torques'].append(float(self.ep_sum_torque[env_idx] / norm_factor))


    def _send_aggregated_factory_metrics(self):
        """Send aggregated factory metrics for all agents (like WandbWrapper)."""
        # Collect all agent metrics first
        all_agent_metrics = {}

        # Initialize metric tensors with zeros for all agents
        metrics_names = [
            'Episode/success_rate', 'Episode/success_time', 'Episode/engagement_rate',
            'Episode/engagement_time', 'Episode/engagement_length',
            'Smoothness/sum_square_velocity', 'Smoothness/sum_square_joint_velocity'
        ]

        for metric_name in metrics_names:
            all_agent_metrics[metric_name] = torch.zeros(self.num_agents, dtype=torch.float32)

        # Add force metrics if available
        force_metrics_names = []
        if self.has_force_data:
            force_metrics_names = ['Smoothness/max_force', 'Smoothness/max_torque',
                                  'Smoothness/avg_force', 'Smoothness/avg_torque']
            for metric_name in force_metrics_names:
                all_agent_metrics[metric_name] = torch.zeros(self.num_agents, dtype=torch.float32)

        # Calculate means for each agent and populate the tensors
        for agent_id in range(self.num_agents):
            agent_data = self.agent_episode_data[agent_id]

            if not agent_data['success_rates']:  # No completed episodes for this agent
                continue

            # Calculate means across completed episodes for this agent
            mean_success_rate = sum(agent_data['success_rates']) / len(agent_data['success_rates'])
            mean_success_time = sum(agent_data['success_times']) / len(agent_data['success_times']) if agent_data['success_times'] else 0.0
            mean_engagement_rate = sum(agent_data['engagement_rates']) / len(agent_data['engagement_rates'])
            mean_engagement_time = sum(agent_data['engagement_times']) / len(agent_data['engagement_times']) if agent_data['engagement_times'] else 0.0
            mean_engagement_length = sum(agent_data['engagement_lengths']) / len(agent_data['engagement_lengths'])
            mean_ssv = sum(agent_data['ssv_values']) / len(agent_data['ssv_values'])
            mean_ssjv = sum(agent_data['ssjv_values']) / len(agent_data['ssjv_values'])


            # Populate tensors for this agent
            all_agent_metrics['Episode/success_rate'][agent_id] = mean_success_rate
            all_agent_metrics['Episode/success_time'][agent_id] = mean_success_time
            all_agent_metrics['Episode/engagement_rate'][agent_id] = mean_engagement_rate
            all_agent_metrics['Episode/engagement_time'][agent_id] = mean_engagement_time
            all_agent_metrics['Episode/engagement_length'][agent_id] = mean_engagement_length
            all_agent_metrics['Smoothness/sum_square_velocity'][agent_id] = mean_ssv
            all_agent_metrics['Smoothness/sum_square_joint_velocity'][agent_id] = mean_ssjv

            if self.has_force_data and agent_data['max_forces']:
                mean_max_force = sum(agent_data['max_forces']) / len(agent_data['max_forces'])
                mean_max_torque = sum(agent_data['max_torques']) / len(agent_data['max_torques'])
                mean_avg_force = sum(agent_data['avg_forces']) / len(agent_data['avg_forces'])
                mean_avg_torque = sum(agent_data['avg_torques']) / len(agent_data['avg_torques'])

                all_agent_metrics['Smoothness/max_force'][agent_id] = mean_max_force
                all_agent_metrics['Smoothness/max_torque'][agent_id] = mean_max_torque
                all_agent_metrics['Smoothness/avg_force'][agent_id] = mean_avg_force
                all_agent_metrics['Smoothness/avg_torque'][agent_id] = mean_avg_torque
        # Send all metrics as num_agents-sized tensors to WandbWrapper
        # This will trigger the "Direct agent assignment" path in _split_by_agent
        self.env.add_metrics(all_agent_metrics)

        # Clear all agent episode data after sending
        for agent_id in range(self.num_agents):
            agent_data = self.agent_episode_data[agent_id]
            agent_data['success_rates'].clear()
            agent_data['success_times'].clear()
            agent_data['engagement_rates'].clear()
            agent_data['engagement_times'].clear()
            agent_data['engagement_lengths'].clear()
            agent_data['ssv_values'].clear()
            agent_data['ssjv_values'].clear()
            if self.has_force_data:
                agent_data['max_forces'].clear()
                agent_data['max_torques'].clear()
                agent_data['avg_forces'].clear()
                agent_data['avg_torques'].clear()

    def _reset_completed_episodes(self, env_ids):
        """Reset tracking variables for completed episodes."""
        self.ep_succeeded[env_ids] = False
        self.ep_success_times[env_ids] = 0
        self.ep_engaged[env_ids] = False
        self.ep_engaged_times[env_ids] = 0
        self.ep_engaged_length[env_ids] = 0
        self.ep_ssv[env_ids] = 0
        self.ep_ssjv[env_ids] = 0

        if self.has_force_data and hasattr(self, 'ep_max_force'):
            self.ep_max_force[env_ids] = 0
            self.ep_max_torque[env_ids] = 0
            self.ep_sum_force[env_ids] = 0
            self.ep_sum_torque[env_ids] = 0

    def get_agent_assignment(self):
        """Get environment indices assigned to each agent."""
        assignments = {}
        for i in range(self.num_agents):
            start_idx = i * self.envs_per_agent
            end_idx = (i + 1) * self.envs_per_agent
            assignments[i] = list(range(start_idx, end_idx))
        return assignments

    def get_success_stats(self):
        """Get current success statistics."""
        return {
            'success_rate': self.ep_succeeded.float().mean().item(),
            'avg_success_time': self.ep_success_times[self.ep_succeeded].float().mean().item() if self.ep_succeeded.any() else 0.0,
            'engagement_rate': self.ep_engaged.float().mean().item(),
            'avg_engagement_time': self.ep_engaged_times[self.ep_engaged].float().mean().item() if self.ep_engaged.any() else 0.0,
            'avg_engagement_length': self.ep_engaged_length[self.ep_engaged].float().mean().item() if self.ep_engaged.any() else 0.0,
        }

    def get_smoothness_stats(self):
        """Get current smoothness statistics."""
        stats = {
            'avg_ssv': self.ep_ssv.mean().item(),
            'std_ssv': self.ep_ssv.std().item(),
            'avg_ssjv': self.ep_ssjv.mean().item(),
            'std_ssjv': self.ep_ssjv.std().item(),
        }

        if self.has_force_data:
            stats.update({
                'avg_max_force': self.ep_max_force.mean().item(),
                'avg_max_torque': self.ep_max_torque.mean().item(),
                'avg_sum_force': self.ep_sum_force.mean().item(),
                'avg_sum_torque': self.ep_sum_torque.mean().item(),
            })

        return stats

    def get_agent_metrics(self, agent_id):
        """Get metrics for a specific agent."""
        if agent_id >= self.num_agents:
            raise ValueError(f"Agent ID ({agent_id}) must be less than number of agents ({self.num_agents})")

        start_idx = agent_id * self.envs_per_agent
        end_idx = (agent_id + 1) * self.envs_per_agent

        metrics = {
            'success_rate': self.ep_succeeded[start_idx:end_idx].float().mean().item(),
            'engagement_rate': self.ep_engaged[start_idx:end_idx].float().mean().item(),
            'avg_ssv': self.ep_ssv[start_idx:end_idx].mean().item(),
            'avg_ssjv': self.ep_ssjv[start_idx:end_idx].mean().item(),
        }

        if self.has_force_data:
            metrics.update({
                'avg_max_force': self.ep_max_force[start_idx:end_idx].mean().item(),
                'avg_max_torque': self.ep_max_torque[start_idx:end_idx].mean().item(),
            })

        return metrics

    def step(self, action):
        """Step environment and collect factory metrics."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Collect step metrics
        self._collect_step_metrics()

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
                max_step_mask = episode_lengths == self.max_episode_length
                max_step_env_ids = max_step_mask.nonzero(as_tuple=False).squeeze(-1).tolist()


                # Show per-agent breakdown for verification
                for agent_id in range(self.num_agents):
                    start_idx = agent_id * self.envs_per_agent
                    end_idx = (agent_id + 1) * self.envs_per_agent
                    agent_max_step = [env_id for env_id in max_step_env_ids if start_idx <= env_id < end_idx]

                if torch.any(max_step_mask):
                    self._store_completed_episodes(max_step_mask)

                # Publish all accumulated metrics and reset all tracking
                self._send_aggregated_factory_metrics()
                self._reset_all_tracking_variables()
            else:
                # Partial reset - these environments terminated
                env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids

                # Show per-agent breakdown for verification
                for agent_id in range(self.num_agents):
                    start_idx = agent_id * self.envs_per_agent
                    end_idx = (agent_id + 1) * self.envs_per_agent
                    agent_terminated = [env_id for env_id in env_ids_list if start_idx <= env_id < end_idx]

                # Store all environments in env_ids as completed episodes
                completed_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                completed_mask[env_ids] = True
                self._store_completed_episodes(completed_mask)

                # Reset env-specific tracking only for completed environments
                self._reset_completed_episodes(env_ids)

        # Call original _reset_idx to maintain wrapper chain
        if self._original_reset_idx is not None:
            self._original_reset_idx(env_ids)

    def reset(self, **kwargs):
        """Reset environment - now just calls super().reset()."""
        return super().reset(**kwargs)

    def _reset_all_tracking_variables(self):
        """Reset all tracking variables to initial state."""
        # Reset episode tracking
        self.ep_succeeded.fill_(False)
        self.ep_success_times.fill_(0)
        self.ep_engaged.fill_(False)
        self.ep_engaged_times.fill_(0)
        self.ep_engaged_length.fill_(0)
        self.ep_ssv.fill_(0)
        self.ep_ssjv.fill_(0)

        # Reset force tracking if available
        if self.has_force_data and hasattr(self, 'ep_max_force'):
            self.ep_max_force.fill_(0)
            self.ep_max_torque.fill_(0)
            self.ep_sum_force.fill_(0)
            self.ep_sum_torque.fill_(0)

        # Clear all agent episode data
        for agent_id in range(self.num_agents):
            agent_data = self.agent_episode_data[agent_id]
            agent_data['success_rates'].clear()
            agent_data['success_times'].clear()
            agent_data['engagement_rates'].clear()
            agent_data['engagement_times'].clear()
            agent_data['engagement_lengths'].clear()
            agent_data['ssv_values'].clear()
            agent_data['ssjv_values'].clear()
            if self.has_force_data:
                if agent_data['max_forces'] is not None:
                    agent_data['max_forces'].clear()
                if agent_data['max_torques'] is not None:
                    agent_data['max_torques'].clear()
                if agent_data['avg_forces'] is not None:
                    agent_data['avg_forces'].clear()
                if agent_data['avg_torques'] is not None:
                    agent_data['avg_torques'].clear()

    def close(self):
        """Close wrapper."""
        if hasattr(super(), 'close'):
            super().close()