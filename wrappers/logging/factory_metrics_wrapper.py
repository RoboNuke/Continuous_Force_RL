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

    def __init__(self, env, num_agents=1, publish_to_wandb=True, engagement_reward_scale=1.0, success_reward_scale=1.0, timeout_penalty=0.0):
        """
        Initialize factory metrics wrapper.

        Args:
            env: Base environment to wrap (must have GenericWandbLoggingWrapper in chain if publish_to_wandb=True)
            num_agents: Number of agents for static assignment
            publish_to_wandb: Whether to publish metrics to WandB (default: True)
            engagement_reward_scale: Scale factor for engagement rewards (default: 1.0)
            success_reward_scale: Scale factor for success rewards (default: 1.0)
            timeout_penalty: Penalty applied when episode times out (default: 0.0)
        """
        super().__init__(env)

        self.publish_to_wandb = publish_to_wandb

        # Validate that we have access to add_metrics (WandbWrapper in chain) only if publishing
        if self.publish_to_wandb and not self._find_wandb_wrapper():
            raise ValueError(
                "Factory Metrics Wrapper requires GenericWandbLoggingWrapper to be applied first when publish_to_wandb=True. "
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
        self.last_pubbed_agent_metrics = None
        self.agent_episode_data = {}
        for i in range(self.num_agents):
            self.agent_episode_data[i] = {
                'success_rates': [],
                'success_times': [],
                'is_timeouts': [],  # Whether episode timed out (only meaningful if not succeeded)
                'engagement_rates': [],
                'engagement_times': [],
                'engagement_lengths': [],  # Kept for backwards compat (total steps engaged)
                'engagement_period_counts': [],  # Number of continuous periods per episode
                'avg_engagement_period_lengths': [],  # Average period length per episode
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

        # Configurable reward scalers for engagement and success (default 1.0 for backward compatibility)
        self.engagement_reward_scale = engagement_reward_scale
        self.success_reward_scale = success_reward_scale
        self.timeout_penalty = timeout_penalty

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
        self.ep_engaged_length = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)  # Total steps engaged (backwards compat)

        # Engagement period tracking (for continuous engagement periods)
        self.prev_engaged = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.current_engagement_length = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.engagement_period_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.engagement_period_total_length = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

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

        # Update force/torque metrics if available - ONLY when in contact
        if self.has_force_data and hasattr(self.unwrapped, 'robot_force_torque'):
            force_magnitude = torch.linalg.norm(self.unwrapped.robot_force_torque[:, :3], axis=1)
            torque_magnitude = torch.linalg.norm(self.unwrapped.robot_force_torque[:, 3:], axis=1)

            # Determine which environments are in contact (any axis)
            if hasattr(self.unwrapped, 'in_contact'):
                # in_contact[:, :3] are force contact flags - any True means contact
                in_contact_mask = self.unwrapped.in_contact[:, :3].any(dim=1)
            else:
                # Fallback: count all steps if in_contact not available
                in_contact_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

            # Only accumulate for environments in contact
            self.ep_sum_force[in_contact_mask] += force_magnitude[in_contact_mask]
            self.ep_sum_torque[in_contact_mask] += torque_magnitude[in_contact_mask]
            self.ep_max_force = torch.max(self.ep_max_force, force_magnitude)  # Max always updates
            self.ep_max_torque = torch.max(self.ep_max_torque, torque_magnitude)

            # Welford's algorithm for running variance - only for environments in contact
            self.ep_force_count[in_contact_mask] += 1

            # Only update Welford stats for environments in contact
            if torch.any(in_contact_mask):
                count_safe = self.ep_force_count.clone()
                count_safe[count_safe == 0] = 1  # Avoid division by zero

                delta_force = force_magnitude - self.ep_mean_force
                self.ep_mean_force[in_contact_mask] += delta_force[in_contact_mask] / count_safe[in_contact_mask]
                delta2_force = force_magnitude - self.ep_mean_force
                self.ep_m2_force[in_contact_mask] += (delta_force * delta2_force)[in_contact_mask]

                delta_torque = torque_magnitude - self.ep_mean_torque
                self.ep_mean_torque[in_contact_mask] += delta_torque[in_contact_mask] / count_safe[in_contact_mask]
                delta2_torque = torque_magnitude - self.ep_mean_torque
                self.ep_m2_torque[in_contact_mask] += (delta_torque * delta2_torque)[in_contact_mask]

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
                self.ep_mean_force = torch.zeros((self.num_envs,), device=self.device)
                self.ep_mean_torque = torch.zeros((self.num_envs,), device=self.device)
                self.ep_m2_force = torch.zeros((self.num_envs,), device=self.device)
                self.ep_m2_torque = torch.zeros((self.num_envs,), device=self.device)
                self.ep_force_count = torch.zeros((self.num_envs,), device=self.device)

                # Update agent episode data to include force metrics
                for i in range(self.num_agents):
                    self.agent_episode_data[i]['max_forces'] = []
                    self.agent_episode_data[i]['max_torques'] = []
                    self.agent_episode_data[i]['avg_forces'] = []
                    self.agent_episode_data[i]['avg_torques'] = []
                    self.agent_episode_data[i]['std_forces'] = []
                    self.agent_episode_data[i]['std_torques'] = []

        else:
            self.has_force_data = False

    def _update_rew_buf_for_env(self, env, curr_successes):
        """Compute reward at current timestep."""
        # Calculate base rewards from scratch (no double-counting)
        # Note: HybridControl now wraps _get_rewards instead of _update_rew_buf,
        # so _original_update_rew_buf points directly to BaseEnv (if it exists)
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

        #print(f"Squishing Terms - Baseline:({a0},{b0})\tCoarse:({a1},{b1})\tFine:({a2},{b2})")

        # Action penalties.
        rew_dict["action_penalty"] = torch.norm(env.actions, p=2, dim=-1)
        rew_dict["action_grad_penalty"] = torch.norm(env.actions - env.prev_actions, p=2, dim=-1)
        rew_dict["curr_engaged"] = (
            env._get_curr_successes(success_threshold=env.cfg_task.engage_threshold, check_rot=False).clone().float()
        )
        rew_dict["curr_successes"] = curr_successes.clone().float()

        # Calculate factory reward components with configurable scalers
        rew_buf = (
            rew_dict["kp_coarse"]
            + rew_dict["kp_baseline"]
            + rew_dict["kp_fine"]
            - rew_dict["action_penalty"] * env.cfg_task.action_penalty_scale
            - rew_dict["action_grad_penalty"] * env.cfg_task.action_grad_penalty_scale
            + rew_dict["curr_engaged"] * self.engagement_reward_scale
            + rew_dict["curr_successes"] * self.success_reward_scale
        )

        # Apply timeout penalty only when max steps reached (not early termination)
        if self.timeout_penalty != 0.0 and hasattr(env, 'episode_length_buf'):
            timeout_mask = env.episode_length_buf >= (self.max_episode_length - 1)
            rew_dict["timeout_penalty"] = timeout_mask.float() * self.timeout_penalty
            rew_buf = rew_buf + rew_dict["timeout_penalty"]

        for rew_name, rew in rew_dict.items():
            env.extras[f"logs_rew_{rew_name}"] = rew

        # Override logged values for scaled rewards to reflect actual contribution
        env.extras["logs_rew_curr_engaged"] = rew_dict["curr_engaged"] * self.engagement_reward_scale
        env.extras["logs_rew_curr_successes"] = rew_dict["curr_successes"] * self.success_reward_scale

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
                if check_rot and not hasattr(self.unwrapped, 'curr_yaw'):
                    self.unwrapped.curr_yaw = torch.zeros(
                        self.unwrapped.num_envs, device=self.unwrapped.device
                    )
                curr_successes = self.unwrapped._get_curr_successes(success_threshold, check_rot)

            if hasattr(self.unwrapped.cfg_task, 'engage_threshold'):
                engage_threshold = self.unwrapped.cfg_task.engage_threshold
                curr_engaged = self.unwrapped._get_curr_successes(engage_threshold, False)

        # Store current states as attributes for external access (e.g., evaluation scripts)
        self.successes = curr_successes
        self.curr_engaged = curr_engaged

        # Update success tracking
        # Only count success if episode_length_buf > 0 (prevents false positives at step 0 after reset)
        if hasattr(self.unwrapped, 'episode_length_buf'):
            valid_step_mask = self.unwrapped.episode_length_buf > 0
            curr_successes = curr_successes & valid_step_mask
            curr_engaged = curr_engaged & valid_step_mask

        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = True
        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        if len(first_success_ids) > 0 and hasattr(self.unwrapped, 'episode_length_buf'):
            self.ep_success_times[first_success_ids] = self.unwrapped.episode_length_buf[first_success_ids].clone()

        # Update engagement tracking (first engagement time)
        first_engaged = torch.logical_and(curr_engaged, torch.logical_not(self.ep_engaged))
        first_engaged_ids = first_engaged.nonzero(as_tuple=False).squeeze(-1)
        if len(first_engaged_ids) > 0 and hasattr(self.unwrapped, 'episode_length_buf'):
            self.ep_engaged_times[first_engaged_ids] = self.unwrapped.episode_length_buf[first_engaged_ids].clone()

        self.ep_engaged[curr_engaged] = True
        self.ep_engaged_length[curr_engaged] += 1  # Total steps engaged (backwards compat)

        # Track continuous engagement periods
        # Detect transitions (vectorized - no Python loops)
        just_engaged = torch.logical_and(curr_engaged, torch.logical_not(self.prev_engaged))
        just_disengaged = torch.logical_and(self.prev_engaged, torch.logical_not(curr_engaged))

        # Start new periods (first step of engagement)
        self.current_engagement_length[just_engaged] = 1

        # Continue existing periods
        continuing = torch.logical_and(curr_engaged, self.prev_engaged)
        self.current_engagement_length[continuing] += 1

        # Complete periods (disengagement) - record to totals
        if torch.any(just_disengaged):
            self.engagement_period_count[just_disengaged] += 1
            self.engagement_period_total_length[just_disengaged] += self.current_engagement_length[just_disengaged]
            self.current_engagement_length[just_disengaged] = 0

        # Update previous state
        self.prev_engaged = curr_engaged.clone()

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

            # Check if this environment timed out vs early termination
            env_is_timeout = episode_length >= (self.max_episode_length - 1)

            # Finalize open engagement period (only for early termination)
            if not env_is_timeout and self.prev_engaged[env_idx] and self.current_engagement_length[env_idx] > 0:
                self.engagement_period_count[env_idx] += 1
                self.engagement_period_total_length[env_idx] += self.current_engagement_length[env_idx]

            # Calculate average period length for this episode
            period_count = self.engagement_period_count[env_idx].item()
            total_period_length = self.engagement_period_total_length[env_idx].item()
            avg_period_length = total_period_length / period_count if period_count > 0 else 0.0

            # Store factory metrics in agent's episode lists
            self.agent_episode_data[agent_id]['success_rates'].append(float(self.ep_succeeded[env_idx]))
            self.agent_episode_data[agent_id]['success_times'].append(float(self.ep_success_times[env_idx]))
            self.agent_episode_data[agent_id]['is_timeouts'].append(env_is_timeout)
            self.agent_episode_data[agent_id]['engagement_rates'].append(float(self.ep_engaged[env_idx]))
            self.agent_episode_data[agent_id]['engagement_times'].append(float(self.ep_engaged_times[env_idx]))
            self.agent_episode_data[agent_id]['engagement_lengths'].append(float(self.ep_engaged_length[env_idx]))  # Backwards compat
            self.agent_episode_data[agent_id]['engagement_period_counts'].append(period_count)
            self.agent_episode_data[agent_id]['avg_engagement_period_lengths'].append(avg_period_length)
            self.agent_episode_data[agent_id]['ssv_values'].append(float(self.ep_ssv[env_idx]))
            self.agent_episode_data[agent_id]['ssjv_values'].append(float(self.ep_ssjv[env_idx]))

            if self.has_force_data and hasattr(self, 'ep_max_force'):
                max_force_val = float(self.ep_max_force[env_idx])

                # Use contact step count for averaging (not total episode length)
                contact_count = float(self.ep_force_count[env_idx])
                avg_force_val = float(self.ep_sum_force[env_idx] / contact_count) if contact_count > 0 else 0.0
                avg_torque_val = float(self.ep_sum_torque[env_idx] / contact_count) if contact_count > 0 else 0.0

                # Calculate std dev from Welford's M2 (sum of squared deviations)
                std_force_val = float(torch.sqrt(self.ep_m2_force[env_idx] / contact_count)) if contact_count > 0 else 0.0
                std_torque_val = float(torch.sqrt(self.ep_m2_torque[env_idx] / contact_count)) if contact_count > 0 else 0.0

                self.agent_episode_data[agent_id]['max_forces'].append(max_force_val)
                self.agent_episode_data[agent_id]['max_torques'].append(float(self.ep_max_torque[env_idx]))
                self.agent_episode_data[agent_id]['avg_forces'].append(avg_force_val)
                self.agent_episode_data[agent_id]['avg_torques'].append(avg_torque_val)
                self.agent_episode_data[agent_id]['std_forces'].append(std_force_val)
                self.agent_episode_data[agent_id]['std_torques'].append(std_torque_val)


    def _send_aggregated_factory_metrics(self):
        """Send aggregated factory metrics for all agents (like WandbWrapper)."""
        # Collect all agent metrics first
        all_agent_metrics = {}

        # Initialize metric tensors with zeros for all agents
        metrics_names = [
            'Episode/success_rate', 'Episode/success_time', 'Episode/engagement_rate',
            'Episode/engagement_time', 'Episode/engagement_length',
            'Episode/success_count', 'Episode/engagement_count', 'Episode/engagement_period_count',
            'Episode/break_rate', 'Episode/break_count',  # Early termination without success
            'Episode/timeout_rate', 'Episode/timeout_count',  # Max steps without success
            'Smoothness/sum_square_velocity', 'Smoothness/sum_square_joint_velocity'
        ]

        for metric_name in metrics_names:
            all_agent_metrics[metric_name] = torch.zeros(self.num_agents, dtype=torch.float32)

        # Add force metrics if available
        force_metrics_names = []
        if self.has_force_data:
            force_metrics_names = ['Smoothness/max_force', 'Smoothness/max_torque',
                                  'Smoothness/avg_force', 'Smoothness/avg_torque',
                                  'Smoothness/std_force', 'Smoothness/std_torque']
            for metric_name in force_metrics_names:
                all_agent_metrics[metric_name] = torch.zeros(self.num_agents, dtype=torch.float32)

        # Calculate means for each agent and populate the tensors
        for agent_id in range(self.num_agents):
            agent_data = self.agent_episode_data[agent_id]

            if not agent_data['success_rates']:  # No completed episodes for this agent
                continue

            # Calculate means across completed episodes for this agent
            mean_success_rate = sum(agent_data['success_rates']) / len(agent_data['success_rates'])
            mean_engagement_rate = sum(agent_data['engagement_rates']) / len(agent_data['engagement_rates'])

            # Only average success_time over successful episodes
            successful_times = [t for t, s in zip(agent_data['success_times'], agent_data['success_rates']) if s == 1.0]
            mean_success_time = sum(successful_times) / len(successful_times) if successful_times else 0.0

            # Only average engagement_time over engaged episodes
            engaged_times = [t for t, e in zip(agent_data['engagement_times'], agent_data['engagement_rates']) if e == 1.0]
            mean_engagement_time = sum(engaged_times) / len(engaged_times) if engaged_times else 0.0

            # Average engagement period length (average of per-episode averages, only for episodes with periods)
            avg_period_lengths = [l for l in agent_data['avg_engagement_period_lengths'] if l > 0]
            mean_engagement_length = sum(avg_period_lengths) / len(avg_period_lengths) if avg_period_lengths else 0.0

            mean_ssv = sum(agent_data['ssv_values']) / len(agent_data['ssv_values'])
            mean_ssjv = sum(agent_data['ssjv_values']) / len(agent_data['ssjv_values'])

            # Raw counts
            total_successes = sum(agent_data['success_rates'])
            total_engagements = sum(agent_data['engagement_rates'])
            total_engagement_periods = sum(agent_data['engagement_period_counts'])

            # Episode outcomes: success, break, or timeout (mutually exclusive)
            # Break = early termination WITHOUT success
            # Timeout = max steps reached WITHOUT success
            total_episodes = len(agent_data['success_rates'])
            total_breaks = sum(1 for s, t in zip(agent_data['success_rates'], agent_data['is_timeouts'])
                              if s == 0.0 and not t)
            total_timeouts = sum(1 for s, t in zip(agent_data['success_rates'], agent_data['is_timeouts'])
                                if s == 0.0 and t)
            break_rate = total_breaks / total_episodes if total_episodes > 0 else 0.0
            timeout_rate = total_timeouts / total_episodes if total_episodes > 0 else 0.0

            # Populate tensors for this agent
            all_agent_metrics['Episode/success_rate'][agent_id] = mean_success_rate
            all_agent_metrics['Episode/success_time'][agent_id] = mean_success_time
            all_agent_metrics['Episode/engagement_rate'][agent_id] = mean_engagement_rate
            all_agent_metrics['Episode/engagement_time'][agent_id] = mean_engagement_time
            all_agent_metrics['Episode/engagement_length'][agent_id] = mean_engagement_length
            all_agent_metrics['Episode/success_count'][agent_id] = total_successes
            all_agent_metrics['Episode/engagement_count'][agent_id] = total_engagements
            all_agent_metrics['Episode/engagement_period_count'][agent_id] = total_engagement_periods
            all_agent_metrics['Episode/break_rate'][agent_id] = break_rate
            all_agent_metrics['Episode/break_count'][agent_id] = total_breaks
            all_agent_metrics['Episode/timeout_rate'][agent_id] = timeout_rate
            all_agent_metrics['Episode/timeout_count'][agent_id] = total_timeouts
            all_agent_metrics['Smoothness/sum_square_velocity'][agent_id] = mean_ssv
            all_agent_metrics['Smoothness/sum_square_joint_velocity'][agent_id] = mean_ssjv

            if self.has_force_data and agent_data['max_forces']:
                max_max_force = max(agent_data['max_forces'])
                max_max_torque = max(agent_data['max_torques'])
                mean_avg_force = sum(agent_data['avg_forces']) / len(agent_data['avg_forces'])
                mean_avg_torque = sum(agent_data['avg_torques']) / len(agent_data['avg_torques'])
                mean_std_force = sum(agent_data['std_forces']) / len(agent_data['std_forces'])
                mean_std_torque = sum(agent_data['std_torques']) / len(agent_data['std_torques'])

                all_agent_metrics['Smoothness/max_force'][agent_id] = max_max_force
                all_agent_metrics['Smoothness/max_torque'][agent_id] = max_max_torque
                all_agent_metrics['Smoothness/avg_force'][agent_id] = mean_avg_force
                all_agent_metrics['Smoothness/avg_torque'][agent_id] = mean_avg_torque
                all_agent_metrics['Smoothness/std_force'][agent_id] = mean_std_force
                all_agent_metrics['Smoothness/std_torque'][agent_id] = mean_std_torque

        # Send all metrics as num_agents-sized tensors to WandbWrapper (only if publishing enabled)
        # This will trigger the "Direct agent assignment" path in _split_by_agent
        self.last_pubbed_agent_metrics = all_agent_metrics
        if self.publish_to_wandb:
            self.env.add_metrics(all_agent_metrics)

        # Clear all agent episode data after sending
        for agent_id in range(self.num_agents):
            agent_data = self.agent_episode_data[agent_id]
            agent_data['success_rates'].clear()
            agent_data['success_times'].clear()
            agent_data['is_timeouts'].clear()
            agent_data['engagement_rates'].clear()
            agent_data['engagement_times'].clear()
            agent_data['engagement_lengths'].clear()
            agent_data['engagement_period_counts'].clear()
            agent_data['avg_engagement_period_lengths'].clear()
            agent_data['ssv_values'].clear()
            agent_data['ssjv_values'].clear()
            if self.has_force_data:
                agent_data['max_forces'].clear()
                agent_data['max_torques'].clear()
                agent_data['avg_forces'].clear()
                agent_data['avg_torques'].clear()
                agent_data['std_forces'].clear()
                agent_data['std_torques'].clear()

    def _reset_completed_episodes(self, env_ids):
        """Reset tracking variables for completed episodes."""
        self.ep_succeeded[env_ids] = False
        self.ep_success_times[env_ids] = 0
        self.ep_engaged[env_ids] = False
        self.ep_engaged_times[env_ids] = 0
        self.ep_engaged_length[env_ids] = 0
        self.ep_ssv[env_ids] = 0
        self.ep_ssjv[env_ids] = 0

        # Reset engagement period tracking
        self.prev_engaged[env_ids] = False
        self.current_engagement_length[env_ids] = 0
        self.engagement_period_count[env_ids] = 0
        self.engagement_period_total_length[env_ids] = 0

        if self.has_force_data and hasattr(self, 'ep_max_force'):
            self.ep_max_force[env_ids] = 0
            self.ep_max_torque[env_ids] = 0
            self.ep_sum_force[env_ids] = 0
            self.ep_sum_torque[env_ids] = 0
            self.ep_mean_force[env_ids] = 0
            self.ep_mean_torque[env_ids] = 0
            self.ep_m2_force[env_ids] = 0
            self.ep_m2_torque[env_ids] = 0
            self.ep_force_count[env_ids] = 0

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

        # Add smoothness metrics to info dict for evaluation
        info['smoothness'] = {
            'ssv': self.ep_ssv.clone(),  # [num_envs] tensor
            'ssjv': self.ep_ssjv.clone(),  # [num_envs] tensor
        }

        # Add force/torque metrics if available
        if self.has_force_data and hasattr(self, 'ep_max_force'):
            info['smoothness']['max_force'] = self.ep_max_force.clone()
            info['smoothness']['max_torque'] = self.ep_max_torque.clone()
            info['smoothness']['sum_force'] = self.ep_sum_force.clone()
            info['smoothness']['sum_torque'] = self.ep_sum_torque.clone()

        return obs, reward, terminated, truncated, info

    def _wrapped_reset_idx(self, env_ids):
        """Handle environment resets via _reset_idx calls."""
        # Store original env_ids for passing to original _reset_idx
        original_env_ids = env_ids

        # Convert to tensor if needed for our processing
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        # CRITICAL: Before storing metrics, check if any terminating environments
        # have just succeeded/engaged THIS step (success detection happens after super().step()
        # returns, but _reset_idx is called from WITHIN super().step()).
        # We must capture success/engagement time NOW before episode_length_buf is reset.
        if hasattr(self.unwrapped, '_get_curr_successes') and hasattr(self.unwrapped, 'cfg_task'):
            # Early success capture
            if hasattr(self.unwrapped.cfg_task, 'success_threshold'):
                success_threshold = self.unwrapped.cfg_task.success_threshold
                check_rot = getattr(self.unwrapped.cfg_task, 'name', '') == "nut_thread"
                if check_rot and not hasattr(self.unwrapped, 'curr_yaw'):
                    self.unwrapped.curr_yaw = torch.zeros(
                        self.unwrapped.num_envs, device=self.unwrapped.device
                    )
                curr_successes = self.unwrapped._get_curr_successes(success_threshold, check_rot)

                # For terminating envs that are currently successful but haven't been marked yet
                first_success_in_terminating = curr_successes[env_ids] & ~self.ep_succeeded[env_ids]
                if torch.any(first_success_in_terminating):
                    terminating_first_success_ids = env_ids[first_success_in_terminating]
                    if hasattr(self.unwrapped, 'episode_length_buf'):
                        self.ep_success_times[terminating_first_success_ids] = self.unwrapped.episode_length_buf[terminating_first_success_ids].clone()
                        self.ep_succeeded[terminating_first_success_ids] = True

            # Early engagement capture
            if hasattr(self.unwrapped.cfg_task, 'engage_threshold'):
                engage_threshold = self.unwrapped.cfg_task.engage_threshold
                curr_engaged = self.unwrapped._get_curr_successes(engage_threshold, False)

                # For terminating envs that are currently engaged but haven't been marked yet
                first_engaged_in_terminating = curr_engaged[env_ids] & ~self.ep_engaged[env_ids]
                if torch.any(first_engaged_in_terminating):
                    terminating_first_engaged_ids = env_ids[first_engaged_in_terminating]
                    if hasattr(self.unwrapped, 'episode_length_buf'):
                        self.ep_engaged_times[terminating_first_engaged_ids] = self.unwrapped.episode_length_buf[terminating_first_engaged_ids].clone()
                        self.ep_engaged[terminating_first_engaged_ids] = True
                        # Also update engagement period tracking for first engagement on termination step
                        self.engagement_period_count[terminating_first_engaged_ids] += 1
                        self.engagement_period_total_length[terminating_first_engaged_ids] += 1  # At least 1 step

        # Skip processing on first reset (all data would be zeros)
        if self._first_reset:
            self._first_reset = False
        else:
            if len(env_ids) == self.num_envs:
                # Full reset - ALL environments get their episodes closed out
                # This happens at max_steps boundary - store ALL environments, not just timeouts
                all_envs_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                self._store_completed_episodes(all_envs_mask)

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

        # Call original _reset_idx to maintain wrapper chain (use original format)
        if self._original_reset_idx is not None:
            self._original_reset_idx(original_env_ids)

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

        # Reset engagement period tracking
        self.prev_engaged.fill_(False)
        self.current_engagement_length.fill_(0)
        self.engagement_period_count.fill_(0)
        self.engagement_period_total_length.fill_(0)

        # Reset force tracking if available
        if self.has_force_data and hasattr(self, 'ep_max_force'):
            self.ep_max_force.fill_(0)
            self.ep_max_torque.fill_(0)
            self.ep_sum_force.fill_(0)
            self.ep_sum_torque.fill_(0)
            self.ep_mean_force.fill_(0)
            self.ep_mean_torque.fill_(0)
            self.ep_m2_force.fill_(0)
            self.ep_m2_torque.fill_(0)
            self.ep_force_count.fill_(0)

        # Clear all agent episode data
        for agent_id in range(self.num_agents):
            agent_data = self.agent_episode_data[agent_id]
            agent_data['success_rates'].clear()
            agent_data['success_times'].clear()
            agent_data['is_timeouts'].clear()
            agent_data['engagement_rates'].clear()
            agent_data['engagement_times'].clear()
            agent_data['engagement_lengths'].clear()
            agent_data['engagement_period_counts'].clear()
            agent_data['avg_engagement_period_lengths'].clear()
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
                if agent_data['std_forces'] is not None:
                    agent_data['std_forces'].clear()
                if agent_data['std_torques'] is not None:
                    agent_data['std_torques'].clear()

    def close(self):
        """Close wrapper."""
        if hasattr(super(), 'close'):
            super().close()