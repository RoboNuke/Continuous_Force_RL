"""
Factory-Specific Metrics Wrapper

This wrapper adds factory task-specific metrics tracking including success rates,
engagement tracking, smoothness metrics, and force/torque statistics.

Extracted from:
- MultiWandbLoggerPPO.py lines 110-148, 179-202 (engagement tracking)
- factory_env.py lines 414, 634-639 (smoothness metrics)
- factory_env.py lines 614-623 (success time tracking)
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
    - Factory-specific reward component processing
    """

    def __init__(self, env, num_agents=1):
        """
        Initialize factory metrics wrapper.

        Args:
            env: Base environment to wrap
            num_agents: Number of agents for static assignment
        """
        super().__init__(env)

        self.num_agents = num_agents
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # Validate agent assignment
        if self.num_envs % self.num_agents != 0:
            raise ValueError(f"Number of environments ({self.num_envs}) must be divisible by number of agents ({self.num_agents})")

        self.envs_per_agent = self.num_envs // self.num_agents

        # Validate that the wandb wrapper is present and has required functions
        # Check both env.unwrapped and env itself (for wrapper chain)
        if not (hasattr(self.env.unwrapped, 'add_metrics') or hasattr(self.env, 'add_metrics')):
            raise ValueError(
                "Factory Metrics Wrapper requires GenericWandbLoggingWrapper to be applied first. "
                "The environment must have an 'add_metrics' method. "
                "Make sure wandb_logging wrapper is enabled and applied before this wrapper."
            )

        print(f"[INFO]: Factory Metrics Wrapper - Wandb integration validated ✓")

        # Initialize tracking variables
        self._init_tracking_variables()

        # Store original methods
        self._original_reset_buffers = None
        self._original_pre_physics_step = None
        self._original_get_rewards = None

        # Initialize wrapper
        self._wrapper_initialized = False
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

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

        # Force/torque metrics (if available)
        self.has_force_data = False
        if hasattr(self.unwrapped, 'robot_force_torque'):
            self.has_force_data = True
            self.ep_max_force = torch.zeros((self.num_envs,), device=self.device)
            self.ep_max_torque = torch.zeros((self.num_envs,), device=self.device)
            self.ep_sum_force = torch.zeros((self.num_envs,), device=self.device)
            self.ep_sum_torque = torch.zeros((self.num_envs,), device=self.device)

    def _initialize_wrapper(self):
        """Initialize wrapper by overriding environment methods."""
        if self._wrapper_initialized:
            return

        # Store and override methods
        if hasattr(self.unwrapped, '_reset_buffers'):
            self._original_reset_buffers = self.unwrapped._reset_buffers
            self.unwrapped._reset_buffers = self._wrapped_reset_buffers

        if hasattr(self.unwrapped, '_pre_physics_step'):
            self._original_pre_physics_step = self.unwrapped._pre_physics_step
            self.unwrapped._pre_physics_step = self._wrapped_pre_physics_step

        if hasattr(self.unwrapped, '_get_rewards'):
            self._original_get_rewards = self.unwrapped._get_rewards
            self.unwrapped._get_rewards = self._wrapped_get_rewards

        self._wrapper_initialized = True

    def _wrapped_reset_buffers(self, env_ids):
        """Reset factory metrics buffers."""
        # Call original reset
        if self._original_reset_buffers:
            self._original_reset_buffers(env_ids)

        # Reset factory-specific metrics
        self.ep_succeeded[env_ids] = False
        self.ep_success_times[env_ids] = 0
        self.ep_engaged[env_ids] = False
        self.ep_engaged_times[env_ids] = 0
        self.ep_engaged_length[env_ids] = 0

        self.ep_ssv[env_ids] = 0
        self.ep_ssjv[env_ids] = 0

        if self.has_force_data:
            self.ep_max_force[env_ids] = 0
            self.ep_max_torque[env_ids] = 0
            self.ep_sum_force[env_ids] = 0
            self.ep_sum_torque[env_ids] = 0

    def _wrapped_pre_physics_step(self, action):
        """Update metrics during physics step."""
        # Call original pre-physics step
        if self._original_pre_physics_step:
            self._original_pre_physics_step(action)

        # Update smoothness metrics
        if hasattr(self.unwrapped, 'ee_linvel_fd'):
            self.ep_ssv += torch.linalg.norm(self.unwrapped.ee_linvel_fd, axis=1)

        # Update sum squared joint velocity (ssjv) from robot data
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

    def _wrapped_get_rewards(self):
        """Update rewards and compute factory-specific statistics."""
        # Get original rewards
        rew_buf = self._original_get_rewards() if self._original_get_rewards else torch.zeros(self.num_envs, device=self.device)

        # Update success and engagement tracking
        self._update_success_engagement_tracking()

        # Add factory metrics to extras
        self._update_extras()

        return rew_buf

    def _update_success_engagement_tracking(self):
        """Update success and engagement tracking."""
        # Get current successes and engagements if available
        curr_successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        curr_engaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Try to get from environment methods - require explicit configuration
        if hasattr(self.unwrapped, '_get_curr_successes'):
            try:
                # Require explicit task configuration
                if not hasattr(self.unwrapped, 'cfg_task'):
                    raise ValueError(
                        "Environment must have cfg_task attribute for factory metrics. "
                        "Please ensure environment configuration includes task parameters."
                    )

                if not hasattr(self.unwrapped.cfg_task, 'success_threshold'):
                    raise ValueError(
                        "Environment cfg_task must have 'success_threshold' parameter. "
                        "Example: cfg_task.success_threshold = 0.02"
                    )

                success_threshold = self.unwrapped.cfg_task.success_threshold
                check_rot = getattr(self.unwrapped.cfg_task, 'name', '') == "nut_thread"
                curr_successes = self.unwrapped._get_curr_successes(success_threshold, check_rot)
            except Exception as e:
                print(f"Warning: Could not get success status: {e}")

        if hasattr(self.unwrapped, '_get_curr_successes'):
            try:
                # Require explicit engagement threshold
                if not hasattr(self.unwrapped.cfg_task, 'engage_threshold'):
                    raise ValueError(
                        "Environment cfg_task must have 'engage_threshold' parameter for engagement tracking. "
                        "Example: cfg_task.engage_threshold = 0.05"
                    )

                engage_threshold = self.unwrapped.cfg_task.engage_threshold
                curr_engaged = self.unwrapped._get_curr_successes(engage_threshold, False)
            except Exception as e:
                print(f"Warning: Could not get engagement status: {e}")

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

        # Store current states for other wrappers
        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['current_engagements'] = curr_engaged.clone().float()
            self.unwrapped.extras['current_successes'] = curr_successes.clone().float()

    def _update_extras(self):
        """Update extras with factory-specific metrics."""
        if not hasattr(self.unwrapped, 'extras'):
            return

        # Check if we should log (typically on timeout/truncation)
        should_log = False
        if hasattr(self.unwrapped, '_get_dones'):
            try:
                _, time_out = self.unwrapped._get_dones()
                should_log = torch.any(time_out)
            except:
                # Fallback - always log
                should_log = True
        else:
            should_log = True

        if should_log:
            # Episode metrics
            self.unwrapped.extras['Episode / successes'] = self.ep_succeeded
            self.unwrapped.extras['Episode / success_times'] = self.ep_success_times
            self.unwrapped.extras['Episode / engaged'] = self.ep_engaged
            self.unwrapped.extras['Episode / engage_times'] = self.ep_engaged_times
            self.unwrapped.extras['Episode / engage_lengths'] = self.ep_engaged_length

            # Smoothness metrics
            self.unwrapped.extras['smoothness'] = {}
            self.unwrapped.extras['smoothness']['Smoothness / Sum Square Velocity'] = self.ep_ssv
            self.unwrapped.extras['smoothness']['Smoothness / Sum Square Joint Velocity'] = self.ep_ssjv

            # Force/torque metrics if available
            if self.has_force_data:
                # Calculate averages based on episode length
                episode_length = getattr(self.unwrapped, 'max_episode_length', 1)
                decimation = getattr(self.unwrapped.cfg, 'decimation', 1)
                norm_factor = decimation * episode_length

                self.unwrapped.extras['smoothness']['Smoothness / Avg Force'] = self.ep_sum_force / norm_factor
                self.unwrapped.extras['smoothness']['Smoothness / Max Force'] = self.ep_max_force
                self.unwrapped.extras['smoothness']['Smoothness / Avg Torque'] = self.ep_sum_torque / norm_factor
                self.unwrapped.extras['smoothness']['Smoothness / Max Torque'] = self.ep_max_torque

    def _collect_factory_metrics(self):
        """Collect factory metrics for wandb logging."""
        metrics = {}

        # Check if we should collect metrics (typically on timeout/truncation)
        should_collect = False
        if hasattr(self.unwrapped, '_get_dones'):
            try:
                _, time_out = self.unwrapped._get_dones()
                should_collect = torch.any(time_out)
            except Exception as e:
                # Fallback - collect current state metrics only
                should_collect = False
        else:

        # Always provide current engagement and success states
        if hasattr(self.unwrapped, 'extras') and 'current_engagements' in self.unwrapped.extras:
            metrics['current_engagements'] = self.unwrapped.extras['current_engagements']
        if hasattr(self.unwrapped, 'extras') and 'current_successes' in self.unwrapped.extras:
            metrics['current_successes'] = self.unwrapped.extras['current_successes']

        # Collect episode metrics when episodes complete
        if should_collect:
            # Episode metrics
            metrics['Episode/successes'] = self.ep_succeeded.float()
            metrics['Episode/success_times'] = self.ep_success_times.float()
            metrics['Episode/engaged'] = self.ep_engaged.float()
            metrics['Episode/engage_times'] = self.ep_engaged_times.float()
            metrics['Episode/engage_lengths'] = self.ep_engaged_length.float()

            # Smoothness metrics
            metrics['Smoothness/Sum_Square_Velocity'] = self.ep_ssv
            metrics['Smoothness/Sum_Square_Joint_Velocity'] = self.ep_ssjv

            # Force/torque metrics if available
            if self.has_force_data:
                # Calculate averages based on episode length
                episode_length = getattr(self.unwrapped, 'max_episode_length', 1)
                decimation = getattr(self.unwrapped.cfg, 'decimation', 1)
                norm_factor = decimation * episode_length

                metrics['Smoothness/Avg_Force'] = self.ep_sum_force / norm_factor
                metrics['Smoothness/Max_Force'] = self.ep_max_force
                metrics['Smoothness/Avg_Torque'] = self.ep_sum_torque / norm_factor
                metrics['Smoothness/Max_Torque'] = self.ep_max_torque

        return metrics

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
        """Step environment and ensure wrapper is initialized."""
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        obs, reward, terminated, truncated, info = super().step(action)

        # Check if wandb wrapper is available (check both env.unwrapped and env)
        add_metrics_target = None

        if hasattr(self.env.unwrapped, 'add_metrics'):
            add_metrics_target = self.env.unwrapped
        elif hasattr(self.env, 'add_metrics'):
            add_metrics_target = self.env

        if add_metrics_target:
            # Collect factory metrics and send to wandb wrapper
            factory_metrics = self._collect_factory_metrics()
            if factory_metrics:
                add_metrics_target.add_metrics(factory_metrics)
            else:
        else:
            # Fallback: Copy relevant extras to info for downstream wrappers
            if hasattr(self.unwrapped, 'extras') and self.unwrapped.extras:
                # Copy current states that should be available every step
                for key in ['current_engagements', 'current_successes']:
                    if key in self.unwrapped.extras:
                        info[key] = self.unwrapped.extras[key]

                # Copy smoothness data if available
                if 'smoothness' in self.unwrapped.extras:
                    info['smoothness'] = self.unwrapped.extras['smoothness']

                # Copy episode metrics when available
                for key in ['Episode / successes', 'Episode / success_times', 'Episode / engaged',
                           'Episode / engage_times', 'Episode / engage_lengths']:
                    if key in self.unwrapped.extras:
                        info[key] = self.unwrapped.extras[key]

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and ensure wrapper is initialized."""
        obs, info = super().reset(**kwargs)
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()
        return obs, info