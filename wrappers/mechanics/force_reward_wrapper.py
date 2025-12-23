"""
Force Reward Wrapper for Factory Environment

This wrapper adds 8 distinct force-based reward functions to factory environments,
building on force sensor data from the force_torque_wrapper. All rewards are
configurable and can be individually enabled/disabled.

Reward Functions:
1. Force Magnitude: Reward based on force magnitude relative to base force
2. Alignment Award: Dot product reward for force alignment with goal direction
3. Force Action Error: Penalty for difference between commanded and actual force
4. Contact Consistency: Exponential reward for consistent force application
5. Oscillation Penalty: Penalty for force command oscillations
6. Contact Transition Reward: Reward for smooth contact transitions
7. Efficiency: Progress-to-energy ratio reward
8. Force Ratio: Reward for optimal force direction ratios

The wrapper integrates with Isaac Lab's _reset_idx pattern for proper state management
and logs component rewards to wandb via the extras system.
"""

import torch
import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, List


class ForceRewardWrapper(gym.Wrapper):
    """
    Wrapper that adds 8 configurable force-based reward functions to factory environments.

    This wrapper requires force_torque_wrapper to be applied first to provide force sensor data.
    All reward functions can be individually enabled/disabled and have configurable weights.
    The wrapper properly handles Isaac Lab's _reset_idx pattern for selective environment resets.

    Features:
    - 8 distinct force-based reward functions
    - Individual enable/disable and weight control for each function
    - Contact detection system for contact-dependent rewards
    - Per-environment state management compatible with _reset_idx
    - Component reward logging to wandb via extras system
    - Comprehensive configuration via ForceRewardConfig
    """

    def __init__(self, env, config: Dict[str, Any]):
        """
        Initialize the force reward wrapper.

        Args:
            env: Base environment to wrap (must have force_torque_wrapper applied)
            config: Dictionary containing all force reward configuration parameters
        """
        super().__init__(env)

        # Store configuration
        self.config = config
        self.enabled = config.get('enabled', False)

        # Initialize environment attributes first
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # Store original methods for wrapping
        self._original_get_rewards = None
        self._original_reset_idx = None

        # Lazy initialization flag
        self._wrapper_initialized = False

        # Early exit if wrapper is disabled
        if not self.enabled:
            return

        # Dependency checking
        self._check_force_torque_dependency()

        # Extract configuration parameters
        self._extract_config_parameters()

        # Initialize state management
        self._init_state_management()

        # Initialize if environment is ready
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _check_force_torque_dependency(self):
        """
        Check that force_torque_wrapper is already applied in the wrapper chain.

        Raises:
            ValueError: If force_torque_wrapper is not found in wrapper chain
        """
        # Check if force_torque data is available
        if not hasattr(self.unwrapped, 'robot_force_torque'):
            # Walk up the wrapper chain to check for force_torque_wrapper
            current_env = self.env
            found_force_torque = False

            while current_env is not None:
                if hasattr(current_env, '__class__') and 'ForceTorqueWrapper' in str(current_env.__class__):
                    found_force_torque = True
                    break
                if hasattr(current_env, 'env'):
                    current_env = current_env.env
                else:
                    break

            if not found_force_torque:
                raise ValueError(
                    "ERROR: ForceRewardWrapper requires ForceTorqueWrapper to be applied first.\n"
                    "\n"
                    "SOLUTION: Apply ForceTorqueWrapper before ForceRewardWrapper.\n"
                    "Correct order:\n"
                    "  1. Apply ForceTorqueWrapper first\n"
                    "  2. Apply ForceRewardWrapper after\n"
                    "  3. Apply other wrappers as needed\n"
                    "\n"
                    "ForceRewardWrapper needs access to robot_force_torque data."
                )

    def _extract_config_parameters(self):
        """Extract and store configuration parameters from config dictionary."""
        # Contact detection parameters
        self.contact_force_threshold = self.config.get('contact_force_threshold', 1.0)
        self.contact_window_size = self.config.get('contact_window_size', 10)

        # Force Magnitude Reward
        self.enable_force_magnitude_reward = self.config.get('enable_force_magnitude_reward', False)
        self.force_magnitude_reward_weight = self.config.get('force_magnitude_reward_weight', 1.0)
        self.force_magnitude_base_force = self.config.get('force_magnitude_base_force', 5.0)
        self.force_magnitude_keep_sign = self.config.get('force_magnitude_keep_sign', True)

        # Alignment Award
        self.enable_alignment_award = self.config.get('enable_alignment_award', False)
        self.alignment_award_reward_weight = self.config.get('alignment_award_reward_weight', 1.0)
        self.alignment_goal_orientation = torch.tensor(
            self.config.get('alignment_goal_orientation', [0.0, 0.0, -1.0]),
            dtype=torch.float32, device=self.device
        )
        if self.alignment_goal_orientation.dim() == 1:
            self.alignment_goal_orientation = self.alignment_goal_orientation.unsqueeze(0).expand(self.num_envs, -1)

        # Force Action Error
        self.enable_force_action_error = self.config.get('enable_force_action_error', False)
        self.force_action_error_reward_weight = self.config.get('force_action_error_reward_weight', 1.0)

        # Contact Consistency
        self.enable_contact_consistency = self.config.get('enable_contact_consistency', False)
        self.contact_consistency_reward_weight = self.config.get('contact_consistency_reward_weight', 1.0)
        self.contact_consistency_beta = self.config.get('contact_consistency_beta', 1.0)
        self.contact_consistency_use_ema = self.config.get('contact_consistency_use_ema', True)
        self.contact_consistency_ema_alpha = self.config.get('contact_consistency_ema_alpha', 0.1)

        # Oscillation Penalty
        self.enable_oscillation_penalty = self.config.get('enable_oscillation_penalty', False)
        self.oscillation_penalty_reward_weight = self.config.get('oscillation_penalty_reward_weight', 1.0)
        self.oscillation_penalty_window_size = self.config.get('oscillation_penalty_window_size', 5)

        # Contact Transition Reward
        self.enable_contact_transition_reward = self.config.get('enable_contact_transition_reward', False)
        self.contact_transition_reward_weight = self.config.get('contact_transition_reward_weight', 1.0)

        # Efficiency
        self.enable_efficiency = self.config.get('enable_efficiency', False)
        self.efficiency_reward_weight = self.config.get('efficiency_reward_weight', 1.0)

        # Force Ratio
        self.enable_force_ratio = self.config.get('enable_force_ratio', False)
        self.force_ratio_reward_weight = self.config.get('force_ratio_reward_weight', 1.0)

        self.enable_contact_rew = self.config.get('enable_contact_reward', False)
        self.contact_rew_weight = self.config.get('contact_reward_weight', 1.0)

        # squared velocity 
        self.enable_square_vel = self.config.get('enable_square_vel', False)
        self.square_vel_weight = self.config.get('square_vel_weight', 1.0)

    def _init_state_management(self):
        """Initialize per-environment state variables for reward calculations."""
        # Contact detection state
        self.force_history = torch.zeros(
            (self.num_envs, self.contact_window_size),
            dtype=torch.float32, device=self.device
        )
        self.force_history_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.in_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Contact consistency state
        if self.enable_contact_consistency:
            if self.contact_consistency_use_ema:
                self.force_ema = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            else:
                max_window = 100  # Maximum window for running average
                self.force_consistency_history = torch.zeros(
                    (self.num_envs, max_window, 3),
                    dtype=torch.float32, device=self.device
                )
                self.force_consistency_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Oscillation penalty state
        if self.enable_oscillation_penalty:
            self.force_action_history = torch.zeros(
                (self.num_envs, self.oscillation_penalty_window_size, 3),
                dtype=torch.float32, device=self.device
            )
            self.force_action_history_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Contact transition state
        if self.enable_contact_transition_reward:
            self.force_magnitude_history = torch.zeros(
                (self.num_envs, self.contact_window_size),
                dtype=torch.float32, device=self.device
            )

        # Efficiency state
        if self.enable_efficiency:
            self.starting_keypoint_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.energy_used = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.episode_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _initialize_wrapper(self):
        """Initialize the wrapper after the base environment is set up."""
        if self._wrapper_initialized or not self.enabled:
            return

        # Store and override _get_rewards method
        if hasattr(self.unwrapped, '_get_rewards'):
            self._original_get_rewards = self.unwrapped._get_rewards
            self.unwrapped._get_rewards = self._wrapped_get_rewards
        else:
            raise ValueError("Environment missing required _get_rewards method")

        # Store and override _reset_idx method for state management
        if hasattr(self.unwrapped, '_reset_idx'):
            self._original_reset_idx = self.unwrapped._reset_idx
            self.unwrapped._reset_idx = self._wrapped_reset_idx

        self._wrapper_initialized = True

    def _wrapped_reset_idx(self, env_ids):
        """Reset force reward state for specific environments."""
        if not self.enabled:
            if self._original_reset_idx:
                self._original_reset_idx(env_ids)
            return

        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        # Reset contact detection state
        self.force_history[env_ids] = 0.0
        self.force_history_idx[env_ids] = 0
        self.in_contact[env_ids] = False
        self.prev_contact[env_ids] = False

        # Reset contact consistency state
        if self.enable_contact_consistency:
            if self.contact_consistency_use_ema:
                self.force_ema[env_ids] = 0.0
            else:
                self.force_consistency_history[env_ids] = 0.0
                self.force_consistency_count[env_ids] = 0

        # Reset oscillation penalty state
        if self.enable_oscillation_penalty:
            self.force_action_history[env_ids] = 0.0
            self.force_action_history_idx[env_ids] = 0

        # Reset contact transition state
        if self.enable_contact_transition_reward:
            self.force_magnitude_history[env_ids] = 0.0

        # Reset efficiency state
        if self.enable_efficiency:
            self.starting_keypoint_dist[env_ids] = 0.0
            self.energy_used[env_ids] = 0.0
            self.episode_initialized[env_ids] = False

        # Call original _reset_idx
        if self._original_reset_idx:
            self._original_reset_idx(env_ids)

    def step(self, action):
        """Execute one environment step with force reward processing."""
        # Initialize wrapper if not done yet
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        return super().step(action)

    def reset(self, **kwargs):
        """Reset the environment and ensure wrapper is initialized."""
        obs, info = super().reset(**kwargs)

        # Initialize wrapper if not done yet
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        return obs, info

    def _wrapped_get_rewards(self):
        """Calculate rewards including force-based rewards."""
        # Get original rewards - use the environment's original function if available
        if hasattr(self.unwrapped, '_get_rewards') and self._original_get_rewards is None:
            # If this is called before initialization, call the unwrapped version directly
            base_rewards = self.unwrapped._get_rewards()
        else:
            base_rewards = self._original_get_rewards() if self._original_get_rewards else torch.zeros(self.num_envs, device=self.device)

        if not self.enabled:
            return base_rewards

        # Update contact detection and state
        self._update_contact_detection()
        self._update_efficiency_state()

        # Calculate force-based rewards
        total_force_reward = torch.zeros_like(base_rewards)

        # 1. Force Magnitude Reward (always active)
        if self.enable_force_magnitude_reward:
            force_mag_reward = self._calculate_force_magnitude_reward()
            total_force_reward += force_mag_reward * self.force_magnitude_reward_weight
            self.unwrapped.extras['logs_rew_force_magnitude'] = force_mag_reward * self.force_magnitude_reward_weight

        # 2. Alignment Award (contact only)
        if self.enable_alignment_award:
            alignment_reward = self._calculate_alignment_award()
            total_force_reward += alignment_reward * self.alignment_award_reward_weight
            self.unwrapped.extras['logs_rew_alignment_award'] = alignment_reward * self.alignment_award_reward_weight

        # 3. Force Action Error (always active)
        if self.enable_force_action_error:
            action_error_reward = self._calculate_force_action_error()
            total_force_reward += action_error_reward * self.force_action_error_reward_weight
            self.unwrapped.extras['logs_rew_force_action_error'] = action_error_reward * self.force_action_error_reward_weight

        # 4. Contact Consistency (contact only)
        if self.enable_contact_consistency:
            consistency_reward = self._calculate_contact_consistency()
            total_force_reward += consistency_reward * self.contact_consistency_reward_weight
            self.unwrapped.extras['logs_rew_contact_consistency'] = consistency_reward * self.contact_consistency_reward_weight

        # 5. Oscillation Penalty (always active)
        if self.enable_oscillation_penalty:
            oscillation_reward = self._calculate_oscillation_penalty()
            total_force_reward += oscillation_reward * self.oscillation_penalty_reward_weight
            self.unwrapped.extras['logs_rew_oscillation_penalty'] = oscillation_reward * self.oscillation_penalty_reward_weight

        # 6. Contact Transition Reward (on transitions)
        if self.enable_contact_transition_reward:
            transition_reward = self._calculate_contact_transition_reward()
            total_force_reward += transition_reward * self.contact_transition_reward_weight
            self.unwrapped.extras['logs_rew_contact_transition'] = transition_reward * self.contact_transition_reward_weight

        # 7. Efficiency (always active)
        if self.enable_efficiency:
            efficiency_reward = self._calculate_efficiency()
            total_force_reward += efficiency_reward * self.efficiency_reward_weight
            self.unwrapped.extras['logs_rew_efficiency'] = efficiency_reward * self.efficiency_reward_weight

        # 8. Force Ratio (contact only)
        if self.enable_force_ratio:
            ratio_reward = self._calculate_force_ratio()
            total_force_reward += ratio_reward * self.force_ratio_reward_weight
            self.unwrapped.extras['logs_rew_force_ratio'] = ratio_reward * self.force_ratio_reward_weight

        if self.enable_contact_rew:
            contact_rew = self._calculate_contact_reward()
            total_force_reward += contact_rew * self.contact_rew_weight
            self.unwrapped.extras['logs_rew_contact_rew'] = contact_rew * self.contact_rew_weight

        if self.enable_square_vel:
            square_vel_rew = self._calculate_square_vel_reward()
            total_force_reward += square_vel_rew * self.square_vel_weight
            self.unwrapped.extras['logs_rew_square_vel_rew'] = square_vel_rew * self.square_vel_weight

        # Return combined rewards
        return base_rewards + total_force_reward

    def _update_contact_detection(self): # TODO: NOT SURE THIS TAKES FORCE HISTORY EVEN IF INDES EMPTY
        """Update contact detection state based on force magnitude moving average."""
        # Get current force magnitude
        current_force = self.unwrapped.robot_force_torque[:, :3]
        force_magnitude = torch.linalg.norm(current_force, dim=1)

        # Update force history (circular buffer)
        self.force_history[torch.arange(self.num_envs), self.force_history_idx] = force_magnitude
        self.force_history_idx = (self.force_history_idx + 1) % self.contact_window_size

        # Calculate moving average
        #force_moving_avg = self.force_history.mean(dim=1)

        # Update contact state
        self.prev_contact = self.in_contact.clone()
        #self.in_contact = force_moving_avg > self.contact_force_threshold
        self.in_contact = self.unwrapped.in_contact[:, :3]

    def _update_efficiency_state(self):
        """Update efficiency calculation state."""
        if not self.enable_efficiency:
            return

        # Initialize starting keypoint distance for new episodes
        if hasattr(self.unwrapped, '_get_keypoint_dist'):
            current_keypoint_dist = self._get_keypoint_dist()

            # Initialize for environments that haven't been initialized yet
            new_episodes = ~self.episode_initialized
            self.starting_keypoint_dist[new_episodes] = current_keypoint_dist[new_episodes]
            self.episode_initialized[new_episodes] = True

        # Update energy usage  
        if hasattr(self.unwrapped, 'joint_vel') and hasattr(self.unwrapped, 'joint_torque'):
            joint_vel = self.unwrapped.joint_vel # REAL
            joint_torque = self.unwrapped.joint_torque # REAL (calculated in ctrl action logic)
            energy_step = torch.linalg.norm(joint_vel * joint_torque, dim=1)
            self.energy_used += energy_step

    def _get_keypoint_dist(self): 
        """Get current keypoint distance from environment."""
        # Try to access factory environment's keypoint distance calculation
        return self.unwrapped.keypoint_dist

    def _get_force_action_from_env(self): #TODO NEEDS TO KNOW WHAT IDXS ARE FORCE RELATED
        """Extract force action commands from environment action space."""
        # This is a placeholder - actual implementation depends on environment's action space structure
        # For factory environments, force commands are typically in actions[:, 6:9] or similar
        if hasattr(self.unwrapped, 'actions'):
            actions = self.unwrapped.actions
            if actions.shape[1] >= 9:  # Assuming force is in positions 6:9
                return actions[:, 6:9]
            else:
                return torch.zeros((self.num_envs, 3), device=self.device)
        else:
            return torch.zeros((self.num_envs, 3), device=self.device)

    # Reward Function Implementations
    def _calculate_force_magnitude_reward(self) -> torch.Tensor:
        """
        Calculate force magnitude reward.

        r = sign(base_force - ||F||) * (base_force - ||F||)² if keep_sign=True
        r = (base_force - ||F||)² if keep_sign=False
        """
        current_force = self.unwrapped.robot_force_torque[:, :3]
        force_magnitude = torch.linalg.norm(current_force, dim=1)

        # ensures 1 at 0 force and 0 at magnitude
        # force magnitude must be a positive number
        norm = self.force_magnitude_base_force ** 3 
        diff = self.force_magnitude_base_force - force_magnitude

        if self.force_magnitude_keep_sign:
            reward = torch.sign(diff) * (diff ** 2) / norm
        else:
            reward = diff ** 2 / norm

        reward = torch.where(torch.any(self.in_contact, dim=-1), reward, torch.zeros_like(reward))
        return reward
    
    def _calculate_contact_reward(self) -> torch.Tensor:
        return self.in_contact[:,2] 
    
    def _calculate_alignment_award(self) -> torch.Tensor: #TODO SHOULD GET CURRENT FORCE ONCE, 
        """
        Calculate alignment award (contact only).

        r = F · goal_orientation (dot product)
        """
        current_force = self.unwrapped.robot_force_torque[:, :3]

        # Calculate dot product
        reward = torch.sum(current_force * self.alignment_goal_orientation, dim=1)

        # Apply only when in contact
        reward = torch.where(torch.any(self.in_contact, dim=-1), reward, torch.zeros_like(reward))

        return reward

    def _calculate_force_action_error(self) -> torch.Tensor:
        """
        Calculate force action error.

        r = -||F_action - F_ee|| (negative L2 norm)
        """
        current_force = self.unwrapped.robot_force_torque[:, :3]
        force_action = self._get_force_action_from_env()

        error = torch.linalg.norm(force_action - current_force, dim=1)
        reward = -error  # Negative because it's a penalty

        return reward

    def _calculate_contact_consistency(self) -> torch.Tensor:
        """
        Calculate contact consistency reward (contact only).

        r = exp(-beta * ||F - F_ra||)
        """
        current_force = self.unwrapped.robot_force_torque[:, :3]

        if self.contact_consistency_use_ema:
            # Update EMA
            self.force_ema = (self.contact_consistency_ema_alpha * current_force +
                             (1 - self.contact_consistency_ema_alpha) * self.force_ema)
            running_avg = self.force_ema
        else:
            # Use true running average (simplified implementation) #TODO NEED REAL RUNNING AVG
            running_avg = current_force  # Placeholder - would need proper circular buffer

        # Calculate consistency reward
        diff = torch.linalg.norm(current_force - running_avg, dim=1)
        reward = torch.exp(-self.contact_consistency_beta * diff)

        # Apply only when in contact
        reward = torch.where(torch.any(self.in_contact, dim=-1), reward, torch.zeros_like(reward))

        return reward

    def _calculate_oscillation_penalty(self) -> torch.Tensor:
        """
        Calculate oscillation penalty.

        r = -||F_action - F_action_window_avg||
        """
        force_action = self._get_force_action_from_env()

        # Update force action history 
        self.force_action_history[torch.arange(self.num_envs), self.force_action_history_idx] = force_action
        self.force_action_history_idx = (self.force_action_history_idx + 1) % self.oscillation_penalty_window_size

        # Calculate window average
        window_avg = self.force_action_history.mean(dim=1)

        # Calculate penalty
        diff = torch.linalg.norm(force_action - window_avg, dim=1)
        reward = -diff  # Negative because it's a penalty

        return reward

    def _calculate_contact_transition_reward(self) -> torch.Tensor:
        """
        Calculate contact transition reward.

        r = max(||F_acceleration||) over window when transitioning
        """
        current_force = self.unwrapped.robot_force_torque[:, :3]
        force_magnitude = torch.linalg.norm(current_force, dim=1)

        # Update force magnitude history 
        self.force_magnitude_history[torch.arange(self.num_envs), self.force_history_idx] = force_magnitude

        # Detect transitions (not-in-contact -> in-contact)
        transitions = self.in_contact & ~self.prev_contact

        # Calculate max acceleration over window for transitioning environments
        if torch.any(transitions): #TODO NEED TO CALCULATE ACTUAL ACCELERATIONS
            # Calculate acceleration (simplified as magnitude differences)
            force_diffs = torch.diff(self.force_magnitude_history, dim=1)
            max_acceleration = torch.max(torch.abs(force_diffs), dim=1)[0]

            reward = torch.where(transitions, max_acceleration, torch.zeros_like(max_acceleration))
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def _calculate_efficiency(self) -> torch.Tensor:
        """
        Calculate efficiency reward.

        r = progress / energy_used
        where progress = 1 - current_keypoint_dist / starting_keypoint_dist
        """
        current_keypoint_dist = self._get_keypoint_dist()

        # Calculate progress
        progress = 1.0 - (current_keypoint_dist / (self.starting_keypoint_dist + 1e-8))  # Add small epsilon to avoid division by zero
        progress = torch.clamp(progress, 0.0, 1.0)  # Clamp to [0, 1]

        # Calculate efficiency
        energy_with_epsilon = self.energy_used + 1e-8  # Avoid division by zero
        efficiency = progress / energy_with_epsilon

        # Only apply to initialized episodes
        reward = torch.where(self.episode_initialized, efficiency, torch.zeros_like(efficiency))

        return reward

    def _calculate_force_ratio(self) -> torch.Tensor:
        """
        Calculate force ratio reward (contact only).

        r = |F_z| / ||F_xy||
        """
        current_force = self.unwrapped.robot_force_torque[:, :3]

        # Extract force components
        f_z = torch.abs(current_force[:, 2])  # Z component
        f_xy = torch.linalg.norm(current_force[:, :2], dim=1)  # XY magnitude

        # Calculate ratio (avoid division by zero)
        ratio = f_z / (f_xy + 1e-8)

        # Apply only when in contact
        reward = torch.where(torch.any(self.in_contact, dim=-1),ratio, torch.zeros_like(ratio))

        return reward
    
    def _calculate_square_vel_reward(self) -> torch.Tensor:                                                                                                                                                  
        """                                                                                                                                                                                                  
        Calculate squared velocity reward.                                                                                                                                                                   
                                                                                                                                                                                                            
        r = -(v_x^2 + v_y^2 + v_z^2) for each env                                                                                                                                                            
        Negative because we want to penalize high velocities.                                                                                                                                                
        """                                                                                                                                                                                                  
        # Get end-effector velocity                                                                                                                                                                          
        ee_vel = self.unwrapped.fingertip_midpoint_linvel  # [num_envs, 3]                                                                                                                                                                                  
        # Sum of squared components per env                                                                                                                                                                  
        square_vel_sum = torch.sum(ee_vel ** 2, dim=1)  # [num_envs]                                                                                                                                                                    
        return -square_vel_sum  # Negative as penalty  

