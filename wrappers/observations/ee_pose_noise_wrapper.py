#!/usr/bin/env python3
"""
End-Effector Pose Noise Wrapper

This wrapper adds Gaussian noise to end-effector position observations with noise
naturally propagating to finite difference velocity calculations.

Key Features:
- Per-step Gaussian noise on EE position (applied in _compute_intermediate_values)
- FD velocity computed from noisy positions
- Noise propagates through (noisy_pos_t - noisy_pos_t-1) / dt
- Creates separate noisy tensors (Forge-style approach)
- Clean values remain available for control and rewards
- Compatible with Isaac Lab factory environment configurations
- Works with history wrapper when placed before it in the stack
"""

import torch
import gymnasium as gym


class EEPoseNoiseWrapper(gym.Wrapper):
    """
    Wrapper that adds per-step Gaussian noise to end-effector position observations.

    The noise is applied in _compute_intermediate_values by creating separate
    noisy_fingertip_pos and noisy_ee_linvel_fd tensors. Clean values remain
    unchanged for control and rewards. Only observations see noisy values.

    Features:
    - Per-step Gaussian noise generation
    - Noise scaled by cfg.obs_rand.ee_pos configuration
    - FD velocity calculation from noisy positions
    - Maintains prev_noisy_ee_pos state for FD computation
    - Forge-style separate noisy tensors
    """

    def __init__(self, env):
        """
        Initialize end-effector pose noise wrapper.

        Args:
            env: Base environment to wrap
        """
        super().__init__(env)

        # Check that no history wrapper is already applied - history must be the last wrapper
        self._check_no_history_wrapper()

        # Get device and number of environments
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # Get noise scale from config
        if not hasattr(self.unwrapped.cfg, 'obs_rand'):
            raise ValueError(
                "Environment config missing 'obs_rand' attribute. "
                "Please add ExtendedObsRandCfg to your environment configuration."
            )

        if not hasattr(self.unwrapped.cfg.obs_rand, 'ee_pos'):
            raise ValueError(
                "obs_rand config missing 'ee_pos' attribute. "
                "Please add ee_pos field to ExtendedObsRandCfg."
            )

        # Convert noise scale to tensor
        self.ee_pos_noise_scale = torch.tensor(
            self.unwrapped.cfg.obs_rand.ee_pos,
            dtype=torch.float32,
            device=self.device
        )

        # Initialize previous noisy position for FD calculation
        # Will be properly initialized on first reset
        self.prev_noisy_ee_pos = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device
        )

        # Flag to track if wrapper is initialized
        self._wrapper_initialized = False

        # Store original methods
        self._original_compute_intermediate_values = None
        self._original_get_observations = None
        self._original_reset_idx = None

        # Initialize after the base environment is ready
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        print(f" EE Pose Noise Wrapper initialized")
        print(f"  - Noise scale (ee_pos): {self.unwrapped.cfg.obs_rand.ee_pos}")
        print(f"  - Noise timing: per-step (in _compute_intermediate_values)")
        print(f"  - FD velocity: computed from noisy positions")
        print(f"  - Style: Forge-style separate noisy tensors")

    def _check_no_history_wrapper(self):
        """
        Check that no history wrapper is already applied in the wrapper chain.

        History wrapper must be the last wrapper applied to ensure proper observation
        space calculation and buffer management. If a history wrapper is detected,
        raise an error with clear instructions.
        """
        current_env = self.env
        while current_env is not None:
            # Check if current wrapper is a history wrapper
            if hasattr(current_env, '__class__') and 'HistoryObservationWrapper' in str(current_env.__class__):
                raise ValueError(
                    "ERROR: History wrapper detected in wrapper chain before EEPoseNoiseWrapper.\n"
                    "\n"
                    "SOLUTION: History wrapper must be applied LAST in the wrapper chain.\n"
                    "Correct order:\n"
                    "  1. Apply sensor wrappers (ForceTorqueWrapper, etc.)\n"
                    "  2. Apply EEPoseNoiseWrapper\n"
                    "  3. Apply HistoryObservationWrapper last\n"
                    "\n"
                    "Current wrapper chain violates this requirement.\n"
                    "Please reorder your wrapper application to ensure HistoryObservationWrapper is applied last."
                )

            # Move to next wrapper in chain
            if hasattr(current_env, 'env'):
                current_env = current_env.env
            else:
                break

    def reset(self, **kwargs):
        """
        Override reset to ensure wrapper is initialized before first use.

        The wrapper may not be fully initialized during __init__ because the
        environment's _robot attribute might not exist yet. This method ensures
        initialization happens on the first reset call when _robot is available.

        Args:
            **kwargs: Arguments passed to the base environment's reset method

        Returns:
            Observation and info from base environment's reset
        """
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()
        return super().reset(**kwargs)

    def _initialize_wrapper(self):
        """
        Initialize the wrapper after the base environment is set up.

        This method performs lazy initialization by overriding environment methods.
        It's called automatically when the environment's robot attribute is detected
        or during first step/reset.
        """
        if self._wrapper_initialized:
            return

        # Store and override _compute_intermediate_values method
        if hasattr(self.unwrapped, '_compute_intermediate_values'):
            self._original_compute_intermediate_values = self.unwrapped._compute_intermediate_values
            self.unwrapped._compute_intermediate_values = self._wrapped_compute_intermediate_values
        else:
            raise ValueError(
                "Factory environment missing required '_compute_intermediate_values' method. "
                "This wrapper requires Isaac Lab factory environment."
            )

        # Store and override _get_observations method
        if hasattr(self.unwrapped, '_get_observations'):
            self._original_get_observations = self.unwrapped._get_observations
            self.unwrapped._get_observations = self._wrapped_get_observations
        else:
            raise ValueError(
                "Factory environment missing required '_get_observations' method. "
                "This wrapper requires Isaac Lab factory environment."
            )

        # Store and override _reset_idx method
        if hasattr(self.unwrapped, '_reset_idx'):
            self._original_reset_idx = self.unwrapped._reset_idx
            self.unwrapped._reset_idx = self._wrapped_reset_idx
        else:
            raise ValueError(
                "Factory environment missing required '_reset_idx' method. "
                "This wrapper requires Isaac Lab factory environment."
            )

        # Initialize noisy tensors
        self.unwrapped.noisy_fingertip_pos = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device
        )
        self.unwrapped.noisy_ee_linvel_fd = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device
        )

        self._wrapper_initialized = True

    def _wrapped_compute_intermediate_values(self, dt):
        """
        Wrap _compute_intermediate_values to create noisy EE position and velocity.

        This method:
        1. Calls original _compute_intermediate_values (computes clean values)
        2. Generates per-step Gaussian noise
        3. Creates noisy_fingertip_pos = clean_pos + noise
        4. Computes noisy_ee_linvel_fd from noisy positions
        5. Updates prev_noisy_ee_pos for next step

        Clean values remain unchanged for control and rewards.
        Only observations use noisy values.

        Args:
            dt (float): Physics simulation timestep
        """
        # Call original computation (computes clean values)
        if self._original_compute_intermediate_values:
            self._original_compute_intermediate_values(dt)

        # Generate per-step Gaussian noise
        # noise ~ N(0, 1), then scale by ee_pos_noise_scale
        ee_pos_noise = torch.randn(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device
        )
        ee_pos_noise = ee_pos_noise * self.ee_pos_noise_scale

        # Create noisy EE position (clean value remains unchanged)
        self.unwrapped.noisy_fingertip_pos = self.unwrapped.fingertip_midpoint_pos + ee_pos_noise

        # Compute FD linear velocity from noisy positions
        # noisy_ee_linvel_fd = (noisy_pos_t - noisy_pos_t-1) / dt
        self.unwrapped.noisy_ee_linvel_fd = (self.unwrapped.noisy_fingertip_pos - self.prev_noisy_ee_pos) / dt

        # Update previous noisy position for next step
        self.prev_noisy_ee_pos = self.unwrapped.noisy_fingertip_pos.clone()

    def _wrapped_get_observations(self):
        """
        Override _get_observations to use noisy EE position and velocity.

        This method builds observation dicts similar to ForceTorqueWrapper but uses
        noisy values for policy observations and clean values for critic state.

        Returns:
            dict: Observation dict with 'policy' and 'critic' keys
        """
        # Get noisy fixed position from environment
        noisy_fixed_pos = self.unwrapped.fixed_pos_obs_frame + self.unwrapped.init_fixed_pos_obs_noise

        # Get previous actions
        prev_actions = self.unwrapped.actions.clone()

        # Build observation dict with NOISY EE position and velocity for policy
        obs_dict = {
            "fingertip_pos": self.unwrapped.noisy_fingertip_pos,  # Noisy position
            "fingertip_pos_rel_fixed": self.unwrapped.noisy_fingertip_pos - noisy_fixed_pos,  # Noisy relative
            "fingertip_quat": self.unwrapped.fingertip_midpoint_quat,  # Clean orientation
            "ee_linvel": self.unwrapped.noisy_ee_linvel_fd,  # Noisy FD velocity
            "ee_angvel": self.unwrapped.ee_angvel_fd,  # Clean angular velocity
            "prev_actions": prev_actions,
        }

        # Build state dict with CLEAN values for critic
        state_dict = {
            "fingertip_pos": self.unwrapped.fingertip_midpoint_pos,  # Clean position
            "fingertip_pos_rel_fixed": self.unwrapped.fingertip_midpoint_pos - self.unwrapped.fixed_pos_obs_frame,
            "fingertip_quat": self.unwrapped.fingertip_midpoint_quat,
            "ee_linvel": self.unwrapped.fingertip_midpoint_linvel,  # Clean velocity
            "ee_angvel": self.unwrapped.fingertip_midpoint_angvel,
            "joint_pos": self.unwrapped.joint_pos[:, 0:7],
            "held_pos": self.unwrapped.held_pos,
            "held_pos_rel_fixed": self.unwrapped.held_pos - self.unwrapped.fixed_pos_obs_frame,
            "held_quat": self.unwrapped.held_quat,
            "fixed_pos": self.unwrapped.fixed_pos,
            "fixed_quat": self.unwrapped.fixed_quat,
            "task_prop_gains": self.unwrapped.cfg.ctrl.default_task_prop_gains,
            "pos_threshold": self.unwrapped.cfg.ctrl.pos_action_threshold,
            "rot_threshold": self.unwrapped.cfg.ctrl.rot_action_threshold,
            "prev_actions": prev_actions,
        }

        # Add force_torque if it exists in obs/state order
        if hasattr(self.unwrapped.cfg, 'obs_order') and 'force_torque' in self.unwrapped.cfg.obs_order:
            if hasattr(self.unwrapped, 'robot_force_torque'):
                obs_dict['force_torque'] = self.unwrapped.robot_force_torque

        if hasattr(self.unwrapped.cfg, 'state_order') and 'force_torque' in self.unwrapped.cfg.state_order:
            if hasattr(self.unwrapped, 'robot_force_torque'):
                state_dict['force_torque'] = self.unwrapped.robot_force_torque

        # Add in_contact if it exists in obs/state order
        if hasattr(self.unwrapped.cfg, 'obs_order') and 'in_contact' in self.unwrapped.cfg.obs_order:
            if hasattr(self.unwrapped, 'in_contact'):
                obs_dict['in_contact'] = self.unwrapped.in_contact[:, :3]

        if hasattr(self.unwrapped.cfg, 'state_order') and 'in_contact' in self.unwrapped.cfg.state_order:
            if hasattr(self.unwrapped, 'in_contact'):
                state_dict['in_contact'] = self.unwrapped.in_contact[:, :3]

        # Concatenate observations in the correct order
        obs_tensors = [obs_dict[obs_name] for obs_name in self.unwrapped.cfg.obs_order + ["prev_actions"]]
        obs_tensors = torch.cat(obs_tensors, dim=-1)

        state_tensors = [state_dict[state_name] for state_name in self.unwrapped.cfg.state_order + ["prev_actions"]]
        state_tensors = torch.cat(state_tensors, dim=-1)

        return {"policy": obs_tensors, "critic": state_tensors}

    def _wrapped_reset_idx(self, env_ids):
        """
        Reset wrapper state for specified environments.

        This method wraps the environment's original _reset_idx method and
        additionally initializes prev_noisy_ee_pos for reset environments.

        Args:
            env_ids (torch.Tensor): Indices of environments to reset
        """
        # Call original reset
        if self._original_reset_idx:
            self._original_reset_idx(env_ids)

        # Initialize prev_noisy_ee_pos for reset environments
        # Use clean position + initial noise as starting point
        num_reset_envs = len(env_ids)

        # Generate initial noise for reset environments
        initial_noise = torch.randn(
            (num_reset_envs, 3),
            dtype=torch.float32,
            device=self.device
        )
        initial_noise = initial_noise * self.ee_pos_noise_scale

        # Set prev_noisy_ee_pos = clean_pos + noise for reset envs
        # Note: fingertip_midpoint_pos has already been updated by original _reset_idx
        clean_ee_pos = self.unwrapped.fingertip_midpoint_pos[env_ids]
        self.prev_noisy_ee_pos[env_ids] = clean_ee_pos + initial_noise


# Example usage
if __name__ == "__main__":
    """Example usage of EE pose noise wrapper."""

    print("EE Pose Noise Wrapper - Example")
    print("=" * 50)

    print("\n✓ Key features:")
    print("  - Per-step Gaussian noise on EE position")
    print("  - Applied in _compute_intermediate_values")
    print("  - FD velocity computed from noisy positions")
    print("  - Noise propagates: (noisy_pos_t - noisy_pos_t-1) / dt")
    print("  - Forge-style: separate noisy tensors")
    print("  - Policy sees noisy values, control/rewards use clean values")
    print("  - Compatible with history wrapper")
    print("  - Configurable via cfg.obs_rand.ee_pos")

    print("\n✓ Configuration:")
    print("  Add to your environment config:")
    print("  obs_rand.ee_pos = [0.00025, 0.00025, 0.00025]  # 0.25mm per axis")

    print("\n✓ Usage:")
    print("  env = create_factory_env(...)")
    print("  env = ForceTorqueWrapper(env)  # If using force-torque")
    print("  env = EEPoseNoiseWrapper(env)")
    print("  env = HistoryObservationWrapper(env)  # History must be last!")
