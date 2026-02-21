"""
Variable Impedance Control (VIC) Pose Wrapper

This wrapper extends pure pose control by letting the model output 3 additional
actions representing translational Kp gains (x, y, z). Rotational gains remain fixed.

Action space: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, kp_x, kp_y, kp_z] = 9 dims

The gain actions are mapped from [-1, 1] (tanh output) to [kp_min, kp_max] via linear
interpolation. Kd gains are auto-derived as 2 * sqrt(Kp) for critical damping.
"""

import torch
import gymnasium as gym
import numpy as np


class VICPoseWrapper(gym.Wrapper):
    """
    Wrapper implementing Variable Impedance Control for pose-only environments.

    The model controls 6 pose actions + 3 translational stiffness (Kp) gains.
    Rotational gains remain fixed at their default values.
    """

    # Number of gain dimensions (translational only)
    GAIN_DIMS = 3

    def __init__(self, env, ctrl_cfg, apply_ema_to_gains=False):
        """
        Initialize VIC pose wrapper.

        Args:
            env: Base environment to wrap (must be pose-only, no hybrid control)
            ctrl_cfg: ExtendedCtrlCfg instance with VIC gain bounds
            apply_ema_to_gains: If True, apply EMA smoothing to gain actions.
                                If False, use raw gain actions each step.
        """
        # Store original action space size before modification
        self._original_action_size = getattr(env.unwrapped.cfg, 'action_space', 6)

        super().__init__(env)

        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.apply_ema_to_gains = apply_ema_to_gains

        # Require explicit configuration
        if ctrl_cfg is None:
            raise ValueError(
                "ctrl_cfg cannot be None. Please provide an ExtendedCtrlCfg instance "
                "with VIC gain bounds (vic_gain_min_pos, vic_gain_max_pos)."
            )
        self.ctrl_cfg = ctrl_cfg

        # Validate VIC gain bounds
        if ctrl_cfg.vic_gain_min_pos is None or ctrl_cfg.vic_gain_max_pos is None:
            raise ValueError(
                "VIC gain bounds must be set. Please provide vic_gain_min_pos and "
                "vic_gain_max_pos in ctrl_cfg."
            )

        # Store gain bounds as tensors [num_envs, 3]
        self.kp_min = torch.tensor(
            ctrl_cfg.vic_gain_min_pos, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.kp_max = torch.tensor(
            ctrl_cfg.vic_gain_max_pos, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        # Validate min < max
        if not all(lo < hi for lo, hi in zip(ctrl_cfg.vic_gain_min_pos, ctrl_cfg.vic_gain_max_pos)):
            raise ValueError(
                f"vic_gain_min_pos={ctrl_cfg.vic_gain_min_pos} must be element-wise less than "
                f"vic_gain_max_pos={ctrl_cfg.vic_gain_max_pos}"
            )

        # Store default gains for reset
        self.default_kp_pos = torch.tensor(
            ctrl_cfg.default_task_prop_gains[:3], device=self.device
        )

        # Validate default gains are within bounds
        kp_min_list = ctrl_cfg.vic_gain_min_pos
        kp_max_list = ctrl_cfg.vic_gain_max_pos
        default_kp_list = ctrl_cfg.default_task_prop_gains[:3]
        for i, (lo, hi, default) in enumerate(zip(kp_min_list, kp_max_list, default_kp_list)):
            if default < lo or default > hi:
                print(f"[VIC] WARNING: default_task_prop_gains[{i}]={default} is outside "
                      f"VIC bounds [{lo}, {hi}]. Reset gains will be clamped.")

        # EMA factor from ctrl config
        self.ema_factor = ctrl_cfg.ema_factor

        # EMA state for gain actions
        self.ema_gain_actions = torch.zeros(
            (self.num_envs, self.GAIN_DIMS), device=self.device
        )

        # New action space size: original + 3 gain dims
        self._new_action_size = self._original_action_size + self.GAIN_DIMS

        # Update action/observation/state space dimensions
        self._update_dimensions()

        # Store original methods
        self._original_pre_physics_step = None
        self._original_reset_idx = None

        # Initialize wrapper
        self._wrapper_initialized = False
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()
        if hasattr(self.unwrapped, 'extras'):
            if 'to_log' not in self.unwrapped.extras.keys():
                self.unwrapped.extras['to_log'] = {}

    def _update_dimensions(self):
        """Update action, observation, and state space dimensions for added gain actions."""
        action_diff = self._new_action_size - self._original_action_size

        if hasattr(self.unwrapped.cfg, 'observation_space'):
            self.unwrapped.cfg.observation_space += action_diff

        if hasattr(self.unwrapped.cfg, 'state_space'):
            self.unwrapped.cfg.state_space += action_diff

        if hasattr(self.unwrapped.cfg, 'action_space'):
            self.unwrapped.cfg.action_space = self._new_action_size

        # Call _configure_gym_env_spaces first (it may rebuild gym spaces from cfg),
        # then overwrite with our correct Box shape after.
        # This matches HybridForcePositionWrapper's ordering.
        if hasattr(self.unwrapped, '_configure_gym_env_spaces'):
            try:
                self.unwrapped._configure_gym_env_spaces()
            except Exception:
                raise RuntimeError("Unable to configure gym env spaces in VIC wrapper")

        # Overwrite gym action_space with correct shape AFTER _configure_gym_env_spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._new_action_size,), dtype=np.float32
        )
        self.unwrapped.action_space = self.action_space

        # Validate consistency
        cfg_size = getattr(self.unwrapped.cfg, 'action_space', None)
        gym_size = self.unwrapped.action_space.shape[0] if hasattr(self.unwrapped.action_space, 'shape') else None
        if cfg_size != gym_size:
            raise ValueError(
                f"Action space mismatch: cfg.action_space={cfg_size} != "
                f"gym.action_space.shape[0]={gym_size}"
            )

    def _initialize_wrapper(self):
        """Initialize wrapper by overriding environment methods."""
        if self._wrapper_initialized:
            return

        # Store and override methods
        if hasattr(self.unwrapped, '_pre_physics_step'):
            self._original_pre_physics_step = self.unwrapped._pre_physics_step
            self.unwrapped._pre_physics_step = self._wrapped_pre_physics_step
        else:
            raise RuntimeError("[VIC] _pre_physics_step not found on unwrapped env!")

        if hasattr(self.unwrapped, '_reset_idx'):
            self._original_reset_idx = self.unwrapped._reset_idx
            self.unwrapped._reset_idx = self._wrapped_reset_idx
        else:
            raise RuntimeError("[VIC] _reset_idx not found on unwrapped env!")

        self._wrapper_initialized = True

    def _map_gain_actions_to_kp(self, gain_actions):
        """Map gain actions from [-1, 1] to [kp_min, kp_max] via linear interpolation.

        Clamps input to [-1, 1] first because sampled actions from the Gaussian
        can exceed tanh bounds (noise added after tanh on mean).

        Args:
            gain_actions: Tensor of shape [num_envs, 3] (may exceed [-1, 1] from sampling)

        Returns:
            Kp gains tensor of shape [num_envs, 3] in range [kp_min, kp_max]
        """
        # Clamp to [-1, 1] — Gaussian samples can exceed tanh bounds
        clamped = torch.clamp(gain_actions, -1.0, 1.0)

        # Linear map: kp = kp_min + (action + 1) / 2 * (kp_max - kp_min)
        kp = self.kp_min + (clamped + 1.0) / 2.0 * (self.kp_max - self.kp_min)

        return kp

    def _wrapped_pre_physics_step(self, action):
        """Process actions: split into pose and gain components, apply gains, then delegate."""
        if action.shape[1] != self._new_action_size:
            raise ValueError(
                f"[VIC] Action dimension mismatch! Got {action.shape[1]}, "
                f"expected {self._new_action_size}"
            )

        # Handle reset environments
        env_ids = self.unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_vic_state(env_ids)

        # Split action into pose (6) and gain (3) components
        pose_actions = action[:, :self._original_action_size]
        gain_actions = action[:, self._original_action_size:]

        # Apply EMA to gain actions if configured
        if self.apply_ema_to_gains:
            self.ema_gain_actions = (
                self.ema_factor * gain_actions +
                (1 - self.ema_factor) * self.ema_gain_actions
            )
            effective_gain_actions = self.ema_gain_actions
        else:
            effective_gain_actions = gain_actions
            # Still store for observation consistency
            self.ema_gain_actions = gain_actions.clone()

        # Map gain actions to Kp values
        kp_pos = self._map_gain_actions_to_kp(effective_gain_actions)

        # Set translational gains (rotational gains remain unchanged)
        self.unwrapped.task_prop_gains[:, 0:3] = kp_pos

        # Recalculate translational derivative gains: kd = 2 * sqrt(kp)
        self.unwrapped.task_deriv_gains[:, 0:3] = 2.0 * torch.sqrt(kp_pos)

        # Log commanded gains
        if hasattr(self.unwrapped, 'extras'):
            self.unwrapped.extras['to_log']['VIC / Commanded Kp X'] = kp_pos[:, 0]
            self.unwrapped.extras['to_log']['VIC / Commanded Kp Y'] = kp_pos[:, 1]
            self.unwrapped.extras['to_log']['VIC / Commanded Kp Z'] = kp_pos[:, 2]

            kd_pos = self.unwrapped.task_deriv_gains[:, 0:3]
            self.unwrapped.extras['to_log']['VIC / Commanded Kd X'] = kd_pos[:, 0]
            self.unwrapped.extras['to_log']['VIC / Commanded Kd Y'] = kd_pos[:, 1]
            self.unwrapped.extras['to_log']['VIC / Commanded Kd Z'] = kd_pos[:, 2]

            # Log raw gain actions
            self.unwrapped.extras['to_log']['VIC / Raw Gain Action X'] = gain_actions[:, 0]
            self.unwrapped.extras['to_log']['VIC / Raw Gain Action Y'] = gain_actions[:, 1]
            self.unwrapped.extras['to_log']['VIC / Raw Gain Action Z'] = gain_actions[:, 2]

        # Call original pre_physics_step with full 9-dim action.
        # The base env does: self.actions = ema * action + (1-ema) * self.actions
        # so we must pass the same shape every step to avoid shape mismatch.
        # The base env only reads [:3] and [3:6] for control, so gain dims at [6:9]
        # are carried harmlessly through its EMA.
        if self._original_pre_physics_step:
            self._original_pre_physics_step(action)

        # Overwrite self.unwrapped.actions with our properly constructed 9-dim tensor.
        # The base env EMA'd the full 9-dim action, but we want our gain actions
        # (raw or VIC-EMA'd) in positions [6:9], not the base env's EMA of them.
        # This matches the pattern used by HybridForcePositionWrapper.
        if hasattr(self.unwrapped, 'actions'):
            base_pose_actions = self.unwrapped.actions[:, :self._original_action_size]
            self.unwrapped.actions = torch.cat([
                base_pose_actions, effective_gain_actions
            ], dim=-1)

    def _reset_vic_state(self, env_ids):
        """Reset VIC state for given environment IDs."""
        # Guard: task_prop_gains doesn't exist until _init_tensors() runs during first reset.
        # On that first reset the base env sets gains to defaults, so skipping is safe.
        if not hasattr(self.unwrapped, 'task_prop_gains'):
            self.ema_gain_actions[env_ids] = 0.0
            return

        # Reset EMA gain state
        self.ema_gain_actions[env_ids] = 0.0

        # Reset gains to nominal defaults
        self.unwrapped.task_prop_gains[env_ids, 0:3] = self.default_kp_pos
        self.unwrapped.task_deriv_gains[env_ids, 0:3] = 2.0 * torch.sqrt(self.default_kp_pos)

    def _wrapped_reset_idx(self, env_ids):
        """Reset wrapper state, then call original _reset_idx."""
        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self._reset_vic_state(env_ids)

        if self._original_reset_idx is not None:
            self._original_reset_idx(env_ids)

    def step(self, action):
        """Step environment and ensure wrapper is initialized."""
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()
        return super().step(action)

    def reset(self, **kwargs):
        """Reset environment and ensure wrapper is initialized."""
        obs, info = super().reset(**kwargs)
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()
        return obs, info
