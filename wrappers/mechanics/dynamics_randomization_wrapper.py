"""
Dynamics Randomization Wrapper

This wrapper implements domain randomization for physics dynamics:
- Friction coefficients (held asset only)
- Controller gains (position and rotation)
- Force/torque gains (for hybrid control)
- Action thresholds

All parameters are randomized per-environment on each reset using the reset_idxs interface.
"""

import torch
import gymnasium as gym
from typing import Dict, Any

# Import Isaac Lab utilities
try:
    import isaaclab_tasks.direct.factory.factory_control as factory_utils
except ImportError:
    try:
        import omni.isaac.lab_tasks.direct.factory.factory_control as factory_utils
    except ImportError:
        raise ImportError("Could not import Isaac Lab factory utilities")


class DynamicsRandomizationWrapper(gym.Wrapper):
    """
    Wrapper that randomizes physics dynamics on each environment reset.

    Features:
    - Per-environment friction randomization (held asset only)
    - Per-environment controller gain randomization (pos/rot separate)
    - Per-environment force/torque gain randomization (hybrid control)
    - Per-environment action threshold randomization
    - Applied on partial resets using reset_idxs interface
    """

    def __init__(self, env, config: Dict[str, Any]):
        """
        Initialize the dynamics randomization wrapper.

        Args:
            env: Base environment to wrap
            config: Configuration dictionary with randomization parameters
        """
        super().__init__(env)

        self.config = config
        self.enabled = config.get('enabled', False)

        if not self.enabled:
            return

        # Environment info
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # Store config flags
        self.randomize_friction = config.get('randomize_friction', False)
        self.randomize_gains = config.get('randomize_gains', False)
        self.randomize_force_gains = config.get('randomize_force_gains', False)
        self.randomize_pos_threshold = config.get('randomize_pos_threshold', False)
        self.randomize_rot_threshold = config.get('randomize_rot_threshold', False)
        self.randomize_force_threshold = config.get('randomize_force_threshold', False)
        self.randomize_torque_bounds = config.get('randomize_torque_bounds', False)

        # Store ranges
        self.friction_range = config.get('friction_range', [0.5, 1.5])
        self.pos_gains_range = config.get('pos_gains_range', [80.0, 120.0])
        self.rot_gains_range = config.get('rot_gains_range', [20.0, 40.0])
        self.force_gains_range = config.get('force_gains_range', [0.08, 0.12])
        self.torque_gains_range = config.get('torque_gains_range', [0.0008, 0.0012])
        self.pos_threshold_range = config.get('pos_threshold_range', [0.015, 0.025])
        self.rot_threshold_range = config.get('rot_threshold_range', [0.08, 0.12])
        self.force_threshold_range = config.get('force_threshold_range', [8.0, 12.0])
        self.torque_bounds_range = config.get('torque_bounds_range', [0.4, 0.6])

        # Initialize per-environment storage tensors
        # Friction (scalar per env)
        self.current_friction = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        # Controller gains (6-DOF: 3 pos + 3 rot)
        self.current_prop_gains = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)

        # Force/torque gains (6-DOF: 3 force + 3 torque, for hybrid control)
        self.current_force_gains = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)

        # Action thresholds (stored as scalars, will be replicated to dims when applied)
        self.current_pos_threshold = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.current_rot_threshold = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.current_force_threshold = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.current_torque_bounds = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        # Store original methods
        self._original_reset_idx = None
        self._wrapper_initialized = False

        # Initialize wrapper if environment is ready
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        print(f"[DynamicsRandomizationWrapper] Initialized")
        print(f"  Friction randomization: {self.randomize_friction}")
        print(f"  Gains randomization: {self.randomize_gains}")
        print(f"  Force gains randomization: {self.randomize_force_gains}")
        print(f"  Threshold randomization: pos={self.randomize_pos_threshold}, rot={self.randomize_rot_threshold}, "
              f"force={self.randomize_force_threshold}, torque={self.randomize_torque_bounds}")

    def _initialize_wrapper(self):
        """Initialize the wrapper after the base environment is set up."""
        if self._wrapper_initialized or not self.enabled:
            return

        # Store and override _reset_idx method
        if hasattr(self.unwrapped, '_reset_idx'):
            self._original_reset_idx = self.unwrapped._reset_idx
            self.unwrapped._reset_idx = self._wrapped_reset_idx
        else:
            raise ValueError("Environment missing required _reset_idx method")

        self._wrapper_initialized = True
        print("[DynamicsRandomizationWrapper] Wrapper initialized and injected into reset chain")

    def _wrapped_reset_idx(self, env_ids):
        """
        Reset specified environments with dynamics randomization.

        Randomizes dynamics BEFORE calling original reset so new parameters
        are in place when environment state is initialized.
        """
        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Randomize dynamics for the environments being reset
        self._randomize_dynamics(env_ids)

        # Call original reset
        if self._original_reset_idx is not None:
            self._original_reset_idx(env_ids)

    def _randomize_dynamics(self, env_ids):
        """
        Randomize dynamics parameters for specified environments.

        Args:
            env_ids: Tensor of environment indices to randomize
        """
        num_reset_envs = len(env_ids)

        # 1. Randomize friction (held asset only)
        if self.randomize_friction:
            friction_min, friction_max = self.friction_range
            sampled_friction = torch.rand(num_reset_envs, device=self.device) * (friction_max - friction_min) + friction_min
            self.current_friction[env_ids] = sampled_friction

            # Apply friction to held asset
            # Note: factory_utils.set_friction applies to ALL environments, so we need a workaround
            # We'll set the material properties directly per environment
            self._set_friction_per_env(self.unwrapped._held_asset, env_ids, sampled_friction)

        # 2. Randomize controller gains (position and rotation)
        if self.randomize_gains:
            # Sample position gains (same for all 3 position dims)
            pos_gain_min, pos_gain_max = self.pos_gains_range
            sampled_pos_gains = torch.rand(num_reset_envs, device=self.device) * (pos_gain_max - pos_gain_min) + pos_gain_min

            # Sample rotation gains (same for all 3 rotation dims)
            rot_gain_min, rot_gain_max = self.rot_gains_range
            sampled_rot_gains = torch.rand(num_reset_envs, device=self.device) * (rot_gain_max - rot_gain_min) + rot_gain_min

            # Replicate to all dimensions: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
            self.current_prop_gains[env_ids, 0:3] = sampled_pos_gains.unsqueeze(1).repeat(1, 3)
            self.current_prop_gains[env_ids, 3:6] = sampled_rot_gains.unsqueeze(1).repeat(1, 3)

            # Apply gains to environment
            self._apply_controller_gains(env_ids)

        # 3. Randomize force/torque gains (hybrid control only)
        if self.randomize_force_gains:
            # Sample force gains (same for all 3 force dims)
            force_gain_min, force_gain_max = self.force_gains_range
            sampled_force_gains = torch.rand(num_reset_envs, device=self.device) * (force_gain_max - force_gain_min) + force_gain_min

            # Sample torque gains (same for all 3 torque dims)
            torque_gain_min, torque_gain_max = self.torque_gains_range
            sampled_torque_gains = torch.rand(num_reset_envs, device=self.device) * (torque_gain_max - torque_gain_min) + torque_gain_min

            # Replicate to all dimensions: [force_x, force_y, force_z, torque_x, torque_y, torque_z]
            self.current_force_gains[env_ids, 0:3] = sampled_force_gains.unsqueeze(1).repeat(1, 3)
            self.current_force_gains[env_ids, 3:6] = sampled_torque_gains.unsqueeze(1).repeat(1, 3)

            # Apply force gains to environment
            self._apply_force_gains(env_ids)

        # 4. Randomize action thresholds
        if self.randomize_pos_threshold:
            pos_thresh_min, pos_thresh_max = self.pos_threshold_range
            sampled_pos_thresh = torch.rand(num_reset_envs, device=self.device) * (pos_thresh_max - pos_thresh_min) + pos_thresh_min
            self.current_pos_threshold[env_ids] = sampled_pos_thresh
            self._apply_pos_threshold(env_ids)

        if self.randomize_rot_threshold:
            rot_thresh_min, rot_thresh_max = self.rot_threshold_range
            sampled_rot_thresh = torch.rand(num_reset_envs, device=self.device) * (rot_thresh_max - rot_thresh_min) + rot_thresh_min
            self.current_rot_threshold[env_ids] = sampled_rot_thresh
            self._apply_rot_threshold(env_ids)

        if self.randomize_force_threshold:
            force_thresh_min, force_thresh_max = self.force_threshold_range
            sampled_force_thresh = torch.rand(num_reset_envs, device=self.device) * (force_thresh_max - force_thresh_min) + force_thresh_min
            self.current_force_threshold[env_ids] = sampled_force_thresh
            self._apply_force_threshold(env_ids)

        if self.randomize_torque_bounds:
            torque_bounds_min, torque_bounds_max = self.torque_bounds_range
            sampled_torque_bounds = torch.rand(num_reset_envs, device=self.device) * (torque_bounds_max - torque_bounds_min) + torque_bounds_min
            self.current_torque_bounds[env_ids] = sampled_torque_bounds
            self._apply_torque_bounds(env_ids)

    def _set_friction_per_env(self, asset, env_ids, friction_values):
        """
        Set friction for specific environments of an asset.

        Args:
            asset: Articulation object to modify
            env_ids: Environment indices to modify
            friction_values: Friction coefficients for each environment
        """
        # Access the physics simulation view to set material properties per environment
        # This is a low-level operation that modifies the simulation materials directly
        try:
            # Get the rigid body view
            if hasattr(asset, '_body_physx_view'):
                body_view = asset._body_physx_view

                # Set friction for all bodies in the asset for specified environments
                # Isaac Lab uses set_material_properties for per-environment friction
                for env_idx, friction in zip(env_ids, friction_values):
                    # Set both static and dynamic friction to the same value
                    body_view.set_material_properties(
                        static_friction=friction.item(),
                        dynamic_friction=friction.item(),
                        indices=[env_idx.item()]
                    )
            else:
                # Fallback: use factory_utils for global setting (not per-env)
                # This is less ideal but works if per-env setting is not available
                avg_friction = friction_values.mean().item()
                factory_utils.set_friction(asset, avg_friction, self.num_envs)
        except Exception as e:
            print(f"[WARNING] Failed to set per-env friction: {e}. Using global friction setting.")
            avg_friction = friction_values.mean().item()
            factory_utils.set_friction(asset, avg_friction, self.num_envs)

    def _apply_controller_gains(self, env_ids):
        """
        Apply randomized controller gains to specified environments.

        Directly sets task_prop_gains and recalculates task_deriv_gains.
        """
        # Set proportional gains
        self.unwrapped.task_prop_gains[env_ids] = self.current_prop_gains[env_ids]

        # Recalculate derivative gains as 2 * sqrt(prop_gains)
        # Using factory_utils or manual calculation
        rot_deriv_scale = 1.0  # Default scaling for rotation derivative gains
        if hasattr(self.unwrapped.cfg.ctrl, 'default_rot_deriv_scale'):
            rot_deriv_scale = self.unwrapped.cfg.ctrl.default_rot_deriv_scale

        # Calculate derivative gains: 2 * sqrt(prop_gains)
        deriv_gains = 2.0 * torch.sqrt(self.current_prop_gains[env_ids])

        # Scale rotation components if needed
        if rot_deriv_scale != 1.0:
            deriv_gains[:, 3:6] *= rot_deriv_scale

        self.unwrapped.task_deriv_gains[env_ids] = deriv_gains

    def _apply_force_gains(self, env_ids):
        """
        Apply randomized force/torque gains to specified environments.

        Note: This assumes the environment has task_force_gains attribute (hybrid control).
        """
        if hasattr(self.unwrapped, 'task_force_gains'):
            self.unwrapped.task_force_gains[env_ids] = self.current_force_gains[env_ids]
        else:
            print("[WARNING] Environment does not have task_force_gains. Force gain randomization skipped.")

    def _apply_pos_threshold(self, env_ids):
        """Apply randomized position action thresholds."""
        # The cfg.ctrl attributes are shared across all envs, so we need to store per-env
        # and apply during action processing if the env supports it
        # For now, we'll store it and hope the environment uses per-env thresholds
        # Otherwise, we'd need to modify action processing in the wrapper

        # Check if environment supports per-env thresholds
        if hasattr(self.unwrapped, 'pos_action_threshold_per_env'):
            # Replicate scalar to 3 dims
            self.unwrapped.pos_action_threshold_per_env[env_ids] = self.current_pos_threshold[env_ids].unsqueeze(1).repeat(1, 3)
        else:
            # Fallback: modify global config (affects all envs - not ideal)
            # We'll just store it for now and print a warning
            print("[WARNING] Environment does not support per-env pos_action_threshold. Using global threshold.")

    def _apply_rot_threshold(self, env_ids):
        """Apply randomized rotation action thresholds."""
        if hasattr(self.unwrapped, 'rot_action_threshold_per_env'):
            self.unwrapped.rot_action_threshold_per_env[env_ids] = self.current_rot_threshold[env_ids].unsqueeze(1).repeat(1, 3)
        else:
            print("[WARNING] Environment does not support per-env rot_action_threshold. Using global threshold.")

    def _apply_force_threshold(self, env_ids):
        """Apply randomized force action thresholds."""
        if hasattr(self.unwrapped, 'force_action_threshold_per_env'):
            self.unwrapped.force_action_threshold_per_env[env_ids] = self.current_force_threshold[env_ids].unsqueeze(1).repeat(1, 3)
        else:
            print("[WARNING] Environment does not support per-env force_action_threshold. Using global threshold.")

    def _apply_torque_bounds(self, env_ids):
        """Apply randomized torque action bounds."""
        if hasattr(self.unwrapped, 'torque_action_bounds_per_env'):
            self.unwrapped.torque_action_bounds_per_env[env_ids] = self.current_torque_bounds[env_ids].unsqueeze(1).repeat(1, 3)
        else:
            print("[WARNING] Environment does not support per-env torque_action_bounds. Using global bounds.")

    def step(self, action):
        """Step environment and ensure wrapper is initialized."""
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info

    def get_dynamics_stats(self):
        """Get current dynamics randomization statistics."""
        if not self.enabled:
            return {}

        stats = {
            'mean_friction': self.current_friction.mean().item(),
            'mean_pos_gain': self.current_prop_gains[:, 0].mean().item(),
            'mean_rot_gain': self.current_prop_gains[:, 3].mean().item(),
        }

        if self.randomize_force_gains:
            stats['mean_force_gain'] = self.current_force_gains[:, 0].mean().item()
            stats['mean_torque_gain'] = self.current_force_gains[:, 3].mean().item()

        return stats
