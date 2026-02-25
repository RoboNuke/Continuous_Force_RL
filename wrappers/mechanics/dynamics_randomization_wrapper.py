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
        self.randomize_held_mass = config.get('randomize_held_mass', False)
        self.randomize_gains = config.get('randomize_gains', False)
        self.randomize_force_gains = config.get('randomize_force_gains', False)
        self.randomize_pos_threshold = config.get('randomize_pos_threshold', False)
        self.randomize_rot_threshold = config.get('randomize_rot_threshold', False)
        self.randomize_force_threshold = config.get('randomize_force_threshold', False)

        # Store ranges
        self.friction_range = config.get('friction_range', [0.5, 1.5])
        self.held_mass_range = config.get('held_mass_range', [0.5, 2.0])
        self.pos_gains_range = config.get('pos_gains_range', [80.0, 120.0])
        self.rot_gains_range = config.get('rot_gains_range', [20.0, 40.0])
        self.force_gains_range = config.get('force_gains_range', [0.08, 0.12])
        self.torque_gains_range = config.get('torque_gains_range', [0.0008, 0.0012])
        self.pos_threshold_range = config.get('pos_threshold_range', [0.015, 0.025])
        self.rot_threshold_range = config.get('rot_threshold_range', [0.08, 0.12])
        self.force_threshold_range = config.get('force_threshold_range', [8.0, 12.0])

        # Initialize per-environment storage tensors (will be properly initialized in _initialize_wrapper)
        # Use default device/dtype for now, will match target tensors during initialization
        self.current_friction = None
        self.current_held_mass = None
        self.current_prop_gains = None
        self.current_force_gains = None
        self.current_pos_threshold = None
        self.current_rot_threshold = None
        self.current_force_threshold = None

        # Store original methods
        self._original_reset_idx = None
        self._wrapper_initialized = False

        # Note: Initialization is delayed until first reset when environment parameters are ready
        # Do NOT initialize here - task_prop_gains and other parameters don't exist until first _reset_idx

        print(f"[DynamicsRandomizationWrapper] Created (initialization deferred until first reset)")
        print(f"  Friction randomization: {self.randomize_friction}")
        print(f"  Mass randomization: {self.randomize_held_mass}")
        print(f"  Gains randomization: {self.randomize_gains}")
        print(f"  Force gains randomization: {self.randomize_force_gains}")
        print(f"  Threshold randomization: pos={self.randomize_pos_threshold}, rot={self.randomize_rot_threshold}, "
              f"force={self.randomize_force_threshold}")

    def _initialize_wrapper(self):
        """Initialize the wrapper after the base environment is set up."""
        if self._wrapper_initialized or not self.enabled:
            return

        # Validate required attributes exist
        if not hasattr(self.unwrapped, '_reset_idx'):
            raise RuntimeError("Environment missing required _reset_idx method")

        # Validate friction randomization requirements
        if self.randomize_friction:
            if not hasattr(self.unwrapped, '_held_asset'):
                raise RuntimeError(
                    "Friction randomization requires _held_asset attribute on environment. "
                    "Ensure the base environment has this attribute or disable friction randomization."
                )
            if not hasattr(self.unwrapped._held_asset, 'root_physx_view'):
                raise RuntimeError(
                    "Friction randomization requires PhysX root_physx_view on _held_asset. "
                    "Ensure asset is properly initialized or disable friction randomization."
                )

        # Validate mass randomization requirements
        if self.randomize_held_mass:
            if not hasattr(self.unwrapped, '_held_asset'):
                raise RuntimeError(
                    "Mass randomization requires _held_asset attribute on environment. "
                    "Ensure the base environment has this attribute or disable mass randomization."
                )
            if not hasattr(self.unwrapped._held_asset, 'root_physx_view'):
                raise RuntimeError(
                    "Mass randomization requires PhysX root_physx_view on _held_asset. "
                    "Ensure asset is properly initialized or disable mass randomization."
                )

        # Validate gain randomization requirements
        if self.randomize_gains:
            if not hasattr(self.unwrapped, 'task_prop_gains'):
                raise RuntimeError(
                    "Gain randomization requires task_prop_gains attribute on environment. "
                    "Ensure the base environment has this attribute or disable gain randomization."
                )

        # Validate force/torque randomization requirements (needs hybrid wrapper)
        if self.randomize_force_gains or self.randomize_force_threshold:
            hybrid_wrapper = self._find_hybrid_wrapper()
            if hybrid_wrapper is None:
                enabled_features = []
                if self.randomize_force_gains:
                    enabled_features.append("randomize_force_gains")
                if self.randomize_force_threshold:
                    enabled_features.append("randomize_force_threshold")
                raise RuntimeError(
                    f"Force/torque randomization enabled but HybridForcePositionWrapper not found in wrapper chain.\n"
                    f"Enabled features: {', '.join(enabled_features)}\n"
                    f"Either disable these features or add HybridForcePositionWrapper to your wrapper stack."
                )

        # Initialize all thresholds/bounds as tensors
        self._initialize_thresholds_bounds_tensors()

        # Initialize storage tensors with correct dtype and device
        self._initialize_storage_tensors()

        # Store and override _reset_idx method
        self._original_reset_idx = self.unwrapped._reset_idx
        self.unwrapped._reset_idx = self._wrapped_reset_idx

        self._wrapper_initialized = True
        print("[DynamicsRandomizationWrapper] Initialized and injected into reset chain")

    def _wrapped_reset_idx(self, env_ids):
        """
        Reset specified environments with dynamics randomization.

        Two-phase randomization:
        - Phase 1 (BEFORE reset): PhysX properties (mass, friction) that need the
          base env's step_sim_no_action() to flush into the simulation.
        - Phase 2 (AFTER reset): Gains and thresholds that are pure tensor assignments.
          These MUST come after reset because the base env's randomize_initial_state()
          overwrites task_prop_gains and task_deriv_gains back to defaults.
        """
        # Initialize wrapper on first reset if not already done
        if not self._wrapper_initialized:
            self._initialize_wrapper()

        # Convert to tensor if needed and ensure on correct device (GPU for general operations)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            # Ensure env_ids are on GPU (test may pass CPU tensors)
            env_ids = env_ids.to(device=self.device)

        # Phase 1: PhysX properties BEFORE reset (flushed by step_sim_no_action inside base reset)
        self._randomize_physx_properties(env_ids)

        # Call original reset (flushes PhysX properties, but overwrites gains to defaults)
        if self._original_reset_idx is not None:
            self._original_reset_idx(env_ids)

        # Phase 2: Gains and thresholds AFTER reset (so base env can't overwrite them)
        self._randomize_gains_and_thresholds(env_ids)

        # Verify all randomized values actually took effect
        self._verify_randomization(env_ids)

    def _randomize_physx_properties(self, env_ids):
        """
        Randomize PhysX-level properties (friction, mass/inertia) for specified environments.

        These are set BEFORE the base reset so that step_sim_no_action() flushes them
        into the PhysX simulation.

        Args:
            env_ids: Tensor of environment indices to randomize
        """
        num_reset_envs = len(env_ids)

        # 1. Randomize friction (held asset only)
        if self.randomize_friction:
            friction_min, friction_max = self.friction_range
            sampled_friction = (
                torch.rand(num_reset_envs, device=self.current_friction.device)
                * (friction_max - friction_min) + friction_min
            )
            self.current_friction[env_ids] = sampled_friction

            # Apply friction to held asset
            self._set_friction_per_env(self.unwrapped._held_asset, env_ids, sampled_friction)

        # 2. Randomize held asset mass (scale factors)
        if self.randomize_held_mass:
            mass_scale_min, mass_scale_max = self.held_mass_range
            sampled_mass_scales = (
                torch.rand(num_reset_envs, device=self.current_held_mass.device)
                * (mass_scale_max - mass_scale_min) + mass_scale_min
            )
            self.current_held_mass[env_ids] = sampled_mass_scales

            # Apply mass scale to held asset
            self._set_mass_per_env(self.unwrapped._held_asset, env_ids, sampled_mass_scales)

    def _randomize_gains_and_thresholds(self, env_ids):
        """
        Randomize controller gains and action thresholds for specified environments.

        These are set AFTER the base reset because the base env's randomize_initial_state()
        overwrites task_prop_gains and task_deriv_gains back to defaults. By applying
        our randomization after, our values are the final ones used for the episode.

        Args:
            env_ids: Tensor of environment indices to randomize
        """
        num_reset_envs = len(env_ids)

        # 1. Randomize controller gains (position and rotation)
        if self.randomize_gains:
            # Sample position gains (same for all 3 position dims)
            pos_gain_min, pos_gain_max = self.pos_gains_range
            sampled_pos_gains = (
                torch.rand(num_reset_envs, device=self.current_prop_gains.device)
                * (pos_gain_max - pos_gain_min) + pos_gain_min
            )

            # Sample rotation gains (same for all 3 rotation dims)
            rot_gain_min, rot_gain_max = self.rot_gains_range
            sampled_rot_gains = (
                torch.rand(num_reset_envs, device=self.current_prop_gains.device)
                * (rot_gain_max - rot_gain_min) + rot_gain_min
            )

            # Replicate to all dimensions: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
            self.current_prop_gains[env_ids, 0:3] = sampled_pos_gains.unsqueeze(1).repeat(1, 3)
            self.current_prop_gains[env_ids, 3:6] = sampled_rot_gains.unsqueeze(1).repeat(1, 3)

            # Apply gains to environment
            self._apply_controller_gains(env_ids)

        # 2. Randomize force/torque gains (hybrid control only)
        if self.randomize_force_gains:
            # Sample force gains (same for all 3 force dims)
            force_gain_min, force_gain_max = self.force_gains_range
            sampled_force_gains = (
                torch.rand(num_reset_envs, device=self.current_force_gains.device)
                * (force_gain_max - force_gain_min) + force_gain_min
            )

            # Sample torque gains (same for all 3 torque dims)
            torque_gain_min, torque_gain_max = self.torque_gains_range
            sampled_torque_gains = (
                torch.rand(num_reset_envs, device=self.current_force_gains.device)
                * (torque_gain_max - torque_gain_min) + torque_gain_min
            )

            # Replicate to all dimensions: [force_x, force_y, force_z, torque_x, torque_y, torque_z]
            self.current_force_gains[env_ids, 0:3] = sampled_force_gains.unsqueeze(1).repeat(1, 3)
            self.current_force_gains[env_ids, 3:6] = sampled_torque_gains.unsqueeze(1).repeat(1, 3)

            # Apply force gains to environment
            self._apply_force_gains(env_ids)

        # 3. Randomize action thresholds
        if self.randomize_pos_threshold:
            pos_thresh_min, pos_thresh_max = self.pos_threshold_range
            sampled_pos_thresh = (
                torch.rand(num_reset_envs, device=self.current_pos_threshold.device)
                * (pos_thresh_max - pos_thresh_min) + pos_thresh_min
            )
            self.current_pos_threshold[env_ids] = sampled_pos_thresh
            self._apply_pos_threshold(env_ids)

        if self.randomize_rot_threshold:
            rot_thresh_min, rot_thresh_max = self.rot_threshold_range
            sampled_rot_thresh = (
                torch.rand(num_reset_envs, device=self.current_rot_threshold.device)
                * (rot_thresh_max - rot_thresh_min) + rot_thresh_min
            )
            self.current_rot_threshold[env_ids] = sampled_rot_thresh
            self._apply_rot_threshold(env_ids)

        if self.randomize_force_threshold:
            force_thresh_min, force_thresh_max = self.force_threshold_range
            sampled_force_thresh = (
                torch.rand(num_reset_envs, device=self.current_force_threshold.device)
                * (force_thresh_max - force_thresh_min) + force_thresh_min
            )
            self.current_force_threshold[env_ids] = sampled_force_thresh
            self._apply_force_threshold(env_ids)

    def _verify_randomization(self, env_ids):
        """
        Verify all randomized parameters actually took effect after reset.

        Reads back values from PhysX views and environment tensors and asserts they
        match what we set. This catches silent IsaacSim bugs where PhysX ignores
        property changes at runtime.

        Args:
            env_ids: Tensor of environment indices that were randomized
        """
        atol = 1e-4

        # Verify friction
        if self.randomize_friction:
            asset = self.unwrapped._held_asset
            materials = asset.root_physx_view.get_material_properties()
            mat_device = materials.device
            env_ids_cpu = env_ids.to(mat_device)
            expected_friction = self.current_friction[env_ids].to(mat_device)

            # Check static friction (index 0) for first shape
            actual_static = materials[env_ids_cpu, 0, 0]
            if not torch.allclose(actual_static, expected_friction, atol=atol):
                raise RuntimeError(
                    f"[DynamicsRandomizationWrapper] FRICTION VERIFICATION FAILED!\n"
                    f"  Expected static friction: {expected_friction[:3].tolist()}...\n"
                    f"  Actual static friction:   {actual_static[:3].tolist()}...\n"
                    f"  This is a known IsaacSim bug where friction changes are silently ignored.\n"
                    f"  See: https://forums.developer.nvidia.com/t/runtime-friction-and-restitution-randomization-bug"
                )

            # Check dynamic friction (index 1) for first shape
            actual_dynamic = materials[env_ids_cpu, 0, 1]
            if not torch.allclose(actual_dynamic, expected_friction, atol=atol):
                raise RuntimeError(
                    f"[DynamicsRandomizationWrapper] FRICTION VERIFICATION FAILED!\n"
                    f"  Expected dynamic friction: {expected_friction[:3].tolist()}...\n"
                    f"  Actual dynamic friction:   {actual_dynamic[:3].tolist()}...\n"
                    f"  This is a known IsaacSim bug where friction changes are silently ignored.\n"
                    f"  See: https://forums.developer.nvidia.com/t/runtime-friction-and-restitution-randomization-bug"
                )

        # Verify mass and inertia
        if self.randomize_held_mass:
            asset = self.unwrapped._held_asset
            masses = asset.root_physx_view.get_masses()
            inertias = asset.root_physx_view.get_inertias()
            physx_device = masses.device
            env_ids_cpu = env_ids.to(physx_device)
            mass_scales = self.current_held_mass[env_ids].to(physx_device)

            # Compute expected masses from defaults * scale
            default_masses = asset.data.default_mass
            expected_masses = default_masses[env_ids_cpu] * mass_scales.unsqueeze(-1)
            actual_masses = masses[env_ids_cpu]

            if not torch.allclose(actual_masses, expected_masses, atol=atol):
                raise RuntimeError(
                    f"[DynamicsRandomizationWrapper] MASS VERIFICATION FAILED!\n"
                    f"  Expected masses (env 0): {expected_masses[0].tolist()}\n"
                    f"  Actual masses (env 0):   {actual_masses[0].tolist()}\n"
                    f"  Mass scales applied: {mass_scales[:3].tolist()}...\n"
                    f"  PhysX may have silently ignored the mass change."
                )

            # Compute expected inertias from defaults * scale
            default_inertias = asset.data.default_inertia
            expected_inertias = default_inertias[env_ids_cpu] * mass_scales.unsqueeze(-1).unsqueeze(-1)
            actual_inertias = inertias[env_ids_cpu]

            if not torch.allclose(actual_inertias, expected_inertias, atol=atol):
                raise RuntimeError(
                    f"[DynamicsRandomizationWrapper] INERTIA VERIFICATION FAILED!\n"
                    f"  Expected inertia (env 0): {expected_inertias[0, 0, :3].tolist()}...\n"
                    f"  Actual inertia (env 0):   {actual_inertias[0, 0, :3].tolist()}...\n"
                    f"  Mass scales applied: {mass_scales[:3].tolist()}...\n"
                    f"  PhysX may have silently ignored the inertia change."
                )

        # Verify controller gains
        if self.randomize_gains:
            expected_prop = self.current_prop_gains[env_ids].to(
                dtype=self.unwrapped.task_prop_gains.dtype,
                device=self.unwrapped.task_prop_gains.device
            )
            actual_prop = self.unwrapped.task_prop_gains[env_ids]

            if not torch.allclose(actual_prop, expected_prop, atol=atol):
                raise RuntimeError(
                    f"[DynamicsRandomizationWrapper] CONTROLLER GAIN VERIFICATION FAILED!\n"
                    f"  Expected prop gains (env 0): {expected_prop[0].tolist()}\n"
                    f"  Actual prop gains (env 0):   {actual_prop[0].tolist()}\n"
                    f"  The base environment's _reset_idx likely overwrote the randomized gains.\n"
                    f"  Gains must be applied AFTER the base reset, not before."
                )

            # Also verify derivative gains
            expected_deriv = 2.0 * torch.sqrt(expected_prop)
            rot_deriv_scale = 1.0
            if hasattr(self.unwrapped.cfg.ctrl, 'default_rot_deriv_scale'):
                rot_deriv_scale = self.unwrapped.cfg.ctrl.default_rot_deriv_scale
            if rot_deriv_scale != 1.0:
                expected_deriv[:, 3:6] *= rot_deriv_scale
            actual_deriv = self.unwrapped.task_deriv_gains[env_ids]

            if not torch.allclose(actual_deriv, expected_deriv.to(dtype=actual_deriv.dtype), atol=atol):
                raise RuntimeError(
                    f"[DynamicsRandomizationWrapper] DERIVATIVE GAIN VERIFICATION FAILED!\n"
                    f"  Expected deriv gains (env 0): {expected_deriv[0].tolist()}\n"
                    f"  Actual deriv gains (env 0):   {actual_deriv[0].tolist()}\n"
                    f"  Derivative gains should be 2*sqrt(prop_gains)."
                )

    def _set_friction_per_env(self, asset, env_ids, friction_values):
        """
        Set friction for specific environments using PhysX API.

        Args:
            asset: Articulation object (e.g., _held_asset)
            env_ids: Tensor of environment indices
            friction_values: Tensor of friction coefficients for each env (shape: num_reset_envs)
        """
        if not hasattr(asset, 'root_physx_view'):
            raise RuntimeError(
                f"Asset does not have root_physx_view attribute required for per-environment friction setting. "
                f"Ensure the asset is properly initialized before dynamics randomization."
            )

        # Get current material properties
        # Shape: (num_envs, num_shapes, num_properties) where last dim = [static, dynamic, restitution]
        materials = asset.root_physx_view.get_material_properties()

        # Move tensors to match materials device (PhysX materials are on CPU)
        mat_device = materials.device
        env_ids_mat = env_ids.to(mat_device)
        friction_mat = friction_values.to(mat_device)

        # Set friction for specified environments (matching factory_utils.set_friction pattern)
        # Use ellipsis to broadcast across shape dimension: friction_values shape (num_reset_envs,) -> (num_reset_envs, num_shapes)
        materials[env_ids_mat, ..., 0] = friction_mat.unsqueeze(-1)  # Static friction
        materials[env_ids_mat, ..., 1] = friction_mat.unsqueeze(-1)  # Dynamic friction

        # Apply updated materials (env_ids must be on CPU per factory_utils pattern)
        asset.root_physx_view.set_material_properties(materials, env_ids_mat)

    def _set_mass_per_env(self, asset, env_ids, mass_scale_factors):
        """
        Set mass for specific environments using scale factors with automatic inertia recomputation.

        Args:
            asset: Articulation object (e.g., _held_asset)
            env_ids: Tensor of environment indices
            mass_scale_factors: Tensor of mass scale factors for each env (shape: num_reset_envs)
        """
        if not hasattr(asset, 'root_physx_view'):
            raise RuntimeError(
                f"Asset does not have root_physx_view attribute required for per-environment mass setting. "
                f"Ensure the asset is properly initialized before dynamics randomization."
            )

        # Get current mass and inertia properties (CPU tensors)
        # Shape: (num_envs, num_bodies)
        masses = asset.root_physx_view.get_masses()
        # Shape: (num_envs, num_bodies, 9) - 3x3 inertia tensor flattened
        inertias = asset.root_physx_view.get_inertias()

        # Move tensors to match PhysX device (CPU)
        physx_device = masses.device
        env_ids_cpu = env_ids.to(physx_device)
        mass_scales_cpu = mass_scale_factors.to(physx_device)

        # Get default masses and inertias for resetting environments
        default_masses = asset.data.default_mass  # (num_envs, num_bodies)
        default_inertias = asset.data.default_inertia  # (num_envs, num_bodies, 9)

        # Apply scale factors to all bodies of specified environments
        # Broadcast mass scale across body dimension: (num_reset_envs,) -> (num_reset_envs, num_bodies)
        new_masses = default_masses[env_ids_cpu] * mass_scales_cpu.unsqueeze(-1)
        masses[env_ids_cpu] = new_masses

        # Recompute inertias (scale by mass ratio assuming uniform density)
        # For uniform density, inertia scales linearly with mass
        # Broadcast across body and inertia dimensions: (num_reset_envs,) -> (num_reset_envs, num_bodies, 9)
        mass_ratios = mass_scales_cpu.unsqueeze(-1).unsqueeze(-1)
        new_inertias = default_inertias[env_ids_cpu] * mass_ratios
        inertias[env_ids_cpu] = new_inertias

        # Apply updated masses and inertias to PhysX (env_ids must be on CPU)
        asset.root_physx_view.set_masses(masses, env_ids_cpu)
        asset.root_physx_view.set_inertias(inertias, env_ids_cpu)

    def _find_hybrid_wrapper(self):
        """Search wrapper chain for HybridForcePositionWrapper."""
        current = self
        while current is not None:
            if current.__class__.__name__ == 'HybridForcePositionWrapper':
                return current
            if hasattr(current, 'env'):
                current = current.env
            else:
                break
        return None

    def _initialize_storage_tensors(self):
        """Initialize storage tensors with correct device to match target tensors.
        Always use float32 dtype since we sample from uniform distribution (float32)."""

        # Friction: Match materials tensor device (CPU for PhysX), always use float32
        if self.randomize_friction:
            materials = self.unwrapped._held_asset.root_physx_view.get_material_properties()
            self.current_friction = torch.ones((self.num_envs,), dtype=torch.float32, device=materials.device)

        # Mass: Match masses tensor device (CPU for PhysX), always use float32
        if self.randomize_held_mass:
            masses = self.unwrapped._held_asset.root_physx_view.get_masses()
            self.current_held_mass = torch.ones((self.num_envs,), dtype=torch.float32, device=masses.device)

        # Controller gains: Match task_prop_gains device, use float32
        if self.randomize_gains:
            self.current_prop_gains = torch.zeros(
                (self.num_envs, 6),
                dtype=torch.float32,
                device=self.unwrapped.task_prop_gains.device
            )

        # Position threshold: Match base env pos_threshold device, use float32
        if self.randomize_pos_threshold:
            self.current_pos_threshold = torch.zeros(
                (self.num_envs,),
                dtype=torch.float32,
                device=self.unwrapped.pos_threshold.device
            )

        # Rotation threshold: Match base env rot_threshold device, use float32
        if self.randomize_rot_threshold:
            self.current_rot_threshold = torch.zeros(
                (self.num_envs,),
                dtype=torch.float32,
                device=self.unwrapped.rot_threshold.device
            )

        # Force/torque gains and threshold: Match hybrid wrapper device, use float32
        hybrid_wrapper = self._find_hybrid_wrapper()
        if self.randomize_force_gains and hybrid_wrapper is not None:
            self.current_force_gains = torch.zeros(
                (self.num_envs, 6),
                dtype=torch.float32,
                device=hybrid_wrapper.kp.device
            )

        if self.randomize_force_threshold and hybrid_wrapper is not None:
            self.current_force_threshold = torch.zeros(
                (self.num_envs,),
                dtype=torch.float32,
                device=hybrid_wrapper.force_threshold.device
            )

    def _initialize_thresholds_bounds_tensors(self):
        """
        Initialize all threshold variables as per-environment tensors.
        Pulls initial values from cfg and replicates across all environments.
        Note: Bounds are NOT initialized here - they remain as fixed hard limits in cfg.
        """
        # 1. Initialize base env thresholds (if not already tensors)
        if not hasattr(self.unwrapped, 'pos_threshold'):
            if hasattr(self.unwrapped.cfg.ctrl, 'pos_action_threshold'):
                pos_thresh = self.unwrapped.cfg.ctrl.pos_action_threshold
                self.unwrapped.pos_threshold = torch.tensor(pos_thresh, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            else:
                raise ValueError("pos_action_threshold not found in cfg.ctrl. Required for position control.")

        if not hasattr(self.unwrapped, 'rot_threshold'):
            if hasattr(self.unwrapped.cfg.ctrl, 'rot_action_threshold'):
                rot_thresh = self.unwrapped.cfg.ctrl.rot_action_threshold
                self.unwrapped.rot_threshold = torch.tensor(rot_thresh, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            else:
                raise ValueError("rot_action_threshold not found in cfg.ctrl. Required for rotation control.")

        # 2. Find hybrid wrapper in chain (if randomizing force/torque parameters)
        if self.randomize_force_threshold or self.randomize_force_gains:
            hybrid_wrapper = self._find_hybrid_wrapper()

            if hybrid_wrapper is not None:
                # Hybrid wrapper should have already initialized these in its __init__
                # But if not, we initialize them here from cfg

                if not hasattr(hybrid_wrapper, 'force_threshold'):
                    if hasattr(self.unwrapped.cfg.ctrl, 'force_action_threshold'):
                        force_thresh = self.unwrapped.cfg.ctrl.force_action_threshold
                        hybrid_wrapper.force_threshold = torch.tensor(force_thresh, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
                    else:
                        raise ValueError("force_action_threshold not found in cfg.ctrl. Required for hybrid control.")

                # Check if torque control is enabled (force_tz or force_torque mode)
                if hasattr(hybrid_wrapper, 'ctrl_mode') and hybrid_wrapper.ctrl_mode in ["force_tz", "force_torque"]:
                    if not hasattr(hybrid_wrapper, 'torque_threshold'):
                        if hasattr(self.unwrapped.cfg.ctrl, 'torque_action_threshold'):
                            torque_thresh = self.unwrapped.cfg.ctrl.torque_action_threshold
                            hybrid_wrapper.torque_threshold = torch.tensor(torque_thresh, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
                        else:
                            raise ValueError("torque_action_threshold not found in cfg.ctrl. Required for torque control.")

    def _apply_controller_gains(self, env_ids):
        """
        Apply randomized controller gains to specified environments.

        Directly sets task_prop_gains and recalculates task_deriv_gains.
        """
        # Set proportional gains (convert dtype to match target if needed)
        self.unwrapped.task_prop_gains[env_ids] = self.current_prop_gains[env_ids].to(
            dtype=self.unwrapped.task_prop_gains.dtype
        )

        # Recalculate derivative gains as 2 * sqrt(prop_gains)
        rot_deriv_scale = 1.0  # Default scaling for rotation derivative gains
        if hasattr(self.unwrapped.cfg.ctrl, 'default_rot_deriv_scale'):
            rot_deriv_scale = self.unwrapped.cfg.ctrl.default_rot_deriv_scale

        # Calculate derivative gains: 2 * sqrt(prop_gains)
        deriv_gains = 2.0 * torch.sqrt(self.current_prop_gains[env_ids])

        # Scale rotation components if needed
        if rot_deriv_scale != 1.0:
            deriv_gains[:, 3:6] *= rot_deriv_scale

        self.unwrapped.task_deriv_gains[env_ids] = deriv_gains.to(
            dtype=self.unwrapped.task_deriv_gains.dtype
        )

    def _apply_force_gains(self, env_ids):
        """
        Apply randomized force/torque gains to HybridForcePositionWrapper's kp.
        """
        hybrid_wrapper = self._find_hybrid_wrapper()

        if hybrid_wrapper is None:
            raise RuntimeError(
                "Force gain randomization enabled but HybridForcePositionWrapper not found in wrapper chain. "
                "Either disable force gain randomization (set randomize_force_gains=False) or ensure "
                "HybridForcePositionWrapper is in the wrapper stack."
            )

        # Modify the wrapper's kp attribute (convert dtype to match target if needed)
        hybrid_wrapper.kp[env_ids] = self.current_force_gains[env_ids].to(dtype=hybrid_wrapper.kp.dtype)

    def _apply_pos_threshold(self, env_ids):
        """Apply randomized position thresholds by overwriting base env attribute."""
        # Replicate scalar to 3 dims and assign directly to base env (convert dtype if needed)
        thresh = self.current_pos_threshold[env_ids].unsqueeze(1).repeat(1, 3)
        self.unwrapped.pos_threshold[env_ids] = thresh.to(dtype=self.unwrapped.pos_threshold.dtype)

    def _apply_rot_threshold(self, env_ids):
        """Apply randomized rotation thresholds by overwriting base env attribute."""
        thresh = self.current_rot_threshold[env_ids].unsqueeze(1).repeat(1, 3)
        self.unwrapped.rot_threshold[env_ids] = thresh.to(dtype=self.unwrapped.rot_threshold.dtype)

    def _apply_force_threshold(self, env_ids):
        """Apply randomized force thresholds to hybrid wrapper."""
        hybrid_wrapper = self._find_hybrid_wrapper()
        if hybrid_wrapper is None:
            raise RuntimeError(
                "Force threshold randomization enabled but HybridForcePositionWrapper not found. "
                "Either disable force_threshold randomization or add HybridForcePositionWrapper to wrapper stack."
            )
        thresh = self.current_force_threshold[env_ids].unsqueeze(1).repeat(1, 3)
        hybrid_wrapper.force_threshold[env_ids] = thresh.to(dtype=hybrid_wrapper.force_threshold.dtype)

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

        stats = {}

        if self.randomize_friction:
            stats['mean_friction'] = self.current_friction.mean().item()

        if self.randomize_held_mass:
            stats['mean_held_mass_scale'] = self.current_held_mass.mean().item()

        if self.randomize_gains:
            stats['mean_pos_gain'] = self.current_prop_gains[:, 0].mean().item()
            stats['mean_rot_gain'] = self.current_prop_gains[:, 3].mean().item()

        if self.randomize_force_gains:
            stats['mean_force_gain'] = self.current_force_gains[:, 0].mean().item()
            stats['mean_torque_gain'] = self.current_force_gains[:, 3].mean().item()

        return stats
