"""
Efficient Reset Wrapper for Factory Environment

This wrapper implements efficient environment resetting by caching initial states
and shuffling them for individual environment resets without requiring simulation steps.

Extracted from: factory_env.py lines 708-741
"""

import torch
import gymnasium as gym


class EfficientResetWrapper(gym.Wrapper):
    """
    Wrapper that provides efficient environment resetting using state caching and shuffling.

    Features:
    - Caches initial states after full environment reset
    - Performs partial resets by shuffling cached states
    - No simulation steps required for individual environment resets
    - Maintains scene state consistency
    """

    def __init__(self, env):
        """
        Initialize the efficient reset wrapper.

        Args:
            env: Base environment to wrap
        """
        super().__init__(env)

        # State management
        self.start_state = None
        self._wrapper_initialized = False

        # Store original methods
        self._original_factory_reset_idx = None
        self._directrl_reset_idx = None

        # Initialize wrapper if environment is ready
        if hasattr(self.unwrapped, 'scene'):
            self._initialize_wrapper()

    def _initialize_wrapper(self):
        """Initialize the wrapper after the base environment is set up."""
        if self._wrapper_initialized:
            return

        # Store and override _reset_idx method
        if hasattr(self.unwrapped, '_reset_idx'):
            # Store original FactoryEnv method for full resets
            self._original_factory_reset_idx = self.unwrapped._reset_idx

            # Find DirectRLEnv's _reset_idx for partial resets
            self._directrl_reset_idx = self._find_directrlenv_reset_method()

            # Override with our wrapper method
            self.unwrapped._reset_idx = self._wrapped_reset_idx
        
        # Store and override _get_dones method
        if hasattr(self.unwrapped, '_get_dones'):
            print(f"[DEBUG] EfficientResetWrapper: Overwriting _get_dones")
            print(f"[DEBUG]   Original was: {self.unwrapped._get_dones}")
            self._original_get_dones = self.unwrapped._get_dones
            self.unwrapped._get_dones = self._wrapped_get_dones
            print(f"[DEBUG]   Now is: {self.unwrapped._get_dones}")

        self._wrapper_initialized = True

    def _find_directrlenv_reset_method(self):
        """
        Find DirectRLEnv's _reset_idx method in the inheritance chain.

        This ensures we call the lightweight DirectRLEnv reset logic instead of
        the expensive factory environment reset, while maintaining proper access
        to all instance variables and the correct inheritance chain.
        """
        try:
            # First, try to find DirectRLEnv in the MRO and use its method directly
            for cls in type(self.unwrapped).__mro__:
                if cls.__name__ == 'DirectRLEnv' and hasattr(cls, '_reset_idx'):
                    # Found DirectRLEnv - bind its method to our environment instance
                    return cls._reset_idx.__get__(self.unwrapped, type(self.unwrapped))

            # If DirectRLEnv not found by name, look for it by import
            try:
                from isaaclab.envs.direct_rl_env import DirectRLEnv
            except ImportError:
                from omni.isaac.lab.envs.direct_rl_env import DirectRLEnv
            if isinstance(self.unwrapped, DirectRLEnv):
                # Our environment inherits from DirectRLEnv, so we can use super() approach
                # Find the factory env class and skip it
                for i, cls in enumerate(type(self.unwrapped).__mro__):
                    if 'Factory' in cls.__name__ or 'factory' in cls.__name__.lower():
                        # Found factory env, get the next class in MRO (should be DirectRLEnv)
                        if i + 1 < len(type(self.unwrapped).__mro__):
                            parent_cls = type(self.unwrapped).__mro__[i + 1]
                            if hasattr(parent_cls, '_reset_idx'):
                                return parent_cls._reset_idx.__get__(self.unwrapped, type(self.unwrapped))
                        break

        except ImportError:
            print("Error in find directrlenv_reset method")

        # Final fallback: use the environment's own method (preserves original behavior)
        return self.unwrapped._reset_idx
    
    def _wrapped_get_dones(self):
        """Make sure that every max_episode_length we reset all envs so we don't overfit."""
        term, time_out = self._original_get_dones()
        if self.unwrapped.common_step_counter > 1:
            time_out = torch.ones_like(self.unwrapped.episode_length_buf) * ((self.unwrapped.common_step_counter % self.unwrapped.max_episode_length ) ==0)
        return term, time_out
    
    def _wrapped_reset_idx(self, env_ids):
        """
        Reset specified environments efficiently using state shuffling.

        This method implements two reset strategies:
        1. Full reset: When all environments are reset simultaneously (uses original FactoryEnv method)
        2. Partial reset: When only some environments are reset (uses DirectRLEnv + state shuffling)
        """
        if env_ids is None:
            env_ids = torch.arange(self.unwrapped.num_envs, device=self.unwrapped.device)

        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.unwrapped.device)

        # Check if this is a full reset (all environments)
        if len(env_ids) == self.unwrapped.num_envs:
            # Full reset - use original FactoryEnv method to ensure complete initialization
            if self._original_factory_reset_idx:
                self._original_factory_reset_idx(env_ids)

            # Cache the initial state after full reset
            if hasattr(self.unwrapped, 'scene') and hasattr(self.unwrapped.scene, 'get_state'):
                self.start_state = self.unwrapped.scene.get_state()
        else:
            # Partial reset - use DirectRLEnv method + efficient state shuffling
            if self._directrl_reset_idx:
                self._directrl_reset_idx(env_ids)

            # Ensure episode_length_buf is reset (safety measure)
            if hasattr(self.unwrapped, 'episode_length_buf'):
                self.unwrapped.episode_length_buf[env_ids] = 0

            # Use efficient state shuffling for partial resets
            if self.start_state is not None:
                self._perform_efficient_reset(env_ids)

    def _perform_efficient_reset(self, env_ids):
        """
        Perform efficient reset by shuffling cached states.

        Args:
            env_ids: Tensor of environment indices to reset
        """
        # Generate random indices to shuffle from
        num_reset_envs = len(env_ids)
        source_idxs = torch.randint(
            low=0,
            high=self.unwrapped.num_envs,
            size=(num_reset_envs,),
            device=self.unwrapped.device
        )

        # Reset articulations by shuffling their states
        if hasattr(self.unwrapped, 'scene') and hasattr(self.unwrapped.scene, 'articulations'):
            for art_name, articulation in self.unwrapped.scene.articulations.items():
                if art_name in self.start_state.get('articulation', {}):
                    art_state = self.start_state['articulation'][art_name]

                    # Shuffle root poses
                    if 'root_pose' in art_state:
                        pose = art_state['root_pose'][source_idxs, :].clone()
                        # Adjust positions for different environment origins
                        pose[:, :3] = (pose[:, :3] -
                                     self.unwrapped.scene.env_origins[source_idxs] +
                                     self.unwrapped.scene.env_origins[env_ids])
                        articulation.write_root_pose_to_sim(pose, env_ids=env_ids)

                    # Shuffle root velocities
                    if 'root_velocity' in art_state:
                        vel = art_state['root_velocity'][source_idxs, :].clone()
                        articulation.write_root_velocity_to_sim(vel, env_ids=env_ids)

                    # Shuffle joint positions
                    if 'joint_position' in art_state:
                        jnt_pos = art_state['joint_position'][source_idxs, :].clone()
                        jnt_vel = art_state.get('joint_velocity', torch.zeros_like(jnt_pos))[source_idxs, :].clone()
                        articulation.write_joint_state_to_sim(jnt_pos, jnt_vel, env_ids=env_ids)

                    # Reset the articulation
                    articulation.reset(env_ids)

    def step(self, action):
        """Step the environment and ensure wrapper is initialized."""
        # Initialize wrapper if not done yet
        if not self._wrapper_initialized and hasattr(self.unwrapped, 'scene'):
            self._initialize_wrapper()

        return super().step(action)

    def reset(self, **kwargs):
        """Reset the environment and ensure wrapper is initialized."""
        obs, info = super().reset(**kwargs)

        # Initialize wrapper if not done yet
        if not self._wrapper_initialized and hasattr(self.unwrapped, 'scene'):
            self._initialize_wrapper()

        # Cache the initial state after a full reset
        if hasattr(self.unwrapped, 'scene') and hasattr(self.unwrapped.scene, 'get_state'):
            self.start_state = self.unwrapped.scene.get_state()

        return obs, info

    def has_cached_state(self):
        """Check if initial state is cached and available for efficient resets."""
        return self.start_state is not None

    def clear_cached_state(self):
        """Clear the cached initial state (forces full reset on next reset call)."""
        self.start_state = None

    def get_reset_efficiency_stats(self):
        """Get statistics about reset efficiency."""
        return {
            'has_cached_state': self.has_cached_state(),
            'supports_efficient_reset': hasattr(self.unwrapped, 'scene'),
        }