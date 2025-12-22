"""
Trajectory Evaluation Module

Captures detailed per-timestep trajectory data for analysis including:
- Phase labeling (approaching, initial_contact, insertion)
- Per policy step: contact force, contact state, control selection/probability, velocity, etc.
- Per sim step (for breaks only): high-resolution data for 10 policy steps leading to break

File output: .pkl files uploaded to wandb
"""

import torch
import pickle
import tempfile
import os
import gymnasium as gym
from typing import Dict, Any, List, Optional
from collections import defaultdict


# ===== TRAJECTORY EVALUATION CONSTANTS =====

TRAJECTORY_MODE_TOTAL_ENVS = 100

# Number of policy steps to capture at sim-step granularity before a break
BREAK_HISTORY_POLICY_STEPS = 10

# Phase labels
PHASE_APPROACHING = "approaching"
PHASE_INITIAL_CONTACT = "initial_contact"
PHASE_INSERTION = "insertion"


class TrajectoryEvalWrapper(gym.Wrapper):
    """
    Wrapper to capture detailed trajectory data at policy and sim step levels.

    Captures per-timestep:
    - contact_force: Force on peg [3]
    - contact_state: Boolean contact state per axis [3]
    - control_selection: Binary control mode (0=pos, 1=force) [3]
    - control_probability: Raw network output before thresholding [3]
    - velocity: End-effector linear velocity [3]
    - terminated: Boolean if break occurred this step
    - position_error: Position tracking error when pos control active [3]
    - force_error: Force tracking error when force control active [3]
    - component_rewards: Dict of all logs_rew_* values
    - phase: Current phase label

    For breaks, also captures sim-step data for the 10 policy steps leading to break.
    """

    def __init__(self, env):
        """
        Initialize trajectory evaluation wrapper.

        Args:
            env: Base environment (should have hybrid controller in wrapper chain)
        """
        super().__init__(env)

        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # Get decimation for sim-step buffer sizing
        self.decimation = getattr(env.unwrapped.cfg, 'decimation', 4)

        # Per-environment phase tracking (monotonic: approaching -> initial_contact -> insertion)
        # 0 = approaching, 1 = initial_contact, 2 = insertion
        self.current_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Per-environment trajectory data storage
        self.trajectory_data = {env_id: [] for env_id in range(self.num_envs)}

        # Per-environment break sim-step data
        self.break_sim_data = {env_id: None for env_id in range(self.num_envs)}

        # Rolling buffer for sim-step data (last BREAK_HISTORY_POLICY_STEPS * decimation steps)
        # Structure: {env_id: deque of sim-step dicts}
        self.sim_step_buffer_size = BREAK_HISTORY_POLICY_STEPS * self.decimation
        self.sim_step_buffers = {env_id: [] for env_id in range(self.num_envs)}

        # Track current policy step within episode for each env
        self.policy_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Flag to track if wrapper is initialized
        self._wrapper_initialized = False
        self._original_apply_action = None
        self._hybrid_wrapper = None

        # Initialize wrapper if environment is ready
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

    def _initialize_wrapper(self):
        """Initialize the wrapper after the base environment is set up."""
        if self._wrapper_initialized:
            return

        # Find hybrid controller wrapper in chain
        self._hybrid_wrapper = self._find_hybrid_wrapper()
        if self._hybrid_wrapper is None:
            raise RuntimeError(
                "TrajectoryEvalWrapper requires HybridForcePositionWrapper in the wrapper chain. "
                "This wrapper is only compatible with hybrid control evaluation."
            )

        # Override _apply_action to capture sim-step data
        if hasattr(self._hybrid_wrapper, '_wrapped_apply_action'):
            self._original_apply_action = self._hybrid_wrapper._wrapped_apply_action
            self._hybrid_wrapper._wrapped_apply_action = self._trajectory_wrapped_apply_action
        else:
            raise RuntimeError(
                "HybridForcePositionWrapper missing _wrapped_apply_action method"
            )

        self._wrapper_initialized = True
        print("[TrajectoryEvalWrapper] Initialized successfully")

    def _find_hybrid_wrapper(self):
        """Find HybridForcePositionWrapper in the wrapper chain."""
        current = self.env
        while current is not None:
            if hasattr(current, 'sel_matrix') and hasattr(current, 'target_force_for_control'):
                return current
            if hasattr(current, 'env'):
                current = current.env
            else:
                break
        return None

    def _get_phase_label(self, phase_idx: int) -> str:
        """Convert phase index to label string."""
        if phase_idx == 0:
            return PHASE_APPROACHING
        elif phase_idx == 1:
            return PHASE_INITIAL_CONTACT
        else:
            return PHASE_INSERTION

    def _compute_is_centered(self) -> torch.Tensor:
        """
        Compute is_centered condition from base environment.

        Returns:
            Boolean tensor [num_envs] indicating if peg is centered over hole
        """
        # Access base environment data
        target_held_base_pos = self.unwrapped.target_held_base_pos
        held_base_pos = self.unwrapped.held_base_pos

        xy_dist = torch.linalg.vector_norm(
            target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1
        )
        is_centered = xy_dist < 0.0025  # 2.5mm threshold
        return is_centered

    def _update_phases(self) -> None:
        """
        Update phase state for all environments based on current conditions.
        Phases are monotonic: can only advance, never regress.
        """
        # Get current contact state (any axis)
        in_contact = self.unwrapped.in_contact[:, :3].any(dim=1)  # [num_envs]

        # Get is_centered condition
        is_centered = self._compute_is_centered()  # [num_envs]

        # Phase transitions (monotonic):
        # approaching (0) -> initial_contact (1): when in_contact becomes True
        # approaching (0) -> insertion (2): when is_centered becomes True (can skip initial_contact)
        # initial_contact (1) -> insertion (2): when is_centered becomes True

        # Advance from approaching to initial_contact
        advance_to_contact = (self.current_phase == 0) & in_contact & ~is_centered
        self.current_phase[advance_to_contact] = 1

        # Advance to insertion (from either approaching or initial_contact)
        advance_to_insertion = (self.current_phase < 2) & is_centered
        self.current_phase[advance_to_insertion] = 2

    def _collect_reward_components(self) -> Dict[str, torch.Tensor]:
        """
        Collect all reward components from extras.

        Returns:
            Dictionary of reward name -> per-env tensor
        """
        rewards = {}
        if hasattr(self.unwrapped, 'extras'):
            for key, value in self.unwrapped.extras.items():
                if key.startswith('logs_rew_'):
                    reward_name = key[9:]  # Strip 'logs_rew_' prefix
                    if isinstance(value, torch.Tensor):
                        rewards[reward_name] = value.clone()
                    else:
                        # Scalar value, expand to per-env tensor
                        rewards[reward_name] = torch.full(
                            (self.num_envs,), float(value), device=self.device
                        )
        return rewards

    def _trajectory_wrapped_apply_action(self):
        """
        Wrapped _apply_action that captures sim-step data before calling original.
        Called once per sim step (decimation times per policy step).
        """
        # Capture sim-step data BEFORE the action is applied
        sim_step_data = self._capture_sim_step_data()

        # Store in rolling buffers for each environment
        for env_id in range(self.num_envs):
            buffer = self.sim_step_buffers[env_id]
            buffer.append({k: v[env_id].clone() if isinstance(v, torch.Tensor) else v
                          for k, v in sim_step_data.items()})

            # Trim buffer to max size
            if len(buffer) > self.sim_step_buffer_size:
                buffer.pop(0)

        # Call original apply_action
        self._original_apply_action()

    def _capture_sim_step_data(self) -> Dict[str, Any]:
        """
        Capture data at sim-step granularity.

        Returns:
            Dictionary of sim-step data
        """
        hybrid = self._hybrid_wrapper

        data = {
            'contact_force': self.unwrapped.robot_force_torque[:, :3].clone(),
            'contact_state': self.unwrapped.in_contact[:, :3].clone(),
            'control_selection': hybrid.sel_matrix[:, :3].clone(),
            'velocity': self.unwrapped.fingertip_midpoint_linvel.clone(),
        }

        # Compute tracking errors
        position_error = (
            self.unwrapped.ctrl_target_fingertip_midpoint_pos -
            self.unwrapped.fingertip_midpoint_pos
        )
        force_error = (
            hybrid.target_force_for_control[:, :3] -
            self.unwrapped.robot_force_torque[:, :3]
        )

        data['position_error'] = position_error.clone()
        data['force_error'] = force_error.clone()

        return data

    def _capture_policy_step_data(self, action: torch.Tensor, terminated: torch.Tensor) -> Dict[str, Any]:
        """
        Capture data at policy-step granularity.

        Args:
            action: Raw action from policy [num_envs, action_dim]
            terminated: Boolean tensor indicating termination [num_envs]

        Returns:
            Dictionary of policy-step data
        """
        hybrid = self._hybrid_wrapper

        # Raw control probability (first 3 elements of action)
        control_probability = action[:, :3].clone()

        # Compute tracking errors
        position_error = (
            self.unwrapped.ctrl_target_fingertip_midpoint_pos -
            self.unwrapped.fingertip_midpoint_pos
        )
        force_error = (
            hybrid.target_force_for_control[:, :3] -
            self.unwrapped.robot_force_torque[:, :3]
        )

        # Collect reward components
        rewards = self._collect_reward_components()

        data = {
            'step': self.policy_step_count.clone(),
            'phase': self.current_phase.clone(),  # Will be converted to string per-env later
            'contact_force': self.unwrapped.robot_force_torque[:, :3].clone(),
            'contact_state': self.unwrapped.in_contact[:, :3].clone(),
            'control_selection': hybrid.sel_matrix[:, :3].clone(),
            'control_probability': control_probability,
            'velocity': self.unwrapped.fingertip_midpoint_linvel.clone(),
            'terminated': terminated.clone(),
            'position_error': position_error.clone(),
            'force_error': force_error.clone(),
            'rewards': rewards,
        }

        return data

    def step(self, action):
        """
        Step environment and capture trajectory data.
        """
        # Initialize wrapper if not done yet
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        # Step the environment
        obs, reward, terminated, truncated, info = super().step(action)

        # Ensure tensors are 1D
        if terminated.dim() > 1:
            terminated = terminated.squeeze(-1)
        if truncated.dim() > 1:
            truncated = truncated.squeeze(-1)

        # Update phases based on current state
        self._update_phases()

        # Capture policy-step data
        policy_data = self._capture_policy_step_data(action, terminated)

        # Store data for each environment
        for env_id in range(self.num_envs):
            # Convert tensors to CPU scalars/lists for storage
            step_data = {
                'step': self.policy_step_count[env_id].item(),
                'phase': self._get_phase_label(self.current_phase[env_id].item()),
                'contact_force': policy_data['contact_force'][env_id].cpu().tolist(),
                'contact_state': policy_data['contact_state'][env_id].cpu().tolist(),
                'control_selection': policy_data['control_selection'][env_id].cpu().tolist(),
                'control_probability': policy_data['control_probability'][env_id].cpu().tolist(),
                'velocity': policy_data['velocity'][env_id].cpu().tolist(),
                'terminated': policy_data['terminated'][env_id].item(),
                'position_error': policy_data['position_error'][env_id].cpu().tolist(),
                'force_error': policy_data['force_error'][env_id].cpu().tolist(),
                'rewards': {k: v[env_id].item() if isinstance(v, torch.Tensor) else v
                          for k, v in policy_data['rewards'].items()},
            }
            self.trajectory_data[env_id].append(step_data)

            # Check for break (terminated but not timeout)
            is_break = terminated[env_id].item() and not truncated[env_id].item()
            if is_break and self.break_sim_data[env_id] is None:
                # Save the sim-step buffer for this break
                self.break_sim_data[env_id] = self._convert_sim_buffer(env_id)

        # Increment policy step count
        self.policy_step_count += 1

        return obs, reward, terminated, truncated, info

    def _convert_sim_buffer(self, env_id: int) -> List[Dict]:
        """
        Convert sim-step buffer to serializable format.

        Args:
            env_id: Environment ID

        Returns:
            List of sim-step data dicts
        """
        buffer = self.sim_step_buffers[env_id]
        converted = []
        for step_data in buffer:
            converted_step = {}
            for key, value in step_data.items():
                if isinstance(value, torch.Tensor):
                    converted_step[key] = value.cpu().tolist()
                else:
                    converted_step[key] = value
            converted.append(converted_step)
        return converted

    def reset(self, **kwargs):
        """Reset environment and trajectory tracking."""
        obs, info = super().reset(**kwargs)

        # Initialize wrapper if not done yet
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        # Reset phase tracking
        self.current_phase.zero_()

        # Reset policy step counts
        self.policy_step_count.zero_()

        # Clear trajectory data
        self.trajectory_data = {env_id: [] for env_id in range(self.num_envs)}

        # Clear break data
        self.break_sim_data = {env_id: None for env_id in range(self.num_envs)}

        # Clear sim-step buffers
        self.sim_step_buffers = {env_id: [] for env_id in range(self.num_envs)}

        return obs, info

    def get_trajectory_data(self) -> Dict[str, Any]:
        """
        Get all collected trajectory data.

        Returns:
            Dictionary with per-environment trajectory data
        """
        result = {}
        for env_id in range(self.num_envs):
            result[f'env_{env_id}'] = {
                'policy_steps': self.trajectory_data[env_id],
                'break_sim_steps': self.break_sim_data[env_id],
            }
        return result


def run_trajectory_evaluation(
    env: Any,
    agent: Any,
    configs: Dict[str, Any],
    max_rollout_steps: int,
    show_progress: bool = False,
    eval_seed: int = None
) -> Dict[str, Any]:
    """
    Run trajectory evaluation and collect detailed per-timestep data.

    Args:
        env: Environment instance (must have TrajectoryEvalWrapper applied)
        agent: Agent instance with loaded checkpoint
        configs: Configuration dictionary
        max_rollout_steps: Maximum steps per episode
        show_progress: Whether to show progress bar
        eval_seed: Seed for deterministic evaluation

    Returns:
        Dictionary containing:
            - 'trajectories': Per-environment trajectory data
            - 'metadata': Summary info including total_rollouts, total_policy_steps, total_breaks
    """
    import tqdm
    from skrl.utils import set_seed

    print("  Running trajectory evaluation...")

    # Set seed for deterministic evaluation
    if eval_seed is not None:
        set_seed(eval_seed, deterministic=True)
        print(f"    Set eval seed: {eval_seed}")

    # Get device and num_envs
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    # Find trajectory wrapper in chain
    traj_wrapper = None
    current = env
    while current is not None:
        if isinstance(current, TrajectoryEvalWrapper):
            traj_wrapper = current
            break
        if hasattr(current, 'env'):
            current = current.env
        else:
            break

    if traj_wrapper is None:
        raise RuntimeError("TrajectoryEvalWrapper not found in environment wrapper chain")

    # Episode tracking
    completed_episodes = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_succeeded = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # Reset environment
    obs_dict, info = env.reset()

    # Rollout loop
    step_count = 0
    progress_bar = tqdm.tqdm(total=max_rollout_steps + 1, desc="Steps completed", disable=not show_progress)

    while not completed_episodes.all():
        # Get actions from agent (deterministic evaluation)
        with torch.no_grad():
            outputs = agent.act(obs_dict, timestep=step_count, timesteps=max_rollout_steps)[-1]
            actions = outputs['mean_actions']

        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions)

        # Ensure tensors are 1D
        if terminated.dim() > 1:
            terminated = terminated.squeeze(-1)
        if truncated.dim() > 1:
            truncated = truncated.squeeze(-1)

        # Track success from factory metrics wrapper
        factory_wrapper = env
        while hasattr(factory_wrapper, 'env'):
            if hasattr(factory_wrapper, 'successes'):
                episode_succeeded |= factory_wrapper.successes
                break
            factory_wrapper = factory_wrapper.env

        # Track completed episodes
        newly_completed = (terminated | truncated) & ~completed_episodes
        completed_episodes |= newly_completed
        episode_terminated[newly_completed & terminated] = True

        # Update progress
        if show_progress:
            progress_bar.update(1)

        step_count += 1

        # Safety check
        if step_count > max_rollout_steps * 2:
            raise RuntimeError(f"Evaluation exceeded maximum steps ({max_rollout_steps * 2})")

    progress_bar.close()
    print(f"    Completed {num_envs} episodes in {step_count} steps")

    # Compute basic metrics
    num_successes = episode_succeeded.sum().item()
    num_breaks = episode_terminated.sum().item()
    num_timeouts = num_envs - num_successes - num_breaks

    # Handle case where success and termination overlap
    successful_completions = (episode_succeeded & ~episode_terminated).sum().item()

    metrics = {
        'total_episodes': num_envs,
        'num_successful_completions': successful_completions,
        'num_breaks': num_breaks,
        'num_failed_timeouts': num_timeouts,
        'success_rate': successful_completions / num_envs,
        'break_rate': num_breaks / num_envs,
    }

    # Get trajectory data from wrapper
    raw_trajectory_data = traj_wrapper.get_trajectory_data()

    # Count total policy steps across all environments
    total_policy_steps = sum(
        len(raw_trajectory_data[f'env_{env_id}']['policy_steps'])
        for env_id in range(num_envs)
    )

    # Build output structure with metadata
    output = {
        'trajectories': raw_trajectory_data,
        'metadata': {
            'total_rollouts': num_envs,
            'total_policy_steps': total_policy_steps,
            'total_breaks': num_breaks,
            'num_envs': num_envs,
            'max_rollout_steps': max_rollout_steps,
            'eval_seed': eval_seed,
            'metrics': metrics,
        }
    }

    print(f"    Success rate: {metrics['success_rate']:.2%}")
    print(f"    Break rate: {metrics['break_rate']:.2%}")

    return output


def save_trajectory_data(trajectory_data: Dict[str, Any], file_path: str) -> None:
    """
    Save trajectory data to a .pkl file.

    Args:
        trajectory_data: Dictionary containing trajectory data for all environments
        file_path: Path to save the .pkl file
    """
    # Save using pickle
    with open(file_path, 'wb') as f:
        pickle.dump(trajectory_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Get file size for logging
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"    Saved trajectory data ({file_size_mb:.2f} MB)")
