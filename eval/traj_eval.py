"""
Trajectory Evaluation Module

Captures detailed per-timestep trajectory data for analysis including:
- Phase labeling (approaching, initial_contact, insertion)
- Per policy step: contact force, contact state, control selection/probability, velocity, etc.
- Per sim step (for breaks only): high-resolution data for 10 policy steps leading to break

File output: .pkl files saved locally to --traj_output_dir
"""

import torch
import pickle
import os
import gymnasium as gym
from typing import Dict, Any, List, Optional
from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper


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
    - position_error: Position tracking error [3]
    - force_error: Force tracking error [3]
    - component_rewards: Dict of all logs_rew_* values
    - phase: Current phase label

    For breaks, also captures sim-step data for the 10 policy steps leading to break.
    """

    def __init__(self, env):
        """
        Initialize trajectory evaluation wrapper.

        Args:
            env: Environment with hybrid controller wrapper applied
        """
        super().__init__(env)

        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device

        # Get decimation for sim-step buffer sizing
        self.decimation = getattr(self.unwrapped.cfg, 'decimation', 4)

        # Per-environment phase tracking (monotonic: approaching -> initial_contact -> insertion)
        # 0 = approaching, 1 = initial_contact, 2 = insertion
        self.current_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Per-environment trajectory data storage
        # This attribute is used by run_trajectory_evaluation to find this wrapper
        self.trajectory_data = {env_id: [] for env_id in range(self.num_envs)}

        # Per-environment break sim-step data
        self.break_sim_data = {env_id: None for env_id in range(self.num_envs)}

        # Rolling buffer for sim-step data (last BREAK_HISTORY_POLICY_STEPS * decimation steps)
        self.sim_step_buffer_size = BREAK_HISTORY_POLICY_STEPS * self.decimation
        self.sim_step_buffers = {env_id: [] for env_id in range(self.num_envs)}

        # Track current policy step within episode for each env
        self.policy_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Track which environments have completed (terminated | truncated)
        self.env_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Track which environments succeeded (cumulative OR with factory successes)
        self.episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Track which environments terminated (for break detection)
        self.episode_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Find and cache the hybrid controller wrapper (has sel_matrix and target_force_for_control)
        self._hybrid_wrapper = self._find_hybrid_wrapper()

        # Setup sim-step capture by patching _apply_action
        self._original_apply_action = self.unwrapped._apply_action
        self.unwrapped._apply_action = self._trajectory_apply_action

        print("[TrajectoryEvalWrapper] Initialized")

    def _find_hybrid_wrapper(self):
        """Find the HybridForcePositionWrapper in the chain (cached once at init)."""
        current = self.env
        while hasattr(current, 'env'):
            if isinstance(current, HybridForcePositionWrapper):
                return current
            current = current.env
        raise AttributeError("HybridForcePositionWrapper not found in wrapper chain")

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
                        rewards[reward_name] = torch.full(
                            (self.num_envs,), float(value), device=self.device
                        )
        return rewards

    def _trajectory_apply_action(self):
        """
        Patched _apply_action that captures sim-step data before calling original.
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
        data = {
            'contact_force': self.unwrapped.robot_force_torque[:, :3].clone(),
            'contact_state': self.unwrapped.in_contact[:, :3].clone(),
            'control_selection': self._hybrid_wrapper.sel_matrix[:, :3].clone(),
            'velocity': self.unwrapped.fingertip_midpoint_linvel.clone(),
        }

        # Compute tracking errors
        position_error = (
            self.unwrapped.ctrl_target_fingertip_midpoint_pos -
            self.unwrapped.fingertip_midpoint_pos
        )
        force_error = (
            self._hybrid_wrapper.target_force_for_control[:, :3] -
            self.unwrapped.robot_force_torque[:, :3]
        )

        data['position_error'] = position_error.clone()
        data['force_error'] = force_error.clone()

        return data

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

    def _capture_policy_step_data(self, action: torch.Tensor, terminated: torch.Tensor) -> Dict[str, Any]:
        """
        Capture data at policy-step granularity.

        Args:
            action: Raw action from policy [num_envs, action_dim]
            terminated: Boolean tensor indicating termination [num_envs]

        Returns:
            Dictionary of policy-step data
        """
        # Raw control probability (first 3 elements of action)
        control_probability = action[:, :3].clone()

        # Compute tracking errors
        position_error = (
            self.unwrapped.ctrl_target_fingertip_midpoint_pos -
            self.unwrapped.fingertip_midpoint_pos
        )
        force_error = (
            self._hybrid_wrapper.target_force_for_control[:, :3] -
            self.unwrapped.robot_force_torque[:, :3]
        )

        # Collect reward components
        rewards = self._collect_reward_components()

        data = {
            'step': self.policy_step_count.clone(),
            'phase': self.current_phase.clone(),
            'contact_force': self.unwrapped.robot_force_torque[:, :3].clone(),
            'contact_state': self.unwrapped.in_contact[:, :3].clone(),
            'control_selection': self._hybrid_wrapper.sel_matrix[:, :3].clone(),
            'control_probability': control_probability,
            'velocity': self.unwrapped.fingertip_midpoint_linvel.clone(),
            'terminated': terminated.clone(),
            'position_error': position_error.clone(),
            'force_error': force_error.clone(),
            'rewards': rewards,
        }

        return data

    def _find_factory_wrapper(self):
        """Find the factory metrics wrapper with 'successes' attribute."""
        current = self.env
        while hasattr(current, 'env'):
            if hasattr(current, 'successes'):
                return current
            current = current.env
        raise AttributeError("Could not find factory wrapper with 'successes' attribute")

    def step(self, action):
        """
        Step environment and capture trajectory data.
        Only records data for environments that haven't completed yet.
        """
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

        # Track success from factory wrapper (cumulative OR for active envs)
        factory_wrapper = self._find_factory_wrapper()
        active_mask = ~self.env_completed
        curr_successes = factory_wrapper.successes
        self.episode_succeeded[curr_successes & active_mask] = True

        # Check for newly completed episodes
        # For trajectory eval: stop recording on SUCCESS or TERMINATION (terminated | truncated)
        # This differs from wandb_eval.py which only uses terminated|truncated for completion
        newly_completed_by_done = (terminated | truncated) & ~self.env_completed
        newly_completed_by_success = curr_successes & ~self.env_completed
        newly_completed = newly_completed_by_done | newly_completed_by_success

        # Track which completed via terminated (for break detection)
        # Only count as terminated if it wasn't a success
        self.episode_terminated[newly_completed_by_done & terminated & ~curr_successes] = True

        # Mark completed
        self.env_completed = self.env_completed | newly_completed

        # Store data for each environment ONLY if not already completed (before this step)
        for env_id in range(self.num_envs):
            # Skip if this env was already completed before this step
            if not active_mask[env_id]:
                continue

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

            # If this env just completed with a break, save sim-step buffer
            if newly_completed[env_id] and terminated[env_id] and not curr_successes[env_id]:
                self.break_sim_data[env_id] = self._convert_sim_buffer(env_id)

        # Increment policy step count
        self.policy_step_count += 1

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and trajectory tracking."""
        obs, info = super().reset(**kwargs)

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

        # Reset completion tracking
        self.env_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return obs, info

    def get_trajectory_data(self) -> Dict[str, Any]:
        """
        Get all collected trajectory data including outcomes.

        Outcome logic (matches wandb_eval.py lines 1765-1771):
        - Success: episode_succeeded & ~episode_terminated
        - Break: episode_terminated
        - Timeout: ~episode_terminated & ~episode_succeeded

        Returns:
            Dictionary with per-environment trajectory data and outcomes
        """
        result = {}
        for env_id in range(self.num_envs):
            # Compute outcome from tracking flags
            succeeded = self.episode_succeeded[env_id].item()
            terminated = self.episode_terminated[env_id].item()

            if succeeded and not terminated:
                outcome = 'success'
            elif terminated:
                outcome = 'break'
            else:
                outcome = 'timeout'

            result[f'env_{env_id}'] = {
                'policy_steps': self.trajectory_data[env_id],
                'break_sim_steps': self.break_sim_data[env_id],
                'outcome': outcome,
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

    # Get device and num_envs from unwrapped
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    # Find trajectory wrapper in chain using standard pattern
    traj_wrapper = env
    while hasattr(traj_wrapper, 'env'):
        if hasattr(traj_wrapper, 'trajectory_data') and hasattr(traj_wrapper, 'break_sim_data'):
            break
        traj_wrapper = traj_wrapper.env

    if not hasattr(traj_wrapper, 'trajectory_data'):
        raise RuntimeError("TrajectoryEvalWrapper not found in environment wrapper chain. "
                          "Make sure TrajectoryEvalWrapper is applied before AsyncCriticIsaacLabWrapper.")

    # Reset environment
    obs_dict, info = env.reset()

    # Rollout loop - use the wrapper's env_completed tracking
    step_count = 0
    progress_bar = tqdm.tqdm(total=max_rollout_steps + 1, desc="Steps completed", disable=not show_progress)

    while not traj_wrapper.env_completed.all():
        # Get actions from agent (deterministic evaluation)
        with torch.no_grad():
            outputs = agent.act(obs_dict, timestep=step_count, timesteps=max_rollout_steps)[-1]
            actions = outputs['mean_actions']

        # Step environment (wrapper handles completion tracking internally)
        obs_dict, rewards, terminated, truncated, info = env.step(actions)

        # Update progress
        if show_progress:
            progress_bar.update(1)

        step_count += 1

        # Safety check
        if step_count > max_rollout_steps * 2:
            raise RuntimeError(f"Evaluation exceeded maximum steps ({max_rollout_steps * 2})")

    progress_bar.close()
    print(f"    Completed {num_envs} episodes in {step_count} steps")

    # Get trajectory data from wrapper
    raw_trajectory_data = traj_wrapper.get_trajectory_data()

    # Compute metrics from outcomes stored in trajectory data
    outcome_counts = {'success': 0, 'break': 0, 'timeout': 0}
    for env_id in range(num_envs):
        outcome = raw_trajectory_data[f'env_{env_id}']['outcome']
        if outcome in outcome_counts:
            outcome_counts[outcome] += 1
        else:
            raise RuntimeError(f"Unexpected outcome '{outcome}' for env_{env_id}")

    metrics = {
        'total_episodes': num_envs,
        'num_successes': outcome_counts['success'],
        'num_breaks': outcome_counts['break'],
        'num_timeouts': outcome_counts['timeout'],
        'success_rate': outcome_counts['success'] / num_envs,
        'break_rate': outcome_counts['break'] / num_envs,
    }

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
            'total_breaks': outcome_counts['break'],
            'num_envs': num_envs,
            'max_rollout_steps': max_rollout_steps,
            'eval_seed': eval_seed,
            'metrics': metrics,
        }
    }

    print(f"    Success rate: {metrics['success_rate']:.2%}")
    print(f"    Break rate: {metrics['break_rate']:.2%}")
    print(f"    Timeout rate: {metrics['num_timeouts'] / num_envs:.2%}")

    return output


def save_trajectory_data(trajectory_data: Dict[str, Any], file_path: str) -> None:
    """
    Save trajectory data to a .pkl file.

    Args:
        trajectory_data: Dictionary containing trajectory data for all environments
        file_path: Path to save the .pkl file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(trajectory_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"    Saved trajectory data ({file_size_mb:.2f} MB)")
