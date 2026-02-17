"""
Real Robot Evaluation Script

Evaluates trained RL policies on a physical Franka Panda for peg-in-hole insertion.
No Isaac Sim dependency - uses pylibfranka robot interface and pure PyTorch control.

Usage:
    python eval/real_robot_eval.py --tag "MATCH:2024-01-15_10:00" --num_episodes 20
    python eval/real_robot_eval.py --tag "MATCH:2024-01-15_10:00" --no_wandb --run_id abc123
"""

import argparse
import os
import sys
import time
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.SimBa import SimBaNet
from real_robot_exps.robot_interface import FrankaInterface, make_ee_target_pose
from real_robot_exps.observation_builder import ObservationBuilder, ObservationNormalizer
from real_robot_exps.hybrid_controller import RealRobotController


# ============================================================================
# Config loading
# ============================================================================

def load_real_robot_config(config_path: str, overrides: Optional[List[str]] = None) -> dict:
    """Load real robot config from YAML and apply CLI overrides.

    Args:
        config_path: Path to config.yaml.
        overrides: List of "key=value" override strings (e.g. "task.hole_height=0.03").

    Returns:
        Config dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if overrides:
        for override in overrides:
            if '=' not in override:
                raise ValueError(f"Override must be 'key=value', got: {override}")
            key_path, value_str = override.split('=', 1)
            keys = key_path.split('.')

            # Navigate to parent
            parent = config
            for k in keys[:-1]:
                if k not in parent:
                    raise ValueError(f"Config key not found: {key_path}")
                parent = parent[k]

            # Parse value (try numeric types first)
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    if value_str.lower() == 'true':
                        value = True
                    elif value_str.lower() == 'false':
                        value = False
                    else:
                        value = value_str

            parent[keys[-1]] = value
            print(f"  Override: {key_path} = {value}")

    return config


# ============================================================================
# obs_order reconstruction from training config
# ============================================================================

def reconstruct_obs_order(configs: dict) -> list:
    """Reconstruct the obs_order that was used during training.

    The base factory environment starts with:
        ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]

    Wrappers append additional components based on config flags:
        - force_torque_sensor.add_force_obs -> appends "force_torque"
        - force_torque_sensor.add_contact_obs -> appends "in_contact"
        - obs_rand.use_yaw_noise -> appends "fingertip_yaw_rel_fixed"

    Args:
        configs: Training configuration dict from WandB.

    Returns:
        List of observation component names in training order.
    """
    # Base obs_order from IsaacLab factory environment
    obs_order = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]

    # Check force-torque sensor additions
    ft_cfg = configs['wrappers'].force_torque_sensor
    if getattr(ft_cfg, 'add_force_obs', False):
        obs_order.append("force_torque")
    if getattr(ft_cfg, 'add_contact_obs', False):
        obs_order.append("in_contact")

    # Check yaw observation
    env_cfg = configs['environment']
    if hasattr(env_cfg, 'obs_rand') and getattr(env_cfg.obs_rand, 'use_yaw_noise', False):
        obs_order.append("fingertip_yaw_rel_fixed")

    print(f"[reconstruct_obs_order] Reconstructed: {obs_order}")
    return obs_order


# ============================================================================
# Policy loading (no sim env needed)
# ============================================================================

def load_single_agent_policy(
    policy_path: str,
    configs: dict,
    obs_dim: int,
    device: str = "cpu",
) -> Tuple[SimBaNet, ObservationNormalizer, dict]:
    """Load a trained policy network and normalizer from checkpoint.

    Creates a standalone SimBaNet directly - no SKRL agent infrastructure needed.

    Args:
        policy_path: Path to policy .pt checkpoint file.
        configs: Training config dict from WandB.
        obs_dim: Policy observation dimension (from obs_order + action_dim).
                 The preprocessor may have more dimensions (policy + critic),
                 so only the first obs_dim elements are used for normalization.
        device: Torch device.

    Returns:
        Tuple of (policy_net, normalizer, model_info) where:
        - policy_net: SimBaNet in eval mode
        - normalizer: ObservationNormalizer with frozen training stats
        - model_info: Dict with sigma_idx, action_dim, use_state_dependent_std
    """
    checkpoint = torch.load(policy_path, map_location=device, weights_only=False)

    # Validate checkpoint contents
    if 'net_state_dict' not in checkpoint:
        raise RuntimeError(f"Policy checkpoint missing 'net_state_dict': {policy_path}")
    if 'state_preprocessor' not in checkpoint:
        raise RuntimeError(f"Policy checkpoint missing 'state_preprocessor': {policy_path}")

    # Validate obs_dim matches the network's input layer
    net_input_dim = checkpoint['net_state_dict']['input.0.weight'].shape[1]
    if net_input_dim != obs_dim:
        raise RuntimeError(
            f"obs_dim mismatch: obs_order+action_dim gives {obs_dim} but "
            f"network input layer expects {net_input_dim}. "
            f"Check that obs_order reconstruction matches training config."
        )

    # Model architecture from training config
    actor_n = configs['model'].actor.n
    actor_latent = configs['model'].actor.latent_size
    use_state_dependent_std = getattr(configs['model'].actor, 'use_state_dependent_std', False)

    # Determine sigma_idx and action_dim
    hybrid_enabled = configs['wrappers'].hybrid_control.enabled
    if hybrid_enabled:
        from configs.cfg_exts.ctrl_mode import get_force_size
        ctrl_mode = getattr(configs['primary'], 'ctrl_mode', 'force_only')
        force_size = get_force_size(ctrl_mode)
        sigma_idx = force_size
        action_dim = 2 * force_size + 6
    else:
        sigma_idx = 0
        action_dim = 6

    # Determine tan_out and network output size
    tan_out = (sigma_idx == 0)

    if use_state_dependent_std:
        std_out_dim = action_dim - sigma_idx
    else:
        std_out_dim = 0

    out_size = action_dim + std_out_dim

    print(f"[load_single_agent_policy] obs_dim={obs_dim}, action_dim={action_dim}, "
          f"sigma_idx={sigma_idx}, tan_out={tan_out}, out_size={out_size}, "
          f"state_dependent_std={use_state_dependent_std}")

    # Create and load SimBaNet
    policy_net = SimBaNet(
        n=actor_n,
        in_size=obs_dim,
        out_size=out_size,
        latent_size=actor_latent,
        device=device,
        tan_out=tan_out,
    )
    policy_net.load_state_dict(checkpoint['net_state_dict'])
    policy_net.eval()

    # Create normalizer — slice to policy obs_dim only (preprocessor includes
    # both policy and critic observations, we only need the policy portion)
    normalizer = ObservationNormalizer(
        checkpoint['state_preprocessor'], device=device, obs_dim=obs_dim
    )

    model_info = {
        'sigma_idx': sigma_idx,
        'action_dim': action_dim,
        'use_state_dependent_std': use_state_dependent_std,
        'obs_dim': obs_dim,
    }

    return policy_net, normalizer, model_info


# ============================================================================
# Deterministic inference
# ============================================================================

@torch.no_grad()
def get_deterministic_action(
    policy_net: SimBaNet,
    normalizer: ObservationNormalizer,
    obs: torch.Tensor,
    model_info: dict,
) -> torch.Tensor:
    """Get deterministic action from policy (mean, no sampling).

    Handles both pose-only and hybrid models.

    Args:
        policy_net: Trained SimBaNet in eval mode.
        normalizer: ObservationNormalizer with frozen stats.
        obs: [obs_dim] raw observation tensor.
        model_info: Dict with sigma_idx, action_dim, use_state_dependent_std.

    Returns:
        [action_dim] action tensor with appropriate activations applied.
    """
    # Normalize and batch
    norm_obs = normalizer.normalize(obs.unsqueeze(0))  # [1, obs_dim]

    # Forward pass
    raw_output = policy_net(norm_obs)  # [1, out_size]
    mean_action = raw_output[0, :model_info['action_dim']]

    sigma_idx = model_info['sigma_idx']

    if sigma_idx == 0:
        # POSE-ONLY: SimBaNet has tan_out=True, tanh already applied
        return mean_action  # [6] in [-1, 1]
    else:
        # HYBRID: SimBaNet has tan_out=False, apply activations manually
        selection = torch.sigmoid(mean_action[:sigma_idx])  # [0, 1]
        components = torch.tanh(mean_action[sigma_idx:])    # [-1, 1]
        return torch.cat([selection, components])


# ============================================================================
# Detection logic (matches IsaacLab factory env)
# ============================================================================

def check_success(
    ee_pos: torch.Tensor,
    ee_to_peg_base_offset: torch.Tensor,
    target_peg_base_pos: torch.Tensor,
    xy_centering_threshold: float,
    hole_height: float,
    threshold: float,
) -> bool:
    """Check if peg is successfully inserted.

    Matches IsaacLab factory env _get_curr_successes().

    Args:
        ee_pos: [3] EE position.
        ee_to_peg_base_offset: [3] offset from EE to peg base.
        target_peg_base_pos: [3] target peg base position when fully inserted.
        xy_centering_threshold: XY centering threshold (meters).
        hole_height: Hole height for Z threshold scaling.
        threshold: success_threshold or engage_threshold multiplier.

    Returns:
        True if success/engagement condition met.
    """
    peg_base_pos = ee_pos + ee_to_peg_base_offset
    xy_dist = torch.norm(peg_base_pos[:2] - target_peg_base_pos[:2])
    z_disp = peg_base_pos[2] - target_peg_base_pos[2]

    is_centered = xy_dist < xy_centering_threshold
    is_inserted = z_disp < hole_height * threshold
    return bool(is_centered and is_inserted)


def check_break(
    force_torque: torch.Tensor,
    break_force_threshold: float,
) -> bool:
    """Check if force exceeds break threshold.

    Matches FragileObjectWrapper break detection.

    Args:
        force_torque: [6] force/torque readings.
        break_force_threshold: Maximum allowed L2 force norm (N).

    Returns:
        True if force exceeds threshold.
    """
    force_magnitude = torch.norm(force_torque[:3])
    return bool(force_magnitude >= break_force_threshold)


# ============================================================================
# Metrics computation
# ============================================================================

def compute_real_robot_metrics(episode_results: List[dict]) -> Dict[str, float]:
    """Compute CORE_METRICS from sequential episode results.

    Args:
        episode_results: List of per-episode result dicts, each containing:
            'succeeded', 'terminated', 'length', 'ssv', 'ssjv',
            'max_force', 'sum_force_in_contact', 'contact_steps', 'energy',
            'success_step', 'termination_step'

    Returns:
        Dict of aggregated CORE_METRICS.
    """
    n = len(episode_results)
    if n == 0:
        raise RuntimeError("No episode results to compute metrics from")

    metrics = {}
    metrics['total_episodes'] = n

    # Outcome classification (matching sim logic exactly)
    num_success = 0
    num_breaks = 0
    num_timeouts = 0
    success_steps = []
    break_steps = []

    for ep in episode_results:
        succeeded = ep['succeeded']
        terminated = ep['terminated']

        if succeeded and not terminated:
            num_success += 1
        elif terminated:
            num_breaks += 1
        else:
            num_timeouts += 1

        # Steps to success/break (matching sim's mutually exclusive logic)
        if succeeded and (not terminated or ep['success_step'] < ep['termination_step']):
            success_steps.append(ep['success_step'])
        if terminated and (not succeeded or ep['termination_step'] <= ep['success_step']):
            break_steps.append(ep['termination_step'])

    metrics['num_successful_completions'] = num_success
    metrics['num_breaks'] = num_breaks
    metrics['num_failed_timeouts'] = num_timeouts

    # Sanity check
    if num_success + num_breaks + num_timeouts != n:
        raise RuntimeError(
            f"Episode outcome counts don't sum: {num_success}+{num_breaks}+{num_timeouts} != {n}"
        )

    # Average episode length
    metrics['episode_length'] = sum(ep['length'] for ep in episode_results) / n

    # Steps to success/break
    metrics['avg_steps_to_success'] = sum(success_steps) / len(success_steps) if success_steps else 0.0
    metrics['avg_steps_to_break'] = sum(break_steps) / len(break_steps) if break_steps else 0.0

    # Smoothness
    metrics['ssv'] = sum(ep['ssv'] for ep in episode_results) / n
    metrics['ssjv'] = sum(ep['ssjv'] for ep in episode_results) / n

    # Force
    metrics['max_force'] = max(ep['max_force'] for ep in episode_results)
    total_force_in_contact = sum(ep['sum_force_in_contact'] for ep in episode_results)
    total_contact_steps = sum(ep['contact_steps'] for ep in episode_results)
    metrics['avg_force_in_contact'] = (
        total_force_in_contact / total_contact_steps if total_contact_steps > 0 else 0.0
    )

    # Energy
    metrics['energy'] = sum(ep['energy'] for ep in episode_results) / n

    return metrics


# ============================================================================
# Single episode execution
# ============================================================================

def run_episode(
    robot: FrankaInterface,
    policy_net: SimBaNet,
    normalizer: ObservationNormalizer,
    model_info: dict,
    obs_builder: ObservationBuilder,
    controller: RealRobotController,
    real_config: dict,
    goal_pos_noise_scale: torch.Tensor,
    goal_yaw_noise_scale: float,
    hand_init_pos: torch.Tensor,
    hand_init_pos_noise: torch.Tensor,
    hand_init_orn: list,
    hand_init_orn_noise: list,
    device: str = "cpu",
) -> dict:
    """Run a single evaluation episode on the real robot.

    Args:
        robot: Connected robot interface.
        policy_net: Trained policy in eval mode.
        normalizer: Observation normalizer.
        model_info: Model info dict.
        obs_builder: Observation builder.
        controller: Hybrid/pose controller.
        real_config: Real robot config dict.
        goal_pos_noise_scale: [3] per-episode goal position noise std.
        goal_yaw_noise_scale: Per-episode goal yaw noise std (rad).
        hand_init_pos: [3] nominal EE start offset relative to fixed_asset_position.
        hand_init_pos_noise: [3] uniform noise range for start position.
        hand_init_orn: [3] nominal EE start orientation (RPY).
        hand_init_orn_noise: [3] uniform noise range for start orientation (RPY).
        device: Torch device.

    Returns:
        Episode result dict for metrics computation.
    """
    task_cfg = real_config['task']
    goal_position = torch.tensor(task_cfg['fixed_asset_position'], device=device, dtype=torch.float32)
    target_peg_base_pos = torch.tensor(task_cfg['target_peg_base_position'], device=device, dtype=torch.float32)
    ee_to_peg_base_offset = torch.tensor(task_cfg['ee_to_peg_base_offset'], device=device, dtype=torch.float32)

    xy_centering = task_cfg['xy_centering_threshold']
    hole_height = task_cfg['hole_height']
    success_threshold = task_cfg['success_threshold']
    break_force_threshold = task_cfg['break_force_threshold']
    max_steps = task_cfg['episode_timeout_steps']
    terminate_on_success = task_cfg['terminate_on_success']

    # --- Per-episode randomization ---

    # 1. Sample per-episode goal noise (obs + action frame)
    pos_noise = torch.randn(3, device=device) * goal_pos_noise_scale
    noisy_goal = goal_position + pos_noise

    yaw_offset = 0.0
    if goal_yaw_noise_scale > 0:
        yaw_offset = (torch.randn(1, device=device) * goal_yaw_noise_scale).item()

    # 2. Sample per-episode start pose noise (matching IsaacLab's uniform sampling)
    start_pos_noise = (2 * torch.rand(3, device=device) - 1) * hand_init_pos_noise
    start_yaw_noise = (2 * torch.rand(1, device=device).item() - 1) * hand_init_orn_noise[2]

    # 3. Compute target EE start pose in world frame
    # hand_init_pos is relative to hole tip (top of hole), but goal_position is hole
    # bottom, so add hole_height in Z to get the tip reference point
    hole_tip_offset = torch.tensor([0.0, 0.0, hole_height], device=device, dtype=torch.float32)
    target_ee_pos = goal_position + hole_tip_offset + hand_init_pos + start_pos_noise
    target_rpy = [hand_init_orn[0], hand_init_orn[1], hand_init_orn[2] + start_yaw_noise]
    target_pose = make_ee_target_pose(target_ee_pos.cpu().numpy(), np.array(target_rpy))

    # 4. Retract upward to safely clear the hole before moving to new start pose
    retract_height = real_config['robot']['retract_height_m']
    input("    [WAIT] Press Enter to RETRACT upward...")
    robot.retract_up(retract_height)

    # 5. Reset robot to start pose
    input("    [WAIT] Press Enter to MOVE TO START POSE...")
    robot.reset_to_start_pose(target_pose)

    prev_actions = torch.zeros(model_info['action_dim'], device=device)

    # Episode tracking
    succeeded = False
    terminated = False
    success_step = -1
    termination_step = -1
    ssv_sum = 0.0
    ssjv_sum = 0.0
    max_force = 0.0
    sum_force_in_contact = 0.0
    contact_steps = 0
    energy_sum = 0.0

    contact_force_threshold = obs_builder.contact_force_threshold

    # 6. Initialize controller using cached EE state from reset_to_start_pose
    ee_pos = robot.get_ee_position()
    controller.reset(ee_pos, noisy_goal)

    # 7. Start torque control — go straight into the loop, no work in between
    input("    [WAIT] Press Enter to START ROLLOUT...")
    robot.start_torque_mode()
    for step in range(max_steps):
        # read_state runs 1kHz loop for ~67ms, handles timing
        print(f"    [DEBUG] step={step} held={[f'{t:.2f}' for t in robot._held_torques]} cmd={[f'{t:.2f}' for t in robot._cmd_torques]}")
        robot.read_state()
        robot.check_safety()

        # Build observation (uses noisy_goal for obs frame, yaw_offset for yaw noise)
        obs = obs_builder.build_observation(robot, noisy_goal, prev_actions, fixed_yaw_offset=yaw_offset)

        # Get action
        action = get_deterministic_action(policy_net, normalizer, obs, model_info)

        # Read robot state for controller
        ee_pos = robot.get_ee_position()
        ee_quat = robot.get_ee_orientation()
        ee_linvel = robot.get_ee_linear_velocity()
        ee_angvel = robot.get_ee_angular_velocity()
        force_torque = robot.get_force_torque()
        joint_pos = robot.get_joint_positions()
        joint_vel = robot.get_joint_velocities()
        jacobian = robot.get_jacobian()
        mass_matrix = robot.get_mass_matrix()

        # Compute control (action frame uses noisy goal)
        ctrl_output = controller.compute_action(
            action, ee_pos, ee_quat, ee_linvel, ee_angvel,
            force_torque, joint_pos, joint_vel, jacobian, mass_matrix,
            noisy_goal,
        )

        # Direct torque command
        robot.send_joint_torques(ctrl_output['joint_torques'])

        # Update prev_actions for next observation
        prev_actions = ctrl_output['ema_actions']

        # ---- Metric tracking ----

        # SSV: sum of EE velocity magnitude
        velocity_norm = torch.norm(ee_linvel).item()
        ssv_sum += velocity_norm

        # SSJV: sum of squared joint velocity norm
        ssjv_step = torch.norm(joint_vel * joint_vel).item()
        ssjv_sum += ssjv_step

        # Force metrics
        force_magnitude = torch.norm(force_torque[:3]).item()
        max_force = max(max_force, force_magnitude)

        # Contact detection (matching sim: any axis force > threshold)
        any_contact = (force_torque[:3].abs() >= contact_force_threshold).any().item()
        if any_contact:
            sum_force_in_contact += force_magnitude
            contact_steps += 1

        # Energy: sum |joint_vel * joint_torque| for arm joints
        energy_step = torch.sum(
            torch.abs(joint_vel * ctrl_output['joint_torques'])
        ).item()
        energy_sum += energy_step

        # ---- Check termination conditions ----

        # Success check (uses TRUE target position, not noisy goal)
        if not succeeded:
            is_success = check_success(
                ee_pos, ee_to_peg_base_offset, target_peg_base_pos,
                xy_centering, hole_height, success_threshold,
            )
            if is_success:
                succeeded = True
                success_step = step
                print(f"    SUCCESS at step {step}")
                if terminate_on_success:
                    break

        # Break check
        if not terminated:
            is_break = check_break(force_torque, break_force_threshold)
            if is_break:
                terminated = True
                termination_step = step
                print(f"    BREAK at step {step} (force={force_magnitude:.2f}N)")
                break

    # End torque control session immediately so the robot isn't waiting
    # for 1kHz communication while we compute metrics / print results
    robot.end_control()

    episode_length = step + 1

    # Normalize smoothness by episode length (matching sim: ssv = sum / ep_len)
    ssv = ssv_sum / episode_length if episode_length > 0 else 0.0
    ssjv = ssjv_sum / episode_length if episode_length > 0 else 0.0
    energy = energy_sum  # Energy is total, not averaged per step (matching sim)

    return {
        'succeeded': succeeded,
        'terminated': terminated,
        'length': episode_length,
        'ssv': ssv,
        'ssjv': ssjv,
        'max_force': max_force,
        'sum_force_in_contact': sum_force_in_contact,
        'contact_steps': contact_steps,
        'energy': energy,
        'success_step': success_step if success_step >= 0 else episode_length,
        'termination_step': termination_step if termination_step >= 0 else episode_length,
    }


# ============================================================================
# Main evaluation loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Real Robot Evaluation")
    parser.add_argument("--tag", type=str, required=True, help="WandB experiment tag")
    parser.add_argument("--entity", type=str, default="hur", help="WandB entity")
    parser.add_argument("--project", type=str, default="SG_Exps", help="WandB project")
    parser.add_argument("--num_episodes", type=int, default=20, help="Episodes per agent")
    parser.add_argument("--eval_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--config", type=str, default="real_robot_exps/config.yaml", help="Config path")
    parser.add_argument("--run_id", type=str, default=None, help="Evaluate specific run only")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--override", action="append", default=[], help="Override config values (repeatable)")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.eval_seed)

    print("=" * 80)
    print("REAL ROBOT EVALUATION")
    print("=" * 80)

    # 1. Load real robot config
    print(f"\nLoading config: {args.config}")
    real_config = load_real_robot_config(args.config, args.override)

    # Enable no-sim mode before importing config infrastructure
    from configs.cfg_exts.version_compat import set_no_sim_mode
    set_no_sim_mode(True)

    # 2. Query WandB for training runs
    import wandb
    from eval.checkpoint_utils import (
        query_runs_by_tag,
        reconstruct_config_from_wandb,
        download_checkpoint_pair,
        get_best_checkpoints_for_runs,
    )

    print(f"\nQuerying WandB for runs with tag: {args.tag}")
    runs = query_runs_by_tag(args.tag, args.entity, args.project, args.run_id)

    # 3. Reconstruct training config from first run (all runs share same config)
    print("\nReconstructing training config...")
    configs, temp_dir = reconstruct_config_from_wandb(runs[0])

    # 4. Get best checkpoint for each run
    print("\nFinding best checkpoints...")
    api = wandb.Api(timeout=60)
    best_checkpoints = get_best_checkpoints_for_runs(
        api, runs, args.tag, args.entity, args.project
    )

    # 5. Reconstruct obs_order and determine model properties
    obs_order = reconstruct_obs_order(configs)

    hybrid_enabled = configs['wrappers'].hybrid_control.enabled
    if hybrid_enabled:
        from configs.cfg_exts.ctrl_mode import get_force_size
        ctrl_mode = getattr(configs['primary'], 'ctrl_mode', 'force_only')
        force_size = get_force_size(ctrl_mode)
        action_dim = 2 * force_size + 6
    else:
        action_dim = 6

    # FT sensor config
    ft_cfg = configs['wrappers'].force_torque_sensor
    use_tanh = getattr(ft_cfg, 'use_tanh_scaling', False)
    tanh_scale = getattr(ft_cfg, 'tanh_scale', 0.03)
    contact_threshold = getattr(ft_cfg, 'contact_force_threshold', 1.5)

    # Per-episode noise config: real robot override or WandB training config
    noise_cfg = real_config.get('noise', {})
    use_rr_noise = noise_cfg.get('use_rr_noise', False)

    if use_rr_noise:
        # Load ALL noise values from real robot config (fail-fast if missing)
        print("[main] Using REAL ROBOT noise config (noise.use_rr_noise=true)")
        goal_pos_noise_scale = torch.tensor(noise_cfg['goal_pos_noise'], device=args.device, dtype=torch.float32)
        use_yaw_noise = noise_cfg['use_yaw_noise']
        goal_yaw_noise_scale = noise_cfg['goal_yaw_noise'] if use_yaw_noise else 0.0
        hand_init_pos = torch.tensor(noise_cfg['hand_init_pos'], device=args.device, dtype=torch.float32)
        hand_init_pos_noise = torch.tensor(noise_cfg['hand_init_pos_noise'], device=args.device, dtype=torch.float32)
        hand_init_orn = list(noise_cfg['hand_init_orn'])
        hand_init_orn_noise = list(noise_cfg['hand_init_orn_noise'])
    else:
        # Load noise from WandB training config (matching sim exactly)
        print("[main] Using WANDB training noise config (noise.use_rr_noise=false)")
        obs_rand = configs['environment'].obs_rand
        goal_pos_noise_scale = torch.tensor(obs_rand.fixed_asset_pos, device=args.device, dtype=torch.float32)
        use_yaw_noise = hasattr(obs_rand, 'use_yaw_noise') and obs_rand.use_yaw_noise
        goal_yaw_noise_scale = obs_rand.fixed_asset_yaw if use_yaw_noise else 0.0

        # Get task config for start pose params (field is 'task' on ExtendedFactoryPegEnvCfg)
        cfg_task = getattr(configs['environment'], 'task', None) or configs['environment']
        hand_init_pos = torch.tensor(getattr(cfg_task, 'hand_init_pos', [0.0, 0.0, 0.047]),
                                     device=args.device, dtype=torch.float32)
        hand_init_pos_noise = torch.tensor(getattr(cfg_task, 'hand_init_pos_noise', [0.02, 0.02, 0.01]),
                                           device=args.device, dtype=torch.float32)
        hand_init_orn = list(getattr(cfg_task, 'hand_init_orn', [3.1416, 0.0, 0.0]))
        hand_init_orn_noise = list(getattr(cfg_task, 'hand_init_orn_noise', [0.0, 0.0, 0.785]))

    print(f"[main] goal_pos_noise_scale={goal_pos_noise_scale.tolist()}, "
          f"goal_yaw_noise_scale={goal_yaw_noise_scale}")
    print(f"[main] hand_init_pos={hand_init_pos.tolist()}, "
          f"hand_init_pos_noise={hand_init_pos_noise.tolist()}")
    print(f"[main] hand_init_orn={hand_init_orn}, hand_init_orn_noise={hand_init_orn_noise}")

    # 6. Initialize observation builder
    obs_builder = ObservationBuilder(
        obs_order=obs_order,
        action_dim=action_dim,
        use_tanh_ft_scaling=use_tanh,
        tanh_ft_scale=tanh_scale,
        contact_force_threshold=contact_threshold,
        device=args.device,
    )

    # 7. Initialize robot interface
    print("\nInitializing robot interface...")
    robot = FrankaInterface(real_config, device=args.device)

    # 8. Initialize controller
    print("\nInitializing controller...")
    controller = RealRobotController(configs, real_config, device=args.device)

    # 9. Evaluate each run
    print(f"\n{'=' * 80}")
    print(f"EVALUATING {len(runs)} RUN(S), {args.num_episodes} EPISODES EACH")
    print(f"{'=' * 80}")

    for run_idx, run in enumerate(runs):
        run_id = run.id
        best_step = best_checkpoints[run_id]

        print(f"\n--- Run {run_idx+1}/{len(runs)}: {run.name} (best step: {best_step}) ---")

        # Download checkpoint
        policy_path, critic_path = download_checkpoint_pair(run, best_step)
        download_dir = os.path.dirname(os.path.dirname(policy_path))

        try:
            # Load policy
            policy_net, normalizer, model_info = load_single_agent_policy(
                policy_path, configs, obs_dim=obs_builder.obs_dim, device=args.device,
            )

            # Validate observation dimensions
            obs_builder.validate_against_checkpoint(model_info['obs_dim'])

            # Run episodes
            episode_results = []
            for ep_idx in range(args.num_episodes):
                print(f"  Episode {ep_idx+1}/{args.num_episodes}:", end=" ", flush=True)

                result = run_episode(
                    robot, policy_net, normalizer, model_info,
                    obs_builder, controller, real_config,
                    goal_pos_noise_scale, goal_yaw_noise_scale,
                    hand_init_pos, hand_init_pos_noise,
                    hand_init_orn, hand_init_orn_noise,
                    args.device,
                )

                outcome = "SUCCESS" if result['succeeded'] and not result['terminated'] else \
                          "BREAK" if result['terminated'] else "TIMEOUT"
                print(f"{outcome} (len={result['length']}, max_f={result['max_force']:.2f}N)")

                episode_results.append(result)

            # Compute metrics
            metrics = compute_real_robot_metrics(episode_results)

            # Print summary
            print(f"\n  Results for {run.name}:")
            print(f"    Successes: {metrics['num_successful_completions']}/{metrics['total_episodes']}")
            print(f"    Breaks:    {metrics['num_breaks']}/{metrics['total_episodes']}")
            print(f"    Timeouts:  {metrics['num_failed_timeouts']}/{metrics['total_episodes']}")
            print(f"    Avg Length: {metrics['episode_length']:.1f}")
            print(f"    SSV:       {metrics['ssv']:.4f}")
            print(f"    Max Force: {metrics['max_force']:.2f}N")
            print(f"    Energy:    {metrics['energy']:.2f}")

            # Log to WandB
            if not args.no_wandb:
                eval_run = wandb.init(
                    entity=args.entity,
                    project=args.project,
                    name=f"Eval_RealRobot_{run.name}",
                    tags=["eval_real_robot", args.tag, f"source_run:{run_id}"],
                    config={
                        "source_run_id": run_id,
                        "source_run_name": run.name,
                        "best_step": best_step,
                        "num_episodes": args.num_episodes,
                        "real_robot_config": real_config,
                    },
                )

                # Log with Eval_Core prefix (matching sim eval format)
                prefixed_metrics = {f"Eval_Core/{k}": v for k, v in metrics.items()}
                prefixed_metrics["total_steps"] = best_step
                wandb.log(prefixed_metrics)
                wandb.finish()
                print(f"    Logged to WandB: {eval_run.url}")
            else:
                print("    (WandB logging disabled)")

        finally:
            # Cleanup checkpoint download
            shutil.rmtree(download_dir, ignore_errors=True)

    # 10. Shutdown
    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 80}")

    robot.shutdown()

    # Cleanup config temp dir
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
