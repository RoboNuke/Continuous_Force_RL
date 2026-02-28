"""
Real Robot Evaluation Script — Process-Based Architecture

Same as real_robot_eval.py but uses pro_robot_interface.FrankaInterface which
runs the torque compute loop in a separate process (eliminating GIL contention).

Usage:
    python eval/pro_real_robot_eval.py --tag "MATCH:2024-01-15_10:00" --num_episodes 20
    python eval/pro_real_robot_eval.py --tag "MATCH:2024-01-15_10:00" --no_wandb --run_id abc123
"""

import argparse
import json
import os
import select
import sys
import termios
import threading
import time
import tty
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.SimBa import SimBaNet
from real_robot_exps.pro_robot_interface import FrankaInterface, StateSnapshot, make_ee_target_pose
from real_robot_exps.observation_builder import ObservationBuilder, ObservationNormalizer, OBS_DIM_MAP
from real_robot_exps.hybrid_controller import RealRobotController, get_euler_xyz


# ============================================================================
# Forge eval noise ranges (matching wandb_eval.py NOISE_RANGES)
# ============================================================================

FORGE_NOISE_RANGES = [
    (0.0, 0.001, "0mm-1mm"),        # 0-1mm
    (0.001, 0.0025, "1mm-2.5mm"),   # 1-2.5mm
    (0.0025, 0.005, "2.5mm-5mm"),   # 2.5-5mm
    (0.005, 0.0075, "5mm-7.5mm"),   # 5-7.5mm
]


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
# Checkpoint caching
# ============================================================================

def sanitize_tag(tag: str) -> str:
    """Convert a WandB tag to a filesystem-safe directory name.

    Args:
        tag: WandB tag string (e.g. "MATCH:2024-01-15_10:00").

    Returns:
        Sanitized string safe for use as a directory name.
    """
    return tag.replace(':', '_').replace('/', '_').replace(' ', '_')


def save_to_cache(
    cache_path: str,
    config_temp_dir: str,
    runs: list,
    best_checkpoints: Dict[str, int],
    checkpoint_paths: Dict[str, Tuple[str, str]],
    best_scores: Optional[Dict[str, dict]] = None,
) -> None:
    """Save WandB data to local cache directory.

    Copies config YAMLs, checkpoint .pt files, and run metadata.
    runs.json is written LAST so its presence marks a complete cache.

    Args:
        cache_path: Absolute path to cache directory for this tag.
        config_temp_dir: Temp dir containing config YAML files from WandB.
        runs: List of WandB run objects (need .id, .name).
        best_checkpoints: Dict mapping run_id -> best checkpoint step.
        checkpoint_paths: Dict mapping run_id -> (policy_path, critic_path).
        best_scores: Optional dict mapping run_id -> {'score', 'successes', 'breaks'}.
    """
    os.makedirs(cache_path, exist_ok=True)

    # Copy base config
    base_src = os.path.join(config_temp_dir, 'config_base.yaml')
    if not os.path.exists(base_src):
        raise RuntimeError(f"Base config not found in temp dir: {base_src}")
    shutil.copy2(base_src, os.path.join(cache_path, 'config_base.yaml'))

    # Copy experiment config (update base_config path to point to cache)
    exp_src = os.path.join(config_temp_dir, 'config_experiment.yaml')
    if os.path.exists(exp_src):
        with open(exp_src, 'r') as f:
            exp_data = yaml.safe_load(f)
        exp_data['base_config'] = os.path.join(cache_path, 'config_base.yaml')
        with open(os.path.join(cache_path, 'config_experiment.yaml'), 'w') as f:
            yaml.safe_dump(exp_data, f, default_flow_style=False)

    # Copy checkpoint files
    for run_id, (policy_path, critic_path) in checkpoint_paths.items():
        run_dir = os.path.join(cache_path, run_id)
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy2(policy_path, os.path.join(run_dir, 'policy.pt'))
        shutil.copy2(critic_path, os.path.join(run_dir, 'critic.pt'))

    # Write runs.json LAST (serves as "cache complete" marker)
    runs_data = []
    for r in runs:
        entry = {
            'id': r.id,
            'name': r.name,
            'group': r.group,
            'project': r.project,
            'tags': list(r.tags),
            'best_step': best_checkpoints[r.id],
        }
        if best_scores and r.id in best_scores:
            entry['best_score'] = best_scores[r.id]['score']
            entry['best_successes'] = best_scores[r.id]['successes']
            entry['best_breaks'] = best_scores[r.id]['breaks']
        runs_data.append(entry)
    with open(os.path.join(cache_path, 'runs.json'), 'w') as f:
        json.dump(runs_data, f, indent=2)

    print(f"  Checkpoint cache saved to: {cache_path}")


def load_from_cache(
    cache_path: str,
    run_id_filter: Optional[str] = None,
) -> Tuple[dict, list, Dict[str, int], Dict[str, dict]]:
    """Load cached WandB data from local directory.

    Args:
        cache_path: Absolute path to cache directory for this tag.
        run_id_filter: Optional run ID to filter to a single run.

    Returns:
        Tuple of (configs, run_infos, best_checkpoints, best_scores) where:
        - configs: Config dict from ConfigManagerV3.process_config()
        - run_infos: List of objects with .id and .name attributes
        - best_checkpoints: Dict mapping run_id -> best checkpoint step
        - best_scores: Dict mapping run_id -> {'score', 'successes', 'breaks'} (empty for old caches)

    Raises:
        RuntimeError: If cache is incomplete or requested run not found.
    """
    from types import SimpleNamespace
    from configs.config_manager_v3 import ConfigManagerV3

    # Load run metadata
    runs_json_path = os.path.join(cache_path, 'runs.json')
    with open(runs_json_path, 'r') as f:
        runs_data = json.load(f)

    # Build run info objects, best_checkpoints, and best_scores mappings
    run_infos = []
    best_checkpoints = {}
    best_scores = {}
    for entry in runs_data:
        for required_field in ('group', 'project', 'tags'):
            if required_field not in entry:
                raise RuntimeError(
                    f"Cache is missing '{required_field}' field (old format). "
                    f"Delete the cache directory and re-download: {cache_path}"
                )
        run_infos.append(SimpleNamespace(
            id=entry['id'], name=entry['name'],
            group=entry['group'], project=entry['project'],
            tags=entry['tags'],
        ))
        best_checkpoints[entry['id']] = entry['best_step']
        # Load sim eval scores (may be absent in old caches)
        if 'best_score' in entry:
            best_scores[entry['id']] = {
                'score': entry['best_score'],
                'successes': entry['best_successes'],
                'breaks': entry['best_breaks'],
            }

    # Apply run_id filter
    if run_id_filter is not None:
        run_infos = [r for r in run_infos if r.id == run_id_filter]
        if len(run_infos) == 0:
            available = [e['id'] for e in runs_data]
            raise RuntimeError(
                f"Run ID '{run_id_filter}' not found in cache. "
                f"Available run IDs: {available}. "
                f"Delete the cache directory to re-download: {cache_path}"
            )
        best_checkpoints = {r.id: best_checkpoints[r.id] for r in run_infos}
        best_scores = {r.id: best_scores[r.id] for r in run_infos if r.id in best_scores}

    # Validate checkpoint files exist
    for run_info in run_infos:
        run_dir = os.path.join(cache_path, run_info.id)
        for fname in ['policy.pt', 'critic.pt']:
            fpath = os.path.join(run_dir, fname)
            if not os.path.exists(fpath):
                raise RuntimeError(
                    f"Cached checkpoint missing: {fpath}. "
                    f"Delete the cache directory to re-download: {cache_path}"
                )

    # Load configs from cached YAML files
    config_manager = ConfigManagerV3()
    exp_path = os.path.join(cache_path, 'config_experiment.yaml')
    base_path = os.path.join(cache_path, 'config_base.yaml')

    if os.path.exists(exp_path):
        configs = config_manager.process_config(exp_path)
    elif os.path.exists(base_path):
        configs = config_manager.process_config(base_path)
    else:
        raise RuntimeError(
            f"No config YAML files found in cache: {cache_path}. "
            f"Delete the cache directory to re-download: {cache_path}"
        )

    print(f"  Loaded {len(run_infos)} run(s) from cache:")
    for r in run_infos:
        print(f"    - {r.id} ({r.name}) [best step: {best_checkpoints[r.id]}]")

    return configs, run_infos, best_checkpoints, best_scores


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
        - obs_rand.use_fixed_asset_yaw_noise -> appends "fingertip_yaw_rel_fixed"

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
    if hasattr(env_cfg, 'obs_rand') and getattr(env_cfg.obs_rand, 'use_fixed_asset_yaw_noise', False):
        obs_order.append("fingertip_yaw_rel_fixed")

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
    # Must match training: BlockSimBaActor uses (sigma_idx == 0) and (not squash_actions)
    squash_actions = getattr(configs['model'], 'squash_actions', False)
    tan_out = (sigma_idx == 0) and (not squash_actions)

    if use_state_dependent_std:
        std_out_dim = action_dim - sigma_idx
    else:
        std_out_dim = 0

    out_size = action_dim + std_out_dim

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

    # Load log_std for optional stochastic sampling
    if use_state_dependent_std:
        log_std = None  # computed at runtime from network output
    else:
        if 'log_std' not in checkpoint:
            raise RuntimeError(
                f"Policy checkpoint missing 'log_std' (required for non-state-dependent std): {policy_path}"
            )
        log_std = checkpoint['log_std'].to(device)  # [1, num_components]

    model_info = {
        'sigma_idx': sigma_idx,
        'action_dim': action_dim,
        'use_state_dependent_std': use_state_dependent_std,
        'squash_actions': squash_actions,
        'obs_dim': obs_dim,
        'log_std': log_std,
    }

    return policy_net, normalizer, model_info


# ============================================================================
# Action inference (deterministic or stochastic)
# ============================================================================

@torch.no_grad()
def get_action(
    policy_net: SimBaNet,
    normalizer: ObservationNormalizer,
    obs: torch.Tensor,
    model_info: dict,
    std_scale: float = 0.0,
) -> torch.Tensor:
    """Get action from policy, optionally with stochastic sampling.

    When std_scale <= 0, returns the deterministic mean action.
    When std_scale > 0, samples from the policy distribution with the
    learned std dev scaled by std_scale. E.g. std_scale=0.1 uses 10%
    of the training std dev.

    Selection actions in hybrid mode always stay deterministic (thresholded
    at 0.5) for safety — only continuous components are sampled.

    Noise is added in the same space as training:
    - Standard (tan_out=True): noise in post-tanh space
    - Squashed (squash_actions=True): noise in pre-tanh space, then tanh applied
    - Hybrid non-squash: noise in post-tanh space on components
    - Hybrid squash: noise in pre-tanh space on components, then tanh applied

    Args:
        policy_net: Trained SimBaNet in eval mode.
        normalizer: ObservationNormalizer with frozen stats.
        obs: [obs_dim] raw observation tensor.
        model_info: Dict with sigma_idx, action_dim, use_state_dependent_std,
                     squash_actions, log_std.
        std_scale: Multiplier on learned std dev. 0.0 = deterministic.

    Returns:
        [action_dim] action tensor with appropriate activations applied.
    """
    # Normalize and batch
    norm_obs = normalizer.normalize(obs.unsqueeze(0))  # [1, obs_dim]

    # Forward pass
    raw_output = policy_net(norm_obs)  # [1, out_size]
    mean_action = raw_output[0, :model_info['action_dim']]

    sigma_idx = model_info['sigma_idx']

    # --- Deterministic path ---
    if std_scale <= 0.0:
        if sigma_idx == 0:
            if model_info.get('squash_actions', False):
                return torch.tanh(mean_action)
            else:
                return mean_action
        else:
            selection = torch.sigmoid(mean_action[:sigma_idx])
            components = torch.tanh(mean_action[sigma_idx:])
            return torch.cat([selection, components])

    # --- Stochastic path ---
    # Extract log_std
    if model_info['log_std'] is not None:
        log_std = model_info['log_std'].squeeze(0)  # [num_components]
    else:
        # State-dependent: std is in network output after action_dim
        log_std = raw_output[0, model_info['action_dim']:]

    # Clamp log_std (matching training: min=-20, max=2)
    log_std = torch.clamp(log_std, -20.0, 2.0)
    std = torch.exp(log_std) * std_scale

    if sigma_idx == 0:
        noise = torch.randn_like(mean_action)
        if model_info.get('squash_actions', False):
            # Squashed: noise in pre-tanh space, then tanh
            return torch.tanh(mean_action + std * noise)
        else:
            # Standard: mean already tanh'd by network, noise in that space
            return mean_action + std * noise
    else:
        # Hybrid: selection stays deterministic, noise on components only
        selection = (torch.sigmoid(mean_action[:sigma_idx]) > 0.5).float()
        raw_components = mean_action[sigma_idx:]
        noise = torch.randn_like(raw_components)

        if model_info.get('squash_actions', False):
            # Squashed: noise pre-tanh, then tanh
            components = torch.tanh(raw_components + std * noise)
        else:
            # Non-squash: noise in post-tanh space (matching training)
            components = torch.tanh(raw_components) + std * noise

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
    total_force = sum(ep['sum_force'] for ep in episode_results)
    total_steps = sum(ep['length'] for ep in episode_results)
    metrics['avg_force'] = total_force / total_steps if total_steps > 0 else 0.0
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
# Observation distribution comparison
# ============================================================================

def print_obs_distribution_comparison(
    all_obs: List[torch.Tensor],
    obs_builder: ObservationBuilder,
    normalizer: ObservationNormalizer,
) -> None:
    """Print per-channel comparison of real robot obs vs training distribution.

    Compares the mean and std of raw observations collected during real robot
    rollouts against the running mean/std from the training normalizer.

    Args:
        all_obs: List of [obs_dim] raw observation tensors collected during rollouts.
        obs_builder: ObservationBuilder (for channel labels).
        normalizer: ObservationNormalizer (for training statistics).
    """
    stacked = torch.stack(all_obs)  # [N, obs_dim]
    real_mean = stacked.mean(dim=0)
    real_std = stacked.std(dim=0)

    train_mean = normalizer.running_mean
    train_std = torch.sqrt(normalizer.running_variance + normalizer.eps)

    # Build channel labels from obs_order + prev_actions
    labels = []
    for name in obs_builder.obs_order:
        dim = obs_builder._obs_dim_map[name]
        if dim == 1:
            labels.append(name)
        else:
            for i in range(dim):
                labels.append(f"{name}[{i}]")
    for i in range(obs_builder.action_dim):
        labels.append(f"prev_action[{i}]")

    rp = EvalKeyboardController.raw_print

    # Pre-compute normalized shifts and collect outliers (|shift| > 2)
    outliers = []
    for i, label in enumerate(labels):
        tm = train_mean[i].item()
        ts = train_std[i].item()
        rm = real_mean[i].item()
        rs = real_std[i].item()

        if ts > 1e-8:
            normalized_shift = (rm - tm) / ts
        else:
            normalized_shift = 0.0 if abs(rm - tm) < 1e-8 else float('inf')

        if abs(normalized_shift) > 2.0:
            outliers.append((label, tm, ts, rm, rs, normalized_shift))

    # Only print if there are outliers
    if not outliers:
        return

    n_samples = stacked.shape[0]
    w = 100
    rp(f"{'=' * w}")
    rp(f"OBS DISTRIBUTION OUTLIERS (|shift| > 2σ): {len(outliers)}/{len(labels)} channels, {n_samples} samples")
    rp(f"{'=' * w}")
    rp(f"{'Channel':<28} {'Train Mean':>11} {'Train Std':>11}  |  {'Real Mean':>11} {'Real Std':>11}  |  {'MeanΔ/Tstd':>11}")
    rp(f"{'-' * w}")

    for label, tm, ts, rm, rs, normalized_shift in outliers:
        flag = " <<<" if abs(normalized_shift) > 3.0 else ""
        rp(f"{label:<28} {tm:>11.5f} {ts:>11.5f}  |  {rm:>11.5f} {rs:>11.5f}  |  {normalized_shift:>+11.2f}{flag}")

    rp(f"{'=' * w}")


# ============================================================================
# Non-blocking keyboard controller for eval
# ============================================================================

class EvalKeyboardController:
    """Non-blocking keyboard listener for eval control.

    Runs a daemon thread that reads single keypresses in raw terminal mode.
    Main thread polls state via properties: should_skip, should_pause.

    Keys (during episode):
        's' - skip: end current episode immediately (counted as BREAK)
        'p' - pause: finish current episode, then pause before next

    Keys (while paused):
        'c' - calibrate: move robot to goal XY, 5cm above goal Z
        Enter - resume: continue running episodes

    Keys (any time):
        ESC - quit: end current episode immediately and shut down
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._skip = False
        self._pause = False
        self._resume = False
        self._calibrate = False
        self._quit = False
        self._paused = False  # set by main thread when entering pause state
        self._stop = threading.Event()
        self._old_settings = None

    def start(self):
        """Save terminal settings, set raw mode, start listener thread."""
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        self._stop.clear()
        thread = threading.Thread(target=self._read_loop, daemon=True)
        thread.start()

    def stop(self):
        """Restore terminal settings."""
        self._stop.set()
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    @property
    def should_skip(self) -> bool:
        """Returns True (once) if 's' was pressed during an episode."""
        with self._lock:
            val = self._skip
            self._skip = False
            return val

    @property
    def should_pause(self) -> bool:
        """Returns True if 'p' was pressed (stays True until cleared by resume)."""
        with self._lock:
            return self._pause

    @property
    def should_calibrate(self) -> bool:
        """Returns True (once) if 'c' was pressed while paused."""
        with self._lock:
            val = self._calibrate
            self._calibrate = False
            return val

    @property
    def should_resume(self) -> bool:
        """Returns True (once) if Enter was pressed while paused."""
        with self._lock:
            val = self._resume
            self._resume = False
            return val

    @property
    def should_quit(self) -> bool:
        """Returns True if ESC was pressed."""
        with self._lock:
            return self._quit

    def set_paused(self, paused: bool):
        """Called by main thread to enter/exit pause state."""
        with self._lock:
            self._paused = paused
            if not paused:
                self._pause = False
                self._resume = False
                self._calibrate = False

    @staticmethod
    def raw_print(msg: str):
        """Print with \\r\\n so output isn't garbled in raw terminal mode."""
        sys.stdout.write(msg + "\r\n")
        sys.stdout.flush()

    def _read_loop(self):
        """Background thread: poll stdin for keypresses."""
        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if ready:
                ch = sys.stdin.read(1)
                with self._lock:
                    if ch == '\x1b':
                        self._quit = True
                        self._skip = True  # end current episode immediately
                    elif ch.lower() == 's' and not self._paused:
                        self._skip = True
                    elif ch.lower() == 'p' and not self._paused:
                        self._pause = True
                    elif ch.lower() == 'c' and self._paused:
                        self._calibrate = True
                    elif ch in ('\r', '\n') and self._paused:
                        self._resume = True


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
    episode_noise: dict,
    hand_init_pos: torch.Tensor,
    hand_init_orn: list,
    hand_init_orn_noise: list,
    keyboard: EvalKeyboardController,
    device: str = "cpu",
    log_trajectory: bool = False,
    std_scale: float = 0.0,
) -> Optional[dict]:
    """Run a single evaluation episode on the real robot.

    Args:
        robot: Connected robot interface.
        policy_net: Trained policy in eval mode.
        normalizer: Observation normalizer.
        model_info: Model info dict.
        obs_builder: Observation builder.
        controller: Hybrid/pose controller.
        real_config: Real robot config dict.
        episode_noise: Pre-generated noise for this episode. Keys:
            'goal_pos_noise': [3] goal position noise tensor.
            'goal_yaw_noise': float, goal yaw offset (rad).
            'start_pos_noise': [3] start position noise tensor.
        hand_init_pos: [3] nominal EE start offset relative to fixed_asset_position.
        hand_init_orn: [3] nominal EE start orientation (RPY).
        hand_init_orn_noise: [3] uniform noise range for start orientation (RPY).
        device: Torch device.
        log_trajectory: If True, enable 1kHz trajectory logging and collect
                        15Hz policy data for saving.

    Returns:
        Episode result dict for metrics computation.
    """
    task_cfg = real_config['task']
    fixed_asset_position = torch.tensor(task_cfg['fixed_asset_position'], device=device, dtype=torch.float32)
    # Compute observation frame: fixed_asset_position + hole_height + base_height
    # Matches sim's fixed_pos_obs_frame = fixed_pos + [0, 0, height + base_height]
    obs_frame_z_offset = task_cfg['hole_height'] + task_cfg['fixed_asset_base_height']
    goal_position = fixed_asset_position.clone()
    goal_position[2] += obs_frame_z_offset
    target_peg_base_pos = torch.tensor(task_cfg['target_peg_base_position'], device=device, dtype=torch.float32)
    ee_to_peg_base_offset = torch.tensor(task_cfg['ee_to_peg_base_offset'], device=device, dtype=torch.float32)

    xy_centering = task_cfg['xy_centering_threshold']
    hole_height = task_cfg['hole_height']
    success_threshold = task_cfg['success_threshold']
    engage_threshold = task_cfg['engage_threshold']
    break_force_threshold = task_cfg['break_force_threshold']
    max_steps = task_cfg['episode_timeout_steps']
    terminate_on_success = task_cfg['terminate_on_success']

    # Safety: force selection height gate (hybrid only)
    is_hybrid = model_info['sigma_idx'] > 0
    if is_hybrid:
        if 'force_select_max_height_mm' not in task_cfg:
            raise RuntimeError(
                "task.force_select_max_height_mm is required in config.yaml for hybrid models. "
                "This is the max height (mm) above hole top where force control is allowed."
            )
        if 'force_select_max_xy_dist_mm' not in task_cfg:
            raise RuntimeError(
                "task.force_select_max_xy_dist_mm is required in config.yaml for hybrid models. "
                "This is the max XY distance (mm) from goal where force control is allowed."
            )
        force_select_max_height_m = task_cfg['force_select_max_height_mm'] / 1000.0
        force_select_max_xy_dist_m = task_cfg['force_select_max_xy_dist_mm'] / 1000.0

    # --- Per-episode noise (pre-generated for cross-policy consistency) ---

    pos_noise = episode_noise['goal_pos_noise']
    noisy_goal = goal_position + pos_noise

    yaw_offset = episode_noise['goal_yaw_noise']

    start_pos_noise = episode_noise['start_pos_noise']
    start_yaw_noise = 0.0  # no yaw noise on starting pose

    # 3. Compute target EE start pose in world frame (ABSOLUTE, not relative to EE)
    #    hand_init_pos is the EE offset above the hole tip, matching sim's convention
    #    in spawn_height_curriculum_wrapper: above_fixed_pos = fixed_tip_pos + hand_init_pos
    target_ee_pos = goal_position + hand_init_pos + start_pos_noise
    target_rpy = [hand_init_orn[0], hand_init_orn[1], hand_init_orn[2] + start_yaw_noise]
    target_pose = make_ee_target_pose(target_ee_pos.cpu().numpy(), np.array(target_rpy))

    # 4. Retract + reset with retry logic (motion reflex errors are recoverable)
    MAX_MOTION_RETRIES = 5
    retract_height = real_config['robot']['retract_height_m']
    for attempt in range(MAX_MOTION_RETRIES):
        try:
            robot.retract_up(retract_height)
            robot.reset_to_start_pose(target_pose)
            break
        except RuntimeError as e:
            EvalKeyboardController.raw_print(f"  [MOTION RETRY {attempt+1}/{MAX_MOTION_RETRIES}] {e}")
            try:
                robot.error_recovery()
            except RuntimeError:
                pass
            time.sleep(1.0)
    else:
        EvalKeyboardController.raw_print(f"  [MOTION FAILED] All {MAX_MOTION_RETRIES} retries exhausted")
        return None

    # 5. Calibrate FT bias at the start pose for this episode
    ft_bias = robot.calibrate_ft_bias()
    EvalKeyboardController.raw_print(
        f"  FT bias: [{ft_bias[0]:+.4f}, {ft_bias[1]:+.4f}, {ft_bias[2]:+.4f}, "
        f"{ft_bias[3]:+.4f}, {ft_bias[4]:+.4f}, {ft_bias[5]:+.4f}]"
    )

    prev_actions = torch.zeros(model_info['action_dim'], device=device)

    # Episode tracking
    succeeded = False
    engaged = False
    terminated = False
    success_step = -1
    termination_step = -1
    ssv_sum = 0.0
    ssjv_sum = 0.0
    max_force = 0.0
    sum_force = 0.0
    sum_force_in_contact = 0.0
    contact_steps = 0
    energy_sum = 0.0
    force_blocked_steps = 0

    # Force tracking error (hybrid only)
    sum_force_error_x = 0.0
    sum_cmd_force_x = 0.0
    sum_meas_force_x = 0.0
    force_selected_steps_x = 0
    sum_force_error_y = 0.0
    sum_cmd_force_y = 0.0
    sum_meas_force_y = 0.0
    force_selected_steps_y = 0
    sum_force_error_z = 0.0
    sum_cmd_force_z = 0.0
    sum_meas_force_z = 0.0
    force_selected_steps_z = 0

    # Position tracking error (per-axis, on position-selected steps)
    sum_pos_error_x = 0.0
    sum_cmd_pos_x = 0.0
    sum_meas_pos_x = 0.0
    pos_selected_steps_x = 0
    sum_pos_error_y = 0.0
    sum_cmd_pos_y = 0.0
    sum_meas_pos_y = 0.0
    pos_selected_steps_y = 0
    sum_pos_error_z = 0.0
    sum_cmd_pos_z = 0.0
    sum_meas_pos_z = 0.0
    pos_selected_steps_z = 0

    contact_force_threshold = obs_builder.contact_force_threshold

    # Collect raw observations for distribution analysis
    obs_history = []

    # 6. Initialize controller using cached EE state from reset_to_start_pose
    snap = robot.get_state_snapshot()
    controller.reset(snap.ee_pos, noisy_goal)

    # 7. Warmup policy + controller (triggers PyTorch JIT compilation)
    for _wi in range(3):
        _warmup_obs = obs_builder.build_observation(snap, noisy_goal, prev_actions, fixed_yaw_offset=yaw_offset)
        _warmup_action = get_action(policy_net, normalizer, _warmup_obs, model_info, std_scale=std_scale)
        _warmup_ctrl = controller.compute_action(
            _warmup_action, snap.ee_pos, snap.ee_quat, snap.ee_linvel, snap.ee_angvel,
            snap.force_torque, snap.joint_pos, snap.joint_vel, snap.jacobian, snap.mass_matrix,
            noisy_goal,
        )
    # Reset controller state since warmup modified EMA
    controller.reset(snap.ee_pos, noisy_goal)
    prev_actions = torch.zeros(model_info['action_dim'], device=device)

    # 8. Start torque control — background 1kHz thread keeps robot fed
    robot.start_torque_mode(log_trajectory=log_trajectory)
    _ep_start = time.time()

    # 15Hz trajectory accumulation (only when logging)
    if log_trajectory:
        traj_actions_15hz = []
        traj_sel_matrices_15hz = []
        traj_time_ms_15hz = []
    for step in range(max_steps):
        # Wait for 15Hz policy step timing, then grab latest snapshot
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        robot.check_safety(snap)

        # Check for keyboard skip ('s' key)
        if keyboard.should_skip:
            terminated = True
            termination_step = step
            break

        # Build observation from snapshot (uses noisy_goal for obs frame)
        obs = obs_builder.build_observation(snap, noisy_goal, prev_actions, fixed_yaw_offset=yaw_offset)
        obs_history.append(obs.clone())

        # Get action
        action = get_action(policy_net, normalizer, obs, model_info, std_scale=std_scale)

        # Print achieved error for previous target before computing new one
        if step > 0:
            cr, cpi, cy = get_euler_xyz(snap.ee_quat)
            xy_err = (snap.ee_pos[:2] - prev_tp[:2]).norm().item() * 1000.0
            z_err = (snap.ee_pos[2] - prev_tp[2]).item() * 1000.0
            rpy_err = [np.degrees(cr - prev_tr), np.degrees(cpi - prev_tpi), np.degrees(cy - prev_ty)]
            # sp = snap.ee_pos
            # EvalKeyboardController.raw_print(
            #     f"    [ACHIEVED] step={step-1} xy_err={xy_err:.1f}mm z_err={z_err:.1f}mm "
            #     f"rpy_err=[{rpy_err[0]:.2f}, {rpy_err[1]:.2f}, {rpy_err[2]:.2f}]deg")
            #     # f"pos=[{sp[0]:.4f}, {sp[1]:.4f}, {sp[2]:.4f}]")

        # Compute control from snapshot state (action frame uses noisy goal)
        ctrl_output = controller.compute_action(
            action, snap.ee_pos, snap.ee_quat, snap.ee_linvel, snap.ee_angvel,
            snap.force_torque, snap.joint_pos, snap.joint_vel, snap.jacobian, snap.mass_matrix,
            noisy_goal,
        )

        tp = ctrl_output['target_pos']
        tr, tpi, ty = get_euler_xyz(ctrl_output['target_quat'])
        cr, cpi, cy = get_euler_xyz(snap.ee_quat)
        xy_err = (snap.ee_pos[:2] - tp[:2]).norm().item() * 1000.0
        z_err = (snap.ee_pos[2] - tp[2]).item() * 1000.0
        rpy_err = [np.degrees(cr - tr), np.degrees(cpi - tpi), np.degrees(cy - ty)]
        sel = ctrl_output.get('sel_matrix', None)
        if sel is not None:
            fx = 'F' if sel[0].item() > 0.5 else 'P'
            fy = 'F' if sel[1].item() > 0.5 else 'P'
            fz = 'F' if sel[2].item() > 0.5 else 'P'
            force_str = f" ctrl=[X:{fx} Y:{fy} Z:{fz}]"
        else:
            force_str = ""
        # EvalKeyboardController.raw_print(
        #     f"    [POLICY] step={step} xy_err={xy_err:.1f}mm z_err={z_err:.1f}mm "
        #     f"rpy_err=[{rpy_err[0]:.2f}, {rpy_err[1]:.2f}, {rpy_err[2]:.2f}]deg{force_str}")
        #     # f"tgt=[{tp[0]:.4f}, {tp[1]:.4f}, {tp[2]:.4f}]")
        if sel is not None:
            sx = 'F' if sel[0].item() > 0.5 else 'P'
            sy = 'F' if sel[1].item() > 0.5 else 'P'
            sz = 'F' if sel[2].item() > 0.5 else 'P'
            EvalKeyboardController.raw_print(
                f"    [SEL] step={step} X:{sx} Y:{sy} Z:{sz}")

        # Save target for next step's achieved error
        prev_tp = tp
        prev_tr, prev_tpi, prev_ty = tr, tpi, ty

        # Safety: block force selection when peg tip is too far from hole
        if is_hybrid:
            peg_tip_pos = snap.ee_pos + ee_to_peg_base_offset
            z_above_hole_m = (peg_tip_pos[2] - goal_position[2]).item()
            xy_dist_m = torch.norm(peg_tip_pos[:2] - goal_position[:2]).item()
            sel = ctrl_output['sel_matrix']
            any_force_selected = (sel > 0.5).any().item()

            should_block = (
                any_force_selected
                and (z_above_hole_m > force_select_max_height_m
                     or xy_dist_m > force_select_max_xy_dist_m)
            )
            if should_block:
                force_blocked_steps += 1
                blocked_axes = [ax for ax, s in zip(['X','Y','Z'], sel[:3]) if s.item() > 0.5]
                EvalKeyboardController.raw_print(
                    f"    [FORCE BLOCKED] step={step} axes={blocked_axes} "
                    f"z={z_above_hole_m*1000:.1f}mm (max={force_select_max_height_m*1000:.1f}) "
                    f"xy={xy_dist_m*1000:.1f}mm (max={force_select_max_xy_dist_m*1000:.1f})")
                zero_sel = torch.zeros(6, device=device)
                ctrl_output['sel_matrix'] = zero_sel
                ctrl_output['control_targets'] = ctrl_output['control_targets']._replace(
                    sel_matrix=zero_sel,
                )

        # Set control targets for 1kHz torque recomputation (starts 15Hz timer)
        robot.set_control_targets(ctrl_output['control_targets'])

        # Update prev_actions for next observation
        # Use EMA-smoothed actions to match sim's self.unwrapped.actions.clone()
        # (base env applies EMA in _pre_physics_step; hybrid wrapper overwrites
        #  self.unwrapped.actions = self.ema_actions.clone())
        prev_actions = ctrl_output['ema_actions']

        # Accumulate 15Hz trajectory data
        if log_trajectory:
            traj_actions_15hz.append(action.detach().cpu().clone())
            sel = ctrl_output.get('sel_matrix', None)
            if sel is not None:
                traj_sel_matrices_15hz.append(sel.detach().cpu().clone())
            else:
                traj_sel_matrices_15hz.append(torch.zeros(6))
            traj_time_ms_15hz.append((time.time() - _ep_start) * 1000.0)

        # ---- Metric tracking ----

        # SSV: sum of EE velocity magnitude
        velocity_norm = torch.norm(snap.ee_linvel).item()
        ssv_sum += velocity_norm

        # SSJV: sum of squared joint velocity norm
        ssjv_step = torch.norm(snap.joint_vel * snap.joint_vel).item()
        ssjv_sum += ssjv_step

        # Force metrics
        force_magnitude = torch.norm(snap.force_torque[:3]).item()
        sum_force += force_magnitude
        max_force = max(max_force, force_magnitude)

        # Contact detection (matching sim: any axis force > threshold)
        any_contact = (snap.force_torque[:3].abs() >= contact_force_threshold).any().item()
        if any_contact:
            sum_force_in_contact += force_magnitude
            contact_steps += 1

        # Energy: sum |joint_vel * joint_torque| for arm joints
        energy_step = torch.sum(
            torch.abs(snap.joint_vel * ctrl_output['joint_torques'])
        ).item()
        energy_sum += energy_step

        # Force tracking error (hybrid only, on force-selected steps)
        if is_hybrid:
            sel = ctrl_output['sel_matrix']
            tf = ctrl_output['target_force']
            force_err = (tf - snap.force_torque).abs()
            mf = snap.force_torque
            if sel[0].item() > 0.5:
                sum_force_error_x += force_err[0].item()
                sum_cmd_force_x += abs(tf[0].item())
                sum_meas_force_x += abs(mf[0].item())
                force_selected_steps_x += 1
            if sel[1].item() > 0.5:
                sum_force_error_y += force_err[1].item()
                sum_cmd_force_y += abs(tf[1].item())
                sum_meas_force_y += abs(mf[1].item())
                force_selected_steps_y += 1
            if sel[2].item() > 0.5:
                sum_force_error_z += force_err[2].item()
                sum_cmd_force_z += abs(tf[2].item())
                sum_meas_force_z += abs(mf[2].item())
                force_selected_steps_z += 1

        # Position tracking error (per-axis, on position-selected steps)
        pos_sel = ctrl_output.get('sel_matrix', None)
        pos_err = (tp - snap.ee_pos).abs()
        if pos_sel is None or pos_sel[0].item() <= 0.5:
            sum_pos_error_x += pos_err[0].item()
            sum_cmd_pos_x += abs(tp[0].item())
            sum_meas_pos_x += abs(snap.ee_pos[0].item())
            pos_selected_steps_x += 1
        if pos_sel is None or pos_sel[1].item() <= 0.5:
            sum_pos_error_y += pos_err[1].item()
            sum_cmd_pos_y += abs(tp[1].item())
            sum_meas_pos_y += abs(snap.ee_pos[1].item())
            pos_selected_steps_y += 1
        if pos_sel is None or pos_sel[2].item() <= 0.5:
            sum_pos_error_z += pos_err[2].item()
            sum_cmd_pos_z += abs(tp[2].item())
            sum_meas_pos_z += abs(snap.ee_pos[2].item())
            pos_selected_steps_z += 1

        # ---- Check termination conditions ----

        # Engage check (uses TRUE target position, not noisy goal)
        if not engaged:
            is_engaged = check_success(
                snap.ee_pos, ee_to_peg_base_offset, target_peg_base_pos,
                xy_centering, hole_height, engage_threshold,
            )
            if is_engaged:
                engaged = True

        # Success check (uses TRUE target position, not noisy goal)
        if not succeeded:
            is_success = check_success(
                snap.ee_pos, ee_to_peg_base_offset, target_peg_base_pos,
                xy_centering, hole_height, success_threshold,
            )
            if is_success:
                succeeded = True
                success_step = step
                if terminate_on_success:
                    break

        # Break check
        if not terminated:
            is_break = check_break(snap.force_torque, break_force_threshold)
            if is_break:
                terminated = True
                termination_step = step
                break

    # End torque control session immediately so the robot isn't waiting
    # for 1kHz communication while we compute metrics / print results
    robot.end_control()

    episode_length = step + 1

    # Per-episode force control summary (hybrid only, only axes with actual usage)
    if is_hybrid:
        axis_data = [
            ('X', force_selected_steps_x, sum_cmd_force_x, sum_force_error_x, sum_meas_force_x),
            ('Y', force_selected_steps_y, sum_cmd_force_y, sum_force_error_y, sum_meas_force_y),
            ('Z', force_selected_steps_z, sum_cmd_force_z, sum_force_error_z, sum_meas_force_z),
        ]
        used_axes = [(name, steps, cmd, err, meas) for name, steps, cmd, err, meas in axis_data if steps > 0]
        if used_axes:
            EvalKeyboardController.raw_print("    Force Control Summary (episode):")
            for name, steps, cmd, err, meas in used_axes:
                EvalKeyboardController.raw_print(
                    f"      {name}: {steps} steps | "
                    f"cmd={cmd/steps:.2f}N | err={err/steps:.2f}N | meas={meas/steps:.2f}N")
            if force_blocked_steps > 0:
                EvalKeyboardController.raw_print(
                    f"      Blocked: {force_blocked_steps} steps")

    # Per-episode position control summary (axes with actual usage)
    pos_axis_data = [
        ('X', pos_selected_steps_x, sum_cmd_pos_x, sum_pos_error_x, sum_meas_pos_x),
        ('Y', pos_selected_steps_y, sum_cmd_pos_y, sum_pos_error_y, sum_meas_pos_y),
        ('Z', pos_selected_steps_z, sum_cmd_pos_z, sum_pos_error_z, sum_meas_pos_z),
    ]
    pos_used_axes = [(name, steps, cmd, err, meas) for name, steps, cmd, err, meas in pos_axis_data if steps > 0]
    if pos_used_axes:
        EvalKeyboardController.raw_print("    Position Control Summary (episode):")
        for name, steps, cmd, err, meas in pos_used_axes:
            EvalKeyboardController.raw_print(
                f"      {name}: {steps} steps | "
                f"cmd={cmd/steps*1000:.2f}mm | err={err/steps*1000:.2f}mm | meas={meas/steps*1000:.2f}mm")

    # Normalize smoothness by episode length (matching sim: ssv = sum / ep_len)
    ssv = ssv_sum / episode_length if episode_length > 0 else 0.0
    ssjv = ssjv_sum / episode_length if episode_length > 0 else 0.0
    energy = energy_sum  # Energy is total, not averaged per step (matching sim)

    # Build trajectory data dict if logging
    if log_trajectory:
        traj_1khz = robot.get_last_trajectory()
        trajectory_data = {
            'trajectory_1khz': traj_1khz,
            'actions_15hz': traj_actions_15hz,
            'sel_matrices_15hz': traj_sel_matrices_15hz,
            'time_ms_15hz': traj_time_ms_15hz,
        }
    else:
        trajectory_data = {}

    return {
        'succeeded': succeeded,
        'engaged': engaged,
        'terminated': terminated,
        'length': episode_length,
        'ssv': ssv,
        'ssjv': ssjv,
        'max_force': max_force,
        'sum_force': sum_force,
        'sum_force_in_contact': sum_force_in_contact,
        'contact_steps': contact_steps,
        'energy': energy,
        'success_step': success_step if success_step >= 0 else episode_length,
        'termination_step': termination_step if termination_step >= 0 else episode_length,
        'obs_history': obs_history,
        'avg_force_error_x': sum_force_error_x / force_selected_steps_x if force_selected_steps_x > 0 else 0.0,
        'avg_force_error_y': sum_force_error_y / force_selected_steps_y if force_selected_steps_y > 0 else 0.0,
        'avg_force_error_z': sum_force_error_z / force_selected_steps_z if force_selected_steps_z > 0 else 0.0,
        'avg_cmd_force_x': sum_cmd_force_x / force_selected_steps_x if force_selected_steps_x > 0 else 0.0,
        'avg_cmd_force_y': sum_cmd_force_y / force_selected_steps_y if force_selected_steps_y > 0 else 0.0,
        'avg_cmd_force_z': sum_cmd_force_z / force_selected_steps_z if force_selected_steps_z > 0 else 0.0,
        'avg_meas_force_x': sum_meas_force_x / force_selected_steps_x if force_selected_steps_x > 0 else 0.0,
        'avg_meas_force_y': sum_meas_force_y / force_selected_steps_y if force_selected_steps_y > 0 else 0.0,
        'avg_meas_force_z': sum_meas_force_z / force_selected_steps_z if force_selected_steps_z > 0 else 0.0,
        'force_selected_steps_x': force_selected_steps_x,
        'force_selected_steps_y': force_selected_steps_y,
        'force_selected_steps_z': force_selected_steps_z,
        'force_blocked_steps': force_blocked_steps,
        'avg_pos_error_x': sum_pos_error_x / pos_selected_steps_x if pos_selected_steps_x > 0 else 0.0,
        'avg_pos_error_y': sum_pos_error_y / pos_selected_steps_y if pos_selected_steps_y > 0 else 0.0,
        'avg_pos_error_z': sum_pos_error_z / pos_selected_steps_z if pos_selected_steps_z > 0 else 0.0,
        'avg_cmd_pos_x': sum_cmd_pos_x / pos_selected_steps_x if pos_selected_steps_x > 0 else 0.0,
        'avg_cmd_pos_y': sum_cmd_pos_y / pos_selected_steps_y if pos_selected_steps_y > 0 else 0.0,
        'avg_cmd_pos_z': sum_cmd_pos_z / pos_selected_steps_z if pos_selected_steps_z > 0 else 0.0,
        'avg_meas_pos_x': sum_meas_pos_x / pos_selected_steps_x if pos_selected_steps_x > 0 else 0.0,
        'avg_meas_pos_y': sum_meas_pos_y / pos_selected_steps_y if pos_selected_steps_y > 0 else 0.0,
        'avg_meas_pos_z': sum_meas_pos_z / pos_selected_steps_z if pos_selected_steps_z > 0 else 0.0,
        'pos_selected_steps_x': pos_selected_steps_x,
        'pos_selected_steps_y': pos_selected_steps_y,
        'pos_selected_steps_z': pos_selected_steps_z,
        **trajectory_data,
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
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint_cache",
                        help="Local directory for caching downloaded checkpoints")
    parser.add_argument("--forge_eval", action="store_true", default=False,
                        help="Forge eval mode: spherical goal noise sampling across 4 ranges "
                             "(num_episodes becomes per-range, total = 4 × num_episodes)")
    parser.add_argument("--policy_idx", type=int, default=None,
                        help="Run only the policy at this index (0-based) from the tag's run list")
    parser.add_argument("--use_sim_best", action="store_true", default=False,
                        help="Auto-select the best policy by sim eval score (successes - breaks). "
                             "Without this flag, all policies are run consecutively.")
    parser.add_argument("--start_forge_idx", type=int, default=0,
                        help="Forge eval: skip to this noise range index (0-based). "
                             "Noise is still generated for all ranges to preserve seed consistency.")
    parser.add_argument("--log_trajectories", action="store_true",
                        help="Save per-episode 1kHz trajectory data as .npz files")
    parser.add_argument("--trajectory_dir", type=str, default="./trajectory_logs",
                        help="Directory for trajectory files")
    args = parser.parse_args()

    # Validate --start_forge_idx
    if args.start_forge_idx != 0 and not args.forge_eval:
        raise ValueError("--start_forge_idx requires --forge_eval")
    if args.start_forge_idx < 0 or args.start_forge_idx >= len(FORGE_NOISE_RANGES):
        raise ValueError(
            f"--start_forge_idx {args.start_forge_idx} out of range "
            f"[0, {len(FORGE_NOISE_RANGES) - 1}]"
        )

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

    # 2. Check for cached checkpoints or download from WandB
    cache_path = os.path.abspath(os.path.join(args.checkpoint_dir, sanitize_tag(args.tag)))
    cache_exists = os.path.exists(os.path.join(cache_path, 'runs.json'))

    if not cache_exists:
        print(f"\nNo local cache found. Downloading from WandB...")

        import wandb
        from eval.checkpoint_utils import (
            query_runs_by_tag,
            reconstruct_config_from_wandb,
            download_checkpoint_pair,
            get_best_checkpoints_for_runs,
        )

        # Query ALL runs for this tag (cache stores everything, --run_id filters at load)
        runs_wb = query_runs_by_tag(args.tag, args.entity, args.project, run_id=None)

        # Reconstruct training config from first run (all runs share same config)
        print("\nReconstructing training config...")
        configs_wb, temp_dir = reconstruct_config_from_wandb(runs_wb[0])

        # Get best checkpoint for each run
        print("\nFinding best checkpoints...")
        api = wandb.Api(timeout=60)
        best_checkpoints_wb, best_scores_wb = get_best_checkpoints_for_runs(
            api, runs_wb, args.tag, args.entity, args.project
        )

        # Download all checkpoints
        print("\nDownloading checkpoints...")
        checkpoint_paths = {}
        download_dirs = []
        for run_wb in runs_wb:
            policy_path, critic_path = download_checkpoint_pair(run_wb, best_checkpoints_wb[run_wb.id])
            checkpoint_paths[run_wb.id] = (policy_path, critic_path)
            download_dirs.append(os.path.dirname(os.path.dirname(policy_path)))

        # Save everything to local cache
        save_to_cache(cache_path, temp_dir, runs_wb, best_checkpoints_wb, checkpoint_paths, best_scores_wb)

        # Cleanup temp directories (data is now in cache)
        for d in download_dirs:
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print(f"\nUsing cached checkpoints from: {cache_path}")

    # Load from cache (single code path for both fresh download and cached)
    configs, runs, best_checkpoints, best_scores = load_from_cache(cache_path, args.run_id)

    # Print sim eval performance table
    has_scores = len(best_scores) == len(runs)
    if has_scores:
        print(f"\n{'=' * 90}")
        print("SIM EVAL PERFORMANCE (score = successes - breaks)")
        print(f"{'=' * 90}")
        best_policy_idx = 0
        best_policy_score = -float('inf')
        for i, r in enumerate(runs):
            sc = best_scores[r.id]
            marker = ""
            if sc['score'] > best_policy_score:
                best_policy_score = sc['score']
                best_policy_idx = i
            print(f"  [{i}] {r.name:<40s}  step={best_checkpoints[r.id]:<10d}"
                  f"  succ={sc['successes']:<4d}  brk={sc['breaks']:<4d}  score={sc['score']}")
        # Mark best
        print(f"{'=' * 90}")
        print(f"  Best: [{best_policy_idx}] {runs[best_policy_idx].name} (score={best_policy_score})")
        print(f"{'=' * 90}")
    else:
        print("\n  [WARNING] No sim eval scores in cache. Delete cache and re-download to get scores.")

    # Filter to single policy by index if requested, or auto-select best
    if args.policy_idx is not None:
        if args.policy_idx < 0 or args.policy_idx >= len(runs):
            print(f"\nAvailable policies for tag '{args.tag}':")
            for i, r in enumerate(runs):
                print(f"  [{i}] {r.name} (id={r.id}, best step={best_checkpoints[r.id]})")
            raise ValueError(
                f"--policy_idx {args.policy_idx} out of range [0, {len(runs)-1}]"
            )
        selected = runs[args.policy_idx]
        print(f"\n  --policy_idx {args.policy_idx}: selected '{selected.name}' (id={selected.id})")
        runs = [selected]
        best_checkpoints = {selected.id: best_checkpoints[selected.id]}
    elif args.use_sim_best:
        if not has_scores:
            raise RuntimeError(
                "--use_sim_best requires sim eval scores in cache. "
                "Delete cache and re-download, or use --policy_idx instead."
            )
        selected = runs[best_policy_idx]
        print(f"\n  --use_sim_best: selected [{best_policy_idx}] '{selected.name}' (id={selected.id})")
        runs = [selected]
        best_checkpoints = {selected.id: best_checkpoints[selected.id]}
    else:
        print(f"\n  Running all {len(runs)} policies consecutively")

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
    exclude_torques = getattr(ft_cfg, 'exclude_torques', False)
    ee_pose_noise_enabled = getattr(configs['wrappers'].ee_pose_noise, 'enabled', False)

    # Per-episode noise config: real robot override or WandB training config
    noise_cfg = real_config.get('noise', {})
    use_rr_noise = noise_cfg.get('use_rr_noise', False)

    if use_rr_noise:
        # Load ALL noise values from real robot config (fail-fast if missing)
        goal_pos_noise_scale = torch.tensor(noise_cfg['goal_pos_noise'], device=args.device, dtype=torch.float32)
        use_fixed_asset_yaw_noise = noise_cfg['use_fixed_asset_yaw_noise']
        goal_yaw_noise_scale = noise_cfg['goal_yaw_noise'] if use_fixed_asset_yaw_noise else 0.0
        hand_init_pos = torch.tensor(noise_cfg['hand_init_pos'], device=args.device, dtype=torch.float32)
        hand_init_pos_noise = torch.tensor(noise_cfg['hand_init_pos_noise'], device=args.device, dtype=torch.float32)
        hand_init_orn = list(noise_cfg['hand_init_orn'])
        hand_init_orn_noise = list(noise_cfg['hand_init_orn_noise'])
    else:
        # Load noise from WandB training config (matching sim exactly)
        obs_rand = configs['environment'].obs_rand
        goal_pos_noise_scale = torch.tensor(obs_rand.fixed_asset_pos, device=args.device, dtype=torch.float32)
        use_fixed_asset_yaw_noise = hasattr(obs_rand, 'use_fixed_asset_yaw_noise') and obs_rand.use_fixed_asset_yaw_noise
        goal_yaw_noise_scale = obs_rand.fixed_asset_yaw if use_fixed_asset_yaw_noise else 0.0

        # Get task config for start pose params (field is 'task' on ExtendedFactoryPegEnvCfg)
        cfg_task = getattr(configs['environment'], 'task', None) or configs['environment']
        hand_init_pos = torch.tensor(getattr(cfg_task, 'hand_init_pos', [0.0, 0.0, 0.047]),
                                     device=args.device, dtype=torch.float32)
        hand_init_pos_noise = torch.tensor(getattr(cfg_task, 'hand_init_pos_noise', [0.02, 0.02, 0.01]),
                                           device=args.device, dtype=torch.float32)
        hand_init_orn = list(getattr(cfg_task, 'hand_init_orn', [3.1416, 0.0, 0.0]))
        hand_init_orn_noise = list(getattr(cfg_task, 'hand_init_orn_noise', [0.0, 0.0, 0.785]))

    # 6. Initialize observation builder
    fixed_asset_yaw = real_config['task']['fixed_asset_yaw']
    obs_builder = ObservationBuilder(
        obs_order=obs_order,
        action_dim=action_dim,
        use_tanh_ft_scaling=use_tanh,
        tanh_ft_scale=tanh_scale,
        contact_force_threshold=contact_threshold,
        fixed_asset_yaw=fixed_asset_yaw,
        ee_pose_noise_enabled=ee_pose_noise_enabled,
        exclude_torques=exclude_torques,
        device=args.device,
    )

    # 7. Initialize robot interface
    print("\nInitializing robot interface...")
    robot = FrankaInterface(real_config, device=args.device)

    # 7b. Close gripper with configured force (one-time, before any episodes)
    print("\nClosing gripper...")
    robot.close_gripper()

    # 8. Initialize controller
    print("\nInitializing controller...")
    controller = RealRobotController(configs, real_config, device=args.device)

    # 8c. Policy sampling config
    std_scale = real_config.get('policy', {}).get('std_scale', 0.0)
    if std_scale > 0.0:
        print(f"[Policy] Stochastic sampling ENABLED (std_scale={std_scale})")
    else:
        print("[Policy] Deterministic (mean only)")

    # 8b. Move robot to default null-space joint angles (disabled for now)
    # print(f"\nDefault null-space joint positions: {controller.default_dof_pos.tolist()}")
    # input("    [WAIT] Press Enter to MOVE TO DEFAULT JOINT POSITIONS...")
    # robot.move_to_joint_positions(controller.default_dof_pos, duration_sec=3.0)

    # 9. Compute calibration pose (goal XY, goal Z + 0.05m) for 'c' command
    task_cfg = real_config['task']
    fixed_asset_position = torch.tensor(task_cfg['fixed_asset_position'], device=args.device, dtype=torch.float32)
    obs_frame_z_offset = task_cfg['hole_height'] + task_cfg['fixed_asset_base_height']
    cal_goal = fixed_asset_position.clone()
    cal_goal[2] += obs_frame_z_offset + 0.05  # 5cm above goal Z
    cal_pose = make_ee_target_pose(cal_goal.cpu().numpy(), np.array(hand_init_orn))
    retract_height = real_config['robot']['retract_height_m']

    # 10. Move to calibration pose and wait for user to confirm alignment
    print("\nMoving to calibration pose (goal XY, 5cm above goal Z)...")
    robot.retract_up(retract_height)
    robot.reset_to_start_pose(cal_pose)
    snap = robot.get_state_snapshot()
    print(f"  Calibration pose: xyz=[{snap.ee_pos[0].item():.4f}, "
          f"{snap.ee_pos[1].item():.4f}, {snap.ee_pos[2].item():.4f}]")
    input("  Press Enter to begin experiments...")

    # 11. Start keyboard controller and evaluate
    keyboard = EvalKeyboardController()
    keyboard.start()

    rp = EvalKeyboardController.raw_print

    all_run_summaries = []

    # Pre-generate ALL per-episode noise so every policy sees identical conditions
    if args.forge_eval:
        total_episodes = args.num_episodes * len(FORGE_NOISE_RANGES)
        forge_range_indices = []
        goal_pos_noises = []
        for range_idx, (min_val, max_val, _range_name) in enumerate(FORGE_NOISE_RANGES):
            for _ in range(args.num_episodes):
                direction = torch.randn(3, device=args.device)
                direction = direction / direction.norm()
                radius = torch.rand(1, device=args.device).item() * (max_val - min_val) + min_val
                goal_pos_noises.append(direction * radius)
                forge_range_indices.append(range_idx)
    else:
        total_episodes = args.num_episodes
        forge_range_indices = None
        goal_pos_noises = [
            torch.randn(3, device=args.device) * goal_pos_noise_scale
            for _ in range(total_episodes)
        ]

    episode_noises = []
    for ep_idx in range(total_episodes):
        episode_noises.append({
            'goal_pos_noise': goal_pos_noises[ep_idx],
            'goal_yaw_noise': (torch.randn(1, device=args.device) * goal_yaw_noise_scale).item()
                              if goal_yaw_noise_scale > 0 else 0.0,
            'start_pos_noise': (2 * torch.rand(3, device=args.device) - 1) * hand_init_pos_noise,
        })

    try:
        rp(f"{'=' * 80}")
        if args.forge_eval:
            rp(f"EVALUATING {len(runs)} RUN(S), {args.num_episodes} EPISODES x {len(FORGE_NOISE_RANGES)} RANGES = {total_episodes} TOTAL")
            rp(f"  Forge eval noise ranges: {', '.join(r[2] for r in FORGE_NOISE_RANGES)}")
            if args.start_forge_idx > 0:
                rp(f"  Starting from range index {args.start_forge_idx} ({FORGE_NOISE_RANGES[args.start_forge_idx][2]})")
        else:
            rp(f"EVALUATING {len(runs)} RUN(S), {args.num_episodes} EPISODES EACH")
        rp(f"{'=' * 80}")
        rp("  Keyboard controls:")
        rp("    's' = skip (end episode as BREAK)")
        rp("    'p' = pause (finish episode, then pause)")
        rp("    'c' = calibrate (while paused: move to goal XY, 5cm above goal Z)")
        rp("    Enter = resume (while paused)")
        rp("    ESC = quit (end episode and shut down)")
        rp(f"{'=' * 80}")

        for run_idx, run in enumerate(runs):
            run_id = run.id
            best_step = best_checkpoints[run_id]

            rp(f"--- Run {run_idx+1}/{len(runs)}: {run.name} (best step: {best_step}) ---")

            # Load checkpoint from cache
            policy_path = os.path.join(cache_path, run_id, 'policy.pt')

            # Load policy
            policy_net, normalizer, model_info = load_single_agent_policy(
                policy_path, configs, obs_dim=obs_builder.obs_dim, device=args.device,
            )

            # Validate observation dimensions
            obs_builder.validate_against_checkpoint(model_info['obs_dim'])

            # ================================================================
            # Run episodes — forge eval iterates per-range with per-range
            # WandB logging; non-forge uses a flat loop with one log at end.
            # ================================================================

            if args.forge_eval:
                # --- FORGE EVAL: per-range loop ---
                episode_results = []  # all results across ranges (for final summary)
                forge_range_results = {}
                quit_requested = False
                all_forge_wandb_metrics = {}  # accumulate per-range metrics with range-specific keys

                # Init single WandB run for all forge ranges (matching wandb_eval.py pattern)
                eval_run = None
                if not args.no_wandb:
                    import wandb
                    keyboard.stop()
                    eval_prefix = "Eval_RealRobotForge"
                    eval_tags = [
                        eval_prefix.lower(), f"source_run:{run_id}",
                    ] + list(run.tags)
                    eval_group = f"{eval_prefix}_{run.group}_{args.tag}" if run.group else None
                    eval_run = wandb.init(
                        project=run.project,
                        entity=args.entity,
                        name=f"{eval_prefix}_{run.name}",
                        group=eval_group,
                        tags=eval_tags,
                        reinit="create_new",
                        config={
                            "source_run_id": run_id,
                            "source_run_name": run.name,
                            "source_run_group": run.group,
                            "source_project": run.project,
                            "eval_mode": "real_robot_forge",
                            "eval_seed": args.eval_seed,
                            "best_step": best_step,
                            "num_episodes": args.num_episodes,
                            "forge_ranges": [r[2] for r in FORGE_NOISE_RANGES],
                            "real_robot_config": real_config,
                        },
                    )
                    keyboard.start()

                for range_idx, (_min_val, _max_val, range_name) in enumerate(FORGE_NOISE_RANGES):
                    if range_idx < args.start_forge_idx:
                        rp(f"  [SKIP] Range {range_idx} ({range_name}) — skipped via --start_forge_idx")
                        continue

                    rp(f"  --- Range {range_idx}/{len(FORGE_NOISE_RANGES)-1}: {range_name} ---")

                    range_results = []
                    running_success = 0
                    running_breaks = 0
                    running_breaks_engaged = 0
                    running_timeouts = 0
                    running_timeouts_engaged = 0
                    abort = False

                    for ep_in_range in range(args.num_episodes):
                        ep_idx = range_idx * args.num_episodes + ep_in_range
                        result = run_episode(
                            robot, policy_net, normalizer, model_info,
                            obs_builder, controller, real_config,
                            episode_noises[ep_idx],
                            hand_init_pos,
                            hand_init_orn, hand_init_orn_noise,
                            keyboard,
                            args.device,
                            log_trajectory=args.log_trajectories,
                            std_scale=std_scale,
                        )

                        if result is None:
                            sys.stdout.write("\r\n")
                            rp(f"  [ABORT] Motion failed after retries — skipping remaining episodes in range")
                            abort = True
                            break

                        result['forge_range_idx'] = range_idx

                        if result['succeeded'] and not result['terminated']:
                            running_success += 1
                        elif result['terminated']:
                            running_breaks += 1
                            if result['engaged']:
                                running_breaks_engaged += 1
                        else:
                            running_timeouts += 1
                            if result['engaged']:
                                running_timeouts_engaged += 1

                        status = (f"  [{ep_in_range+1}/{args.num_episodes}] [{range_name}] "
                                  f"S:{running_success} B:{running_breaks}({running_breaks_engaged}) "
                                  f"T:{running_timeouts}({running_timeouts_engaged})")
                        sys.stdout.write(f"\r\x1b[K{status}")
                        sys.stdout.flush()

                        range_results.append(result)

                        # Save trajectory data
                        if args.log_trajectories and result.get('trajectory_1khz') is not None:
                            policy_traj_dir = os.path.join(args.trajectory_dir, run.name)
                            os.makedirs(policy_traj_dir, exist_ok=True)
                            np.savez_compressed(
                                os.path.join(policy_traj_dir, f"traj_{ep_idx:03d}.npz"),
                                **result['trajectory_1khz'],
                                action_15hz=np.stack([a.numpy() for a in result['actions_15hz']]),
                                sel_matrix_15hz=np.stack([s.numpy() for s in result['sel_matrices_15hz']]),
                                time_ms_15hz=np.array(result['time_ms_15hz']),
                            )

                        if keyboard.should_quit:
                            sys.stdout.write("\r\n")
                            rp("  [QUIT] Shutting down...")
                            quit_requested = True
                            break

                        if keyboard.should_pause:
                            sys.stdout.write("\r\n")
                            keyboard.set_paused(True)
                            rp("  [PAUSED] 'c' = calibrate, Enter = resume, ESC = quit")
                            while True:
                                if keyboard.should_quit:
                                    rp("  [QUIT] Shutting down...")
                                    keyboard.set_paused(False)
                                    quit_requested = True
                                    break
                                if keyboard.should_calibrate:
                                    rp("  [CALIBRATING] Moving to goal XY, 5cm above goal Z...")
                                    robot.retract_up(retract_height)
                                    robot.reset_to_start_pose(cal_pose)
                                    snap = robot.get_state_snapshot()
                                    rp(f"  [CALIBRATED] xyz=[{snap.ee_pos[0].item():.4f}, "
                                       f"{snap.ee_pos[1].item():.4f}, {snap.ee_pos[2].item():.4f}]")
                                    rp("  [PAUSED] 'c' = calibrate, Enter = resume, ESC = quit")
                                if keyboard.should_resume:
                                    keyboard.set_paused(False)
                                    rp("  [RESUMED]")
                                    break
                                time.sleep(0.05)
                            if quit_requested:
                                break
                    else:
                        # All episodes in this range completed
                        sys.stdout.write("\r\n")

                    # --- Log completed range results ---
                    if range_results:
                        episode_results.extend(range_results)

                        r_n = len(range_results)
                        r_s = sum(1 for ep in range_results if ep['succeeded'] and not ep['terminated'])
                        r_b = sum(1 for ep in range_results if ep['terminated'])
                        r_be = sum(1 for ep in range_results if ep['terminated'] and ep['engaged'])
                        r_t = r_n - r_s - r_b
                        r_te = sum(1 for ep in range_results if not ep['succeeded'] and not ep['terminated'] and ep['engaged'])
                        forge_range_results[range_name] = {
                            'n': r_n, 'success': r_s, 'breaks': r_b, 'break_eng': r_be,
                            'timeouts': r_t, 'timeout_eng': r_te,
                        }
                        rp(f"    [{range_name:<10}] N={r_n:<3} S:{r_s:<3} B:{r_b}(E:{r_be}) T:{r_t}(E:{r_te})")

                        # Compute per-range metrics and accumulate with range-specific keys
                        range_metrics = compute_real_robot_metrics(range_results)
                        for k, v in range_metrics.items():
                            all_forge_wandb_metrics[f"Noise_Eval({range_name})_Core/{k}"] = v

                    if quit_requested or abort:
                        break

                # Log all accumulated forge metrics to WandB in a single call
                if eval_run is not None and all_forge_wandb_metrics:
                    keyboard.stop()
                    all_forge_wandb_metrics["total_steps"] = best_step
                    eval_run.log(all_forge_wandb_metrics, step=best_step)
                    eval_run.finish()
                    keyboard.start()
                    rp(f"    Logged forge eval to WandB: {eval_run.url}")
                elif eval_run is not None:
                    # No metrics collected but run was opened — close it cleanly
                    keyboard.stop()
                    eval_run.finish()
                    keyboard.start()

                # --- End-of-policy summary for forge eval ---
                if not episode_results:
                    rp(f"  Results for {run.name}: NO EPISODES COMPLETED")
                    all_run_summaries.append({
                        'name': run.name, 'total': 0, 'success': 0,
                        'breaks': 0, 'break_eng': 0, 'timeouts': 0, 'timeout_eng': 0,
                        'avg_len': 0.0, 'ssv': 0.0, 'avg_force': 0.0, 'max_force': 0.0,
                        'energy': 0.0, 'blocked_avg': 0.0, 'blocked_std': 0.0,
                    })
                else:
                    all_obs = []
                    for ep in episode_results:
                        all_obs.extend(ep['obs_history'])
                    print_obs_distribution_comparison(all_obs, obs_builder, normalizer)

                    metrics = compute_real_robot_metrics(episode_results)
                    break_engaged = sum(1 for ep in episode_results if ep['terminated'] and ep['engaged'])
                    timeout_engaged = sum(1 for ep in episode_results
                                          if not ep['succeeded'] and not ep['terminated'] and ep['engaged'])

                    rp(f"  Overall results for {run.name}:")
                    for rn, rr in forge_range_results.items():
                        rp(f"    [{rn:<10}] N={rr['n']:<3} S:{rr['success']:<3} B:{rr['breaks']}(E:{rr['break_eng']}) T:{rr['timeouts']}(E:{rr['timeout_eng']})")

                    blocked_counts = [ep['force_blocked_steps'] for ep in episode_results]
                    blocked_avg = sum(blocked_counts) / len(blocked_counts)
                    blocked_std = (sum((x - blocked_avg) ** 2 for x in blocked_counts) / len(blocked_counts)) ** 0.5

                    rp(f"    Successes: {metrics['num_successful_completions']}/{metrics['total_episodes']}")
                    rp(f"    Breaks:    {metrics['num_breaks']}/{metrics['total_episodes']} ({break_engaged} engaged)")
                    rp(f"    Timeouts:  {metrics['num_failed_timeouts']}/{metrics['total_episodes']} ({timeout_engaged} engaged)")
                    rp(f"    Avg Length: {metrics['episode_length']:.1f}")
                    rp(f"    SSV:       {metrics['ssv']:.4f}")
                    rp(f"    Avg Force: {metrics['avg_force']:.2f}N")
                    rp(f"    Max Force: {metrics['max_force']:.2f}N")
                    rp(f"    Energy:    {metrics['energy']:.2f}")
                    rp(f"    Blocked:   {blocked_avg:.1f} +/- {blocked_std:.1f} steps/ep")

                    summary_entry = {
                        'name': run.name,
                        'total': metrics['total_episodes'],
                        'success': metrics['num_successful_completions'],
                        'breaks': metrics['num_breaks'],
                        'break_eng': break_engaged,
                        'timeouts': metrics['num_failed_timeouts'],
                        'timeout_eng': timeout_engaged,
                        'avg_len': metrics['episode_length'],
                        'ssv': metrics['ssv'],
                        'avg_force': metrics['avg_force'],
                        'max_force': metrics['max_force'],
                        'energy': metrics['energy'],
                        'blocked_avg': blocked_avg,
                        'blocked_std': blocked_std,
                        'forge_range_results': forge_range_results,
                    }
                    all_run_summaries.append(summary_entry)

                    if hybrid_enabled:
                        total_fe_x = 0.0; total_cf_x = 0.0; total_mf_x = 0.0; total_steps_x = 0
                        total_fe_y = 0.0; total_cf_y = 0.0; total_mf_y = 0.0; total_steps_y = 0
                        total_fe_z = 0.0; total_cf_z = 0.0; total_mf_z = 0.0; total_steps_z = 0
                        for ep in episode_results:
                            total_fe_x += ep['avg_force_error_x'] * ep['force_selected_steps_x']
                            total_cf_x += ep['avg_cmd_force_x'] * ep['force_selected_steps_x']
                            total_mf_x += ep['avg_meas_force_x'] * ep['force_selected_steps_x']
                            total_steps_x += ep['force_selected_steps_x']
                            total_fe_y += ep['avg_force_error_y'] * ep['force_selected_steps_y']
                            total_cf_y += ep['avg_cmd_force_y'] * ep['force_selected_steps_y']
                            total_mf_y += ep['avg_meas_force_y'] * ep['force_selected_steps_y']
                            total_steps_y += ep['force_selected_steps_y']
                            total_fe_z += ep['avg_force_error_z'] * ep['force_selected_steps_z']
                            total_cf_z += ep['avg_cmd_force_z'] * ep['force_selected_steps_z']
                            total_mf_z += ep['avg_meas_force_z'] * ep['force_selected_steps_z']
                            total_steps_z += ep['force_selected_steps_z']
                        avg_fe_x = total_fe_x / total_steps_x if total_steps_x > 0 else 0.0
                        avg_cf_x = total_cf_x / total_steps_x if total_steps_x > 0 else 0.0
                        avg_mf_x = total_mf_x / total_steps_x if total_steps_x > 0 else 0.0
                        avg_fe_y = total_fe_y / total_steps_y if total_steps_y > 0 else 0.0
                        avg_cf_y = total_cf_y / total_steps_y if total_steps_y > 0 else 0.0
                        avg_mf_y = total_mf_y / total_steps_y if total_steps_y > 0 else 0.0
                        avg_fe_z = total_fe_z / total_steps_z if total_steps_z > 0 else 0.0
                        avg_cf_z = total_cf_z / total_steps_z if total_steps_z > 0 else 0.0
                        avg_mf_z = total_mf_z / total_steps_z if total_steps_z > 0 else 0.0
                        rp(f"    Force Err X: {avg_fe_x:.3f}N ({total_steps_x} steps) cmd={avg_cf_x:.3f}N meas={avg_mf_x:.3f}N")
                        rp(f"    Force Err Y: {avg_fe_y:.3f}N ({total_steps_y} steps) cmd={avg_cf_y:.3f}N meas={avg_mf_y:.3f}N")
                        rp(f"    Force Err Z: {avg_fe_z:.3f}N ({total_steps_z} steps) cmd={avg_cf_z:.3f}N meas={avg_mf_z:.3f}N")

                if quit_requested:
                    break

            else:
                # --- NON-FORGE: flat episode loop (unchanged) ---
                episode_results = []
                running_success = 0
                running_breaks = 0
                running_breaks_engaged = 0
                running_timeouts = 0
                running_timeouts_engaged = 0
                for ep_idx in range(total_episodes):
                    result = run_episode(
                        robot, policy_net, normalizer, model_info,
                        obs_builder, controller, real_config,
                        episode_noises[ep_idx],
                        hand_init_pos,
                        hand_init_orn, hand_init_orn_noise,
                        keyboard,
                        args.device,
                        log_trajectory=args.log_trajectories,
                        std_scale=std_scale,
                    )

                    if result is None:
                        sys.stdout.write("\r\n")
                        rp(f"  [ABORT] Motion failed after retries — skipping remaining episodes")
                        break

                    if result['succeeded'] and not result['terminated']:
                        running_success += 1
                    elif result['terminated']:
                        running_breaks += 1
                        if result['engaged']:
                            running_breaks_engaged += 1
                    else:
                        running_timeouts += 1
                        if result['engaged']:
                            running_timeouts_engaged += 1

                    status = f"  [{ep_idx+1}/{total_episodes}] S:{running_success} B:{running_breaks}({running_breaks_engaged}) T:{running_timeouts}({running_timeouts_engaged})"
                    sys.stdout.write(f"\r\x1b[K{status}")
                    sys.stdout.flush()

                    episode_results.append(result)

                    if args.log_trajectories and result.get('trajectory_1khz') is not None:
                        policy_traj_dir = os.path.join(args.trajectory_dir, run.name)
                        os.makedirs(policy_traj_dir, exist_ok=True)
                        np.savez_compressed(
                            os.path.join(policy_traj_dir, f"traj_{ep_idx:03d}.npz"),
                            **result['trajectory_1khz'],
                            action_15hz=np.stack([a.numpy() for a in result['actions_15hz']]),
                            sel_matrix_15hz=np.stack([s.numpy() for s in result['sel_matrices_15hz']]),
                            time_ms_15hz=np.array(result['time_ms_15hz']),
                        )

                    if keyboard.should_quit:
                        sys.stdout.write("\r\n")
                        rp("  [QUIT] Shutting down...")
                        break

                    if keyboard.should_pause:
                        sys.stdout.write("\r\n")
                        keyboard.set_paused(True)
                        rp("  [PAUSED] 'c' = calibrate, Enter = resume, ESC = quit")
                        while True:
                            if keyboard.should_quit:
                                rp("  [QUIT] Shutting down...")
                                keyboard.set_paused(False)
                                break
                            if keyboard.should_calibrate:
                                rp("  [CALIBRATING] Moving to goal XY, 5cm above goal Z...")
                                robot.retract_up(retract_height)
                                robot.reset_to_start_pose(cal_pose)
                                snap = robot.get_state_snapshot()
                                rp(f"  [CALIBRATED] xyz=[{snap.ee_pos[0].item():.4f}, "
                                   f"{snap.ee_pos[1].item():.4f}, {snap.ee_pos[2].item():.4f}]")
                                rp("  [PAUSED] 'c' = calibrate, Enter = resume, ESC = quit")
                            if keyboard.should_resume:
                                keyboard.set_paused(False)
                                rp("  [RESUMED]")
                                break
                            time.sleep(0.05)
                        if keyboard.should_quit:
                            break
                else:
                    sys.stdout.write("\r\n")

                if not episode_results:
                    rp(f"  Results for {run.name}: NO EPISODES COMPLETED (motion failures)")
                    all_run_summaries.append({
                        'name': run.name, 'total': 0, 'success': 0,
                        'breaks': 0, 'break_eng': 0, 'timeouts': 0, 'timeout_eng': 0,
                        'avg_len': 0.0, 'ssv': 0.0, 'avg_force': 0.0, 'max_force': 0.0,
                        'energy': 0.0, 'blocked_avg': 0.0, 'blocked_std': 0.0,
                    })
                    if keyboard.should_quit:
                        break
                    continue

                all_obs = []
                for ep in episode_results:
                    all_obs.extend(ep['obs_history'])
                print_obs_distribution_comparison(all_obs, obs_builder, normalizer)

                metrics = compute_real_robot_metrics(episode_results)

                break_engaged = 0
                timeout_engaged = 0
                for ep in episode_results:
                    if ep['succeeded'] and not ep['terminated']:
                        pass
                    elif ep['terminated']:
                        if ep['engaged']:
                            break_engaged += 1
                    else:
                        if ep['engaged']:
                            timeout_engaged += 1

                rp(f"  Results for {run.name}:")

                blocked_counts = [ep['force_blocked_steps'] for ep in episode_results]
                blocked_avg = sum(blocked_counts) / len(blocked_counts)
                blocked_std = (sum((x - blocked_avg) ** 2 for x in blocked_counts) / len(blocked_counts)) ** 0.5

                rp(f"    Successes: {metrics['num_successful_completions']}/{metrics['total_episodes']}")
                rp(f"    Breaks:    {metrics['num_breaks']}/{metrics['total_episodes']} ({break_engaged} engaged)")
                rp(f"    Timeouts:  {metrics['num_failed_timeouts']}/{metrics['total_episodes']} ({timeout_engaged} engaged)")
                rp(f"    Avg Length: {metrics['episode_length']:.1f}")
                rp(f"    SSV:       {metrics['ssv']:.4f}")
                rp(f"    Avg Force: {metrics['avg_force']:.2f}N")
                rp(f"    Max Force: {metrics['max_force']:.2f}N")
                rp(f"    Energy:    {metrics['energy']:.2f}")
                rp(f"    Blocked:   {blocked_avg:.1f} +/- {blocked_std:.1f} steps/ep")

                summary_entry = {
                    'name': run.name,
                    'total': metrics['total_episodes'],
                    'success': metrics['num_successful_completions'],
                    'breaks': metrics['num_breaks'],
                    'break_eng': break_engaged,
                    'timeouts': metrics['num_failed_timeouts'],
                    'timeout_eng': timeout_engaged,
                    'avg_len': metrics['episode_length'],
                    'ssv': metrics['ssv'],
                    'avg_force': metrics['avg_force'],
                    'max_force': metrics['max_force'],
                    'energy': metrics['energy'],
                    'blocked_avg': blocked_avg,
                    'blocked_std': blocked_std,
                }
                all_run_summaries.append(summary_entry)

                if hybrid_enabled:
                    total_fe_x = 0.0; total_cf_x = 0.0; total_mf_x = 0.0; total_steps_x = 0
                    total_fe_y = 0.0; total_cf_y = 0.0; total_mf_y = 0.0; total_steps_y = 0
                    total_fe_z = 0.0; total_cf_z = 0.0; total_mf_z = 0.0; total_steps_z = 0
                    for ep in episode_results:
                        total_fe_x += ep['avg_force_error_x'] * ep['force_selected_steps_x']
                        total_cf_x += ep['avg_cmd_force_x'] * ep['force_selected_steps_x']
                        total_mf_x += ep['avg_meas_force_x'] * ep['force_selected_steps_x']
                        total_steps_x += ep['force_selected_steps_x']
                        total_fe_y += ep['avg_force_error_y'] * ep['force_selected_steps_y']
                        total_cf_y += ep['avg_cmd_force_y'] * ep['force_selected_steps_y']
                        total_mf_y += ep['avg_meas_force_y'] * ep['force_selected_steps_y']
                        total_steps_y += ep['force_selected_steps_y']
                        total_fe_z += ep['avg_force_error_z'] * ep['force_selected_steps_z']
                        total_cf_z += ep['avg_cmd_force_z'] * ep['force_selected_steps_z']
                        total_mf_z += ep['avg_meas_force_z'] * ep['force_selected_steps_z']
                        total_steps_z += ep['force_selected_steps_z']
                    avg_fe_x = total_fe_x / total_steps_x if total_steps_x > 0 else 0.0
                    avg_cf_x = total_cf_x / total_steps_x if total_steps_x > 0 else 0.0
                    avg_mf_x = total_mf_x / total_steps_x if total_steps_x > 0 else 0.0
                    avg_fe_y = total_fe_y / total_steps_y if total_steps_y > 0 else 0.0
                    avg_cf_y = total_cf_y / total_steps_y if total_steps_y > 0 else 0.0
                    avg_mf_y = total_mf_y / total_steps_y if total_steps_y > 0 else 0.0
                    avg_fe_z = total_fe_z / total_steps_z if total_steps_z > 0 else 0.0
                    avg_cf_z = total_cf_z / total_steps_z if total_steps_z > 0 else 0.0
                    avg_mf_z = total_mf_z / total_steps_z if total_steps_z > 0 else 0.0
                    rp(f"    Force Err X: {avg_fe_x:.3f}N ({total_steps_x} steps) cmd={avg_cf_x:.3f}N meas={avg_mf_x:.3f}N")
                    rp(f"    Force Err Y: {avg_fe_y:.3f}N ({total_steps_y} steps) cmd={avg_cf_y:.3f}N meas={avg_mf_y:.3f}N")
                    rp(f"    Force Err Z: {avg_fe_z:.3f}N ({total_steps_z} steps) cmd={avg_cf_z:.3f}N meas={avg_mf_z:.3f}N")

                # Log to WandB
                if not args.no_wandb:
                    import wandb
                    keyboard.stop()
                    eval_tags = [
                        "eval_realrobot", f"source_run:{run_id}",
                    ] + list(run.tags)
                    eval_group = f"Eval_RealRobot_{run.group}_{args.tag}" if run.group else None
                    eval_run = wandb.init(
                        project=run.project,
                        entity=args.entity,
                        name=f"Eval_RealRobot_{run.name}",
                        group=eval_group,
                        tags=eval_tags,
                        reinit="create_new",
                        config={
                            "source_run_id": run_id,
                            "source_run_name": run.name,
                            "source_run_group": run.group,
                            "source_project": run.project,
                            "eval_mode": "real_robot",
                            "eval_seed": args.eval_seed,
                            "best_step": best_step,
                            "num_episodes": args.num_episodes,
                            "real_robot_config": real_config,
                        },
                    )
                    prefixed_metrics = {f"Eval_Core/{k}": v for k, v in metrics.items()}
                    prefixed_metrics["total_steps"] = best_step
                    eval_run.log(prefixed_metrics)
                    eval_run.finish()
                    keyboard.start()
                    rp(f"    Logged to WandB: {eval_run.url}")
                else:
                    rp("    (WandB logging disabled)")

                if keyboard.should_quit:
                    break

        # 11. Final summary table across all policies
        if all_run_summaries:
            rp("")
            if args.forge_eval:
                # Wide table with per-range success rate columns
                range_headers = "".join(f" {r[2]:>10}" for r in FORGE_NOISE_RANGES)
                table_width = 170
                rp(f"{'=' * table_width}")
                rp("FINAL SUMMARY — ALL POLICIES (FORGE EVAL)")
                rp(f"{'=' * table_width}")
                rp(f"{'Policy':<30} {'N':>4}{range_headers} {'Total':>6} {'Brk':>5} {'BrkE':>5} {'TO':>5} {'TOE':>5} {'AvgLen':>7} {'SSV':>8} {'AvgF':>7} {'MaxF':>7} {'Energy':>8} {'BlkAvg':>7} {'BlkStd':>7}")
                rp(f"{'-' * table_width}")
                for s in all_run_summaries:
                    range_cols = ""
                    frr = s.get('forge_range_results', {})
                    for _min_val, _max_val, range_name in FORGE_NOISE_RANGES:
                        rr = frr.get(range_name, {})
                        r_s = rr.get('success', 0)
                        r_n = rr.get('n', 0)
                        cell = f"{r_s}/{r_n}" if r_n > 0 else "–"
                        range_cols += f" {cell:>10}"
                    success_pct = f"{100*s['success']/s['total']:.0f}%" if s['total'] > 0 else "N/A"
                    rp(f"{s['name']:<30} {s['total']:>4}{range_cols} {success_pct:>6} {s['breaks']:>5} {s['break_eng']:>5} {s['timeouts']:>5} {s['timeout_eng']:>5} {s['avg_len']:>7.1f} {s['ssv']:>8.4f} {s['avg_force']:>7.2f} {s['max_force']:>7.2f} {s['energy']:>8.2f} {s['blocked_avg']:>7.1f} {s['blocked_std']:>7.1f}")
                rp(f"{'=' * table_width}")
            else:
                table_width = 143
                rp(f"{'=' * table_width}")
                rp("FINAL SUMMARY — ALL POLICIES")
                rp(f"{'=' * table_width}")
                rp(f"{'Policy':<30} {'N':>4} {'Succ':>5} {'Brk':>5} {'BrkE':>5} {'TO':>5} {'TOE':>5} {'AvgLen':>7} {'SSV':>8} {'AvgF':>7} {'MaxF':>7} {'Energy':>8} {'BlkAvg':>7} {'BlkStd':>7}")
                rp(f"{'-' * table_width}")
                for s in all_run_summaries:
                    rp(f"{s['name']:<30} {s['total']:>4} {s['success']:>5} {s['breaks']:>5} {s['break_eng']:>5} {s['timeouts']:>5} {s['timeout_eng']:>5} {s['avg_len']:>7.1f} {s['ssv']:>8.4f} {s['avg_force']:>7.2f} {s['max_force']:>7.2f} {s['energy']:>8.2f} {s['blocked_avg']:>7.1f} {s['blocked_std']:>7.1f}")
                rp(f"{'=' * table_width}")

        rp("")
        rp(f"{'=' * 80}")
        rp("EVALUATION COMPLETE")
        rp(f"{'=' * 80}")

    finally:
        keyboard.stop()

    robot.shutdown()


if __name__ == "__main__":
    main()
