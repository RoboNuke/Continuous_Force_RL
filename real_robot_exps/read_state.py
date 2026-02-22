"""
Read Robot State — Diagnostic utility mirroring pro_real_robot_eval observation pipeline.

Connects to the robot identically to pro_real_robot_eval.py, reads the current
state, builds the exact raw observation tensor that would be fed to the policy's
state preprocessor (BEFORE normalization), and prints everything labeled by channel.

Requires a cached checkpoint (run eval first to populate the cache, or download
manually). Does NOT download from WandB.

Usage:
    python real_robot_exps/read_state.py --tag "MATCH:2024-01-15_10:00"
    python real_robot_exps/read_state.py --tag "MATCH:..." --config real_robot_exps/config.yaml
    python real_robot_exps/read_state.py --tag "MATCH:..." --run_id abc123
"""

import argparse
import os
import sys
import time

import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_robot_exps.pro_robot_interface import FrankaInterface, StateSnapshot
from real_robot_exps.observation_builder import ObservationBuilder, OBS_DIM_MAP
from eval.pro_real_robot_eval import (
    load_real_robot_config,
    sanitize_tag,
    load_from_cache,
    reconstruct_obs_order,
)


def print_snapshot(snap: StateSnapshot):
    """Print all StateSnapshot fields with formatted values."""
    print("\n" + "=" * 70)
    print("  RAW STATE SNAPSHOT")
    print("=" * 70)
    print(f"  ee_pos:        [{snap.ee_pos[0].item():.6f}, {snap.ee_pos[1].item():.6f}, {snap.ee_pos[2].item():.6f}]")
    print(f"  ee_quat:       [{', '.join(f'{v:.6f}' for v in snap.ee_quat.tolist())}]")
    print(f"  ee_linvel:     [{', '.join(f'{v:.6f}' for v in snap.ee_linvel.tolist())}]")
    print(f"  ee_angvel:     [{', '.join(f'{v:.6f}' for v in snap.ee_angvel.tolist())}]")
    print(f"  force_torque:  [{', '.join(f'{v:.4f}' for v in snap.force_torque.tolist())}]")
    print(f"  joint_pos:     [{', '.join(f'{v:.4f}' for v in snap.joint_pos.tolist())}]")
    print(f"  joint_vel:     [{', '.join(f'{v:.6f}' for v in snap.joint_vel.tolist())}]")
    print(f"  joint_torques: [{', '.join(f'{v:.4f}' for v in snap.joint_torques.tolist())}]")
    print(f"  jacobian:      shape={list(snap.jacobian.shape)}")
    print(f"  mass_matrix:   shape={list(snap.mass_matrix.shape)}")


def print_observation(obs: torch.Tensor, obs_builder: ObservationBuilder, action_dim: int):
    """Print raw observation vector with per-channel labels and index ranges."""
    print("\n" + "=" * 70)
    print("  RAW OBSERVATION (input to state preprocessor, before normalization)")
    print("=" * 70)

    idx = 0
    for name in obs_builder.obs_order:
        dim = OBS_DIM_MAP[name]
        vals = obs[idx:idx + dim]
        print(f"  [{idx:>3}:{idx + dim:<3}] {name:<28} {[f'{v:.6f}' for v in vals.tolist()]}")
        idx += dim

    vals = obs[idx:idx + action_dim]
    print(f"  [{idx:>3}:{idx + action_dim:<3}] {'prev_actions':<28} {[f'{v:.6f}' for v in vals.tolist()]}")
    idx += action_dim

    print(f"\n  obs_dim: {obs.shape[0]}")


def main():
    parser = argparse.ArgumentParser(
        description="Read robot state and print raw observation (mirrors pro_real_robot_eval pipeline)"
    )
    parser.add_argument("--config", type=str, default="real_robot_exps/config.yaml",
                        help="Path to real robot config.yaml")
    parser.add_argument("--tag", type=str, required=True,
                        help="WandB experiment tag (must have local cache)")
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.expanduser("~/ckpts"),
                        help="Local directory for cached checkpoints")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Evaluate specific run only (default: first in cache)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")
    parser.add_argument("--override", action="append", default=[],
                        help="Override config values (repeatable, e.g. robot.use_mock=true)")
    args = parser.parse_args()

    print("=" * 70)
    print("  READ ROBOT STATE")
    print("=" * 70)

    # ---- 1. Load real robot config (same as eval) ----
    print(f"\nLoading config: {args.config}")
    real_config = load_real_robot_config(args.config, args.override)

    # ---- 2. Enable no-sim mode (same as eval) ----
    from configs.cfg_exts.version_compat import set_no_sim_mode
    set_no_sim_mode(True)

    # ---- 3. Load training config from cache (same as eval) ----
    cache_path = os.path.abspath(os.path.join(args.checkpoint_dir, sanitize_tag(args.tag)))
    if not os.path.exists(os.path.join(cache_path, 'runs.json')):
        raise RuntimeError(
            f"No cached checkpoints found at: {cache_path}\n"
            f"Run pro_real_robot_eval.py first to download and cache checkpoints, "
            f"or populate the cache manually."
        )

    configs, runs, best_checkpoints = load_from_cache(cache_path, args.run_id)

    # ---- 4. Reconstruct obs_order (same as eval) ----
    obs_order = reconstruct_obs_order(configs)

    # ---- 5. Determine action_dim and FT config (same as eval lines 1143-1157) ----
    hybrid_enabled = configs['wrappers'].hybrid_control.enabled
    if hybrid_enabled:
        from configs.cfg_exts.ctrl_mode import get_force_size
        ctrl_mode = getattr(configs['primary'], 'ctrl_mode', 'force_only')
        force_size = get_force_size(ctrl_mode)
        action_dim = 2 * force_size + 6
    else:
        action_dim = 6

    ft_cfg = configs['wrappers'].force_torque_sensor
    use_tanh = getattr(ft_cfg, 'use_tanh_scaling', False)
    tanh_scale = getattr(ft_cfg, 'tanh_scale', 0.03)
    contact_threshold = getattr(ft_cfg, 'contact_force_threshold', 1.5)
    ee_pose_noise_enabled = getattr(configs['wrappers'].ee_pose_noise, 'enabled', False)

    # ---- 6. Create ObservationBuilder (same as eval line 1190) ----
    fixed_asset_yaw = real_config['task']['fixed_asset_yaw']
    obs_builder = ObservationBuilder(
        obs_order=obs_order,
        action_dim=action_dim,
        use_tanh_ft_scaling=use_tanh,
        tanh_ft_scale=tanh_scale,
        contact_force_threshold=contact_threshold,
        fixed_asset_yaw=fixed_asset_yaw,
        ee_pose_noise_enabled=ee_pose_noise_enabled,
        device=args.device,
    )

    # ---- 7. Compute goal_position (same as eval run_episode lines 800-805) ----
    task_cfg = real_config['task']
    fixed_asset_position = torch.tensor(
        task_cfg['fixed_asset_position'], device=args.device, dtype=torch.float32
    )
    obs_frame_z_offset = task_cfg['hole_height'] + task_cfg['fixed_asset_base_height']
    goal_position = fixed_asset_position.clone()
    goal_position[2] += obs_frame_z_offset

    print(f"\n  goal_position (obs frame): {goal_position.tolist()}")
    print(f"  action_dim: {action_dim}")
    print(f"  hybrid: {hybrid_enabled}")

    # ---- 8. Connect to robot (same as eval line 1203) ----
    print("\nInitializing robot interface...")
    robot = FrankaInterface(real_config, device=args.device)

    try:
        # ---- 9. Start torque mode for proper snapshots ----
        robot.start_torque_mode()

        # ---- 10. Let F/T EMA stabilize ----
        print("Stabilizing F/T EMA filter (1 second)...")
        time.sleep(1.0)

        # ---- 11. Read snapshot ----
        snap = robot.get_state_snapshot()

        # ---- 12. End torque control (before printing) ----
        robot.end_control()

        # ---- 13. Build observation (same as eval run_episode line 913) ----
        prev_actions = torch.zeros(action_dim, device=args.device)
        obs = obs_builder.build_observation(
            snap, goal_position, prev_actions, fixed_yaw_offset=0.0
        )

        # ---- 14. Print everything ----
        print_snapshot(snap)
        print_observation(obs, obs_builder, action_dim)

    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
