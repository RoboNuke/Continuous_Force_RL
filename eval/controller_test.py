"""
Controller Verification Script

Moves the robot 5cm along each axis (X, Y, Z) using the same torque control
pipeline as pro_real_robot_eval.py, then returns home. Verifies the controller
converges by checking for 10 consecutive frames with position change < 0.1mm.

All gains are loaded from the real robot config (control_gains section).
No WandB tag or training checkpoints required.

Usage:
    python eval/controller_test.py --config real_robot_exps/config.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_robot_exps.pro_robot_interface import FrankaInterface, make_ee_target_pose
from real_robot_exps.hybrid_controller import ControlTargets


CONVERGE_THRESHOLD = 1e-4   # 0.1mm
CONVERGE_FRAMES = 10
MAX_STEPS = 500             # ~33s at 15Hz safety cap
MOVE_DISTANCE = 0.05        # 5cm


def load_gains_from_config(real_config: dict, device: str = "cpu") -> dict:
    """Load controller gains from real robot config control_gains section.

    Returns dict with all tensors needed to build ControlTargets.
    """
    gains_cfg = real_config.get('control_gains', {})

    if 'task_prop_gains' not in gains_cfg:
        raise RuntimeError("control_gains.task_prop_gains not found in config")
    if 'task_deriv_gains' not in gains_cfg:
        raise RuntimeError("control_gains.task_deriv_gains not found in config")

    task_prop_gains = torch.tensor(gains_cfg['task_prop_gains'], device=device, dtype=torch.float32)
    task_deriv_gains = torch.tensor(gains_cfg['task_deriv_gains'], device=device, dtype=torch.float32)
    kp_null = gains_cfg.get('kp_null', 0.0)
    kd_null = gains_cfg.get('kd_null', 0.0)

    if gains_cfg.get('singularity_damping_enabled', False):
        singularity_damping = gains_cfg.get('singularity_damping_lambda', 0.01)
    else:
        singularity_damping = 0.0

    print(f"  task_prop_gains:  {task_prop_gains.tolist()}")
    print(f"  task_deriv_gains: {task_deriv_gains.tolist()}")
    print(f"  kp_null: {kp_null}, kd_null: {kd_null}")
    print(f"  singularity_damping: {singularity_damping}")

    return {
        'task_prop_gains': task_prop_gains,
        'task_deriv_gains': task_deriv_gains,
        'kp_null': kp_null,
        'kd_null': kd_null,
        'singularity_damping': singularity_damping,
    }


def build_position_targets(
    gains: dict,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    default_dof_pos: torch.Tensor,
    device: str = "cpu",
) -> ControlTargets:
    """Build ControlTargets for pure position control to a fixed target.

    Sets goal_position = target_pos so bounds constraint doesn't interfere.
    """
    # pos_bounds set large enough to never clamp
    pos_bounds = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)

    return ControlTargets(
        target_pos=target_pos,
        target_quat=target_quat,
        target_force=torch.zeros(6, device=device),
        sel_matrix=torch.zeros(6, device=device),
        task_prop_gains=gains['task_prop_gains'],
        task_deriv_gains=gains['task_deriv_gains'],
        force_kp=torch.zeros(6, device=device),
        force_di_wrench=torch.zeros(6, device=device),
        default_dof_pos=default_dof_pos,
        kp_null=gains['kp_null'],
        kd_null=gains['kd_null'],
        pos_bounds=pos_bounds,
        goal_position=target_pos,
        ctrl_mode="force_only",
        singularity_damping=gains['singularity_damping'],
    )


def run_move(
    robot: FrankaInterface,
    gains: dict,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    default_dof_pos: torch.Tensor,
    label: str,
    device: str = "cpu",
) -> dict:
    """Run torque control until robot converges to target_pos.

    Returns dict with start_pos, target_pos, achieved_pos, error, steps.
    """
    targets = build_position_targets(gains, target_pos, target_quat, default_dof_pos, device)

    robot.start_torque_mode()

    snap = robot.get_state_snapshot()
    start_pos = snap.ee_pos.clone()
    prev_pos = snap.ee_pos.clone()
    converge_count = 0

    for step in range(MAX_STEPS):
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        robot.check_safety(snap)

        robot.set_control_targets(targets)

        pos_delta = torch.norm(snap.ee_pos - prev_pos).item()
        prev_pos = snap.ee_pos.clone()

        if pos_delta < CONVERGE_THRESHOLD:
            converge_count += 1
        else:
            converge_count = 0

        if converge_count >= CONVERGE_FRAMES:
            break

    robot.end_control()

    achieved_pos = snap.ee_pos.clone()
    error = (achieved_pos - target_pos).tolist()
    error_norm = torch.norm(achieved_pos - target_pos).item()

    steps_used = step + 1
    converged = converge_count >= CONVERGE_FRAMES

    print(f"  [{label}]")
    print(f"    Start:    [{start_pos[0].item():.5f}, {start_pos[1].item():.5f}, {start_pos[2].item():.5f}]")
    print(f"    Target:   [{target_pos[0].item():.5f}, {target_pos[1].item():.5f}, {target_pos[2].item():.5f}]")
    print(f"    Achieved: [{achieved_pos[0].item():.5f}, {achieved_pos[1].item():.5f}, {achieved_pos[2].item():.5f}]")
    print(f"    Error:    [{error[0]:.5f}, {error[1]:.5f}, {error[2]:.5f}] (norm={error_norm*1000:.2f}mm)")
    print(f"    Steps:    {steps_used} ({'converged' if converged else 'MAX STEPS'})")

    return {
        'label': label,
        'start_pos': start_pos,
        'target_pos': target_pos,
        'achieved_pos': achieved_pos,
        'error': error,
        'error_norm': error_norm,
        'steps': steps_used,
        'converged': converged,
    }


def main():
    parser = argparse.ArgumentParser(description="Controller Verification Test")
    parser.add_argument("--config", type=str, default="real_robot_exps/config.yaml", help="Real robot config path")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--override", action="append", default=[], help="Override config values")
    args = parser.parse_args()

    device = args.device

    print("=" * 80)
    print("CONTROLLER VERIFICATION TEST")
    print("=" * 80)

    # 1. Load config
    print(f"\nLoading config: {args.config}")
    with open(args.config, 'r') as f:
        real_config = yaml.safe_load(f)

    if args.override:
        for override in args.override:
            if '=' not in override:
                raise ValueError(f"Override must be 'key=value', got: {override}")
            key_path, value_str = override.split('=', 1)
            keys = key_path.split('.')
            parent = real_config
            for k in keys[:-1]:
                parent = parent[k]
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

    # 2. Load gains from config
    print("\nLoading controller gains...")
    gains = load_gains_from_config(real_config, device)

    # 3. Initialize robot
    print("\nInitializing robot interface...")
    robot = FrankaInterface(real_config, device=device)

    print("\nClosing gripper...")
    robot.close_gripper()

    # 4. Get hand orientation from config
    noise_cfg = real_config.get('noise', {})
    hand_init_orn = list(noise_cfg.get('hand_init_orn', [3.1416, 0.0, 0.0]))

    # 5. Compute home position (same calibration pose as eval: goal XY, 5cm above goal Z)
    task_cfg = real_config['task']
    fixed_asset_position = torch.tensor(task_cfg['fixed_asset_position'], device=device, dtype=torch.float32)
    obs_frame_z_offset = task_cfg['hole_height'] + task_cfg['fixed_asset_base_height']
    home_pos = fixed_asset_position.clone()
    home_pos[2] += obs_frame_z_offset + 0.05
    home_pose_4x4 = make_ee_target_pose(home_pos.cpu().numpy(), np.array(hand_init_orn))
    retract_height = real_config['robot']['retract_height_m']

    # 6. Move to home and wait for user
    print("\nMoving to home position...")
    robot.retract_up(retract_height)
    robot.reset_to_start_pose(home_pose_4x4)
    snap = robot.get_state_snapshot()
    home_actual = snap.ee_pos.clone()
    home_quat = snap.ee_quat.clone()
    default_dof_pos = snap.joint_pos.clone()
    print(f"  Home: [{home_actual[0].item():.5f}, {home_actual[1].item():.5f}, {home_actual[2].item():.5f}]")
    input("  Press Enter to begin controller test...")

    # 7. Run tests
    axis_names = ['X', 'Y', 'Z']
    results = []

    for axis in range(3):
        label = f"+{MOVE_DISTANCE*100:.0f}cm {axis_names[axis]}"
        print(f"\n--- Test: {label} ---")

        # Compute target
        target = home_actual.clone()
        target[axis] += MOVE_DISTANCE

        # Move to target
        result = run_move(robot, gains, target, home_quat, default_dof_pos, label, device)
        results.append(result)

        # Return home
        print(f"  Returning home...")
        robot.retract_up(retract_height)
        robot.reset_to_start_pose(home_pose_4x4)

        # Verify home position
        snap = robot.get_state_snapshot()
        home_err = torch.norm(snap.ee_pos - home_actual).item()
        print(f"  Home error: {home_err*1000:.2f}mm")

    # 8. Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Test':<15} {'Error (mm)':>12} {'Steps':>8} {'Converged':>12}")
    print(f"  {'-'*47}")
    for r in results:
        print(f"  {r['label']:<15} {r['error_norm']*1000:>12.2f} {r['steps']:>8} {'YES' if r['converged'] else 'NO':>12}")
    print(f"{'=' * 80}")

    # 9. Shutdown
    robot.shutdown()


if __name__ == "__main__":
    main()
