"""
Controller Verification Script

Moves the robot 5cm along each axis (X, Y, Z) using the same torque control
pipeline as pro_real_robot_eval.py, then returns home. Verifies the controller
converges by checking for 10 consecutive frames with position change < 0.1mm.

Tracks orientation throughout — the robot should maintain its home orientation
(from hand_init_orn config) during all moves. Reports per-axis RPY error in degrees.

All gains are loaded from the real robot config (control_gains section).
No WandB tag or training checkpoints required.

Usage:
    python -m real_robot_exps.controller_test --config real_robot_exps/config.yaml
"""

import argparse
import math
import time

import numpy as np
import torch
import yaml

from real_robot_exps.pro_robot_interface import FrankaInterface, make_ee_target_pose
from real_robot_exps.hybrid_controller import (
    ControlTargets, get_euler_xyz, compute_pose_error, compute_pose_task_wrench,
)


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

    partial_inertia_decoupling = gains_cfg.get('partial_inertia_decoupling', False)
    sep_ori = gains_cfg.get('sep_ori', False)

    # Mutual exclusion check
    if partial_inertia_decoupling and sep_ori:
        raise RuntimeError(
            "partial_inertia_decoupling and sep_ori are mutually exclusive — "
            "set only one to true in control_gains"
        )

    print(f"  task_prop_gains:  {task_prop_gains.tolist()}")
    print(f"  task_deriv_gains: {task_deriv_gains.tolist()}")
    print(f"  kp_null: {kp_null}, kd_null: {kd_null}")
    print(f"  singularity_damping: {singularity_damping}")
    print(f"  partial_inertia_decoupling: {partial_inertia_decoupling}")
    print(f"  sep_ori: {sep_ori}")

    return {
        'task_prop_gains': task_prop_gains,
        'task_deriv_gains': task_deriv_gains,
        'kp_null': kp_null,
        'kd_null': kd_null,
        'singularity_damping': singularity_damping,
        'partial_inertia_decoupling': partial_inertia_decoupling,
        'sep_ori': sep_ori,
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
        partial_inertia_decoupling=gains['partial_inertia_decoupling'],
        sep_ori=gains['sep_ori'],
    )


def _quat_to_rpy_deg(quat: torch.Tensor) -> list:
    """Convert quaternion (w,x,y,z) to [roll, pitch, yaw] in degrees."""
    roll, pitch, yaw = get_euler_xyz(quat)
    return [math.degrees(roll.item()), math.degrees(pitch.item()), math.degrees(yaw.item())]


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

    Returns dict with position/orientation start, target, achieved, errors, and steps.
    """
    targets = build_position_targets(gains, target_pos, target_quat, default_dof_pos, device)

    robot.start_torque_mode()

    snap = robot.get_state_snapshot()
    start_pos = snap.ee_pos.clone()
    start_quat = snap.ee_quat.clone()
    prev_pos = snap.ee_pos.clone()
    converge_count = 0

    for step in range(MAX_STEPS):
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        robot.check_safety(snap)

        robot.set_control_targets(targets)

        # Debug: replicate wrench computation from compute process for visibility
        if step == 0 or step % 50 == 0 or converge_count == CONVERGE_FRAMES - 1:
            pos_err, aa_err = compute_pose_error(
                snap.ee_pos, snap.ee_quat, target_pos, target_quat,
            )
            wrench = compute_pose_task_wrench(
                snap.ee_pos, snap.ee_quat, snap.ee_linvel, snap.ee_angvel,
                target_pos, target_quat,
                gains['task_prop_gains'], gains['task_deriv_gains'],
            )
            print(f"    [step {step:3d}] orn_error (axis-angle, base frame): "
                  f"[{aa_err[0].item():.6f}, {aa_err[1].item():.6f}, {aa_err[2].item():.6f}]")
            print(f"    [step {step:3d}] wrench [Fx,Fy,Fz,Tx,Ty,Tz] (base frame, pre-Lambda): "
                  f"[{wrench[0].item():.4f}, {wrench[1].item():.4f}, {wrench[2].item():.4f}, "
                  f"{wrench[3].item():.4f}, {wrench[4].item():.4f}, {wrench[5].item():.4f}]")
            print(f"    [step {step:3d}] frame: geometric Jacobian base frame "
                  f"(q_err = q_target * q_current^-1 -> axis-angle)")

        pos_delta = torch.norm(snap.ee_pos - prev_pos).item()
        prev_pos = snap.ee_pos.clone()

        if pos_delta < CONVERGE_THRESHOLD:
            converge_count += 1
        else:
            converge_count = 0

        if converge_count >= CONVERGE_FRAMES:
            break

    robot.end_control()

    # Position results
    achieved_pos = snap.ee_pos.clone()
    pos_error = (achieved_pos - target_pos).tolist()
    pos_error_norm = torch.norm(achieved_pos - target_pos).item()

    # Orientation results
    achieved_quat = snap.ee_quat.clone()
    target_rpy_deg = _quat_to_rpy_deg(target_quat)
    start_rpy_deg = _quat_to_rpy_deg(start_quat)
    achieved_rpy_deg = _quat_to_rpy_deg(achieved_quat)
    orn_error_deg = [(achieved_rpy_deg[i] - target_rpy_deg[i] + 180.0) % 360.0 - 180.0 for i in range(3)]

    steps_used = step + 1
    converged = converge_count >= CONVERGE_FRAMES

    print(f"  [{label}]")
    print(f"    Start Pos:      [{start_pos[0].item():.5f}, {start_pos[1].item():.5f}, {start_pos[2].item():.5f}]")
    print(f"    Target Pos:     [{target_pos[0].item():.5f}, {target_pos[1].item():.5f}, {target_pos[2].item():.5f}]")
    print(f"    Achieved Pos:   [{achieved_pos[0].item():.5f}, {achieved_pos[1].item():.5f}, {achieved_pos[2].item():.5f}]")
    print(f"    Pos Error:      [{pos_error[0]:.5f}, {pos_error[1]:.5f}, {pos_error[2]:.5f}] (norm={pos_error_norm*1000:.2f}mm)")
    print(f"    Orn Target  (RPY deg): [{target_rpy_deg[0]:.2f}, {target_rpy_deg[1]:.2f}, {target_rpy_deg[2]:.2f}]")
    print(f"    Orn Start   (RPY deg): [{start_rpy_deg[0]:.2f}, {start_rpy_deg[1]:.2f}, {start_rpy_deg[2]:.2f}]")
    print(f"    Orn Achieved(RPY deg): [{achieved_rpy_deg[0]:.2f}, {achieved_rpy_deg[1]:.2f}, {achieved_rpy_deg[2]:.2f}]")
    print(f"    Orn Error   (RPY deg): [{orn_error_deg[0]:.2f}, {orn_error_deg[1]:.2f}, {orn_error_deg[2]:.2f}]")
    print(f"    Steps:    {steps_used} ({'converged' if converged else 'MAX STEPS'})")

    return {
        'label': label,
        'start_pos': start_pos,
        'target_pos': target_pos,
        'achieved_pos': achieved_pos,
        'pos_error': pos_error,
        'pos_error_norm': pos_error_norm,
        'target_rpy_deg': target_rpy_deg,
        'start_rpy_deg': start_rpy_deg,
        'achieved_rpy_deg': achieved_rpy_deg,
        'orn_error_deg': orn_error_deg,
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

    # =========================================================================
    # DIAGNOSTIC: Frame transforms + Jacobian + null-space projector verification
    # Uses a temporary direct pylibfranka connection (before FrankaInterface)
    # =========================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC: Frame Transforms & Null-Space Projector Verification")
    print("=" * 80)

    import pylibfranka as plf

    robot_cfg = real_config['robot']
    diag_robot = plf.Robot(robot_cfg['ip'])

    # Set NE_T_EE and EE_T_K exactly as FrankaInterface does
    NE_T_EE_cfg = robot_cfg.get('NE_T_EE', [
        0.7071, -0.7071, 0.0, 0.0,
        0.7071,  0.7071, 0.0, 0.0,
        0.0,     0.0,    1.0, 0.0,
        0.0,     0.0,    0.1034, 1.0,
    ])
    EE_T_K_cfg = robot_cfg.get('EE_T_K', [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
    diag_robot.set_EE(NE_T_EE_cfg)
    diag_robot.set_K(EE_T_K_cfg)

    diag_model = diag_robot.load_model()
    diag_state = diag_robot.read_once()

    # --- Your frame transforms ---
    print("\n--- Frame Transforms (column-major -> 4x4) ---")
    print("F_T_NE (flange -> nominal EE):")
    print(np.array(diag_state.F_T_NE).reshape(4, 4, order='F'))
    print("\nF_T_EE (flange -> EE):")
    print(np.array(diag_state.F_T_EE).reshape(4, 4, order='F'))
    print("\nNE_T_EE (nominal EE -> EE, what we set):")
    print(np.array(diag_state.NE_T_EE).reshape(4, 4, order='F'))
    print("\nO_T_EE (base -> EE):")
    print(np.array(diag_state.O_T_EE).reshape(4, 4, order='F'))

    # --- Config NE_T_EE for comparison ---
    print("\nNE_T_EE from config.yaml (for comparison):")
    print(np.array(NE_T_EE_cfg).reshape(4, 4, order='F'))

    # --- Jacobian ---
    jac_flat = diag_model.zero_jacobian(diag_state)
    mass_flat = diag_model.mass(diag_state)

    # Reshape: pylibfranka returns column-major 6x7
    # .reshape(7,6).T correctly converts column-major to standard 6x7
    J = torch.tensor(jac_flat, dtype=torch.float32).reshape(7, 6).T   # [6, 7]
    M = torch.tensor(mass_flat, dtype=torch.float32).reshape(7, 7)    # [7, 7] (symmetric)

    print("\n--- Jacobian (6x7, zero_jacobian in base frame) ---")
    print("  Reshape method: column-major flat -> .reshape(7,6).T")
    print(f"  Row ordering: [vx, vy, vz, wx, wy, wz]")
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print(J.numpy())

    # Sanity check: Joint 1 rotates about base Z.
    # Row 5 (wz), col 0 (joint 1) should be ~1.0
    # Row 3 (wx), col 0 should be ~0.0
    # Row 4 (wy), col 0 should be ~0.0
    print(f"\n--- Jacobian sanity check ---")
    print(f"  J[5,0] (wz from joint 1, expect ~1.0):  {J[5,0].item():.6f}")
    print(f"  J[3,0] (wx from joint 1, expect ~0.0):  {J[3,0].item():.6f}")
    print(f"  J[4,0] (wy from joint 1, expect ~0.0):  {J[4,0].item():.6f}")
    print(f"  J[2,0] (vz from joint 1, expect ~0.0):  {J[2,0].item():.6f}")

    # --- EE velocity check: J @ dq should match expected ---
    q = torch.tensor(diag_state.q, dtype=torch.float32)
    dq = torch.tensor(diag_state.dq, dtype=torch.float32)
    ee_vel = J @ dq
    print(f"\n--- Joint state ---")
    print(f"  q:  {q.tolist()}")
    print(f"  dq: {dq.tolist()}")
    print(f"  J @ dq = ee_vel: {ee_vel.tolist()}")

    # --- Mass matrix ---
    print(f"\n--- Mass Matrix (7x7) ---")
    print(M.numpy())
    print(f"  Symmetric check (max |M - M^T|): {(M - M.T).abs().max().item():.2e}")

    # --- Null-space projector verification ---
    print(f"\n--- Null-Space Projector Verification ---")
    M_inv = torch.inverse(M)
    J_T = J.T                                         # [7, 6]
    JMJ = J @ M_inv @ J_T                             # [6, 6]
    Lambda = torch.inverse(JMJ)                        # [6, 6] task-space mass
    J_bar = Lambda @ J @ M_inv                         # [6, 7] dyn-consistent pseudoinverse
    N = torch.eye(7) - J_T @ J_bar                     # [7, 7] null-space projector

    print(f"  cond(J @ M^-1 @ J^T): {torch.linalg.cond(JMJ).item():.2f}")

    # The key test: J @ M^-1 @ N @ M @ u should be zero for any u
    # This means null-space torques produce zero EE acceleration
    torch.manual_seed(42)
    max_leak = 0.0
    for trial in range(10):
        u = torch.randn(7)
        null_torque = N @ M @ u
        ee_accel = J @ M_inv @ null_torque
        leak = ee_accel.abs().max().item()
        if leak > max_leak:
            max_leak = leak

    print(f"  Max |J @ M^-1 @ N @ M @ u| over 10 random u: {max_leak:.2e}")
    if max_leak < 1e-4:
        print(f"  PASS: Null-space projector correctly produces zero EE acceleration")
    else:
        print(f"  FAIL: Null-space torques LEAK into task space!")

    # Also test with the actual default_dof_pos -> current offset
    dist = q.clone()  # default = current, so dist = 0 at start
    u_actual = gains['kp_null'] * torch.zeros(7) + gains['kd_null'] * (-dq)
    null_torque_actual = N @ M @ u_actual
    ee_accel_actual = J @ M_inv @ null_torque_actual
    print(f"  Actual null-space EE accel (from current state): {ee_accel_actual.tolist()}")

    # --- Print raw flat arrays for manual inspection ---
    print(f"\n--- Raw flat arrays (for manual cross-check) ---")
    print(f"  jac_flat (first 12 of 42): {jac_flat[:12]}")
    print(f"  mass_flat (first 14 of 49): {mass_flat[:14]}")

    diag_robot.stop()
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

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
    home_rpy_deg = _quat_to_rpy_deg(home_quat)
    print(f"  Home Pos: [{home_actual[0].item():.5f}, {home_actual[1].item():.5f}, {home_actual[2].item():.5f}]")
    print(f"  Home Orn (RPY deg): [{home_rpy_deg[0]:.2f}, {home_rpy_deg[1]:.2f}, {home_rpy_deg[2]:.2f}]")
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

        # Move to target (orientation goal = home_quat, should not change)
        result = run_move(robot, gains, target, home_quat, default_dof_pos, label, device)
        results.append(result)

        # Return home — pause for robot to settle after torque control
        # (Cartesian motion generator rejects if robot has residual velocity)
        time.sleep(1.0)
        print(f"  Returning home...")
        robot.retract_up(retract_height)
        robot.reset_to_start_pose(home_pose_4x4)

        # Verify home position
        snap = robot.get_state_snapshot()
        home_err = torch.norm(snap.ee_pos - home_actual).item()
        print(f"  Home error: {home_err*1000:.2f}mm")

    # 8. Summary
    print(f"\n{'=' * 115}")
    print("SUMMARY")
    print(f"{'=' * 115}")
    print(f"  {'Test':<15} {'dX(mm)':>8} {'dY(mm)':>8} {'dZ(mm)':>8} "
          f"{'dRoll(°)':>10} {'dPitch(°)':>10} {'dYaw(°)':>10} "
          f"{'Steps':>8} {'Converged':>12}")
    print(f"  {'-'*103}")
    for r in results:
        pe = r['pos_error']
        oe = r['orn_error_deg']
        print(f"  {r['label']:<15} {pe[0]*1000:>8.2f} {pe[1]*1000:>8.2f} {pe[2]*1000:>8.2f} "
              f"{oe[0]:>10.2f} {oe[1]:>10.2f} {oe[2]:>10.2f} "
              f"{r['steps']:>8} {'YES' if r['converged'] else 'NO':>12}")
    print(f"{'=' * 115}")

    # 9. Shutdown
    robot.shutdown()


if __name__ == "__main__":
    main()
