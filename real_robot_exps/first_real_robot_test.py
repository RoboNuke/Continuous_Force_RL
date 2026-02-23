"""
First Real Robot Test — Safe, interactive validation of FrankaInterface on real FR3.

Runs 5 sequential phases, printing all values for manual inspection.
Waits for Enter between phases. Robot holds zero torques throughout.

Usage:
    python real_robot_exps/first_real_robot_test.py
    python real_robot_exps/first_real_robot_test.py --config real_robot_exps/config.yaml
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_robot_exps.robot_interface import FrankaInterface, make_ee_target_pose, SafetyViolation


def wait_for_enter(msg: str = "Press Enter to continue to next phase..."):
    """Block until user presses Enter."""
    input(f"\n>>> {msg}\n")


def print_separator(phase: int, title: str):
    print(f"\n{'=' * 70}")
    print(f"  PHASE {phase}: {title}")
    print(f"{'=' * 70}\n")


def phase1_connect_and_read(config: dict):
    """Phase 1: Connect to robot, read raw state (no motion)."""
    print_separator(1, "CONNECT & READ RAW STATE (no motion)")

    robot_cfg = config['robot']
    ip = robot_cfg['ip']

    # Import real pylibfranka (never mock for this test)
    import pylibfranka
    print(f"Connecting to robot at {ip}...")
    raw_robot = pylibfranka.Robot(ip)

    # Set frames
    NE_T_EE = robot_cfg['NE_T_EE']
    EE_T_K = robot_cfg['EE_T_K']
    raw_robot.set_EE(NE_T_EE)
    raw_robot.set_K(EE_T_K)
    print("NE_T_EE and EE_T_K set.")

    # Read raw state (outside control loop)
    state = raw_robot.read_once()

    print("\n--- Raw O_T_EE (column-major, 16 elements) ---")
    T = state.O_T_EE
    for row in range(4):
        cols = [T[row + col * 4] for col in range(4)]
        print(f"  [{cols[0]:+.6f}  {cols[1]:+.6f}  {cols[2]:+.6f}  {cols[3]:+.6f}]")

    print(f"\n--- Raw joint positions q (7) ---")
    print(f"  {[f'{v:+.4f}' for v in state.q]}")

    print(f"\n--- Raw joint velocities dq (7) ---")
    print(f"  {[f'{v:+.6f}' for v in state.dq]}")

    print(f"\n--- Raw O_F_ext_hat_K (6) ---")
    print(f"  {[f'{v:+.4f}' for v in state.O_F_ext_hat_K]}")

    # Extract EE position
    ee_pos = [T[12], T[13], T[14]]
    print(f"\n--- EE position from O_T_EE[12:15] ---")
    print(f"  x={ee_pos[0]:.4f}m, y={ee_pos[1]:.4f}m, z={ee_pos[2]:.4f}m")

    # Sanity check: is position in reasonable workspace?
    x, y, z = ee_pos
    checks = []
    checks.append(("x in [0.1, 0.8]", 0.1 <= x <= 0.8))
    checks.append(("y in [-0.5, 0.5]", -0.5 <= y <= 0.5))
    checks.append(("z in [0.0, 0.8]", 0.0 <= z <= 0.8))

    print("\n--- Workspace sanity checks ---")
    all_ok = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {status}: {name} (value: {eval(name[0])})")

    if not all_ok:
        print("\n  WARNING: EE position outside expected workspace!")
        print("  Visually confirm the robot is in a safe configuration before continuing.")

    print("\n  MANUAL CHECK: Does the printed EE position match where the")
    print("  fingertip midpoint physically is? Measure with a ruler if unsure.")

    # YAML-ready calibration output (cyan for visibility)
    ee_to_peg_base_offset = config['task']['ee_to_peg_base_offset']
    peg_base_pos = [ee_pos[i] + ee_to_peg_base_offset[i] for i in range(3)]
    CYAN = "\033[36m"
    RESET = "\033[0m"
    yaml_val = f"[{peg_base_pos[0]:.4f}, {peg_base_pos[1]:.4f}, {peg_base_pos[2]:.4f}]"
    print(f"\n  {CYAN}--- Copy-paste into config.yaml (ee_pos + ee_to_peg_base_offset) ---{RESET}")
    print(f"  {CYAN}ee_to_peg_base_offset: {ee_to_peg_base_offset}{RESET}")
    print(f"  {CYAN}fixed_asset_position: {yaml_val}{RESET}")
    print(f"  {CYAN}target_peg_base_position: {yaml_val}{RESET}")

    yaml_joints = "[" + ", ".join(f"{v:.4f}" for v in state.q) + "]"
    print(f"\n  {CYAN}--- Joint angles (copy-paste into config.yaml task.joint_angles) ---{RESET}")
    print(f"  {CYAN}joint_angles: {yaml_joints}{RESET}")

    raw_robot.stop()
    return ee_pos


def phase2_frame_validation(config: dict):
    """Phase 2: Validate frame setup and quaternion extraction."""
    print_separator(2, "FRAME VALIDATION")

    # Raw pylibfranka connection for frame/quaternion validation (no FrankaInterface needed)
    import pylibfranka
    raw_robot = pylibfranka.Robot(config['robot']['ip'])
    raw_robot.set_EE(config['robot']['NE_T_EE'])
    raw_robot.set_K(config['robot']['EE_T_K'])
    state = raw_robot.read_once()

    T = np.array(state.O_T_EE)

    print("--- NE_T_EE (as set) ---")
    ne = config['robot']['NE_T_EE']
    for row in range(4):
        cols = [ne[row + col * 4] for col in range(4)]
        print(f"  [{cols[0]:+.6f}  {cols[1]:+.6f}  {cols[2]:+.6f}  {cols[3]:+.6f}]")

    print("\n--- EE_T_K (as set) ---")
    ek = config['robot']['EE_T_K']
    for row in range(4):
        cols = [ek[row + col * 4] for col in range(4)]
        print(f"  [{cols[0]:+.6f}  {cols[1]:+.6f}  {cols[2]:+.6f}  {cols[3]:+.6f}]")

    print(f"\n--- EE position from O_T_EE ---")
    pos = [T[12], T[13], T[14]]
    print(f"  position: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")

    print(f"\n--- Rotation matrix from O_T_EE (column-major) ---")
    R = np.array([
        [T[0], T[4], T[8]],
        [T[1], T[5], T[9]],
        [T[2], T[6], T[10]],
    ])
    for row in range(3):
        print(f"  [{R[row, 0]:+.6f}  {R[row, 1]:+.6f}  {R[row, 2]:+.6f}]")

    # Check rotation matrix validity
    det = np.linalg.det(R)
    RtR = R.T @ R
    ident_err = np.max(np.abs(RtR - np.eye(3)))
    print(f"\n  det(R) = {det:.6f} (expected ~1.0)")
    print(f"  max|R^T R - I| = {ident_err:.8f} (expected ~0.0)")
    print(f"  {'PASS' if abs(det - 1.0) < 0.01 and ident_err < 0.01 else 'FAIL'}: valid rotation matrix")

    # Quaternion via our conversion
    R_torch = torch.tensor(R, dtype=torch.float32)
    from real_robot_exps.robot_interface import _rotation_matrix_to_quat_wxyz
    quat = _rotation_matrix_to_quat_wxyz(R_torch)
    q_norm = torch.norm(quat).item()
    print(f"\n--- Quaternion (w,x,y,z) ---")
    print(f"  {quat.tolist()}")
    print(f"  norm = {q_norm:.6f}")
    print(f"  {'PASS' if abs(q_norm - 1.0) < 1e-5 else 'FAIL'}: unit quaternion")

    # Extract Euler angles for interpretability
    w, x, y, z = quat.tolist()
    # Roll (X), Pitch (Y), Yaw (Z) from quaternion
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)

    print(f"\n--- Euler angles (XYZ intrinsic) ---")
    print(f"  roll  = {roll:.4f} rad ({math.degrees(roll):.2f} deg)")
    print(f"  pitch = {pitch:.4f} rad ({math.degrees(pitch):.2f} deg)")
    print(f"  yaw   = {yaw:.4f} rad ({math.degrees(yaw):.2f} deg)")
    print(f"\n  For peg-in-hole, expect roll ~pi (~180 deg), pitch ~0, yaw varies.")

    raw_robot.stop()


def phase3_torque_control_snapshot(robot: FrankaInterface):
    """Phase 3: Start torque control with background thread, read snapshots."""
    print_separator(3, "TORQUE CONTROL + SNAPSHOT (zero torques, background thread)")

    # Start torque mode (includes warmup + background thread)
    robot.start_torque_mode()

    n_iters = 5
    print(f"Reading {n_iters} snapshots from background thread...\n")

    for i in range(n_iters):
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        # Reset timer for next iteration
        robot.send_joint_torques(torch.zeros(7))

        print(f"--- Iteration {i+1}/{n_iters} ---")
        print(f"  ee_pos:     {snap.ee_pos.tolist()}")
        print(f"  ee_quat:    {[f'{v:.4f}' for v in snap.ee_quat.tolist()]}")
        print(f"  ee_linvel:  {[f'{v:.6f}' for v in snap.ee_linvel.tolist()]}")
        print(f"  ee_angvel:  {[f'{v:.6f}' for v in snap.ee_angvel.tolist()]}")
        print(f"  force_torque: {[f'{v:.4f}' for v in snap.force_torque.tolist()]}")
        print(f"  joint_pos:  {[f'{v:.4f}' for v in snap.joint_pos.tolist()]}")
        print(f"  joint_vel:  {[f'{v:.6f}' for v in snap.joint_vel.tolist()]}")
        print(f"  joint_torq: {[f'{v:.4f}' for v in snap.joint_torques.tolist()]}")

        J = snap.jacobian
        M = snap.mass_matrix
        print(f"  jacobian shape: {J.shape}")
        print(f"  mass_matrix shape: {M.shape}")

        # Mass matrix symmetry check
        sym_err = torch.max(torch.abs(M - M.T)).item()
        print(f"  mass_matrix symmetry error: {sym_err:.8f} {'PASS' if sym_err < 0.01 else 'FAIL'}")

        # Verify J @ dq matches ee velocities
        ee_vel_from_J = J @ snap.joint_vel
        ee_linvel_from_J = ee_vel_from_J[:3]
        ee_angvel_from_J = ee_vel_from_J[3:]
        linvel_err = torch.max(torch.abs(ee_linvel_from_J - snap.ee_linvel)).item()
        angvel_err = torch.max(torch.abs(ee_angvel_from_J - snap.ee_angvel)).item()
        print(f"  J@dq vs ee_linvel error: {linvel_err:.8f} {'PASS' if linvel_err < 1e-6 else 'FAIL'}")
        print(f"  J@dq vs ee_angvel error: {angvel_err:.8f} {'PASS' if angvel_err < 1e-6 else 'FAIL'}")
        print()

    # Safety check
    snap = robot.get_state_snapshot()
    try:
        robot.check_safety(snap)
        print("PASS: check_safety() passed")
    except SafetyViolation as e:
        print(f"FAIL: SafetyViolation: {e}")

    # End torque control session (keep connection alive for next phase)
    robot.end_control()


def phase4_force_torque_sign(robot: FrankaInterface):
    """Phase 4: Validate F/T sign convention with manual push.

    Split into two torque-control sessions so we can safely block for user
    input between them (background thread keeps 1kHz communication active
    while a control session is active).
    """
    print_separator(4, "FORCE/TORQUE SIGN CONVENTION")

    # --- Session 1: Read baseline F/T at rest ---
    robot.start_torque_mode()

    # Let EMA stabilize (background thread runs 1kHz loop)
    print("Stabilizing EMA filter (1 second)...")
    time.sleep(1.0)

    snap_baseline = robot.get_state_snapshot()
    our_ft = snap_baseline.force_torque

    print(f"\n--- Baseline F/T (robot at rest) ---")
    print(f"  Our force_torque (negated EMA): {[f'{v:+.4f}' for v in our_ft.tolist()]}")

    # End torque control before waiting for user input
    robot.end_control()

    print(f"\n  INSTRUCTION: Gently push the end-effector in the POSITIVE X direction")
    print(f"  (away from the robot base, toward the front of the workspace).")
    print(f"  Hold the push steady and press Enter.\n")

    wait_for_enter("Push EE in +X direction, then press Enter...")

    # --- Session 2: Read F/T while user is pushing (same connection) ---
    robot.start_torque_mode()

    # Let EMA stabilize under applied force
    time.sleep(1.0)

    snap_pushed = robot.get_state_snapshot()
    pushed_our_ft = snap_pushed.force_torque

    print(f"\n--- F/T while pushing +X ---")
    print(f"  Our force_torque: {[f'{v:+.4f}' for v in pushed_our_ft.tolist()]}")

    delta_our = pushed_our_ft.numpy() - our_ft.numpy()
    print(f"\n  Delta our Fx: {delta_our[0]:+.4f}")

    print(f"\n  EXPECTED for training convention (force = -joint_forces):")
    print(f"    Push +X -> our force_torque[0] delta should be POSITIVE (negated reaction)")
    print(f"\n  MANUAL CHECK: Do the signs above match the expected convention?")
    print(f"  If our Fx delta is positive when pushing +X, the sign convention is correct.")

    # End torque control session (keep connection alive for next phase)
    robot.end_control()


def phase5_cartesian_reset(robot: FrankaInterface, initial_ee_pos: list):
    """Phase 5: Small Cartesian reset motion (2cm up in Z)."""
    print_separator(5, "CARTESIAN RESET MOTION (2cm up)")

    # Target: current position + 2cm in Z
    target_pos = np.array(initial_ee_pos)
    target_pos[2] += 0.02  # 2cm up

    # Use approximately downward-facing orientation (roll=pi, pitch=0, yaw=0)
    target_rpy = np.array([math.pi, 0.0, 0.0])

    print(f"  Current EE position (from Phase 1): {initial_ee_pos}")
    print(f"  Target EE position:  {target_pos.tolist()}")
    print(f"  Target RPY: {target_rpy.tolist()}")
    print(f"  Motion: +2cm in Z (straight up)")
    print(f"  Duration: {robot._reset_duration_sec}s")

    target_pose = make_ee_target_pose(target_pos, target_rpy)
    print(f"\n  Target 4x4 transform (row-major):")
    for row in range(4):
        print(f"    [{target_pose[row, 0]:+.6f}  {target_pose[row, 1]:+.6f}  "
              f"{target_pose[row, 2]:+.6f}  {target_pose[row, 3]:+.6f}]")

    print(f"\n  This will move the robot 2cm UPWARD. Ensure the path is clear.")
    wait_for_enter("Press Enter to execute Cartesian reset motion...")

    robot.reset_to_start_pose(target_pose)

    print("\n  Reset complete. Reading snapshot from reset...")
    snap = robot.get_state_snapshot()

    new_pos = snap.ee_pos
    print(f"\n--- Post-reset EE position ---")
    print(f"  New position: {new_pos.tolist()}")
    print(f"  Expected Z:   ~{target_pos[2]:.4f}")
    print(f"  Actual Z:     {new_pos[2].item():.4f}")

    z_err = abs(new_pos[2].item() - target_pos[2])
    print(f"  Z error: {z_err*1000:.2f}mm")
    print(f"  {'PASS' if z_err < 0.005 else 'WARN'}: Z within 5mm of target")

    # Verify torque control works after reset
    print("\n  Verifying torque mode starts after reset...")
    robot.start_torque_mode()
    snap = robot.get_state_snapshot()
    print(f"  PASS: torque mode + snapshot works after reset")

    # Print all state for inspection
    print(f"\n--- Full state after reset ---")
    print(f"  ee_pos:     {snap.ee_pos.tolist()}")
    print(f"  ee_quat:    {[f'{v:.4f}' for v in snap.ee_quat.tolist()]}")
    print(f"  joint_pos:  {[f'{v:.4f}' for v in snap.joint_pos.tolist()]}")
    print(f"  force_torque: {[f'{v:.4f}' for v in snap.force_torque.tolist()]}")

    # End torque control (keep connection alive for shutdown in main)
    robot.end_control()


def phase6_torque_recomputation_stress(robot: FrankaInterface):
    """Phase 6: Stress test 1kHz torque recomputation with control targets.

    Sets control targets where target_pos = current EE pos so the PD error
    is near-zero, producing near-zero torques. The 1kHz thread runs the full
    wrench + J^T + null-space computation every cycle. If this phase completes
    without controller_torque_discontinuity, the computation fits in the 1ms budget.
    """
    from real_robot_exps.hybrid_controller import ControlTargets

    print_separator(6, "1kHz TORQUE RECOMPUTATION STRESS TEST")

    robot.start_torque_mode()

    # Let the thread stabilize with zero torques first
    time.sleep(0.5)
    snap = robot.get_state_snapshot()

    # Build control targets: target = current position → PD error ≈ 0 → torques ≈ 0
    targets = ControlTargets(
        target_pos=snap.ee_pos.clone(),
        target_quat=snap.ee_quat.clone(),
        target_force=torch.zeros(6),
        sel_matrix=torch.zeros(6),
        task_prop_gains=torch.tensor([100.0, 100.0, 100.0, 30.0, 30.0, 30.0]),
        task_deriv_gains=torch.tensor([20.0, 20.0, 20.0, 11.0, 11.0, 11.0]),
        force_kp=torch.zeros(6),
        force_di_wrench=torch.zeros(6),
        default_dof_pos=snap.joint_pos.clone(),
        kp_null=10.0,
        kd_null=6.3246,
        pos_bounds=torch.tensor([0.05, 0.05, 0.05]),
        goal_position=snap.ee_pos.clone(),
        ctrl_mode="force_only",
        singularity_damping=0.0,
        partial_inertia_decoupling=False,
        sep_ori=False,
    )

    print(f"  Target EE position: {snap.ee_pos.tolist()}")
    print(f"  Target EE quat:     {[f'{v:.4f}' for v in snap.ee_quat.tolist()]}")
    print(f"  Expected result:     near-zero torques (PD error ≈ 0)")
    print(f"\n  Setting control targets — 1kHz thread will now run full")
    print(f"  wrench + J^T + null-space computation every cycle.")
    print(f"  Running for 5 seconds...\n")

    robot.set_control_targets(targets)

    # Run for 5 seconds, reading snapshots at 15Hz to monitor
    test_duration = 5.0
    start = time.time()
    n_reads = 0

    while time.time() - start < test_duration:
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        robot.check_safety(snap)

        # Update targets to track current position (keep torques near zero)
        targets = ControlTargets(
            target_pos=snap.ee_pos.clone(),
            target_quat=snap.ee_quat.clone(),
            target_force=torch.zeros(6),
            sel_matrix=torch.zeros(6),
            task_prop_gains=torch.tensor([100.0, 100.0, 100.0, 30.0, 30.0, 30.0]),
            task_deriv_gains=torch.tensor([20.0, 20.0, 20.0, 11.0, 11.0, 11.0]),
            force_kp=torch.zeros(6),
            force_di_wrench=torch.zeros(6),
            default_dof_pos=snap.joint_pos.clone(),
            kp_null=10.0,
            kd_null=6.3246,
            pos_bounds=torch.tensor([0.05, 0.05, 0.05]),
            goal_position=snap.ee_pos.clone(),
            ctrl_mode="force_only",
            singularity_damping=0.0,
        )
        robot.set_control_targets(targets)
        n_reads += 1

        # Print status every ~1 second
        elapsed = time.time() - start
        if n_reads % 15 == 0:
            print(f"  [{elapsed:.1f}s] ee_pos={[f'{v:.4f}' for v in snap.ee_pos.tolist()]}  "
                  f"ft={[f'{v:.2f}' for v in snap.force_torque[:3].tolist()]}")

    elapsed = time.time() - start
    print(f"\n  Completed {elapsed:.1f}s, {n_reads} policy steps "
          f"({n_reads/elapsed:.1f} Hz)")
    print(f"  PASS: No controller_torque_discontinuity — 1kHz recomputation fits in budget")

    robot.end_control()


def main():
    parser = argparse.ArgumentParser(description="First Real Robot Test")
    parser.add_argument("--config", type=str, default="real_robot_exps/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--skip-to", type=int, default=1,
                        help="Skip to phase N (1-6)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override to real robot
    if config['robot'].get('use_mock', False):
        print("WARNING: config has use_mock=true. This test requires a REAL robot.")
        print("Overriding to use_mock=false for this test.\n")
        config['robot']['use_mock'] = False

    print("=" * 70)
    print("  FIRST REAL ROBOT TEST")
    print("  Safe, interactive validation of FrankaInterface on real FR3")
    print("  Robot IP:", config['robot']['ip'])
    print("=" * 70)
    print("\n  This test has 6 phases:")
    print("  1. Connect & read raw state (no motion)")
    print("  2. Frame validation (no motion)")
    print("  3. Torque control + snapshot (zero torques, background thread)")
    print("  4. Force/torque sign convention (manual push)")
    print("  5. Cartesian reset motion (2cm up)")
    print("  6. 1kHz torque recomputation stress test")
    print("\n  Each phase waits for Enter before proceeding.")
    print("  You can Ctrl+C at any time to abort.\n")

    initial_ee_pos = None
    robot = None  # shared FrankaInterface for phases 3-5

    try:
        if args.skip_to <= 1:
            wait_for_enter("Press Enter to start Phase 1...")
            initial_ee_pos = phase1_connect_and_read(config)

        if args.skip_to <= 2:
            wait_for_enter("Press Enter to start Phase 2...")
            phase2_frame_validation(config)

        # Create shared FrankaInterface for phases 3-6 (mirrors eval script
        # which reuses one connection across all episodes)
        if args.skip_to <= 6:
            test_config = {**config, 'robot': {**config['robot'], 'use_mock': False}}
            robot = FrankaInterface(test_config, device='cpu')

        if args.skip_to <= 3:
            wait_for_enter("Press Enter to start Phase 3...")
            phase3_torque_control_snapshot(robot)

        if args.skip_to <= 4:
            wait_for_enter("Press Enter to start Phase 4...")
            phase4_force_torque_sign(robot)

        if args.skip_to <= 5:
            if initial_ee_pos is None:
                # Read position from existing connection if we skipped phase 1
                state = robot._robot.read_once()
                T = state.O_T_EE
                initial_ee_pos = [T[12], T[13], T[14]]

            wait_for_enter("Press Enter to start Phase 5 (will move robot 2cm up)...")
            phase5_cartesian_reset(robot, initial_ee_pos)

        if args.skip_to <= 6:
            wait_for_enter("Press Enter to start Phase 6 (1kHz recomputation stress test)...")
            phase6_torque_recomputation_stress(robot)

        print("\n" + "=" * 70)
        print("  ALL PHASES COMPLETE")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nAborted by user (Ctrl+C). Robot should be safe.")
        print("If torque control was active, the robot will stop on communication timeout.")

    except SafetyViolation as e:
        print(f"\n\nSAFETY VIOLATION: {e}")
        print("Robot should stop automatically.")

    except Exception as e:
        print(f"\n\nERROR: {type(e).__name__}: {e}")
        print("Robot should stop automatically on communication timeout.")
        raise

    finally:
        if robot is not None:
            robot.shutdown()


if __name__ == "__main__":
    main()
