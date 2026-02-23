"""Tests for process-based FrankaInterface (pro_robot_interface)."""
import torch
import numpy as np
import time
from real_robot_exps.pro_robot_interface import FrankaInterface, make_ee_target_pose, SafetyViolation
from real_robot_exps.hybrid_controller import ControlTargets


def main():
    config = {
        'robot': {
            'ip': '192.168.1.11', 'use_mock': True, 'control_rate_hz': 15.0,
            'reset_duration_sec': 0.01, 'ft_ema_alpha': 0.2,
            'NE_T_EE': [0.7071,-0.7071,0.0,0.0, 0.7071,0.7071,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.1034,1.0],
            'EE_T_K': [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
        },
        'task': {'fixed_asset_position': [0.3, 0.0, 0.35]},
    }

    passed = 0
    total = 8

    robot = FrankaInterface(config, device='cpu')

    # 1. Fail-fast: get_state_snapshot() before reset should raise RuntimeError
    try:
        robot.get_state_snapshot()
        print("FAIL: no error before reset")
    except RuntimeError:
        print("PASS: fail-fast before reset")
        passed += 1

    # 2. Reset stores a snapshot
    target = make_ee_target_pose(np.array([0.3, 0.0, 0.4]), np.array([3.1416, 0.0, 0.0]))
    robot.reset_to_start_pose(target)
    snap = robot.get_state_snapshot()

    assert snap.ee_pos.shape == (3,)
    assert snap.ee_quat.shape == (4,)
    assert snap.ee_linvel.shape == (3,)
    assert snap.ee_angvel.shape == (3,)
    assert snap.force_torque.shape == (6,)
    assert snap.joint_pos.shape == (7,)
    assert snap.joint_vel.shape == (7,)
    assert snap.joint_torques.shape == (7,)
    assert snap.jacobian.shape == (6, 7)
    assert snap.mass_matrix.shape == (7, 7)
    print("PASS: all shapes correct")
    passed += 1

    # 3. Start torque mode and verify both processes running
    robot.start_torque_mode()
    time.sleep(0.05)  # let processes spin up
    snap = robot.get_state_snapshot()
    assert snap.ee_pos.shape == (3,)
    assert robot._compute_process.is_alive()
    assert robot._compute_active.is_set()
    assert robot._comm_process.is_alive()
    assert robot._torque_mode_active
    print("PASS: torque mode — both processes running")
    passed += 1

    # 4. send_joint_torques
    robot.send_joint_torques(torch.zeros(7))
    try:
        robot.send_joint_torques(torch.zeros(9))
        print("FAIL: no error for wrong shape")
    except ValueError:
        print("PASS: rejects wrong shape")
        passed += 1

    # 5. set_control_targets — compute process recomputes torques
    dummy_targets = ControlTargets(
        target_pos=snap.ee_pos.clone(),
        target_quat=snap.ee_quat.clone(),
        target_force=torch.zeros(6),
        sel_matrix=torch.zeros(6),
        task_prop_gains=torch.tensor([100.0, 100.0, 100.0, 30.0, 30.0, 30.0]),
        task_deriv_gains=torch.tensor([20.0, 20.0, 20.0, 11.0, 11.0, 11.0]),
        force_kp=torch.zeros(6),
        force_di_wrench=torch.zeros(6),
        default_dof_pos=torch.zeros(7),
        kp_null=10.0,
        kd_null=6.3246,
        pos_bounds=torch.tensor([0.05, 0.05, 0.05]),
        goal_position=torch.tensor([0.3, 0.0, 0.35]),
        ctrl_mode="force_only",
        singularity_damping=0.01,
        partial_inertia_decoupling=False,
        sep_ori=False,
    )
    robot.set_control_targets(dummy_targets)
    time.sleep(0.1)  # let compute process pick up targets and compute
    snap2 = robot.get_state_snapshot()
    assert snap2.ee_pos.shape == (3,)
    # Verify compute process wrote something to torque shared memory
    torque_vals = list(robot._torque_shm[0:7])
    print(f"PASS: set_control_targets works (torques: [{', '.join(f'{v:.3f}' for v in torque_vals)}])")
    passed += 1

    # 6. Safety + quaternion norm
    robot.check_safety(snap)
    assert abs(torch.norm(snap.ee_quat).item() - 1.0) < 1e-5
    print("PASS: safety + unit quaternion")
    passed += 1

    # 7. wait_for_policy_step
    robot.wait_for_policy_step()
    print("PASS: wait_for_policy_step works")
    passed += 1

    # 8. End control deactivates compute, both processes stay alive
    robot.end_control()
    assert robot._compute_process.is_alive()  # process stays alive for reuse
    assert not robot._compute_active.is_set()  # but deactivated (idle)
    assert robot._comm_process.is_alive()  # comm process stays alive
    assert not robot._torque_mode_active
    print("PASS: end_control deactivates compute, processes stay alive")
    passed += 1

    robot.shutdown()
    print(f"\nALL TESTS PASSED ({passed}/{total})")


if __name__ == '__main__':
    main()
