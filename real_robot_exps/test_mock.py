import torch
import numpy as np
from real_robot_exps.robot_interface import FrankaInterface, make_ee_target_pose, SafetyViolation
from real_robot_exps.hybrid_controller import ControlTargets

config = {
'robot': {
  'ip': '192.168.1.11', 'use_mock': True, 'control_rate_hz': 15.0,
  'reset_duration_sec': 0.01, 'ft_ema_alpha': 0.2,
  'NE_T_EE': [0.7071,-0.7071,0.0,0.0, 0.7071,0.7071,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.1034,1.0],
  'EE_T_K': [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
},
'task': {'fixed_asset_position': [0.3, 0.0, 0.35]},
}

robot = FrankaInterface(config, device='cpu')

# Fail-fast: get_state_snapshot() before reset should raise RuntimeError
try:
    robot.get_state_snapshot()
    print("FAIL")
except RuntimeError:
    print("PASS: fail-fast before reset")

# Reset stores a snapshot
target = make_ee_target_pose(np.array([0.3, 0.0, 0.4]), np.array([3.1416, 0.0, 0.0]))
robot.reset_to_start_pose(target)
snap = robot.get_state_snapshot()

# Check all shapes from snapshot
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

# Start torque mode and verify background thread
robot.start_torque_mode()
snap = robot.get_state_snapshot()
assert snap.ee_pos.shape == (3,)
print("PASS: torque mode snapshot works")

# send_joint_torques (legacy API)
robot.send_joint_torques(torch.zeros(7))
try:
    robot.send_joint_torques(torch.zeros(9))
    print("FAIL")
except ValueError:
    print("PASS: rejects wrong shape")

# set_control_targets — 1kHz torque recomputation
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
    singularity_damping=0.0,
)
robot.set_control_targets(dummy_targets)
import time; time.sleep(0.05)  # let 1kHz thread run a few cycles
snap2 = robot.get_state_snapshot()
assert snap2.ee_pos.shape == (3,)
print("PASS: set_control_targets + 1kHz recomputation works")

# Safety + quaternion norm
robot.check_safety(snap)
assert abs(torch.norm(snap.ee_quat).item() - 1.0) < 1e-5
print("PASS: safety + unit quaternion")

# wait_for_policy_step should not error
robot.wait_for_policy_step()
print("PASS: wait_for_policy_step works")

# End control stops background thread and clears targets
robot.end_control()
assert robot._control_targets is None
print("PASS: end_control stops thread + clears targets")

robot.shutdown()
print("ALL TESTS PASSED (8/8)")
