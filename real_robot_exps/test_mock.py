import torch
import numpy as np
from real_robot_exps.robot_interface import FrankaInterface, make_ee_target_pose, SafetyViolation

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

# Fail-fast: get_*() before reset should raise RuntimeError
try:
    robot.get_ee_position()
    print("FAIL")
except RuntimeError:
    print("PASS: fail-fast before reset")

# Reset + read_state
target = make_ee_target_pose(np.array([0.3, 0.0, 0.4]), np.array([3.1416, 0.0, 0.0]))
robot.reset_to_start_pose(target)
robot.read_state()

# Check all shapes
assert robot.get_ee_position().shape == (3,)
assert robot.get_ee_orientation().shape == (4,)
assert robot.get_ee_linear_velocity().shape == (3,)
assert robot.get_ee_angular_velocity().shape == (3,)
assert robot.get_force_torque().shape == (6,)
assert robot.get_joint_positions().shape == (7,)
assert robot.get_joint_velocities().shape == (7,)
assert robot.get_joint_torques().shape == (7,)
assert robot.get_jacobian().shape == (6, 7)
assert robot.get_mass_matrix().shape == (7, 7)
print("PASS: all shapes correct")

# send_joint_torques
robot.send_joint_torques(torch.zeros(7))
try:
    robot.send_joint_torques(torch.zeros(9))
    print("FAIL")
except ValueError:
    print("PASS: rejects wrong shape")

# Safety + quaternion norm
robot.check_safety()
assert abs(torch.norm(robot.get_ee_orientation()).item() - 1.0) < 1e-5
print("PASS: safety + unit quaternion")

robot.shutdown()
print("ALL TESTS PASSED")

