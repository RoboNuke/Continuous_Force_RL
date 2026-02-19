"""
Quick diagnostic: move to eval start pose, then read O_F_ext_hat_K to check for gravity bias.

Moves the robot to the same pose used at the start of eval episodes (above the hole),
then holds with zero torques for 60s to measure the F/T bias. This ensures the
bias is measured at the same arm configuration used during evaluation.

If the end-effector payload model is correct, readings should be near zero.
A consistent bias means the payload mass/CoG is wrong in the robot's model.

Usage:
    conda activate hunter_env
    python real_robot_exps/check_ft_bias.py
"""

import math
import numpy as np
import time
import yaml

CONFIG_PATH = "real_robot_exps/config.yaml"
DURATION_SEC = 60
PRINT_INTERVAL_MS = 5000
MOVE_DURATION_SEC = 3.0


def _rotation_matrix_from_rpy(roll, pitch, yaw):
    """RPY (XYZ intrinsic) to 3x3 rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])


def _make_target_pose(pos, rpy):
    """Build 4x4 homogeneous transform from position [3] and RPY [3]."""
    R = _rotation_matrix_from_rpy(rpy[0], rpy[1], rpy[2])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def _pose_to_column_major(T):
    """Convert 4x4 row-major to 16-element column-major list (libfranka format)."""
    return [
        T[0,0], T[1,0], T[2,0], 0.0,
        T[0,1], T[1,1], T[2,1], 0.0,
        T[0,2], T[1,2], T[2,2], 0.0,
        T[0,3], T[1,3], T[2,3], 1.0,
    ]


def _quat_slerp(q0, q1, t):
    """Spherical linear interpolation between two (w,x,y,z) quaternions."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return q0 + t * (q1 - q0)
    theta = math.acos(min(dot, 1.0))
    sin_theta = math.sin(theta)
    return (math.sin((1-t)*theta) / sin_theta) * q0 + (math.sin(t*theta) / sin_theta) * q1


def _rotation_matrix_to_quat(R):
    """3x3 rotation matrix to (w,x,y,z) quaternion."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w, x, y, z = 0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w, x, y, z = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w, x, y, z = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w, x, y, z = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quat_to_rotation_matrix(q):
    """(w,x,y,z) quaternion to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def main():
    import pylibfranka as plf

    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    robot_cfg = config['robot']
    task_cfg = config['task']
    noise_cfg = config.get('noise', {})

    IP = robot_cfg['ip']
    NE_T_EE = robot_cfg['NE_T_EE']
    EE_T_K = robot_cfg['EE_T_K']

    # Compute eval start pose (same as pro_real_robot_eval.py)
    fixed_pos = np.array(task_cfg['fixed_asset_position'])
    obs_frame_z = task_cfg['hole_height'] + task_cfg.get('fixed_asset_base_height', 0.0)
    goal_position = fixed_pos.copy()
    goal_position[2] += obs_frame_z

    hand_init_pos = np.array(noise_cfg.get('hand_init_pos', [0.0, 0.0, 0.047]))
    hand_init_orn = noise_cfg.get('hand_init_orn', [3.1416, 0.0, 0.0])

    target_ee_pos = goal_position + hand_init_pos
    target_rpy = list(hand_init_orn)
    target_pose = _make_target_pose(target_ee_pos, target_rpy)
    target_flat = _pose_to_column_major(target_pose)

    print(f"Connecting to robot at {IP}...")
    robot = plf.Robot(IP)
    robot.set_EE(NE_T_EE)
    robot.set_K(EE_T_K)
    robot.set_collision_behavior(
        [100.0] * 7, [100.0] * 7,
        [100.0] * 6, [100.0] * 6,
    )

    # --- Move to eval start pose ---
    print(f"Target EE pos: {target_ee_pos.tolist()}")
    print(f"Target RPY: {target_rpy}")
    input("Press Enter to MOVE TO EVAL START POSE...")

    ctrl = robot.start_cartesian_pose_control(plf.ControllerMode.JointImpedance)
    state, _ = ctrl.readOnce()
    start_flat = np.array(state.O_T_EE)

    start_t = start_flat[12:15].copy()
    start_R = np.array([
        [start_flat[0], start_flat[4], start_flat[8]],
        [start_flat[1], start_flat[5], start_flat[9]],
        [start_flat[2], start_flat[6], start_flat[10]],
    ])
    start_q = _rotation_matrix_to_quat(start_R)

    target_t = target_ee_pos
    target_R = _rotation_matrix_from_rpy(*target_rpy)
    target_q = _rotation_matrix_to_quat(target_R)

    n_move = int(MOVE_DURATION_SEC * 1000)
    for i in range(n_move):
        alpha = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / n_move))
        interp_t = (1.0 - alpha) * start_t + alpha * target_t
        interp_q = _quat_slerp(start_q, target_q, alpha)
        interp_R = _quat_to_rotation_matrix(interp_q)

        interp_flat = np.array([
            interp_R[0,0], interp_R[1,0], interp_R[2,0], 0.0,
            interp_R[0,1], interp_R[1,1], interp_R[2,1], 0.0,
            interp_R[0,2], interp_R[1,2], interp_R[2,2], 0.0,
            interp_t[0], interp_t[1], interp_t[2], 1.0,
        ])

        pose_cmd = plf.CartesianPose(interp_flat.tolist())
        if i == n_move - 1:
            pose_cmd.motion_finished = True
            ctrl.writeOnce(pose_cmd)
            break
        ctrl.writeOnce(pose_cmd)
        state, _ = ctrl.readOnce()

    robot.stop()
    print("Arrived at eval start pose.")

    # --- Read F/T bias at eval pose ---
    print(f"\nReading O_F_ext_hat_K for {DURATION_SEC}s (robot should be in FREE SPACE, no contact)...")
    print("=" * 80)

    ctrl = robot.start_torque_control()

    n_steps = DURATION_SEC * 1000
    all_readings = []

    for i in range(n_steps):
        state, _ = ctrl.readOnce()
        ft = list(state.O_F_ext_hat_K)
        all_readings.append(ft)

        if i % PRINT_INTERVAL_MS == 0:
            print(f"  t={i/1000:.1f}s  F/T = [Fx={ft[0]:+.3f}, Fy={ft[1]:+.3f}, Fz={ft[2]:+.3f}, "
                  f"Tx={ft[3]:+.3f}, Ty={ft[4]:+.3f}, Tz={ft[5]:+.3f}]")

        ctrl.writeOnce(plf.Torques([0.0] * 7))

    robot.stop()

    # Compute statistics
    readings = np.array(all_readings)
    mean = readings.mean(axis=0)
    std = readings.std(axis=0)
    abs_max = np.abs(readings).max(axis=0)

    print("=" * 80)
    print("SUMMARY (should all be near zero if EE model is correct):")
    print(f"  Mean:    [Fx={mean[0]:+.4f}, Fy={mean[1]:+.4f}, Fz={mean[2]:+.4f}, "
          f"Tx={mean[3]:+.4f}, Ty={mean[4]:+.4f}, Tz={mean[5]:+.4f}]")
    print(f"  Std:     [Fx={std[0]:.4f}, Fy={std[1]:.4f}, Fz={std[2]:.4f}, "
          f"Tx={std[3]:.4f}, Ty={std[4]:.4f}, Tz={std[5]:.4f}]")
    print(f"  AbsMax:  [Fx={abs_max[0]:.4f}, Fy={abs_max[1]:.4f}, Fz={abs_max[2]:.4f}, "
          f"Tx={abs_max[3]:.4f}, Ty={abs_max[4]:.4f}, Tz={abs_max[5]:.4f}]")

    force_bias = np.linalg.norm(mean[:3])
    torque_bias = np.linalg.norm(mean[3:])
    print(f"\n  Force bias magnitude:  {force_bias:.4f} N")
    print(f"  Torque bias magnitude: {torque_bias:.4f} Nm")

    # Print config-ready line
    mean_list = [round(float(v), 4) for v in mean]
    print(f"\n  config.yaml ft_bias value:")
    print(f"  ft_bias: {mean_list}")

    if force_bias > 1.0:
        print(f"\n\033[91m[WARNING] Force bias {force_bias:.2f}N is significant (>1N).\033[0m")
    else:
        print(f"\n\033[92m[OK] Force bias {force_bias:.2f}N is small (<1N). EE model looks correct.\033[0m")


if __name__ == "__main__":
    main()
