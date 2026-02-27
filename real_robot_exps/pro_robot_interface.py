"""
Real Robot Interface (pylibfranka) — Process-Based Architecture

Same API as robot_interface.FrankaInterface, but BOTH the 1kHz comm loop
AND the torque compute loop run in separate PROCESSES (separate GILs),
eliminating all GIL contention.

Architecture:
  Main Process:
    └── Main thread     — 15Hz policy loop

  Comm Process (own GIL):
    └── Owns robot, model, ctrl
        - IDLE: waits for commands (reset, start_torque, retract, etc.)
        - TORQUE: runs 1kHz readOnce/writeOnce loop
        - Packs ALL state into shared memory

  Compute Process (own GIL):
    └── Reads state_shm, computes torques, writes torque_shm

Communication:
  - State: Comm process → Compute process via shared memory (mp.Array)
  - Torques: Compute process → Comm process via shared memory (mp.Array)
  - Control targets: Main thread → Compute process via mp.Queue
  - Commands: Main thread → Comm process via mp.Queue
  - Responses: Comm process → Main thread via mp.Queue

Uses torch.multiprocessing with 'spawn' context:
  - No fork(): fresh Python interpreter, no stale torch/CUDA state
  - torch tensor sharing support (share_memory_())
  - Local context (mp.get_context) avoids global state conflicts
"""

import math
import time
import queue as _queue

import numpy as np
import torch
import torch.multiprocessing as mp

from real_robot_exps.robot_interface import (
    StateSnapshot,
    SafetyViolation,
    _rotation_matrix_to_quat_wxyz,
    _rotation_matrix_to_quat_wxyz_np,
    _quat_slerp,
    _quat_wxyz_to_rotation_matrix_np,
    make_ee_target_pose,
    _FR3_JOINT_POS_LIMITS,
    _FR3_JOINT_VEL_LIMITS,
    _MAX_TORQUE_DELTA,
    _FR3_JOINT_TORQUE_LIMITS,
    _SAFETY_MARGIN_POS,
    _SAFETY_MARGIN_VEL,
    _SAFETY_MARGIN_FORCE,
)


# ============================================================================
# Spawn context — all process/queue/event/array creation uses this
# ============================================================================

_ctx = mp.get_context('spawn')


# ============================================================================
# Shared memory layout (indices into mp.Array of doubles)
# ============================================================================

_SHM_Q       = (0,   7)    # joint positions [7]
_SHM_DQ      = (7,   14)   # joint velocities [7]
_SHM_O_T_EE  = (14,  30)   # EE pose column-major [16]
_SHM_TAU_J   = (30,  37)   # joint torques [7]
_SHM_FT_EMA  = (37,  43)   # EMA-filtered F/T [6]
_SHM_JAC     = (43,  85)   # Jacobian 6x7 flat [42]
_SHM_MASS    = (85,  134)  # Mass matrix 7x7 flat [49]
_SHM_STATE_SIZE = 134

_SHM_TORQUE_SIZE = 7
_SHM_WRENCH = (7, 13)            # task_wrench [6] (compute process writes)
_SHM_TORQUE_WRENCH_SIZE = 13     # total torque_shm size


# ============================================================================
# Build StateSnapshot from shared memory
# ============================================================================

def _build_snapshot_from_shm(state_shm, device):
    """Build StateSnapshot entirely from shared memory data.

    Reads all 134 doubles from state_shm, constructs torch tensors for
    ee_pos, ee_quat, joint_pos, joint_vel, etc. Computes ee_linvel/ee_angvel
    via J @ dq. Negates ft_ema (training convention).

    ft_bias is already subtracted pre-EMA in the comm process, so the
    ft_ema values in shared memory are already bias-corrected.

    Args:
        state_shm: mp.Array of doubles, size _SHM_STATE_SIZE.
        device: Torch device string.

    Returns:
        StateSnapshot namedtuple.
    """
    q = list(state_shm[_SHM_Q[0]:_SHM_Q[1]])
    dq = list(state_shm[_SHM_DQ[0]:_SHM_DQ[1]])
    T = list(state_shm[_SHM_O_T_EE[0]:_SHM_O_T_EE[1]])
    tau_J = list(state_shm[_SHM_TAU_J[0]:_SHM_TAU_J[1]])
    ft_ema = list(state_shm[_SHM_FT_EMA[0]:_SHM_FT_EMA[1]])
    jac_flat = list(state_shm[_SHM_JAC[0]:_SHM_JAC[1]])
    mass_flat = list(state_shm[_SHM_MASS[0]:_SHM_MASS[1]])

    ee_pos = torch.tensor([T[12], T[13], T[14]], device=device, dtype=torch.float32)

    R = torch.tensor([
        [T[0], T[4], T[8]],
        [T[1], T[5], T[9]],
        [T[2], T[6], T[10]],
    ], device=device, dtype=torch.float32)
    ee_quat = _rotation_matrix_to_quat_wxyz(R)

    joint_pos = torch.tensor(q, device=device, dtype=torch.float32)
    joint_vel = torch.tensor(dq, device=device, dtype=torch.float32)
    joint_torques = torch.tensor(tau_J, device=device, dtype=torch.float32)

    jacobian = torch.tensor(jac_flat, device=device, dtype=torch.float32).reshape(7, 6).T
    mass_matrix = torch.tensor(mass_flat, device=device, dtype=torch.float32).reshape(7, 7)

    # EE velocity: J @ dq
    dq_t = joint_vel.unsqueeze(1)  # [7, 1]
    ee_vel = (jacobian @ dq_t).squeeze(1)  # [6]
    ee_linvel = ee_vel[:3]
    ee_angvel = ee_vel[3:]

    # Rotate F/T from base frame to EE/body frame, then negate.
    # Sim reads joint forces in the panda_hand LOCAL frame (rotates with robot).
    # Real robot's O_F_ext_hat_K is in the fixed base frame.
    # NE_T_EE is identity; F_T_NE (from Franka Desk) already contains
    # R_z(-45°) + T_z(0.1034) matching sim's panda_hand body frame,
    # so R from O_T_EE = F_T_NE directly represents the sim body orientation.
    # R^T rotates base -> body. Negation flips robot-on-env to env-on-robot.
    ft_base = torch.tensor(ft_ema, device=device, dtype=torch.float32)
    ft_body = torch.zeros(6, device=device, dtype=torch.float32)
    ft_body[:3] = R.T @ ft_base[:3]
    ft_body[3:6] = R.T @ ft_base[3:6]
    force_torque = -ft_body

    return StateSnapshot(
        ee_pos, ee_quat, ee_linvel, ee_angvel, force_torque,
        joint_pos, joint_vel, joint_torques, jacobian, mass_matrix,
    )


# ============================================================================
# Comm process (runs in its own process with its own GIL)
# ============================================================================

def _comm_process_fn(state_shm, torque_shm, cmd_queue, response_queue,
                     stop_torque, comm_stop, state_ready, config, ft_bias):
    """Comm process: owns robot connection, model, runs all hardware I/O.

    Module-level function. Owns the entire robot hardware lifecycle.
    Created fresh by 'spawn' — no parent state carried over.

    Args:
        state_shm: mp.Array of doubles, size _SHM_STATE_SIZE.
        torque_shm: mp.Array of doubles, size _SHM_TORQUE_SIZE.
        cmd_queue: mp.Queue for receiving commands from main.
        response_queue: mp.Queue for sending responses to main.
        stop_torque: mp.Event, main sets to exit 1kHz loop.
        comm_stop: mp.Event, set at shutdown.
        state_ready: mp.Event, set when state_shm is valid.
        config: Full config dict.
        ft_bias: Optional [6] list — constant bias subtracted from raw
                 O_F_ext_hat_K BEFORE EMA filtering. None to disable.
    """
    import os
    import sys
    import math
    import time
    import queue
    import numpy as np
    from real_robot_exps.robot_interface import (
        _rotation_matrix_to_quat_wxyz_np,
        _quat_slerp,
        _quat_wxyz_to_rotation_matrix_np,
    )

    robot_cfg = config['robot']

    try:
        # --- Init (inside process — no pickling needed) ---
        if robot_cfg.get('use_mock', False):
            import real_robot_exps.mock_pylibfranka as plf
            print("[COMM PROCESS] Using mock pylibfranka")
        else:
            import pylibfranka as plf
            print("[COMM PROCESS] Using real pylibfranka")

        robot = plf.Robot(robot_cfg['ip'])

        NE_T_EE = robot_cfg.get('NE_T_EE', [
            0.7071, -0.7071, 0.0, 0.0,
            0.7071, 0.7071, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.1034, 1.0,
        ])
        EE_T_K = robot_cfg.get('EE_T_K', [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
        robot.set_EE(NE_T_EE)
        robot.set_K(EE_T_K)
        robot.set_collision_behavior(
            [100.0] * 7, [100.0] * 7,
            [100.0] * 6, [100.0] * 6,
        )

        model = robot.load_model()
        Torques = plf.Torques
        CartesianPose = plf.CartesianPose
        JointPositions = plf.JointPositions
        ControllerMode = plf.ControllerMode

        ft_ema_alpha = robot_cfg.get('ft_ema_alpha', 0.2)
        reset_duration_sec = robot_cfg.get('reset_duration_sec', 3.0)

        print(f"[COMM PROCESS] Started (PID={os.getpid()})")
        response_queue.put(("init_done", None))

    except Exception as e:
        import traceback
        traceback.print_exc()
        response_queue.put(("init_error", str(e)))
        return

    # --- Helper: pack state into shared memory ---
    def _pack_state(state, ft_ema, jac_flat, mass_flat):
        state_shm[_SHM_Q[0]:_SHM_Q[1]] = state.q
        state_shm[_SHM_DQ[0]:_SHM_DQ[1]] = state.dq
        state_shm[_SHM_O_T_EE[0]:_SHM_O_T_EE[1]] = state.O_T_EE
        state_shm[_SHM_TAU_J[0]:_SHM_TAU_J[1]] = state.tau_J
        state_shm[_SHM_FT_EMA[0]:_SHM_FT_EMA[1]] = ft_ema
        state_shm[_SHM_JAC[0]:_SHM_JAC[1]] = jac_flat
        state_shm[_SHM_MASS[0]:_SHM_MASS[1]] = mass_flat

    # --- Main command loop ---
    try:
        while not comm_stop.is_set():
            try:
                cmd = cmd_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if cmd[0] == "reset":
                try:
                    target_pose_list = cmd[1]  # nested list from .tolist() of 4x4 array
                    target_pose_4x4 = np.array(target_pose_list)

                    ctrl = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
                    state, _ = ctrl.readOnce()
                    start_flat = np.array(state.O_T_EE)

                    start_R = np.array([
                        [start_flat[0], start_flat[4], start_flat[8]],
                        [start_flat[1], start_flat[5], start_flat[9]],
                        [start_flat[2], start_flat[6], start_flat[10]],
                    ])
                    start_t = start_flat[12:15]

                    target_R = target_pose_4x4[:3, :3]
                    target_t = target_pose_4x4[:3, 3]

                    start_q = _rotation_matrix_to_quat_wxyz_np(start_R)
                    target_q = _rotation_matrix_to_quat_wxyz_np(target_R)

                    n_steps = int(reset_duration_sec * 1000)
                    for i in range(n_steps):
                        alpha = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / n_steps))
                        interp_t = (1.0 - alpha) * start_t + alpha * target_t
                        interp_q = _quat_slerp(start_q, target_q, alpha)
                        interp_R = _quat_wxyz_to_rotation_matrix_np(interp_q)

                        interp_flat = np.array([
                            interp_R[0, 0], interp_R[1, 0], interp_R[2, 0], 0.0,
                            interp_R[0, 1], interp_R[1, 1], interp_R[2, 1], 0.0,
                            interp_R[0, 2], interp_R[1, 2], interp_R[2, 2], 0.0,
                            interp_t[0], interp_t[1], interp_t[2], 1.0,
                        ])

                        pose_cmd = CartesianPose(interp_flat.tolist())
                        if i == n_steps - 1:
                            pose_cmd.motion_finished = True
                            ctrl.writeOnce(pose_cmd)
                            break
                        ctrl.writeOnce(pose_cmd)
                        state, _ = ctrl.readOnce()

                    # Pack final state into shared memory
                    jac_flat = model.zero_jacobian(state)
                    mass_flat = model.mass(state)
                    _pack_state(state, [0.0] * 6, jac_flat, mass_flat)
                    state_ready.set()

                    robot.stop()
                    time.sleep(0.5)
                    response_queue.put(("reset_done", None))
                except Exception as e:
                    sys.stdout.write(f"[COMM PROCESS] Reset motion failed: {e}\r\n")
                    sys.stdout.flush()
                    try:
                        ctrl = None
                        robot.stop()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    response_queue.put(("error", str(e)))

            elif cmd[0] == "start_torque":
                log_trajectory = cmd[1] if len(cmd) > 1 else False

                ctrl = robot.start_torque_control()

                # Warmup (1 step)
                state, _ = ctrl.readOnce()
                ctrl.writeOnce(Torques([0.0] * 7))

                # Compute initial Jacobian and mass matrix
                jac_flat = model.zero_jacobian(state)
                mass_flat = model.mass(state)

                # Pack initial state
                ft_ema = [0.0] * 6
                _pack_state(state, ft_ema, jac_flat, mass_flat)
                state_ready.set()

                response_queue.put(("torque_started", None))
                stop_torque.clear()

                # --- 1kHz trajectory: pre-allocate numpy buffers ---
                if log_trajectory:
                    import gc as _gc
                    _TRAJ_ALLOC = 15000  # 15s at 1kHz, generous margin
                    _buf_time = np.empty(_TRAJ_ALLOC, dtype=np.float64)
                    _buf_O_T_EE = np.empty((_TRAJ_ALLOC, 16), dtype=np.float64)
                    _buf_q = np.empty((_TRAJ_ALLOC, 7), dtype=np.float64)
                    _buf_dq = np.empty((_TRAJ_ALLOC, 7), dtype=np.float64)
                    _buf_ft_raw = np.empty((_TRAJ_ALLOC, 6), dtype=np.float64)
                    _buf_ft_filt = np.empty((_TRAJ_ALLOC, 6), dtype=np.float64)
                    _buf_torques = np.empty((_TRAJ_ALLOC, 7), dtype=np.float64)
                    _buf_wrench = np.empty((_TRAJ_ALLOC, 6), dtype=np.float64)
                    _tidx = 0
                    _t0 = time.time()
                    _gc.disable()  # prevent GC pauses in 1kHz loop

                # --- 1kHz TORQUE LOOP ---
                alpha = ft_ema_alpha
                one_minus_alpha = 1.0 - alpha
                cmd_torques = [0.0] * 7  # ramped torques actually sent to robot

                try:
                    while not stop_torque.is_set() and not comm_stop.is_set():
                        state, _ = ctrl.readOnce()

                        # Read target torques from shared memory (compute process writes these)
                        target_t = list(torque_shm[0:_SHM_TORQUE_SIZE])

                        # Rate-limit toward target (_MAX_TORQUE_DELTA per 1kHz step)
                        for j in range(7):
                            delta = target_t[j] - cmd_torques[j]
                            if delta > _MAX_TORQUE_DELTA:
                                cmd_torques[j] += _MAX_TORQUE_DELTA
                            elif delta < -_MAX_TORQUE_DELTA:
                                cmd_torques[j] -= _MAX_TORQUE_DELTA
                            else:
                                cmd_torques[j] = target_t[j]

                        ctrl.writeOnce(Torques(cmd_torques))

                        # AFTER writeOnce: compute Jacobian and mass matrix
                        jac_flat = model.zero_jacobian(state)
                        mass_flat = model.mass(state)

                        # Capture raw F/T (write to numpy BEFORE bias subtraction)
                        ft = list(state.O_F_ext_hat_K)
                        if log_trajectory and _tidx < _TRAJ_ALLOC:
                            _buf_ft_raw[_tidx] = ft

                        # Subtract F/T bias from raw reading, then EMA filter
                        if ft_bias is not None:
                            for i in range(6):
                                ft[i] -= ft_bias[i]
                        for i in range(6):
                            ft_ema[i] = alpha * ft[i] + one_minus_alpha * ft_ema[i]

                        # Pack state into shared memory
                        _pack_state(state, ft_ema, jac_flat, mass_flat)

                        # Write 1kHz snapshot into pre-allocated numpy buffers
                        # (no Python list objects created — data goes straight to C doubles)
                        if log_trajectory and _tidx < _TRAJ_ALLOC:
                            _buf_time[_tidx] = (time.time() - _t0) * 1000.0
                            _buf_O_T_EE[_tidx] = state.O_T_EE
                            _buf_q[_tidx] = state.q
                            _buf_dq[_tidx] = state.dq
                            _buf_ft_filt[_tidx] = ft_ema
                            _buf_torques[_tidx] = cmd_torques
                            _buf_wrench[_tidx] = torque_shm[_SHM_WRENCH[0]:_SHM_WRENCH[1]]
                            _tidx += 1

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    response_queue.put(("error", str(e)))
                    try:
                        ctrl = None
                        robot.stop()
                        time.sleep(0.5)
                    except Exception:
                        pass
                    state_ready.clear()
                    continue

                # Stop controller FIRST — robot expects 1kHz comms until stopped
                ctrl = None
                robot.stop()
                time.sleep(0.5)
                state_ready.clear()

                # Package trajectory data (safe now — robot is stopped)
                if log_trajectory:
                    _gc.enable()  # re-enable GC now that loop is done

                    # Slice pre-allocated buffers to actual length
                    n = _tidx
                    O_T_EE_arr = _buf_O_T_EE[:n]              # [n, 16]
                    ee_pos_arr = O_T_EE_arr[:, [12, 13, 14]]   # [n, 3]
                    ee_quat_arr = np.empty((n, 4), dtype=np.float64)
                    for i in range(n):
                        T = O_T_EE_arr[i]
                        R = np.array([[T[0], T[4], T[8]],
                                      [T[1], T[5], T[9]],
                                      [T[2], T[6], T[10]]])
                        ee_quat_arr[i] = _rotation_matrix_to_quat_wxyz_np(R)

                    traj_data = {
                        'time_ms': _buf_time[:n].copy(),
                        'ee_pos': ee_pos_arr.copy(),
                        'ee_quat': ee_quat_arr,
                        'joint_pos': _buf_q[:n].copy(),
                        'joint_vel': _buf_dq[:n].copy(),
                        'ft_raw': _buf_ft_raw[:n].copy(),
                        'ft_filtered': _buf_ft_filt[:n].copy(),
                        'joint_torques_cmd': _buf_torques[:n].copy(),
                        'task_wrench': _buf_wrench[:n].copy(),
                    }
                else:
                    traj_data = None

                response_queue.put(("torque_stopped", traj_data))

            elif cmd[0] == "retract":
                try:
                    height_m = cmd[1]

                    ctrl = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
                    state, _ = ctrl.readOnce()
                    start_flat = np.array(state.O_T_EE)
                    start_t = start_flat[12:15].copy()

                    target_t = start_t.copy()
                    target_t[2] += height_m

                    n_steps = 1000
                    for i in range(n_steps):
                        alpha = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / n_steps))
                        interp_t = (1.0 - alpha) * start_t + alpha * target_t

                        interp_flat = start_flat.copy()
                        interp_flat[12] = interp_t[0]
                        interp_flat[13] = interp_t[1]
                        interp_flat[14] = interp_t[2]

                        pose_cmd = CartesianPose(interp_flat.tolist())
                        if i == n_steps - 1:
                            pose_cmd.motion_finished = True
                            ctrl.writeOnce(pose_cmd)
                            break
                        ctrl.writeOnce(pose_cmd)
                        state, _ = ctrl.readOnce()

                    robot.stop()
                    time.sleep(0.5)
                    response_queue.put(("retract_done", None))
                except Exception as e:
                    sys.stdout.write(f"[COMM PROCESS] Retract motion failed: {e}\r\n")
                    sys.stdout.flush()
                    try:
                        ctrl = None
                        robot.stop()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    response_queue.put(("error", str(e)))

            elif cmd[0] == "move_joints":
                target_q = np.array(cmd[1], dtype=np.float64)
                duration_sec = cmd[2]

                ctrl = robot.start_joint_position_control(ControllerMode.JointImpedance)
                state, _ = ctrl.readOnce()
                start_q = np.array(state.q)

                n_steps = int(duration_sec * 1000)
                for i in range(n_steps):
                    alpha = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / n_steps))
                    interp_q = (1.0 - alpha) * start_q + alpha * target_q

                    jcmd = JointPositions(interp_q.tolist())
                    if i == n_steps - 1:
                        jcmd.motion_finished = True
                        ctrl.writeOnce(jcmd)
                        break
                    ctrl.writeOnce(jcmd)
                    state, _ = ctrl.readOnce()

                robot.stop()
                time.sleep(0.5)
                response_queue.put(("move_done", None))

            elif cmd[0] == "calibrate_ft":
                try:
                    duration_sec = cmd[1]
                    n_samples = int(duration_sec * 1000)

                    ctrl = robot.start_joint_position_control(ControllerMode.JointImpedance)
                    state, _ = ctrl.readOnce()
                    q_hold = list(state.q)

                    # Settle 0.5s
                    for _ in range(500):
                        state, _ = ctrl.readOnce()
                        ctrl.writeOnce(JointPositions(q_hold))

                    # Record for configured duration
                    readings = []
                    for _ in range(n_samples):
                        state, _ = ctrl.readOnce()
                        readings.append(list(state.O_F_ext_hat_K))
                        ctrl.writeOnce(JointPositions(q_hold))

                    # Mean
                    mean_bias = [0.0] * 6
                    for r in readings:
                        for i in range(6):
                            mean_bias[i] += r[i]
                    for i in range(6):
                        mean_bias[i] /= len(readings)

                    ft_bias = mean_bias

                    jcmd = JointPositions(q_hold)
                    jcmd.motion_finished = True
                    ctrl.writeOnce(jcmd)

                    robot.stop()
                    time.sleep(0.5)
                    response_queue.put(("calibrate_ft_done", mean_bias))
                except Exception as e:
                    print(f"\n[COMM PROCESS] FT calibration failed: {e}")
                    try:
                        robot.stop()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    response_queue.put(("error", str(e)))

            elif cmd[0] == "error_recovery":
                try:
                    robot.automatic_error_recovery()
                    time.sleep(0.5)
                    response_queue.put(("error_recovery_done", None))
                except Exception as e:
                    sys.stdout.write(f"[COMM PROCESS] Error recovery failed: {e}\r\n")
                    sys.stdout.flush()
                    response_queue.put(("error", str(e)))

            elif cmd[0] == "shutdown":
                break

    except Exception as e:
        import traceback
        sys.stdout.write(f"[COMM PROCESS ERROR] {e}\r\n")
        sys.stdout.flush()
        traceback.print_exc()
        try:
            response_queue.put(("error", str(e)))
        except Exception:
            pass

    try:
        robot.stop()
    except Exception:
        pass
    print("[COMM PROCESS] Exiting")


# ============================================================================
# Compute process (runs in its own process with its own GIL)
# ============================================================================

def _compute_process_fn(state_shm, torque_shm, targets_queue,
                        active_event, stop_event, device_str):
    """Compute process entry point. Lives for the entire FrankaInterface lifetime.

    Idles when active_event is cleared, computes torques when active_event is set.
    Exits when stop_event is set (at shutdown).

    ft_bias is already subtracted pre-EMA in the comm process, so ft_ema
    values read from shared memory are already bias-corrected.

    Args:
        state_shm: mp.Array of doubles, size _SHM_STATE_SIZE.
        torque_shm: mp.Array of doubles, size _SHM_TORQUE_SIZE.
        targets_queue: mp.Queue for receiving packed ControlTargets.
        active_event: mp.Event — set = compute torques, clear = idle.
        stop_event: mp.Event — set = exit process entirely.
        device_str: Torch device string (e.g. "cpu").
    """
    import os
    import sys
    import time
    import math
    import queue as _q
    import torch
    import numpy as np
    from real_robot_exps.hybrid_controller import (
        compute_torques_from_targets, unpack_control_targets,
    )
    from real_robot_exps.robot_interface import _rotation_matrix_to_quat_wxyz

    device = device_str
    targets = None
    was_active = False
    start_time = time.time()

    print("[COMPUTE PROCESS] Started (PID={})".format(os.getpid()))

    try:
        while not stop_event.is_set():
            # --- Idle when not active ---
            if not active_event.is_set():
                was_active = False
                targets = None
                time.sleep(0.01)
                continue

            # --- Just activated: drain stale targets from previous episode ---
            if not was_active:
                while True:
                    try:
                        targets_queue.get_nowait()
                    except _q.Empty:
                        break
                targets = None
                was_active = True
                start_time = time.time()

            # Drain queue for latest targets (keep only the newest)
            new_targets = None
            while True:
                try:
                    new_targets = targets_queue.get_nowait()
                except _q.Empty:
                    break
            if new_targets is not None:
                targets = unpack_control_targets(new_targets, device)

            if targets is None:
                time.sleep(0.001)
                continue

            # Read state from shared memory
            q = list(state_shm[_SHM_Q[0]:_SHM_Q[1]])
            dq = list(state_shm[_SHM_DQ[0]:_SHM_DQ[1]])
            O_T_EE = list(state_shm[_SHM_O_T_EE[0]:_SHM_O_T_EE[1]])
            tau_J = list(state_shm[_SHM_TAU_J[0]:_SHM_TAU_J[1]])
            ft_ema = list(state_shm[_SHM_FT_EMA[0]:_SHM_FT_EMA[1]])
            jac_flat = list(state_shm[_SHM_JAC[0]:_SHM_JAC[1]])
            mass_flat = list(state_shm[_SHM_MASS[0]:_SHM_MASS[1]])

            # Build torch tensors from shared memory data
            T = O_T_EE
            ee_pos = torch.tensor(
                [T[12], T[13], T[14]], device=device, dtype=torch.float32
            )
            R = torch.tensor([
                [T[0], T[4], T[8]],
                [T[1], T[5], T[9]],
                [T[2], T[6], T[10]],
            ], device=device, dtype=torch.float32)
            ee_quat = _rotation_matrix_to_quat_wxyz(R)

            joint_pos = torch.tensor(q, device=device, dtype=torch.float32)
            joint_vel = torch.tensor(dq, device=device, dtype=torch.float32)
            jacobian = torch.tensor(
                jac_flat, device=device, dtype=torch.float32
            ).reshape(7, 6).T
            mass_matrix = torch.tensor(
                mass_flat, device=device, dtype=torch.float32
            ).reshape(7, 7)

            # EE velocity: J @ dq
            dq_t = joint_vel.unsqueeze(1)  # [7, 1]
            ee_vel = (jacobian @ dq_t).squeeze(1)  # [6]
            ee_linvel = ee_vel[:3]
            ee_angvel = ee_vel[3:]

            # Rotate F/T from base frame to EE/body frame, then negate.
            # ft_ema is already bias-corrected (subtracted pre-EMA in comm process).
            ft_base = torch.tensor(ft_ema, device=device, dtype=torch.float32)
            ft_body = torch.zeros(6, device=device, dtype=torch.float32)
            ft_body[:3] = R.T @ ft_base[:3]
            ft_body[3:6] = R.T @ ft_base[3:6]
            force_torque = -ft_body

            # Full wrench + J^T + null-space computation
            torques, task_wrench, jt_torque, null_torque = compute_torques_from_targets(
                ee_pos, ee_quat, ee_linvel, ee_angvel,
                force_torque, joint_pos, joint_vel, jacobian, mass_matrix,
                targets,
            )

            # Write torques and task wrench to shared memory (comm process reads these)
            torque_shm[0:_SHM_TORQUE_SIZE] = torques.detach().cpu().tolist()
            torque_shm[_SHM_WRENCH[0]:_SHM_WRENCH[1]] = task_wrench.detach().cpu().tolist()

    except Exception as e:
        sys.stdout.write(f"[COMPUTE PROCESS ERROR] {e}\r\n")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
    finally:
        print("[COMPUTE PROCESS] Exiting")


# ============================================================================
# FrankaInterface (process-based)
# ============================================================================

class FrankaInterface:
    """pylibfranka interface for Franka FR3 — process-based architecture.

    Both the 1kHz comm loop and the torque compute loop run in separate
    processes (each with their own Python GIL), so nothing the main thread
    does (policy inference, torch ops) can affect the 1kHz timing.

    Args:
        config: Dictionary loaded from real_robot_exps/config.yaml.
        device: Torch device for tensor outputs (default: "cpu").
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self._device = device
        self._config = config
        self._control_rate_hz = config['robot'].get('control_rate_hz', 15.0)
        self._ft_bias = config['robot'].get('ft_bias', None)

        # Shared memory via spawn context
        self._state_shm = _ctx.Array('d', _SHM_STATE_SIZE, lock=False)
        self._torque_shm = _ctx.Array('d', _SHM_TORQUE_WRENCH_SIZE, lock=False)

        # Queues via spawn context
        self._targets_queue = _ctx.Queue()   # main → compute
        self._cmd_queue = _ctx.Queue()       # main → comm
        self._response_queue = _ctx.Queue()  # comm → main

        # Events via spawn context
        self._compute_active = _ctx.Event()
        self._compute_stop = _ctx.Event()
        self._comm_stop = _ctx.Event()
        self._stop_torque = _ctx.Event()
        self._state_ready = _ctx.Event()

        self._last_send_time = None
        self._torque_mode_active = False
        self._last_trajectory = None

        # Start compute process (spawn — fresh interpreter, no fork overhead)
        self._compute_process = _ctx.Process(
            target=_compute_process_fn,
            args=(self._state_shm, self._torque_shm, self._targets_queue,
                  self._compute_active, self._compute_stop, device),
            daemon=True,
        )
        self._compute_process.start()

        # Start comm process (creates robot connection inside — spawn = clean start)
        # ft_bias is passed here so it's subtracted pre-EMA at 1kHz
        self._comm_process = _ctx.Process(
            target=_comm_process_fn,
            args=(self._state_shm, self._torque_shm, self._cmd_queue,
                  self._response_queue, self._stop_torque, self._comm_stop,
                  self._state_ready, config, self._ft_bias),
            daemon=True,
        )
        self._comm_process.start()

        # Wait for robot init to complete
        resp = self._response_queue.get(timeout=30.0)
        if resp[0] == "init_error":
            raise RuntimeError(f"Comm process init failed: {resp[1]}")
        elif resp[0] != "init_done":
            raise RuntimeError(f"Unexpected comm process response: {resp}")

        print(f"[FrankaInterface/PRO] control_rate={self._control_rate_hz}Hz, "
              f"comm process + compute process started")

    # -------------------------------------------------------------------------
    # Core methods
    # -------------------------------------------------------------------------

    def send_joint_torques(self, torques: torch.Tensor):
        """Set target [7] torques directly via shared memory.

        Writes directly to the shared torque buffer that the comm process reads.
        Use this for direct torque control (without the compute process).
        Do NOT use simultaneously with set_control_targets().

        Args:
            torques: [7] joint torques tensor.

        Raises:
            ValueError: If shape is not (7,).
        """
        if torques.shape != (7,):
            raise ValueError(f"Expected [7] torques, got {torques.shape}")
        self._torque_shm[0:_SHM_TORQUE_SIZE] = torques.detach().cpu().tolist()
        self._last_send_time = time.time()

    def set_control_targets(self, targets):
        """Set control targets for process-based torque recomputation.

        Serializes the ControlTargets and sends to the compute process via
        mp.Queue. The compute process recomputes wrenches and J^T torques
        from CURRENT robot state against these FIXED targets each cycle.

        Args:
            targets: ControlTargets namedtuple from controller.compute_action().
        """
        from real_robot_exps.hybrid_controller import pack_control_targets
        self._targets_queue.put_nowait(pack_control_targets(targets))
        self._last_send_time = time.time()

    def get_state_snapshot(self) -> StateSnapshot:
        """Build and return a StateSnapshot from shared memory.

        No longer needs pylibfranka state objects or model access. Builds
        StateSnapshot entirely from shared memory data.

        Returns:
            StateSnapshot namedtuple with all robot state fields.

        Raises:
            RuntimeError: If no state available (call reset_to_start_pose
                or start_torque_mode first).
        """
        if not self._state_ready.is_set():
            raise RuntimeError(
                "No state available. Call reset_to_start_pose() or "
                "start_torque_mode() first."
            )
        return _build_snapshot_from_shm(self._state_shm, self._device)

    def wait_for_policy_step(self):
        """Block until 1/control_rate_hz has elapsed since last send/set call."""
        if self._last_send_time is None:
            return
        target_dt = 1.0 / self._control_rate_hz
        elapsed = time.time() - self._last_send_time
        remaining = target_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def reset_to_start_pose(self, target_pose_4x4: np.ndarray):
        """Move to target pose via Cartesian control, then stop.

        Sends command to comm process which owns the robot. Blocks until
        motion completes. State is packed into shared memory by comm process.

        Args:
            target_pose_4x4: [4, 4] numpy homogeneous transform (row-major).
        """
        self._cmd_queue.put(("reset", target_pose_4x4.tolist()))
        resp = self._response_queue.get(timeout=30.0)
        if resp[0] != "reset_done":
            raise RuntimeError(f"Reset failed: {resp}")

    def start_torque_mode(self, log_trajectory: bool = False):
        """Start torque control mode with 1kHz comm loop in comm process.

        Sends command to comm process, waits for confirmation, then
        activates the compute process.

        Args:
            log_trajectory: If True, the comm process accumulates per-step
                trajectory data at 1kHz. Retrieve after end_control() via
                get_last_trajectory().
        """
        self._torque_shm[0:_SHM_TORQUE_WRENCH_SIZE] = [0.0] * _SHM_TORQUE_WRENCH_SIZE
        self._cmd_queue.put(("start_torque", log_trajectory))
        resp = self._response_queue.get(timeout=10.0)
        if resp[0] != "torque_started":
            raise RuntimeError(f"Start torque failed: {resp}")
        self._compute_active.set()
        self._last_send_time = time.time()
        self._torque_mode_active = True

    def end_control(self):
        """End the active torque control session.

        Deactivates the compute process, signals the comm process to exit
        the 1kHz loop, and waits for confirmation. Both processes stay
        alive for reuse in the next episode.

        If trajectory logging was enabled via start_torque_mode(log_trajectory=True),
        the trajectory data dict is stored and retrievable via get_last_trajectory().

        No-op if not in torque mode.
        """
        if not self._torque_mode_active:
            return
        self._torque_mode_active = False

        # Deactivate compute process (goes back to idle sleep loop)
        self._compute_active.clear()

        # Signal comm process to exit 1kHz loop
        self._stop_torque.set()
        resp = self._response_queue.get(timeout=10.0)  # increased for large trajectory data
        if resp[0] != "torque_stopped":
            raise RuntimeError(f"End control failed: {resp}")

        self._last_trajectory = resp[1]  # None or dict of numpy arrays
        self._last_send_time = None

        # Drain targets queue so stale data doesn't leak into next episode
        while True:
            try:
                self._targets_queue.get_nowait()
            except _queue.Empty:
                break

    def get_last_trajectory(self):
        """Return the 1kHz trajectory data from the last torque control session.

        Returns:
            Dict of numpy arrays if trajectory logging was enabled, else None.
            Keys: time_ms, ee_pos, ee_quat, joint_pos, joint_vel, ft_raw,
            ft_filtered, joint_torques_cmd, task_wrench.
        """
        return self._last_trajectory

    def error_recovery(self):
        """Clear Reflex/error state on the robot.

        Sends automatic_error_recovery() to the comm process's robot object.
        Call this before retrying motions after a reflex error.
        """
        self._cmd_queue.put(("error_recovery",))
        resp = self._response_queue.get(timeout=10.0)
        if resp[0] != "error_recovery_done":
            raise RuntimeError(f"Error recovery failed: {resp}")

    def retract_up(self, height_m: float):
        """Retract EE vertically upward by the specified height.

        Args:
            height_m: Distance to retract upward in meters.

        Raises:
            ValueError: If height_m <= 0.
        """
        if height_m <= 0:
            raise ValueError(f"retract height must be > 0, got {height_m}")
        self._cmd_queue.put(("retract", height_m))
        resp = self._response_queue.get(timeout=10.0)
        if resp[0] != "retract_done":
            raise RuntimeError(f"Retract failed: {resp}")

    def move_to_joint_positions(self, target_q, duration_sec: float = 3.0):
        """Move to target joint positions via joint position control.

        Args:
            target_q: [7] target joint angles (list, numpy array, or torch tensor).
            duration_sec: Duration of the motion in seconds.

        Raises:
            ValueError: If target_q is not length 7 or duration_sec <= 0.
        """
        if hasattr(target_q, 'cpu'):
            target_q = target_q.detach().cpu().numpy()
        target_q = np.asarray(target_q, dtype=np.float64)
        if target_q.shape != (7,):
            raise ValueError(f"Expected [7] joint positions, got {target_q.shape}")
        if duration_sec <= 0:
            raise ValueError(f"duration_sec must be > 0, got {duration_sec}")
        self._cmd_queue.put(("move_joints", target_q.tolist(), duration_sec))
        resp = self._response_queue.get(timeout=duration_sec + 10.0)
        if resp[0] != "move_done":
            raise RuntimeError(f"Move joints failed: {resp}")

    def calibrate_ft_bias(self) -> list:
        """Calibrate FT bias by averaging raw O_F_ext_hat_K at current pose.

        Sends calibrate_ft command to comm process. The comm process holds
        current joint positions, settles 0.5s, records FT data for the
        configured duration, and updates its internal ft_bias for subsequent
        torque control sessions.

        Must be called when robot is idle (not in torque control mode).

        Returns:
            List of 6 floats: the measured FT bias [Fx, Fy, Fz, Tx, Ty, Tz].

        Raises:
            RuntimeError: If calibration fails or robot is in torque mode.
        """
        if self._torque_mode_active:
            raise RuntimeError(
                "Cannot calibrate FT bias while in torque mode. "
                "Call end_control() first."
            )
        duration = self._config['robot']['ft_calibration_duration_sec']
        self._cmd_queue.put(("calibrate_ft", duration))
        resp = self._response_queue.get(timeout=duration + 5.0)
        if resp[0] != "calibrate_ft_done":
            raise RuntimeError(f"FT calibration failed: {resp}")
        return resp[1]

    def close_gripper(self):
        """Close gripper with configured force. Call once before episodes.

        Creates a separate pylibfranka.Gripper connection (independent of
        Robot), commands a grasp with width=0.0 and generous epsilon_outer
        so the fingers clamp down on whatever is already in the gripper.

        Raises:
            ValueError: If configured gripper_force_n exceeds 70N.
            RuntimeError: If grasp command fails.
        """
        force = self._config['robot'].get('gripper_force_n', 50.0)
        if force > 70.0:
            raise ValueError(
                f"Gripper force {force}N exceeds 70N continuous limit. "
                f"Set robot.gripper_force_n <= 70.0 in config.yaml."
            )

        robot_cfg = self._config['robot']
        if robot_cfg.get('use_mock', False):
            import real_robot_exps.mock_pylibfranka as plf
        else:
            import pylibfranka as plf

        max_attempts = 5
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                gripper = plf.Gripper(robot_cfg['ip'])
                success = gripper.grasp(
                    width=0.0, speed=0.04, force=force,
                    epsilon_inner=0.005, epsilon_outer=0.1,
                )
                if not success:
                    raise RuntimeError(
                        "Gripper grasp returned False — check that peg is in gripper"
                    )

                state = gripper.read_once()
                print(f"[FrankaInterface/PRO] Gripper closed: width={state.width:.4f}m, "
                      f"force={force}N, is_grasped={state.is_grasped}")
                return
            except Exception as e:
                last_error = e
                print(f"[FrankaInterface/PRO] Gripper attempt {attempt}/{max_attempts} "
                      f"failed: {e}")

        raise RuntimeError(
            f"Gripper grasp failed after {max_attempts} attempts. "
            f"Last error: {last_error}"
        )

    def check_safety(self, snapshot: StateSnapshot):
        """Check process health, joint pos/vel limits, and force magnitude.

        Args:
            snapshot: StateSnapshot to check.

        Raises:
            SafetyViolation: If any limit exceeded or a process died.
        """
        # Check comm process health
        if not self._comm_process.is_alive():
            raise SafetyViolation("Comm process died unexpectedly")

        # Check compute process health
        if not self._compute_process.is_alive():
            raise SafetyViolation("Compute process died unexpectedly")

        # Joint position limits
        q = snapshot.joint_pos.cpu().numpy()
        for i in range(7):
            low = _FR3_JOINT_POS_LIMITS[i, 0] * _SAFETY_MARGIN_POS
            high = _FR3_JOINT_POS_LIMITS[i, 1] * _SAFETY_MARGIN_POS
            if q[i] < low or q[i] > high:
                raise SafetyViolation(
                    f"Joint {i} position {q[i]:.4f} rad outside safe range "
                    f"[{low:.4f}, {high:.4f}]"
                )

        # Joint velocity limits
        dq = snapshot.joint_vel.cpu().numpy()
        for i in range(7):
            limit = _FR3_JOINT_VEL_LIMITS[i] * _SAFETY_MARGIN_VEL
            if abs(dq[i]) > limit:
                raise SafetyViolation(
                    f"Joint {i} velocity {dq[i]:.4f} rad/s exceeds limit {limit:.4f}"
                )

        # Force magnitude
        force_mag = torch.norm(snapshot.force_torque[:3]).item()
        if force_mag > _SAFETY_MARGIN_FORCE:
            raise SafetyViolation(
                f"Force magnitude {force_mag:.2f}N exceeds limit {_SAFETY_MARGIN_FORCE:.1f}N"
            )

    def shutdown(self):
        """End control, stop both processes, and close robot connection."""
        self.end_control()

        # Tell comm process to exit its command loop
        self._cmd_queue.put(("shutdown",))
        self._comm_stop.set()
        self._comm_process.join(timeout=5.0)
        if self._comm_process.is_alive():
            self._comm_process.terminate()

        # Stop compute process
        self._compute_stop.set()
        self._compute_process.join(timeout=2.0)
        if self._compute_process.is_alive():
            self._compute_process.terminate()

        self._comm_process = None
        self._compute_process = None
        print("[FrankaInterface/PRO] Shutdown complete.")
