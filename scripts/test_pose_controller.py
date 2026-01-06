#!/usr/bin/env python3
"""
Test script for debugging pose control contact forces.

This script:
1. Loads a single environment with fixed peg/hole positions
2. Uses pure pose control to move the peg into the hole
3. Captures detailed force, position, velocity, and control data
4. Generates diagnostic plots including FFT analysis for oscillation detection

Usage (run from project root):
    python -m scripts.test_pose_controller --prefix test1
    python -m scripts.test_pose_controller --prefix test1 --headless
    python -m scripts.test_pose_controller --prefix test1 --descent_depth 0.005
"""

import argparse
import sys

try:
    from isaaclab.app import AppLauncher
except ImportError:
    from omni.isaac.lab.app import AppLauncher

# Parse arguments before launching Isaac Sim
parser = argparse.ArgumentParser(description="Test pose controller and debug contact forces")
parser.add_argument("--config", type=str,
                    default="configs/experiments/testing/force_ctrller.yaml",
                    help="Path to configuration file")
parser.add_argument("--descent_depth", type=float, default=0.003,
                    help="How far below hole surface to push (meters, default: 0.003 = 3mm)")
parser.add_argument("--xy_offset", type=float, default=0.005,
                    help="XY offset from hole center to ensure contact (meters, default: 0.005 = 5mm)")
parser.add_argument("--hold_duration", type=float, default=3.0,
                    help="How long to hold position in hole (seconds, default: 3.0)")
parser.add_argument("--override", action="append", help="Override config values: key=value")
parser.add_argument("--prefix", type=str, required=True,
                    help="Prefix for output folder and filenames")
parser.add_argument("--no-display", action="store_true",
                    help="Skip interactive plot display (just save files)")
parser.add_argument("--force_threshold", type=float, default=10000.0,
                    help="Force threshold to highlight on plots (N, default: 10000)")

# Add Isaac Sim launcher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Enable cameras if not running headless (must be set before AppLauncher)
args_cli.enable_cameras = not args_cli.headless

# Clear sys.argv for any downstream argparse
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
print("\n[INFO]: Launching Isaac Sim...")
print(f"[INFO]: Headless mode: {args_cli.headless}, Cameras enabled: {args_cli.enable_cameras}")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[INFO]: Isaac Sim launched successfully")

# Enable PhysX contact processing
import carb
settings = carb.settings.get_settings()
settings.set("/physics/disableContactProcessing", False)

# Now import everything else (after Isaac Sim is launched)
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

from skrl.utils import set_seed

# Import Isaac Lab components
try:
    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
except ImportError:
    from omni.isaac.lab_tasks.direct.factory.factory_env import FactoryEnv

# Import local modules
from configs.config_manager_v3 import ConfigManagerV3
from wrappers.sensors.factory_env_with_sensors import create_sensor_enabled_factory_env
import learning.launch_utils_v3 as lUtils


def create_test_environment(config_path, overrides=None, headless=True):
    """Create and wrap the factory environment for testing."""
    print("\n" + "=" * 60)
    print("PHASE 0: INITIALIZATION")
    print("Goal: Load configuration and create environment")
    print("=" * 60)

    print("\n[INFO]: Loading configuration...")
    config_manager = ConfigManagerV3()
    configs = config_manager.process_config(config_path, overrides)

    # Apply asset variant if specified
    task_cfg = configs['environment'].task
    if hasattr(task_cfg, 'apply_asset_variant_if_specified'):
        if task_cfg.apply_asset_variant_if_specified():
            print("[INFO]: Asset variant applied successfully")

    # Set seed for reproducibility
    if configs['primary'].seed == -1:
        configs['primary'].seed = random.randint(0, 10000)
    print(f"[INFO]: Setting global seed: {configs['primary'].seed}")
    set_seed(configs['primary'].seed)
    configs['environment'].seed = configs['primary'].seed

    # Force single environment
    configs['environment'].scene.num_envs = 1
    configs['primary'].num_envs_per_agent = 1
    configs['primary'].agents_per_break_force = 1

    # Set max episode length to avoid early resets
    configs['environment'].episode_length_s = 1000.0

    # Handle camera configuration
    if headless:
        if hasattr(configs['environment'].scene, 'tiled_camera'):
            configs['environment'].scene.tiled_camera = None
            print("[INFO]: Camera disabled for headless mode")
    else:
        print("[INFO]: Setting up camera configuration for visualization...")
        try:
            from isaaclab.sensors import TiledCameraCfg
            import isaaclab.sim as sim_utils
        except ImportError:
            from omni.isaac.lab.sensors import TiledCameraCfg
            import omni.isaac.lab.sim as sim_utils

        configs['environment'].scene.tiled_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.35),
                rot=(-0.3535534, 0.6123724, 0.6123724, -0.3535534),
                convention="ros"
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=0.05,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 20.0)
            ),
            width=240,
            height=180,
            debug_vis=False,
        )

    print(f"[INFO]: Creating FactoryEnv with {configs['environment'].scene.num_envs} environment(s)")

    # Use sensor-enabled env if contact sensor is configured
    if (hasattr(configs['environment'].task, 'held_fixed_contact_sensor') and
        configs['wrappers'].force_torque_sensor.use_contact_sensor):
        EnvClass = create_sensor_enabled_factory_env(FactoryEnv)
        print("[INFO]: Using sensor-enabled factory environment")
    else:
        EnvClass = FactoryEnv
        print("[INFO]: Using standard factory environment")

    # Create environment
    env = EnvClass(cfg=configs['environment'], render_mode=None)
    print("[INFO]: Environment created successfully")

    # Enable camera light and set viewport camera (when not headless)
    if not headless:
        try:
            import omni.kit.viewport.utility as vp_utils
            viewport_api = vp_utils.get_active_viewport()
            if viewport_api:
                camera_path = "/World/envs/env_0/Camera"
                viewport_api.set_active_camera(camera_path)
                print(f"[INFO]: Viewport camera set to {camera_path}")

                settings.set("/rtx/useViewLightingMode", True)
                print("[INFO]: Camera light (headlight) enabled via RTX settings")
        except Exception as e:
            print(f"[WARNING]: Could not configure viewport camera: {e}")

    # Apply wrappers
    print("[INFO]: Applying wrapper stack...")
    env = lUtils.apply_wrappers(env, configs)
    print("[INFO]: Wrappers applied successfully")

    print("\n[INFO]: PHASE 0 COMPLETE - Environment initialized")

    return env, configs


def compute_position_action(env, target_x, target_y, target_z, configs):
    """
    Compute action using pure position control on all axes.

    Action format (12 dimensions):
    [sel_x, sel_y, sel_z, dx, dy, dz, drx, dry, drz, fx, fy, fz]
    """
    # Get current state
    current_pos = env.unwrapped.fingertip_midpoint_pos[0].cpu().numpy()

    # Get thresholds from config
    pos_threshold = configs['environment'].ctrl.pos_action_threshold

    # Compute position deltas
    dx = (target_x - current_pos[0]) / pos_threshold[0]
    dy = (target_y - current_pos[1]) / pos_threshold[1]
    dz = (target_z - current_pos[2]) / pos_threshold[2]

    # Clamp position actions to [-1, 1]
    dx = np.clip(dx, -1.0, 1.0)
    dy = np.clip(dy, -1.0, 1.0)
    dz = np.clip(dz, -1.0, 1.0)

    # Build action tensor
    action = torch.zeros((1, 12), device=env.device)

    # Selection matrix: all position control (0 = position, 1 = force)
    action[0, 0] = 0.0  # sel_x = position control
    action[0, 1] = 0.0  # sel_y = position control
    action[0, 2] = 0.0  # sel_z = position control

    # Position deltas
    action[0, 3] = dx
    action[0, 4] = dy
    action[0, 5] = dz

    # Rotation deltas (all zero)
    action[0, 6] = 0.0
    action[0, 7] = 0.0
    action[0, 8] = 0.0

    # Force targets (ignored when selection = 0)
    action[0, 9] = 0.0
    action[0, 10] = 0.0
    action[0, 11] = 0.0

    return action


def run_pose_control_test(env, configs, descent_depth, xy_offset, hold_duration, force_threshold):
    """
    Run the pose control test and collect comprehensive debugging data.

    Args:
        env: The wrapped environment
        configs: Configuration dictionary
        descent_depth: How far below hole surface to push (meters)
        xy_offset: XY offset from hole center to ensure contact (meters)
        hold_duration: How long to hold position (seconds)
        force_threshold: Force threshold to highlight (N)
    """
    dt = 1.0 / 60.0  # Approximate timestep

    # Reset environment
    print("\n[INFO]: Resetting environment...")
    obs, info = env.reset()

    # Get asset dimensions
    task_cfg = configs['environment'].task
    peg_height = task_cfg.held_asset_cfg.height
    hole_height = task_cfg.fixed_asset_cfg.height

    # Get control gains for logging
    prop_gains = env.unwrapped.task_prop_gains[0].cpu().numpy()
    deriv_gains = env.unwrapped.task_deriv_gains[0].cpu().numpy()

    print(f"\n[INFO]: Control Gains:")
    print(f"  Proportional: {prop_gains}")
    print(f"  Derivative:   {deriv_gains}")

    # Calculate positions
    hole_origin = env.unwrapped.fixed_pos[0].cpu().numpy()
    hole_surface_z = hole_origin[2] + hole_height
    hole_center_xy = hole_origin[:2]

    print(f"\n[INFO]: Hole surface at z={hole_surface_z:.4f}")
    print(f"[INFO]: Hole center at xy=({hole_center_xy[0]:.4f}, {hole_center_xy[1]:.4f})")

    # Get fingertip to peg tip offset
    fingertip_pos = env.unwrapped.fingertip_midpoint_pos[0].cpu().numpy()
    peg_tip_pos = env.unwrapped.held_base_pos[0].cpu().numpy()
    fingertip_to_peg_tip_offset = peg_tip_pos - fingertip_pos

    print(f"[INFO]: Fingertip to peg tip offset: {fingertip_to_peg_tip_offset}")

    # Data collection structure
    data = {
        # Time
        'time': [],
        'phase': [],  # 'approach', 'descent', 'hold'

        # Forces (6-DOF)
        'force_x': [],
        'force_y': [],
        'force_z': [],
        'torque_x': [],
        'torque_y': [],
        'torque_z': [],
        'force_magnitude': [],

        # Positions
        'fingertip_x': [],
        'fingertip_y': [],
        'fingertip_z': [],
        'peg_tip_x': [],
        'peg_tip_y': [],
        'peg_tip_z': [],
        'target_x': [],
        'target_y': [],
        'target_z': [],

        # Velocities
        'linvel_x': [],
        'linvel_y': [],
        'linvel_z': [],
        'angvel_x': [],
        'angvel_y': [],
        'angvel_z': [],
        'linvel_magnitude': [],

        # Pose errors
        'pos_error_x': [],
        'pos_error_y': [],
        'pos_error_z': [],
        'pos_error_magnitude': [],

        # Control data
        'action_dx': [],
        'action_dy': [],
        'action_dz': [],

        # Target depth (for incremental descent analysis)
        'target_depth': [],  # Target depth below hole surface (meters)

        # Metadata
        'hole_surface_z': hole_surface_z,
        'hole_center_x': hole_center_xy[0],
        'hole_center_y': hole_center_xy[1],
        'xy_offset': xy_offset,
        'descent_depth': descent_depth,
        'prop_gains': prop_gains.tolist(),
        'deriv_gains': deriv_gains.tolist(),
        'peg_height': peg_height,
        'hole_height': hole_height,
        'force_threshold': force_threshold,
    }

    step_count = 0

    def collect_data(phase, target_fingertip, action, target_depth_m=0.0):
        """Collect all data for current timestep.

        Args:
            phase: Current phase name
            target_fingertip: Target fingertip position
            action: Action tensor
            target_depth_m: Target depth below hole surface in meters
        """
        nonlocal step_count

        # Forces
        ft = env.unwrapped.robot_force_torque[0].cpu().numpy()
        force_mag = np.linalg.norm(ft[:3])

        # Positions
        fingertip = env.unwrapped.fingertip_midpoint_pos[0].cpu().numpy()
        peg_tip = env.unwrapped.held_base_pos[0].cpu().numpy()

        # Velocities
        linvel = env.unwrapped.fingertip_midpoint_linvel[0].cpu().numpy()
        angvel = env.unwrapped.fingertip_midpoint_angvel[0].cpu().numpy()
        linvel_mag = np.linalg.norm(linvel)

        # Pose error
        pos_error = target_fingertip - fingertip
        pos_error_mag = np.linalg.norm(pos_error)

        # Store data
        data['time'].append(step_count * dt)
        data['phase'].append(phase)
        data['target_depth'].append(target_depth_m)

        data['force_x'].append(ft[0])
        data['force_y'].append(ft[1])
        data['force_z'].append(ft[2])
        data['torque_x'].append(ft[3])
        data['torque_y'].append(ft[4])
        data['torque_z'].append(ft[5])
        data['force_magnitude'].append(force_mag)

        data['fingertip_x'].append(fingertip[0])
        data['fingertip_y'].append(fingertip[1])
        data['fingertip_z'].append(fingertip[2])
        data['peg_tip_x'].append(peg_tip[0])
        data['peg_tip_y'].append(peg_tip[1])
        data['peg_tip_z'].append(peg_tip[2])
        data['target_x'].append(target_fingertip[0])
        data['target_y'].append(target_fingertip[1])
        data['target_z'].append(target_fingertip[2])

        data['linvel_x'].append(linvel[0])
        data['linvel_y'].append(linvel[1])
        data['linvel_z'].append(linvel[2])
        data['angvel_x'].append(angvel[0])
        data['angvel_y'].append(angvel[1])
        data['angvel_z'].append(angvel[2])
        data['linvel_magnitude'].append(linvel_mag)

        data['pos_error_x'].append(pos_error[0])
        data['pos_error_y'].append(pos_error[1])
        data['pos_error_z'].append(pos_error[2])
        data['pos_error_magnitude'].append(pos_error_mag)

        action_np = action[0].cpu().numpy()
        data['action_dx'].append(action_np[3])
        data['action_dy'].append(action_np[4])
        data['action_dz'].append(action_np[5])

        step_count += 1

    # =========================================
    # PHASE 1: APPROACH
    # =========================================
    print("\n" + "=" * 60)
    print("PHASE 1: APPROACH")
    print(f"Goal: Move peg tip to 2mm above hole surface, offset {xy_offset*1000:.1f}mm from center")
    print("=" * 60)

    approach_height = 0.002  # 2mm above surface
    # Offset in X direction from hole center to ensure we hit the edge
    peg_tip_target = np.array([
        hole_center_xy[0] + xy_offset,  # Offset in X
        hole_center_xy[1],               # Centered in Y
        hole_surface_z + approach_height
    ])
    fingertip_target = peg_tip_target - fingertip_to_peg_tip_offset

    print(f"[INFO]: Hole center: ({hole_center_xy[0]:.4f}, {hole_center_xy[1]:.4f})")
    print(f"[INFO]: XY offset: {xy_offset*1000:.1f}mm in X direction")
    print(f"[INFO]: Peg tip target: {peg_tip_target}")
    print(f"[INFO]: Fingertip target: {fingertip_target}")

    max_approach_steps = 300
    position_threshold = 0.0005  # 0.5mm

    for step in range(max_approach_steps):
        action = compute_position_action(env, fingertip_target[0], fingertip_target[1], fingertip_target[2], configs)
        obs, reward, terminated, truncated, info = env.step(action)

        collect_data('approach', fingertip_target, action, target_depth_m=-approach_height)

        # Check if reached target
        current_peg_tip = env.unwrapped.held_base_pos[0].cpu().numpy()
        distance = np.linalg.norm(current_peg_tip - peg_tip_target)

        if step % 50 == 0:
            force_z = env.unwrapped.robot_force_torque[0, 2].cpu().item()
            print(f"  Step {step}: distance={distance*1000:.2f}mm, force_z={force_z:.2f}N")

        if distance < position_threshold:
            print(f"  Reached approach position at step {step}")
            break

    # Brief pause
    print("\n[INFO]: Pausing for 0.5s to settle...")
    pause_steps = int(0.5 / dt)
    zero_action = torch.zeros((1, 12), device=env.device)
    for _ in range(pause_steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        collect_data('approach', fingertip_target, zero_action, target_depth_m=-approach_height)

    # =========================================
    # PHASE 2: INCREMENTAL DESCENT
    # =========================================
    print("\n" + "=" * 60)
    print("PHASE 2: INCREMENTAL DESCENT")
    print("Goal: Incrementally increase target depth to observe force vs error response")
    print("  - Start at 1mm below surface, hold 5 seconds")
    print("  - Then increase by 1mm every second until 20mm")
    print(f"  - XY offset: {xy_offset*1000:.1f}mm from hole center")
    print("=" * 60)

    max_force_seen = 0

    # Initial depth: 1mm below surface
    initial_depth = 0.001  # 1mm
    max_depth = 0.020      # 20mm
    depth_increment = 0.001  # 1mm

    initial_hold_duration = 5.0  # 5 seconds at first depth
    increment_hold_duration = 1.0  # 1 second per subsequent depth

    current_depth = initial_depth

    # --- Initial descent to 1mm and hold for 5 seconds ---
    print(f"\n[DEPTH 1mm]: Setting target to {current_depth*1000:.0f}mm below surface, holding {initial_hold_duration:.0f}s...")

    peg_tip_target = np.array([
        hole_center_xy[0] + xy_offset,
        hole_center_xy[1],
        hole_surface_z - current_depth
    ])
    fingertip_target = peg_tip_target - fingertip_to_peg_tip_offset

    initial_hold_steps = int(initial_hold_duration / dt)
    for step in range(initial_hold_steps):
        action = compute_position_action(env, fingertip_target[0], fingertip_target[1], fingertip_target[2], configs)
        obs, reward, terminated, truncated, info = env.step(action)

        collect_data('descent_1mm', fingertip_target, action, target_depth_m=current_depth)

        force_mag = np.linalg.norm(env.unwrapped.robot_force_torque[0, :3].cpu().numpy())
        max_force_seen = max(max_force_seen, force_mag)

        if step % 60 == 0:  # Every ~1 second
            force_z = env.unwrapped.robot_force_torque[0, 2].cpu().item()
            current_peg_tip = env.unwrapped.held_base_pos[0].cpu().numpy()
            z_relative = (current_peg_tip[2] - hole_surface_z) * 1000
            print(f"  t={step*dt:.1f}s: target={current_depth*1000:.0f}mm, actual_z={z_relative:.2f}mm, force_z={force_z:.2f}N, force_mag={force_mag:.1f}N")

    # --- Incremental descent: 2mm to 20mm, 1 second each ---
    current_depth = initial_depth + depth_increment  # Start at 2mm

    while current_depth <= max_depth:
        depth_mm = int(current_depth * 1000)
        print(f"\n[DEPTH {depth_mm}mm]: Setting target to {depth_mm}mm below surface, holding {increment_hold_duration:.0f}s...")

        peg_tip_target = np.array([
            hole_center_xy[0] + xy_offset,
            hole_center_xy[1],
            hole_surface_z - current_depth
        ])
        fingertip_target = peg_tip_target - fingertip_to_peg_tip_offset

        increment_hold_steps = int(increment_hold_duration / dt)
        for step in range(increment_hold_steps):
            action = compute_position_action(env, fingertip_target[0], fingertip_target[1], fingertip_target[2], configs)
            obs, reward, terminated, truncated, info = env.step(action)

            phase_name = f'descent_{depth_mm}mm'
            collect_data(phase_name, fingertip_target, action, target_depth_m=current_depth)

            force_mag = np.linalg.norm(env.unwrapped.robot_force_torque[0, :3].cpu().numpy())
            max_force_seen = max(max_force_seen, force_mag)

        # Print summary for this depth
        force_z = env.unwrapped.robot_force_torque[0, 2].cpu().item()
        current_peg_tip = env.unwrapped.held_base_pos[0].cpu().numpy()
        z_relative = (current_peg_tip[2] - hole_surface_z) * 1000
        pos_error = (hole_surface_z - current_depth) - current_peg_tip[2]
        print(f"  End: target={depth_mm}mm, actual_z={z_relative:.2f}mm, pos_error={pos_error*1000:.2f}mm, force_z={force_z:.2f}N, force_mag={force_mag:.1f}N")

        current_depth += depth_increment

    # --- Final large error tests: 30mm and 40mm (COMMENTED OUT) ---
    # final_depths = [0.030, 0.040]  # 30mm and 40mm
    #
    # for final_depth in final_depths:
    #     depth_mm = int(final_depth * 1000)
    #     print(f"\n[DEPTH {depth_mm}mm]: Setting target to {depth_mm}mm below surface, holding {increment_hold_duration:.0f}s...")
    #
    #     peg_tip_target = np.array([
    #         hole_center_xy[0] + xy_offset,
    #         hole_center_xy[1],
    #         hole_surface_z - final_depth
    #     ])
    #     fingertip_target = peg_tip_target - fingertip_to_peg_tip_offset
    #
    #     increment_hold_steps = int(increment_hold_duration / dt)
    #     for step in range(increment_hold_steps):
    #         action = compute_position_action(env, fingertip_target[0], fingertip_target[1], fingertip_target[2], configs)
    #         obs, reward, terminated, truncated, info = env.step(action)
    #
    #         phase_name = f'descent_{depth_mm}mm'
    #         collect_data(phase_name, fingertip_target, action, target_depth_m=final_depth)
    #
    #         force_mag = np.linalg.norm(env.unwrapped.robot_force_torque[0, :3].cpu().numpy())
    #         max_force_seen = max(max_force_seen, force_mag)
    #
    #     # Print summary for this depth
    #     force_z = env.unwrapped.robot_force_torque[0, 2].cpu().item()
    #     current_peg_tip = env.unwrapped.held_base_pos[0].cpu().numpy()
    #     z_relative = (current_peg_tip[2] - hole_surface_z) * 1000
    #     pos_error = (hole_surface_z - final_depth) - current_peg_tip[2]
    #     print(f"  End: target={depth_mm}mm, actual_z={z_relative:.2f}mm, pos_error={pos_error*1000:.2f}mm, force_z={force_z:.2f}N, force_mag={force_mag:.1f}N")

    # Print incremental descent statistics
    print(f"\n[INFO]: Incremental Descent Complete")
    print(f"  Max force seen: {max_force_seen:.1f} N")

    # =========================================
    # PHASE 3: IN-HOLE RANDOM XY TEST
    # =========================================
    print("\n" + "=" * 60)
    print("PHASE 3: IN-HOLE RANDOM XY TEST")
    print("Goal: Insert peg into hole and apply continuous random XY perturbations")
    print("  - Step 1: Lift peg to 1mm above hole (keep current XY)")
    print("  - Step 2: Center over hole and insert to 10mm below surface")
    print("  - Step 3: Apply random XY delta every control step for 10 seconds")
    print("=" * 60)

    # Separate data collection for random XY test
    random_xy_data = {
        'time': [],
        'delta_x': [],
        'delta_y': [],
        'force_x': [],
        'force_y': [],
        'force_z': [],
        'force_magnitude': [],
        'torque_x': [],
        'torque_y': [],
        'torque_z': [],
        'pos_error_x': [],
        'pos_error_y': [],
        'pos_error_z': [],
        'fingertip_x': [],
        'fingertip_y': [],
        'fingertip_z': [],
        'target_x': [],
        'target_y': [],
        'target_z': [],
    }
    random_xy_step_count = 0

    def collect_random_xy_data(delta_x, delta_y, fingertip_target_local):
        """Collect data for random XY test phase."""
        nonlocal random_xy_step_count

        ft = env.unwrapped.robot_force_torque[0].cpu().numpy()
        force_mag = np.linalg.norm(ft[:3])
        fingertip = env.unwrapped.fingertip_midpoint_pos[0].cpu().numpy()
        pos_error = fingertip_target_local - fingertip

        random_xy_data['time'].append(random_xy_step_count * dt)
        random_xy_data['delta_x'].append(delta_x)
        random_xy_data['delta_y'].append(delta_y)
        random_xy_data['force_x'].append(ft[0])
        random_xy_data['force_y'].append(ft[1])
        random_xy_data['force_z'].append(ft[2])
        random_xy_data['force_magnitude'].append(force_mag)
        random_xy_data['torque_x'].append(ft[3])
        random_xy_data['torque_y'].append(ft[4])
        random_xy_data['torque_z'].append(ft[5])
        random_xy_data['pos_error_x'].append(pos_error[0])
        random_xy_data['pos_error_y'].append(pos_error[1])
        random_xy_data['pos_error_z'].append(pos_error[2])
        random_xy_data['fingertip_x'].append(fingertip[0])
        random_xy_data['fingertip_y'].append(fingertip[1])
        random_xy_data['fingertip_z'].append(fingertip[2])
        random_xy_data['target_x'].append(fingertip_target_local[0])
        random_xy_data['target_y'].append(fingertip_target_local[1])
        random_xy_data['target_z'].append(fingertip_target_local[2])

        random_xy_step_count += 1

    # --- Step 3.1: Lift peg to 1mm above hole, keep current XY ---
    print("\n[STEP 3.1]: Lifting peg to 1mm above hole surface...")
    current_peg_tip = env.unwrapped.held_base_pos[0].cpu().numpy()
    lift_height = 0.001  # 1mm above surface

    peg_tip_target = np.array([
        current_peg_tip[0],  # Keep current X
        current_peg_tip[1],  # Keep current Y
        hole_surface_z + lift_height
    ])
    fingertip_target = peg_tip_target - fingertip_to_peg_tip_offset

    print(f"  Current peg XY: ({current_peg_tip[0]:.4f}, {current_peg_tip[1]:.4f})")
    print(f"  Target Z: {lift_height*1000:.1f}mm above hole surface")

    lift_steps = int(2.0 / dt)  # 2 seconds to lift
    for step in range(lift_steps):
        action = compute_position_action(env, fingertip_target[0], fingertip_target[1], fingertip_target[2], configs)
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 60 == 0:
            peg_pos = env.unwrapped.held_base_pos[0].cpu().numpy()
            z_rel = (peg_pos[2] - hole_surface_z) * 1000
            print(f"  t={step*dt:.1f}s: peg_z={z_rel:.2f}mm above surface")

    # --- Step 3.2: Center over hole and insert to 10mm below surface ---
    print("\n[STEP 3.2]: Centering over hole and inserting to 10mm below surface...")
    insert_depth = 0.010  # 10mm below surface

    peg_tip_target = np.array([
        hole_center_xy[0],  # Center X
        hole_center_xy[1],  # Center Y
        hole_surface_z - insert_depth
    ])
    fingertip_target = peg_tip_target - fingertip_to_peg_tip_offset

    print(f"  Target: center=({hole_center_xy[0]:.4f}, {hole_center_xy[1]:.4f}), depth={insert_depth*1000:.0f}mm")

    insert_steps = int(3.0 / dt)  # 3 seconds to insert
    for step in range(insert_steps):
        action = compute_position_action(env, fingertip_target[0], fingertip_target[1], fingertip_target[2], configs)
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 60 == 0:
            peg_pos = env.unwrapped.held_base_pos[0].cpu().numpy()
            z_rel = (peg_pos[2] - hole_surface_z) * 1000
            xy_dist = np.sqrt((peg_pos[0] - hole_center_xy[0])**2 + (peg_pos[1] - hole_center_xy[1])**2) * 1000
            force_mag = np.linalg.norm(env.unwrapped.robot_force_torque[0, :3].cpu().numpy())
            print(f"  t={step*dt:.1f}s: z={z_rel:.2f}mm, xy_offset={xy_dist:.2f}mm, force_mag={force_mag:.1f}N")

    # --- Step 3.3: Apply random XY position deltas (continuous) ---
    print("\n[STEP 3.3]: Applying continuous random XY perturbations...")

    # Random XY test parameters
    test_duration = 10.0  # 10 seconds of continuous random perturbations
    delta_range = 0.005  # ±5mm random deltas

    # Base position (centered in hole at 10mm depth)
    base_peg_tip = np.array([
        hole_center_xy[0],
        hole_center_xy[1],
        hole_surface_z - insert_depth
    ])

    print(f"  Test duration: {test_duration:.1f}s")
    print(f"  Delta range: ±{delta_range*1000:.1f}mm")
    print(f"  New random delta every control step")

    max_force_random = 0
    total_steps = int(test_duration / dt)

    for step in range(total_steps):
        # Generate NEW random XY delta every step
        delta_x = random.uniform(-delta_range, delta_range)
        delta_y = random.uniform(-delta_range, delta_range)

        # Apply delta to base position
        peg_tip_target = np.array([
            base_peg_tip[0] + delta_x,
            base_peg_tip[1] + delta_y,
            base_peg_tip[2]  # Keep Z constant
        ])
        fingertip_target = peg_tip_target - fingertip_to_peg_tip_offset

        action = compute_position_action(env, fingertip_target[0], fingertip_target[1], fingertip_target[2], configs)
        obs, reward, terminated, truncated, info = env.step(action)

        collect_random_xy_data(delta_x, delta_y, fingertip_target)

        force_mag = np.linalg.norm(env.unwrapped.robot_force_torque[0, :3].cpu().numpy())
        max_force_random = max(max_force_random, force_mag)

        # Print progress every second
        if step % 60 == 0:
            force_z = env.unwrapped.robot_force_torque[0, 2].cpu().item()
            print(f"  t={step*dt:.1f}s: delta=({delta_x*1000:+5.1f}, {delta_y*1000:+5.1f})mm, force_z={force_z:+7.2f}N, force_mag={force_mag:.1f}N")

    print(f"\n[INFO]: Random XY Test Complete")
    print(f"  Total steps: {total_steps}")
    print(f"  Max force seen: {max_force_random:.1f} N")

    # Print aggregated statistics
    force_x_arr = np.array(random_xy_data['force_x'])
    force_y_arr = np.array(random_xy_data['force_y'])
    force_z_arr = np.array(random_xy_data['force_z'])
    force_mag_arr = np.array(random_xy_data['force_magnitude'])

    print(f"\n  Aggregated Force Statistics:")
    print(f"    Force X:   mean={np.mean(force_x_arr):+7.2f}N, std={np.std(force_x_arr):6.2f}N, range=[{np.min(force_x_arr):+7.2f}, {np.max(force_x_arr):+7.2f}]N")
    print(f"    Force Y:   mean={np.mean(force_y_arr):+7.2f}N, std={np.std(force_y_arr):6.2f}N, range=[{np.min(force_y_arr):+7.2f}, {np.max(force_y_arr):+7.2f}]N")
    print(f"    Force Z:   mean={np.mean(force_z_arr):+7.2f}N, std={np.std(force_z_arr):6.2f}N, range=[{np.min(force_z_arr):+7.2f}, {np.max(force_z_arr):+7.2f}]N")
    print(f"    Force Mag: mean={np.mean(force_mag_arr):7.2f}N, std={np.std(force_mag_arr):6.2f}N, range=[{np.min(force_mag_arr):7.2f}, {np.max(force_mag_arr):7.2f}]N")

    # Store random XY data in main data dict for later access
    data['random_xy_test'] = random_xy_data

    return data


def plot_debugging_results(data, output_dir, prefix, force_threshold):
    """Generate comprehensive debugging plots."""

    time = np.array(data['time'])
    target_depth = np.array(data['target_depth']) * 1000  # Convert to mm

    # Get phase colors - approach is green, descent phases are shades of orange/red
    def get_phase_color(phase):
        if phase == 'approach':
            return 'green'
        elif phase.startswith('descent'):
            return 'blue'
        else:
            return 'gray'

    phases = data['phase']
    colors = [get_phase_color(p) for p in phases]

    # Create phase transition markers (only major transitions)
    phase_changes = []
    current_phase = phases[0]
    for i, phase in enumerate(phases):
        # Only mark transition from approach to descent
        if phase != current_phase and current_phase == 'approach':
            phase_changes.append((i, phase))
        current_phase = phase

    # =========================================
    # Figure 1: Forces over time
    # =========================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('Contact Forces During Incremental Descent Test', fontsize=14)

    # Force X, Y, Z
    ax = axes1[0, 0]
    ax.plot(time, data['force_x'], 'r-', label='Force X', linewidth=1)
    ax.plot(time, data['force_y'], 'g-', label='Force Y', linewidth=1)
    ax.plot(time, data['force_z'], 'b-', label='Force Z', linewidth=1)
    for idx, phase in phase_changes:
        ax.axvline(x=time[idx], color='gray', linestyle=':', alpha=0.5, label='Descent start')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.set_title('Force Components (X, Y, Z)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Force magnitude
    ax = axes1[0, 1]
    force_mag = np.array(data['force_magnitude'])
    ax.plot(time, force_mag, 'purple', linewidth=1)
    for idx, phase in phase_changes:
        ax.axvline(x=time[idx], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force Magnitude (N)')
    ax.set_title('Force Magnitude (L2 Norm)')
    ax.grid(True, alpha=0.3)

    # Torques
    ax = axes1[1, 0]
    ax.plot(time, data['torque_x'], 'r-', label='Torque X', linewidth=1)
    ax.plot(time, data['torque_y'], 'g-', label='Torque Y', linewidth=1)
    ax.plot(time, data['torque_z'], 'b-', label='Torque Z', linewidth=1)
    for idx, phase in phase_changes:
        ax.axvline(x=time[idx], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Torque Components (X, Y, Z)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Force Z zoomed (just the Z component)
    ax = axes1[1, 1]
    force_z = np.array(data['force_z'])
    ax.plot(time, force_z, 'b-', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    for idx, phase in phase_changes:
        ax.axvline(x=time[idx], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force Z (N)')
    ax.set_title('Force Z (Vertical Contact Force)')
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_fz = np.mean(force_z)
    std_fz = np.std(force_z)
    ax.text(0.02, 0.98, f'Mean: {mean_fz:.2f} N\nStd: {std_fz:.2f} N',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = f"{output_dir}/{prefix}_forces.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO]: Saved forces plot to {save_path}")

    # =========================================
    # Figure 2: Position and Velocity
    # =========================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Position and Velocity During Pose Control Test', fontsize=14)

    # Peg tip Z relative to hole surface
    ax = axes2[0, 0]
    peg_z = np.array(data['peg_tip_z'])
    hole_surface_z = data['hole_surface_z']
    z_relative = (peg_z - hole_surface_z) * 1000  # mm
    ax.plot(time, z_relative, 'b-', linewidth=1)
    ax.axhline(y=0, color='r', linestyle='--', label='Hole Surface', linewidth=2)
    for idx, phase in phase_changes:
        ax.axvline(x=time[idx], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z Position (mm)')
    ax.set_title('Peg Tip Z Relative to Hole Surface')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # XY position
    ax = axes2[0, 1]
    peg_x = np.array(data['peg_tip_x']) * 1000
    peg_y = np.array(data['peg_tip_y']) * 1000
    # Color by phase
    for i in range(len(peg_x) - 1):
        color = get_phase_color(phases[i])
        ax.plot(peg_x[i:i+2], peg_y[i:i+2], color=color, linewidth=1)
    ax.scatter([peg_x[0]], [peg_y[0]], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter([peg_x[-1]], [peg_y[-1]], c='red', s=100, marker='x', label='End', zorder=5)
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.set_title('Peg Tip XY Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Linear velocity magnitude
    ax = axes2[1, 0]
    linvel_mag = np.array(data['linvel_magnitude']) * 1000  # mm/s
    ax.plot(time, linvel_mag, 'b-', linewidth=1)
    for idx, phase in phase_changes:
        ax.axvline(x=time[idx], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (mm/s)')
    ax.set_title('Linear Velocity Magnitude')
    ax.grid(True, alpha=0.3)

    # Velocity components
    ax = axes2[1, 1]
    ax.plot(time, np.array(data['linvel_x']) * 1000, 'r-', label='Vel X', linewidth=1)
    ax.plot(time, np.array(data['linvel_y']) * 1000, 'g-', label='Vel Y', linewidth=1)
    ax.plot(time, np.array(data['linvel_z']) * 1000, 'b-', label='Vel Z', linewidth=1)
    for idx, phase in phase_changes:
        ax.axvline(x=time[idx], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (mm/s)')
    ax.set_title('Velocity Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{output_dir}/{prefix}_position_velocity.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO]: Saved position/velocity plot to {save_path}")

    # =========================================
    # Figure 3: Force vs Target Depth Analysis (KEY PLOT)
    # =========================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Force vs Target Depth Analysis', fontsize=14)

    # Force Z vs Target Depth (most important plot)
    ax = axes3[0, 0]
    force_z = np.array(data['force_z'])
    ax.scatter(target_depth, force_z, c='blue', alpha=0.3, s=5)
    ax.set_xlabel('Target Depth Below Surface (mm)')
    ax.set_ylabel('Force Z (N)')
    ax.set_title('Force Z vs Target Depth')
    ax.grid(True, alpha=0.3)
    # Add mean force per depth level
    unique_depths = np.unique(target_depth)
    mean_forces = [np.mean(force_z[target_depth == d]) for d in unique_depths]
    ax.plot(unique_depths, mean_forces, 'r-o', linewidth=2, markersize=6, label='Mean Force')
    ax.legend()

    # Force Magnitude vs Target Depth
    ax = axes3[0, 1]
    force_mag = np.array(data['force_magnitude'])
    ax.scatter(target_depth, force_mag, c='purple', alpha=0.3, s=5)
    ax.set_xlabel('Target Depth Below Surface (mm)')
    ax.set_ylabel('Force Magnitude (N)')
    ax.set_title('Force Magnitude vs Target Depth')
    ax.grid(True, alpha=0.3)
    mean_mags = [np.mean(force_mag[target_depth == d]) for d in unique_depths]
    ax.plot(unique_depths, mean_mags, 'r-o', linewidth=2, markersize=6, label='Mean Force Mag')
    ax.legend()

    # Position Error vs Target Depth
    ax = axes3[1, 0]
    pos_error_z = np.array(data['pos_error_z']) * 1000  # mm
    ax.scatter(target_depth, pos_error_z, c='green', alpha=0.3, s=5)
    ax.set_xlabel('Target Depth Below Surface (mm)')
    ax.set_ylabel('Position Error Z (mm)')
    ax.set_title('Position Error Z vs Target Depth')
    ax.grid(True, alpha=0.3)
    mean_errors = [np.mean(pos_error_z[target_depth == d]) for d in unique_depths]
    ax.plot(unique_depths, mean_errors, 'r-o', linewidth=2, markersize=6, label='Mean Error')
    ax.legend()

    # Force vs Position Error scatter
    ax = axes3[1, 1]
    pos_error_mag = np.array(data['pos_error_magnitude']) * 1000  # mm
    scatter = ax.scatter(pos_error_mag, force_mag, c=target_depth, alpha=0.5, s=10, cmap='viridis')
    ax.set_xlabel('Position Error Magnitude (mm)')
    ax.set_ylabel('Force Magnitude (N)')
    ax.set_title('Force vs Position Error (colored by target depth)')
    plt.colorbar(scatter, ax=ax, label='Target Depth (mm)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{output_dir}/{prefix}_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO]: Saved analysis plot to {save_path}")

    # =========================================
    # Figure 4: Pose Error and Control
    # =========================================
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle('Pose Error and Control Analysis', fontsize=14)

    # Pose error components
    ax = axes4[0, 0]
    ax.plot(time, np.array(data['pos_error_x']) * 1000, 'r-', label='Error X', linewidth=1)
    ax.plot(time, np.array(data['pos_error_y']) * 1000, 'g-', label='Error Y', linewidth=1)
    ax.plot(time, np.array(data['pos_error_z']) * 1000, 'b-', label='Error Z', linewidth=1)
    for idx, phase in phase_changes:
        ax.axvline(x=time[idx], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (mm)')
    ax.set_title('Position Error Components (Target - Current)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pose error magnitude
    ax = axes4[0, 1]
    pos_error_mag = np.array(data['pos_error_magnitude']) * 1000
    ax.plot(time, pos_error_mag, 'purple', linewidth=1)
    for idx, phase in phase_changes:
        ax.axvline(x=time[idx], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (mm)')
    ax.set_title('Position Error Magnitude')
    ax.grid(True, alpha=0.3)

    # Action components
    ax = axes4[1, 0]
    ax.plot(time, data['action_dx'], 'r-', label='Action dX', linewidth=1)
    ax.plot(time, data['action_dy'], 'g-', label='Action dY', linewidth=1)
    ax.plot(time, data['action_dz'], 'b-', label='Action dZ', linewidth=1)
    for idx, phase in phase_changes:
        ax.axvline(x=time[idx], color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.3)
    ax.axhline(y=-1.0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Action (normalized)')
    ax.set_title('Position Control Actions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Force Z vs Position Error Z
    ax = axes4[1, 1]
    pos_error_z = np.array(data['pos_error_z']) * 1000
    force_z = np.array(data['force_z'])
    ax.scatter(pos_error_z, force_z, c=colors, alpha=0.5, s=10)
    ax.set_xlabel('Position Error Z (mm)')
    ax.set_ylabel('Force Z (N)')
    ax.set_title('Force Z vs Position Error Z')
    ax.grid(True, alpha=0.3)

    # Add correlation coefficient
    corr = np.corrcoef(pos_error_z, force_z)[0, 1]
    ax.text(0.02, 0.98, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = f"{output_dir}/{prefix}_pose_error.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO]: Saved pose error plot to {save_path}")

    # =========================================
    # Figure 5: Random XY Test Results
    # =========================================
    fig5 = None
    if 'random_xy_test' in data and len(data['random_xy_test']['time']) > 0:
        random_xy = data['random_xy_test']

        fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
        fig5.suptitle('Random XY Perturbation Test (In-Hole)', fontsize=14)

        xy_time = np.array(random_xy['time'])
        delta_x = np.array(random_xy['delta_x']) * 1000  # mm
        delta_y = np.array(random_xy['delta_y']) * 1000  # mm
        force_x = np.array(random_xy['force_x'])
        force_y = np.array(random_xy['force_y'])
        force_z = np.array(random_xy['force_z'])
        force_mag = np.array(random_xy['force_magnitude'])

        # Force XY vs Delta XY scatter
        ax = axes5[0, 0]
        scatter = ax.scatter(delta_x, force_x, c='red', alpha=0.5, s=10, label='Force X')
        ax.scatter(delta_y, force_y, c='blue', alpha=0.5, s=10, label='Force Y')
        ax.set_xlabel('XY Delta Command (mm)')
        ax.set_ylabel('Force (N)')
        ax.set_title('Force X/Y vs XY Delta Commands')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

        # Force magnitude over time
        ax = axes5[0, 1]
        ax.plot(xy_time, force_mag, 'purple', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force Magnitude (N)')
        ax.set_title('Force Magnitude During Random XY Test')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_fm = np.mean(force_mag)
        max_fm = np.max(force_mag)
        ax.text(0.02, 0.98, f'Mean: {mean_fm:.2f} N\nMax: {max_fm:.2f} N',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Force components over time
        ax = axes5[1, 0]
        ax.plot(xy_time, force_x, 'r-', label='Force X', linewidth=1)
        ax.plot(xy_time, force_y, 'g-', label='Force Y', linewidth=1)
        ax.plot(xy_time, force_z, 'b-', label='Force Z', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        ax.set_title('Force Components During Random XY Test')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Delta magnitude vs Force magnitude scatter
        ax = axes5[1, 1]
        delta_mag = np.sqrt(delta_x**2 + delta_y**2)
        ax.scatter(delta_mag, force_mag, c='purple', alpha=0.5, s=10)
        ax.set_xlabel('XY Delta Magnitude (mm)')
        ax.set_ylabel('Force Magnitude (N)')
        ax.set_title('Force Magnitude vs XY Delta Magnitude')
        ax.grid(True, alpha=0.3)

        # Add correlation
        if len(delta_mag) > 1:
            corr = np.corrcoef(delta_mag, force_mag)[0, 1]
            ax.text(0.02, 0.98, f'Correlation: {corr:.3f}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        save_path = f"{output_dir}/{prefix}_random_xy_test.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO]: Saved random XY test plot to {save_path}")

    return fig1, fig2, fig3, fig4, fig5


def main():
    """Main entry point."""
    print("=" * 60)
    print("Pose Controller Debug Test Script")
    print("=" * 60)

    # Create environment
    env, configs = create_test_environment(
        args_cli.config,
        args_cli.override,
        headless=args_cli.headless
    )

    # Create output directory
    output_dir = f"scripts/test_hybrid_results/{args_cli.prefix}_pose_debug"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO]: Output directory: {output_dir}")

    try:
        # Run the test
        data = run_pose_control_test(
            env,
            configs,
            descent_depth=args_cli.descent_depth,
            xy_offset=args_cli.xy_offset,
            hold_duration=args_cli.hold_duration,
            force_threshold=args_cli.force_threshold
        )

        # Generate plots
        print("\n[INFO]: Generating diagnostic plots...")
        plot_debugging_results(data, output_dir, args_cli.prefix, args_cli.force_threshold)

        # Save raw data
        import json

        def convert_to_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            else:
                return obj

        data_to_save = convert_to_serializable(data)
        data_path = f"{output_dir}/{args_cli.prefix}_data.json"
        with open(data_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        print(f"[INFO]: Saved raw data to {data_path}")

        # Display plots
        if not args_cli.no_display:
            print("\n[INFO]: Displaying plots...")
            plt.show()

    finally:
        print("\n[INFO]: Closing environment...")
        env.close()
        print("[INFO]: Done.")


if __name__ == "__main__":
    main()
