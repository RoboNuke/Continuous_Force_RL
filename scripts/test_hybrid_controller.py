#!/usr/bin/env python3
"""
Test script for tuning the hybrid force-position controller.

This script:
1. Loads a single environment with fixed peg/hole positions
2. Moves the peg in a circular trajectory around the hole
3. Maintains a target force in Z direction during the motion
4. Plots measured force vs target force and commanded wrench

Usage (run from project root):
    python -m scripts.test_hybrid_controller
    python -m scripts.test_hybrid_controller --headless
    python -m scripts.test_hybrid_controller --radius 0.008 --force_z 10.0 --loops 3
"""

import argparse
import sys

try:
    from isaaclab.app import AppLauncher
except ImportError:
    from omni.isaac.lab.app import AppLauncher

# Parse arguments before launching Isaac Sim
parser = argparse.ArgumentParser(description="Test hybrid force-position controller")
parser.add_argument("--config", type=str,
                    default="configs/experiments/ctrl_tuning/force_ctrller.yaml",
                    help="Path to configuration file")
parser.add_argument("--radius", type=float, default=0.006,
                    help="Circle radius in meters (default: 0.006 = 6mm)")
parser.add_argument("--force_z", type=float, default=5.0,
                    help="Target downward force magnitude in Newtons (default: 5.0, internally negated)")
parser.add_argument("--loops", type=int, default=2,
                    help="Number of loops around the hole (default: 2)")
parser.add_argument("--speed", type=float, default=0.5,
                    help="Angular velocity in radians per second (default: 0.5)")
parser.add_argument("--override", action="append", help="Override config values: key=value")

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
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

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
    """Create and wrap the factory environment for testing.

    Uses the same pattern as factory_runnerv3.py for consistency.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: INITIALIZATION")
    print("Goal: Load configuration and create environment")
    print("=" * 60)

    print("\n[INFO]: Loading configuration...")
    config_manager = ConfigManagerV3()
    configs = config_manager.process_config(config_path, overrides)

    # Set seed for reproducibility
    if configs['primary'].seed == -1:
        configs['primary'].seed = random.randint(0, 10000)
    print(f"[INFO]: Setting global seed: {configs['primary'].seed}")
    set_seed(configs['primary'].seed)
    configs['environment'].seed = configs['primary'].seed

    # Force single environment (override any config values)
    configs['environment'].scene.num_envs = 1
    configs['primary'].num_envs_per_agent = 1
    configs['primary'].agents_per_break_force = 1

    # Set max episode length to 1000 steps to avoid early resets during test
    configs['environment'].episode_length_s = 1000.0
    print(f"[INFO]: Episode length set to {configs['environment'].episode_length_s}s (prevents early resets)")

    # Handle camera configuration based on headless mode
    if headless:
        # Remove any camera config when running headless to avoid sensor errors
        if hasattr(configs['environment'].scene, 'tiled_camera'):
            configs['environment'].scene.tiled_camera = None
            print("[INFO]: Camera disabled for headless mode")
    else:
        # Setup camera configuration for visualization
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
        print("[INFO]: Camera configured: 240x180, RGB")

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

    # Apply wrappers using the standard utility function
    print("[INFO]: Applying wrapper stack...")
    env = lUtils.apply_wrappers(env, configs)
    print("[INFO]: Wrappers applied successfully")

    # Find the HybridForcePositionWrapper for accessing task_wrench
    hybrid_wrapper = find_hybrid_wrapper(env)
    if hybrid_wrapper is None:
        raise RuntimeError(
            "HybridForcePositionWrapper not found in wrapper chain. "
            "Ensure hybrid_control.enabled=true in config."
        )
    print("[INFO]: HybridForcePositionWrapper found in wrapper chain")

    print("\n[INFO]: PHASE 1 COMPLETE - Environment initialized")

    return env, configs, hybrid_wrapper


def find_hybrid_wrapper(env):
    """Traverse wrapper chain to find HybridForcePositionWrapper."""
    from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

    current = env
    while current is not None:
        if isinstance(current, HybridForcePositionWrapper):
            return current
        if hasattr(current, 'env'):
            current = current.env
        else:
            break
    return None


def generate_circular_trajectory(center_x, center_y, radius, num_loops, num_points_per_loop):
    """Generate points along a circular trajectory."""
    total_points = num_loops * num_points_per_loop
    angles = np.linspace(0, num_loops * 2 * np.pi, total_points)

    x_points = center_x + radius * np.cos(angles)
    y_points = center_y + radius * np.sin(angles)

    return x_points, y_points, angles


def compute_position_action(env, target_x, target_y, target_z, configs):
    """
    Compute action using position control on all axes (X, Y, Z).

    Action format (3DOF force mode, 12 dimensions):
    [sel_x, sel_y, sel_z, dx, dy, dz, drx, dry, drz, fx, fy, fz]
    """
    # Get current state
    current_pos = env.unwrapped.fingertip_midpoint_pos[0].cpu().numpy()

    # Get thresholds from config
    pos_threshold_xy = configs['environment'].ctrl.pos_action_threshold[0]
    pos_threshold_z = configs['environment'].ctrl.pos_action_threshold[2]

    # Compute position deltas
    dx = (target_x - current_pos[0]) / pos_threshold_xy
    dy = (target_y - current_pos[1]) / pos_threshold_xy
    dz = (target_z - current_pos[2]) / pos_threshold_z

    # Clamp position actions to [-1, 1]
    dx = np.clip(dx, -1.0, 1.0)
    dy = np.clip(dy, -1.0, 1.0)
    dz = np.clip(dz, -1.0, 1.0)

    # Build action tensor
    action = torch.zeros((1, 12), device=env.device)

    # Selection matrix: all position control
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


def compute_force_action(env, target_x, target_y, target_force_z, configs):
    """
    Compute action with position control on X,Y and force control on Z.

    Action format (3DOF force mode, 12 dimensions):
    [sel_x, sel_y, sel_z, dx, dy, dz, drx, dry, drz, fx, fy, fz]

    - Selection: 0 = position control, 1 = force control
    - Position: delta scaled by pos_threshold
    - Force: absolute targeting, action = target_force / force_bounds
    """
    # Get current state
    current_pos = env.unwrapped.fingertip_midpoint_pos[0].cpu().numpy()

    # Get thresholds/bounds from config
    pos_threshold = configs['environment'].ctrl.pos_action_threshold[0]
    force_bounds = configs['environment'].ctrl.force_action_bounds[2]

    # Compute position deltas (for X and Y - position control)
    dx = (target_x - current_pos[0]) / pos_threshold
    dy = (target_y - current_pos[1]) / pos_threshold
    dz = 0.0  # Z controlled by force

    # Clamp position actions to [-1, 1]
    dx = np.clip(dx, -1.0, 1.0)
    dy = np.clip(dy, -1.0, 1.0)

    # Compute force action (for Z - force control)
    # Absolute force targeting: target = action * force_bounds
    # So: action = target / force_bounds
    # Note: Pushing down is negative Z force in the robot's force sensing frame
    desired_force_z = -abs(target_force_z)  # Always negative = pushing down
    fz_action = desired_force_z / force_bounds
    fz_action = np.clip(fz_action, -1.0, 1.0)

    # Build action tensor
    action = torch.zeros((1, 12), device=env.device)

    # Selection matrix: position control for X,Y; force control for Z
    action[0, 0] = 0.0  # sel_x = position control
    action[0, 1] = 0.0  # sel_y = position control
    action[0, 2] = 1.0  # sel_z = force control

    # Position deltas
    action[0, 3] = dx
    action[0, 4] = dy
    action[0, 5] = dz

    # Rotation deltas (all zero - keep orientation fixed)
    action[0, 6] = 0.0
    action[0, 7] = 0.0
    action[0, 8] = 0.0

    # Force targets
    action[0, 9] = 0.0   # fx
    action[0, 10] = 0.0  # fy
    action[0, 11] = fz_action  # fz

    return action


def print_progress_bar(current_degrees, total_degrees, bar_length=40):
    """Print a progress bar showing degrees traversed."""
    progress = current_degrees / total_degrees
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\r  Progress: [{bar}] {current_degrees:.1f}°/{total_degrees:.1f}° ({progress*100:.1f}%)", end="", flush=True)


def run_trajectory_test(env, configs, hybrid_wrapper, radius, target_force_z, num_loops, speed):
    """
    Run the circular trajectory test and collect data.

    Args:
        env: The wrapped environment
        configs: Configuration dictionary
        hybrid_wrapper: Reference to HybridForcePositionWrapper for accessing task_wrench
        radius: Circle radius in meters
        target_force_z: Target force in Z direction (Newtons)
        num_loops: Number of loops around the hole
        speed: Angular velocity (rad/s)
    """
    # Reset environment
    print("\n[INFO]: Resetting environment...")
    obs, info = env.reset()

    # Debug: Print all relevant coordinate frames
    print("\n[DEBUG]: Coordinate Frame Analysis:")
    print(f"  fixed_pos (hole origin):        {env.unwrapped.fixed_pos[0].cpu().numpy()}")
    print(f"  fixed_pos_obs_frame (hole top): {env.unwrapped.fixed_pos_obs_frame[0].cpu().numpy()}")
    print(f"  fixed_pos_action_frame:         {env.unwrapped.fixed_pos_action_frame[0].cpu().numpy()}")
    print(f"  scene.env_origins:              {env.unwrapped.scene.env_origins[0].cpu().numpy()}")

    # Get hole surface position (top of the hole, not the origin)
    hole_surface_pos = env.unwrapped.fixed_pos_obs_frame[0].cpu().numpy()
    print(f"\n[INFO]: Hole surface position: x={hole_surface_pos[0]:.4f}, y={hole_surface_pos[1]:.4f}, z={hole_surface_pos[2]:.4f}")

    # Get peg tip position (bottom of peg)
    peg_tip_pos = env.unwrapped.held_base_pos[0].cpu().numpy()
    print(f"[INFO]: Initial peg tip position: x={peg_tip_pos[0]:.4f}, y={peg_tip_pos[1]:.4f}, z={peg_tip_pos[2]:.4f}")

    # Calculate offset from fingertip to peg tip (this offset is used to convert peg tip targets to fingertip targets)
    fingertip_pos = env.unwrapped.fingertip_midpoint_pos[0].cpu().numpy()
    fingertip_to_peg_tip_offset = peg_tip_pos - fingertip_pos
    print(f"[INFO]: Fingertip position: x={fingertip_pos[0]:.4f}, y={fingertip_pos[1]:.4f}, z={fingertip_pos[2]:.4f}")
    print(f"[INFO]: Fingertip to peg tip offset: dx={fingertip_to_peg_tip_offset[0]*1000:.2f}mm, dy={fingertip_to_peg_tip_offset[1]*1000:.2f}mm, dz={fingertip_to_peg_tip_offset[2]*1000:.2f}mm")

    # Check position bounds
    if hasattr(env, 'pos_bounds'):
        print(f"[INFO]: Position bounds: {env.pos_bounds[0].cpu().numpy()}")

    # Get current end-effector position
    ee_pos = env.unwrapped.fingertip_midpoint_pos[0].cpu().numpy()
    print(f"[INFO]: Initial EE position: x={ee_pos[0]:.4f}, y={ee_pos[1]:.4f}, z={ee_pos[2]:.4f}")

    # Calculate trajectory parameters
    dt = 1.0 / 60.0  # Approximate timestep (60 Hz control rate)
    points_per_loop = int(2 * np.pi / (speed * dt))
    total_points = num_loops * points_per_loop
    total_degrees = num_loops * 360.0

    print(f"\n[INFO]: Trajectory parameters:")
    print(f"  - Radius: {radius*1000:.1f} mm")
    print(f"  - Target force Z: {target_force_z:.1f} N")
    print(f"  - Number of loops: {num_loops}")
    print(f"  - Angular speed: {speed:.2f} rad/s")
    print(f"  - Points per loop: {points_per_loop}")
    print(f"  - Total steps: {total_points}")

    # Generate trajectory (centered on hole surface)
    x_traj, y_traj, angles = generate_circular_trajectory(
        hole_surface_pos[0], hole_surface_pos[1], radius, num_loops, points_per_loop
    )

    # Data collection
    data = {
        'time': [],
        'measured_force_z': [],
        'target_force_z': [],
        'commanded_wrench_z': [],
        'position_x': [],
        'position_y': [],
        'position_z': [],
        'target_x': [],
        'target_y': [],
        'angle': [],
        'hole_surface_z': hole_surface_pos[2]  # Store for plotting
    }

    # =========================================
    # PHASE 2: APPROACHING
    # =========================================
    # Target for PEG TIP: circle start position, 1mm above hole surface
    peg_tip_target_x, peg_tip_target_y = x_traj[0], y_traj[0]
    approach_height = 0.001  # 1mm above hole surface
    peg_tip_target_z = hole_surface_pos[2] + approach_height

    # Convert peg tip target to fingertip target (what we actually control)
    # fingertip_target = peg_tip_target - fingertip_to_peg_tip_offset
    fingertip_target_x = peg_tip_target_x - fingertip_to_peg_tip_offset[0]
    fingertip_target_y = peg_tip_target_y - fingertip_to_peg_tip_offset[1]
    fingertip_target_z = peg_tip_target_z - fingertip_to_peg_tip_offset[2]

    position_threshold = 0.001  # Switch to force control when within 1mm

    print("\n" + "=" * 60)
    print("PHASE 2: APPROACHING")
    print(f"Goal: Move peg tip to start position and establish {target_force_z:.1f}N contact")
    print(f"Peg Tip Target: x={peg_tip_target_x:.4f}, y={peg_tip_target_y:.4f}, z={peg_tip_target_z:.4f}")
    print(f"  (This is {approach_height*1000:.1f}mm above hole surface at z={hole_surface_pos[2]:.4f})")
    print(f"Fingertip Target: x={fingertip_target_x:.4f}, y={fingertip_target_y:.4f}, z={fingertip_target_z:.4f}")
    print(f"(Hole center at: x={hole_surface_pos[0]:.4f}, y={hole_surface_pos[1]:.4f}, offset by {radius*1000:.1f}mm)")
    print("=" * 60)

    # Save current gains (in case we want to restore later)
    original_prop_gains = env.unwrapped.task_prop_gains.clone()
    original_deriv_gains = env.unwrapped.task_deriv_gains.clone()

    # Scale proportional gains to 90% to reduce overshoot
    scaled_prop_gains = original_prop_gains * 1.0
    #scaled_deriv_gains = 2 * torch.sqrt(scaled_prop_gains)

    # Apply scaled gains
    env.unwrapped.task_prop_gains = scaled_prop_gains
    #env.unwrapped.task_deriv_gains = scaled_deriv_gains

    print(f"[INFO]: Using scaled gains for approach (90% of default):")
    print(f"  Original prop gains: {original_prop_gains[0].tolist()}")
    print(f"  Scaled prop gains:   {scaled_prop_gains[0].tolist()}")
    #print(f"  Scaled deriv gains:  {scaled_deriv_gains[0].tolist()}")

    # -----------------------------------------
    # Sub-phase 2a: Position control approach
    # -----------------------------------------
    print(f"\n[SUB-PHASE 2a]: Position control to {approach_height*1000:.1f}mm above hole")
    print(f"  Peg tip target: x={peg_tip_target_x:.4f}, y={peg_tip_target_y:.4f}, z={peg_tip_target_z:.4f}")

    max_position_steps = 200
    velocity_threshold = 0.0005  # 5 mm/s

    for step in range(max_position_steps):
        # Control fingertip position to achieve desired peg tip position
        action = compute_position_action(env, fingertip_target_x, fingertip_target_y, fingertip_target_z, configs)
        obs, reward, terminated, truncated, info = env.step(action)

        # Check peg tip position (what we actually care about)
        current_peg_tip = env.unwrapped.held_base_pos[0].cpu().numpy()
        distance = np.sqrt((current_peg_tip[0] - peg_tip_target_x)**2 +
                          (current_peg_tip[1] - peg_tip_target_y)**2 +
                          (current_peg_tip[2] - peg_tip_target_z)**2)

        # Check velocity
        current_vel = env.unwrapped.fingertip_midpoint_linvel[0].cpu().numpy()
        vel_magnitude = np.sqrt(np.sum(current_vel**2))

        if step % 20 == 0:
            print(f"  Step {step}: Peg tip=({current_peg_tip[0]:.4f}, {current_peg_tip[1]:.4f}, {current_peg_tip[2]:.4f}), "
                  f"Distance={distance*1000:.2f}mm, Vel={vel_magnitude*1000:.1f}mm/s")

        if distance < position_threshold and vel_magnitude < velocity_threshold:
            print(f"  Reached position target at step {step} (distance={distance*1000:.2f}mm, vel={vel_magnitude*1000:.1f}mm/s)")
            break

    # Print position approach results
    current_peg_tip = env.unwrapped.held_base_pos[0].cpu().numpy()
    current_fingertip = env.unwrapped.fingertip_midpoint_pos[0].cpu().numpy()
    print(f"\n[SUB-PHASE 2a Positioning Complete]: Position Approach Results:")
    print(f"  Peg Tip Target:   x={peg_tip_target_x:.4f}, y={peg_tip_target_y:.4f}, z={peg_tip_target_z:.4f}")
    print(f"  Peg Tip Achieved: x={current_peg_tip[0]:.4f}, y={current_peg_tip[1]:.4f}, z={current_peg_tip[2]:.4f}")
    print(f"  Peg Tip Error:    dx={(current_peg_tip[0]-peg_tip_target_x)*1000:.2f}mm, "
          f"dy={(current_peg_tip[1]-peg_tip_target_y)*1000:.2f}mm, "
          f"dz={(current_peg_tip[2]-peg_tip_target_z)*1000:.2f}mm")

    # -----------------------------------------
    # 1 second pause with zero actions
    # -----------------------------------------
    print(f"\n[PAUSE]: Sending zero actions for 1 seconds to observe settling behavior...")
    pause_duration = 2.0/15.0  # seconds
    pause_steps = int(pause_duration / dt)
    zero_action = torch.zeros((1, 12), device=env.device)

    for step in range(pause_steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)

        if step % 120 == 0:  # Print every ~2 seconds
            current_peg_tip = env.unwrapped.held_base_pos[0].cpu().numpy()
            current_force = env.unwrapped.robot_force_torque[0, 2].cpu().item()
            elapsed = step * dt
            print(f"  t={elapsed:.1f}s: Peg tip=({current_peg_tip[0]:.4f}, {current_peg_tip[1]:.4f}, {current_peg_tip[2]:.4f}), "
                  f"Force Z={current_force:.2f}N")

    print(f"[PAUSE COMPLETE]: 1 second observation period finished")

    # Print position approach results
    current_peg_tip = env.unwrapped.held_base_pos[0].cpu().numpy()
    current_fingertip = env.unwrapped.fingertip_midpoint_pos[0].cpu().numpy()
    print(f"\n[SUB-PHASE 2a COMPLETE]: Position Approach Results:")
    print(f"  Peg Tip Target:   x={peg_tip_target_x:.4f}, y={peg_tip_target_y:.4f}, z={peg_tip_target_z:.4f}")
    print(f"  Peg Tip Achieved: x={current_peg_tip[0]:.4f}, y={current_peg_tip[1]:.4f}, z={current_peg_tip[2]:.4f}")
    print(f"  Peg Tip Error:    dx={(current_peg_tip[0]-peg_tip_target_x)*1000:.2f}mm, "
          f"dy={(current_peg_tip[1]-peg_tip_target_y)*1000:.2f}mm, "
          f"dz={(current_peg_tip[2]-peg_tip_target_z)*1000:.2f}mm")
    # -----------------------------------------
    # Sub-phase 2b: Force control contact
    # -----------------------------------------
    print(f"\n[SUB-PHASE 2b]: Force control to establish {target_force_z:.1f}N contact")

    # Debug: Print initial state before force control
    initial_force = env.unwrapped.robot_force_torque[0].cpu().numpy()
    print(f"[DEBUG 2b START]: Measured force = {initial_force[:3]}")
    actual_target_force = -abs(target_force_z)
    print(f"[DEBUG 2b START]: Target force Z = {actual_target_force:.1f}N")
    print(f"[DEBUG 2b START]: Force error = {actual_target_force - initial_force[2]:.2f}N")

    num_force_steps = 100
    for step in range(num_force_steps):
        # Use fingertip X/Y targets with force control on Z
        action = compute_force_action(env, fingertip_target_x, fingertip_target_y, target_force_z, configs)
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 20 == 0:
            current_peg_tip = env.unwrapped.held_base_pos[0].cpu().numpy()
            current_force = env.unwrapped.robot_force_torque[0, 2].cpu().item()
            print(f"  Step {step}/{num_force_steps}: "
                  f"Peg tip=({current_peg_tip[0]:.4f}, {current_peg_tip[1]:.4f}, {current_peg_tip[2]:.4f}), "
                  f"Force Z={current_force:.2f}N (target: {actual_target_force:.1f}N)")

    # Print force control results
    final_peg_tip = env.unwrapped.held_base_pos[0].cpu().numpy()
    current_force = env.unwrapped.robot_force_torque[0, 2].cpu().item()
    pos_error_x = (final_peg_tip[0] - peg_tip_target_x) * 1000  # Convert to mm
    pos_error_y = (final_peg_tip[1] - peg_tip_target_y) * 1000

    print(f"\n[SUB-PHASE 2b COMPLETE]: Force Contact Results:")
    print(f"  Peg Tip Position: x={final_peg_tip[0]:.4f}, y={final_peg_tip[1]:.4f}, z={final_peg_tip[2]:.4f}")
    print(f"  Peg Tip XY Error: dx={pos_error_x:.2f}mm, dy={pos_error_y:.2f}mm")
    print(f"  Force Z:  {current_force:.2f}N (target: {actual_target_force:.1f}N)")

    # -----------------------------------------
    # Sub-phase 2c: Stationary Force Control Test
    # -----------------------------------------
    print(f"\n[SUB-PHASE 2c]: Stationary force control test (no XY motion)")
    print(f"  Holding position while maintaining {target_force_z:.1f}N force for 5 seconds")

    stationary_duration = 5.0  # seconds
    stationary_steps = int(stationary_duration / dt)

    # Data collection for stationary phase
    stationary_data = {
        'time': [],
        'measured_force_z': [],
        'target_force_z': [],
        'commanded_wrench_z': [],
        'position_z': [],
        'hole_surface_z': hole_surface_pos[2]
    }

    # Get current position to hold
    hold_fingertip_x = env.unwrapped.fingertip_midpoint_pos[0, 0].cpu().item()
    hold_fingertip_y = env.unwrapped.fingertip_midpoint_pos[0, 1].cpu().item()

    for step in range(stationary_steps):
        # Force control in Z, position control in XY at CURRENT position (no motion)
        action = compute_force_action(env, hold_fingertip_x, hold_fingertip_y, target_force_z, configs)
        obs, reward, terminated, truncated, info = env.step(action)

        # Collect data
        current_peg_tip = env.unwrapped.held_base_pos[0].cpu().numpy()
        measured_force_z = env.unwrapped.robot_force_torque[0, 2].cpu().item()
        commanded_wrench_z = hybrid_wrapper.task_wrench[0, 2].cpu().item()

        stationary_data['time'].append(step * dt)
        stationary_data['measured_force_z'].append(measured_force_z)
        stationary_data['target_force_z'].append(-abs(target_force_z))
        stationary_data['commanded_wrench_z'].append(commanded_wrench_z)
        stationary_data['position_z'].append(current_peg_tip[2])

        if step % 60 == 0:  # Print every ~1 second
            print(f"  t={step*dt:.1f}s: Force Z={measured_force_z:.2f}N, "
                  f"Z rel={((current_peg_tip[2] - hole_surface_pos[2])*1000):.2f}mm")

    # Print stationary phase statistics
    stationary_forces = np.array(stationary_data['measured_force_z'])
    print(f"\n[SUB-PHASE 2c COMPLETE]: Stationary Force Control Results:")
    print(f"  Mean Force Z: {np.mean(stationary_forces):.3f} N (target: {-abs(target_force_z):.1f} N)")
    print(f"  Std Force Z:  {np.std(stationary_forces):.3f} N")
    print(f"  Min Force Z:  {np.min(stationary_forces):.3f} N")
    print(f"  Max Force Z:  {np.max(stationary_forces):.3f} N")

    # Plot stationary phase results
    plot_stationary_phase(stationary_data, target_force_z)

    # # Restore original gains for circle traversal
    env.unwrapped.task_prop_gains = original_prop_gains
    env.unwrapped.task_deriv_gains = original_deriv_gains
    print(f"\n[INFO]: Restored original gains for circle traversal")
    print(f"  Final Prop gains: {env.unwrapped.task_prop_gains[0].tolist()}")

    print(f"\n[INFO]: PHASE 2 COMPLETE")

    # =========================================
    # PHASE 3: CIRCLE TRAVERSAL
    # =========================================
    print("\n" + "=" * 60)
    print("PHASE 3: CIRCLE TRAVERSAL")
    print(f"Goal: Complete {num_loops} loops while maintaining {target_force_z:.1f}N force")
    print("=" * 60 + "\n")

    # Pre-compute fingertip trajectory from peg tip trajectory
    # fingertip_target = peg_tip_target - fingertip_to_peg_tip_offset
    x_traj_fingertip = x_traj - fingertip_to_peg_tip_offset[0]
    y_traj_fingertip = y_traj - fingertip_to_peg_tip_offset[1]

    # Main trajectory loop
    for step, (peg_target_x, peg_target_y, fingertip_x, fingertip_y, angle) in enumerate(
            zip(x_traj, y_traj, x_traj_fingertip, y_traj_fingertip, angles)):
        # Compute and apply action using fingertip targets
        action = compute_force_action(env, fingertip_x, fingertip_y, target_force_z, configs)
        obs, reward, terminated, truncated, info = env.step(action)

        # Collect data - track peg tip position (what we actually care about)
        current_peg_tip = env.unwrapped.held_base_pos[0].cpu().numpy()
        measured_force_z = env.unwrapped.robot_force_torque[0, 2].cpu().item()
        commanded_wrench_z = hybrid_wrapper.task_wrench[0, 2].cpu().item()

        data['time'].append(step * dt)
        data['measured_force_z'].append(measured_force_z)
        data['target_force_z'].append(-abs(target_force_z))  # Negative for downward contact
        data['commanded_wrench_z'].append(commanded_wrench_z)
        data['position_x'].append(current_peg_tip[0])
        data['position_y'].append(current_peg_tip[1])
        data['position_z'].append(current_peg_tip[2])
        data['target_x'].append(peg_target_x)
        data['target_y'].append(peg_target_y)
        data['angle'].append(angle)

        # Progress bar update
        current_degrees = np.degrees(angle)
        print_progress_bar(current_degrees, total_degrees)

        if terminated or truncated:
            print(f"\n[WARNING]: Episode terminated at step {step}")
            break

    # Final newline after progress bar
    print()
    print(f"\n[INFO]: PHASE 3 COMPLETE - Collected {len(data['time'])} data points")

    # Print summary statistics
    measured_forces = np.array(data['measured_force_z'])
    print(f"\n[INFO]: Force Statistics:")
    print(f"  - Mean Force Z: {np.mean(measured_forces):.3f} N (target: {-abs(target_force_z):.1f} N)")
    print(f"  - Std Force Z:  {np.std(measured_forces):.3f} N")
    print(f"  - Min Force Z:  {np.min(measured_forces):.3f} N")
    print(f"  - Max Force Z:  {np.max(measured_forces):.3f} N")

    return data


def plot_stationary_phase(data, target_force_z):
    """Generate plots for the stationary force control test phase."""
    time = np.array(data['time'])
    measured_force = np.array(data['measured_force_z'])
    target_force = np.array(data['target_force_z'])
    commanded_wrench = np.array(data['commanded_wrench_z'])
    pos_z = np.array(data['position_z'])
    hole_surface_z = data['hole_surface_z']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Phase 2c: Stationary Force Control Test (No XY Motion)', fontsize=14)

    # Plot 1: Z Position relative to hole surface
    ax1 = axes[0, 0]
    z_relative = (pos_z - hole_surface_z) * 1000  # Convert to mm
    ax1.plot(time, z_relative, 'b-', linewidth=1.5)
    ax1.axhline(y=0, color='r', linestyle='--', label='Hole Surface (z=0)', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Z Position (mm)')
    ax1.set_title('Peg Tip Z Position Relative to Hole Surface')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add Z statistics
    mean_z = np.mean(z_relative)
    std_z = np.std(z_relative)
    ax1.text(0.02, 0.98, f'Mean: {mean_z:.3f} mm\nStd: {std_z:.3f} mm',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Measured Force Z vs Target
    ax2 = axes[0, 1]
    ax2.plot(time, measured_force, 'b-', label='Measured Force Z', linewidth=1.5)
    actual_target = -abs(target_force_z)
    ax2.axhline(y=actual_target, color='r', linestyle='--', label=f'Target ({actual_target:.1f} N)', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Force (N)')
    ax2.set_title('Measured Force Z vs Target Force')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add force statistics
    mean_f = np.mean(measured_force)
    std_f = np.std(measured_force)
    ax2.text(0.02, 0.98, f'Mean: {mean_f:.3f} N\nStd: {std_f:.3f} N',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Force Error
    ax3 = axes[1, 0]
    force_error = measured_force - target_force
    ax3.plot(time, force_error, 'r-', linewidth=1.5)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.fill_between(time, force_error, alpha=0.3, color='red')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Force Error (N)')
    ax3.set_title('Force Tracking Error (Measured - Target)')
    ax3.grid(True, alpha=0.3)

    # Add error statistics
    rms_error = np.sqrt(np.mean(force_error**2))
    mean_error = np.mean(force_error)
    ax3.text(0.02, 0.98, f'RMS: {rms_error:.3f} N\nMean: {mean_error:.3f} N',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Commanded Wrench vs Measured Force
    ax4 = axes[1, 1]
    ax4.plot(time, commanded_wrench, 'g-', label='Commanded Wrench Z', linewidth=1.5)
    ax4.plot(time, measured_force, 'b-', label='Measured Force Z', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Force / Wrench (N)')
    ax4.set_title('Commanded Wrench Z vs Measured Force Z')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    print("\n[INFO]: Displaying stationary phase plots...")
    plt.show()


def plot_results(data, target_force_z):
    """Generate plots of the test results."""
    time = np.array(data['time'])
    measured_force = np.array(data['measured_force_z'])
    target_force = np.array(data['target_force_z'])
    commanded_wrench = np.array(data['commanded_wrench_z'])
    pos_x = np.array(data['position_x'])
    pos_y = np.array(data['position_y'])
    pos_z = np.array(data['position_z'])
    target_x = np.array(data['target_x'])
    target_y = np.array(data['target_y'])
    hole_surface_z = data['hole_surface_z']

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Plot 1: Measured Force Z vs Target Force Z
    ax1 = axes[0, 0]
    ax1.plot(time, measured_force, 'b-', label='Measured Force Z', linewidth=1.5)
    actual_target = -abs(target_force_z)
    ax1.axhline(y=actual_target, color='r', linestyle='--', label=f'Target ({actual_target:.1f} N)', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Force (N)')
    ax1.set_title('Measured Force Z vs Target Force')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Measured Force Z vs Commanded Wrench Z
    ax2 = axes[0, 1]
    ax2.plot(time, measured_force, 'b-', label='Measured Force Z', linewidth=1.5)
    ax2.plot(time, commanded_wrench, 'g-', label='Commanded Wrench Z', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Force / Wrench (N)')
    ax2.set_title('Measured Force Z vs Commanded Wrench Z')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: X/Y Trajectory (top-down view) - Peg Tip Position
    ax3 = axes[1, 0]
    ax3.plot(pos_x * 1000, pos_y * 1000, 'b-', label='Actual Peg Tip Path', linewidth=1.5)
    ax3.plot(target_x * 1000, target_y * 1000, 'r--', label='Target Peg Tip Path', linewidth=1, alpha=0.5)
    ax3.scatter([pos_x[0] * 1000], [pos_y[0] * 1000], c='green', s=100, marker='o', label='Start', zorder=5)
    ax3.scatter([pos_x[-1] * 1000], [pos_y[-1] * 1000], c='red', s=100, marker='x', label='End', zorder=5)
    ax3.set_xlabel('Peg Tip X Position (mm)')
    ax3.set_ylabel('Peg Tip Y Position (mm)')
    ax3.set_title('Peg Tip Trajectory (Top-Down View)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    # Plot 4: Force tracking error
    ax4 = axes[1, 1]
    force_error = measured_force - target_force
    ax4.plot(time, force_error, 'r-', linewidth=1.5)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax4.fill_between(time, force_error, alpha=0.3, color='red')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Force Error (N)')
    ax4.set_title('Force Tracking Error (Measured - Target)')
    ax4.grid(True, alpha=0.3)

    # Add statistics
    rms_error = np.sqrt(np.mean(force_error**2))
    max_error = np.max(np.abs(force_error))
    mean_error = np.mean(force_error)
    ax4.text(0.02, 0.98, f'RMS Error: {rms_error:.3f} N\nMax Error: {max_error:.3f} N\nMean Error: {mean_error:.3f} N',
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 5: Z Position relative to hole surface
    ax5 = axes[2, 0]
    z_relative = (pos_z - hole_surface_z) * 1000  # Convert to mm
    ax5.plot(time, z_relative, 'b-', label='Peg Tip Z (relative to hole surface)', linewidth=1.5)
    ax5.axhline(y=0, color='r', linestyle='--', label='Hole Surface (z=0)', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Z Position (mm)')
    ax5.set_title('Peg Tip Z Position Relative to Hole Surface')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Add statistics for Z
    mean_z = np.mean(z_relative)
    min_z = np.min(z_relative)
    max_z = np.max(z_relative)
    ax5.text(0.02, 0.98, f'Mean: {mean_z:.2f} mm\nMin: {min_z:.2f} mm\nMax: {max_z:.2f} mm',
             transform=ax5.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 6: Z Position and Force Z combined (dual axis)
    ax6 = axes[2, 1]
    ax6.plot(time, z_relative, 'b-', label='Z Position (mm)', linewidth=1.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Z Position (mm)', color='b')
    ax6.tick_params(axis='y', labelcolor='b')
    ax6.axhline(y=0, color='b', linestyle=':', alpha=0.5)

    ax6_twin = ax6.twinx()
    ax6_twin.plot(time, measured_force, 'r-', label='Force Z (N)', linewidth=1.5, alpha=0.7)
    ax6_twin.set_ylabel('Force Z (N)', color='r')
    ax6_twin.tick_params(axis='y', labelcolor='r')

    ax6.set_title('Z Position and Force Z (Correlation)')
    ax6.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.suptitle('Hybrid Force Controller Tuning Test', fontsize=14, y=1.02)

    print("\n[INFO]: Displaying plots...")
    plt.show()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Hybrid Force Controller Test Script")
    print("=" * 60)

    # Create environment (pass headless flag for camera setup)
    env, configs, hybrid_wrapper = create_test_environment(
        args_cli.config,
        args_cli.override,
        headless=args_cli.headless
    )

    try:
        # Run the trajectory test
        data = run_trajectory_test(
            env,
            configs,
            hybrid_wrapper,
            radius=args_cli.radius,
            target_force_z=args_cli.force_z,
            num_loops=args_cli.loops,
            speed=args_cli.speed
        )

        # Plot results
        plot_results(data, args_cli.force_z)

    finally:
        print("\n[INFO]: Closing environment...")
        env.close()
        print("[INFO]: Done.")


if __name__ == "__main__":
    main()
