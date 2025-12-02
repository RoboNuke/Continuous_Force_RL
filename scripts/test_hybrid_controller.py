#!/usr/bin/env python3
"""
Test script for tuning the hybrid force-position controller.

This script:
1. Loads a single environment with fixed peg/hole positions
2. Moves the peg in a circular trajectory around the hole
3. Maintains a target force in Z direction during the motion
4. Plots measured force vs target force and commanded wrench

Usage:
    python scripts/test_hybrid_controller.py --config configs/experiments/ctrl_tuning/force_ctrller.yaml
    python scripts/test_hybrid_controller.py --headless  # Run without visualization
    python scripts/test_hybrid_controller.py --radius 0.008 --force_z 10.0 --loops 3
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
                    help="Target force in Z direction in Newtons (default: 5.0)")
parser.add_argument("--loops", type=int, default=2,
                    help="Number of loops around the hole (default: 2)")
parser.add_argument("--speed", type=float, default=0.5,
                    help="Angular velocity in radians per second (default: 0.5)")
parser.add_argument("--override", action="append", help="Override config values: key=value")

# Add Isaac Sim launcher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for any downstream argparse
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
print("\n[INFO]: Launching Isaac Sim...")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[INFO]: Isaac Sim launched successfully")

# Enable PhysX contact processing
import carb
settings = carb.settings.get_settings()
settings.set("/physics/disableContactProcessing", False)

# Now import everything else (after Isaac Sim is launched)
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict

# Import Isaac Lab components
try:
    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
except ImportError:
    from omni.isaac.lab_tasks.direct.factory.factory_env import FactoryEnv

# Import local modules
from configs.config_manager_v3 import ConfigManagerV3
from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper
from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper
from wrappers.mechanics.efficient_reset_wrapper import EfficientResetWrapper


def create_test_environment(config_path, overrides=None):
    """Create and wrap the factory environment for testing."""
    print("\n[INFO]: Loading configuration...")
    config_manager = ConfigManagerV3()
    configs = config_manager.process_config(config_path, overrides)

    # Force single environment
    configs['environment'].scene.num_envs = 1
    configs['primary'].num_envs_per_agent = 1
    configs['primary'].agents_per_break_force = 1

    print(f"[INFO]: Creating FactoryEnv with {configs['environment'].scene.num_envs} environment(s)")
    env = FactoryEnv(cfg=configs['environment'], render_mode=None)

    # Apply essential wrappers in correct order
    wrappers_config = configs['wrappers']

    # Efficient reset (optional but useful)
    if wrappers_config.efficient_reset.enabled:
        print("[INFO]: Applying EfficientResetWrapper")
        env = EfficientResetWrapper(env, config=wrappers_config.efficient_reset)

    # Force-torque sensor (REQUIRED before hybrid control)
    print("[INFO]: Applying ForceTorqueWrapper")
    env = ForceTorqueWrapper(
        env,
        use_tanh_scaling=wrappers_config.force_torque_sensor.use_tanh_scaling,
        tanh_scale=wrappers_config.force_torque_sensor.tanh_scale,
        add_force_obs=wrappers_config.force_torque_sensor.add_force_obs,
        add_contact_obs=wrappers_config.force_torque_sensor.add_contact_obs,
        add_contact_state=wrappers_config.force_torque_sensor.add_contact_state,
        contact_force_threshold=wrappers_config.force_torque_sensor.contact_force_threshold,
        contact_torque_threshold=wrappers_config.force_torque_sensor.contact_torque_threshold,
        log_contact_state=wrappers_config.force_torque_sensor.log_contact_state,
        use_contact_sensor=wrappers_config.force_torque_sensor.use_contact_sensor
    )

    # Hybrid force-position control
    print("[INFO]: Applying HybridForcePositionWrapper")
    env = HybridForcePositionWrapper(
        env,
        ctrl_torque=configs['primary'].ctrl_torque,
        reward_type=wrappers_config.hybrid_control.reward_type,
        ctrl_cfg=configs['environment'].ctrl,
        task_cfg=configs['environment'].hybrid_task,
        num_agents=1,
        use_ground_truth_selection=wrappers_config.hybrid_control.use_ground_truth_selection
    )

    return env, configs


def generate_circular_trajectory(center_x, center_y, radius, num_loops, num_points_per_loop):
    """Generate points along a circular trajectory."""
    total_points = num_loops * num_points_per_loop
    angles = np.linspace(0, num_loops * 2 * np.pi, total_points)

    x_points = center_x + radius * np.cos(angles)
    y_points = center_y + radius * np.sin(angles)

    return x_points, y_points, angles


def compute_action(env, target_x, target_y, target_force_z, configs):
    """
    Compute the action to move toward target position with target force in Z.

    Action format (3DOF force mode, 12 dimensions):
    [sel_x, sel_y, sel_z, dx, dy, dz, drx, dry, drz, fx, fy, fz]

    - Selection: 0 = position control, 1 = force control
    - Position: delta scaled by pos_threshold
    - Force: delta from current force, scaled by force_threshold
    """
    # Get current state
    current_pos = env.unwrapped.fingertip_midpoint_pos[0].cpu().numpy()
    current_force = env.unwrapped.robot_force_torque[0, :3].cpu().numpy()

    # Get thresholds from config
    pos_threshold = configs['environment'].ctrl.pos_action_threshold[0]  # Assume same for x,y,z
    force_threshold = configs['environment'].ctrl.force_action_threshold[2]  # Z threshold

    # Compute position deltas (for X and Y - position control)
    dx = (target_x - current_pos[0]) / pos_threshold
    dy = (target_y - current_pos[1]) / pos_threshold
    dz = 0.0  # Z controlled by force

    # Clamp position actions to [-1, 1]
    dx = np.clip(dx, -1.0, 1.0)
    dy = np.clip(dy, -1.0, 1.0)

    # Compute force action (for Z - force control)
    # Force is delta-based: target = action * threshold + current_force
    # So: action = (target - current) / threshold
    # Note: Pushing down is negative Z force
    desired_force_z = -abs(target_force_z)  # Negative = pushing down
    fz_action = (desired_force_z - current_force[2]) / force_threshold
    fz_action = np.clip(fz_action, -1.0, 1.0)

    # Build action tensor
    # [sel_x, sel_y, sel_z, dx, dy, dz, drx, dry, drz, fx, fy, fz]
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


def run_trajectory_test(env, configs, radius, target_force_z, num_loops, speed):
    """
    Run the circular trajectory test and collect data.
    """
    print("\n[INFO]: Resetting environment...")
    obs, info = env.reset()

    # Get hole center position (fixed asset position)
    hole_pos = env.unwrapped.fixed_pos[0].cpu().numpy()
    print(f"[INFO]: Hole position: x={hole_pos[0]:.4f}, y={hole_pos[1]:.4f}, z={hole_pos[2]:.4f}")

    # Get current end-effector position
    ee_pos = env.unwrapped.fingertip_midpoint_pos[0].cpu().numpy()
    print(f"[INFO]: Initial EE position: x={ee_pos[0]:.4f}, y={ee_pos[1]:.4f}, z={ee_pos[2]:.4f}")

    # Calculate trajectory parameters
    # Assume 60 Hz control rate, adjust based on decimation
    dt = 1.0 / 60.0  # Approximate timestep
    points_per_loop = int(2 * np.pi / (speed * dt))
    total_points = num_loops * points_per_loop

    print(f"\n[INFO]: Trajectory parameters:")
    print(f"  - Radius: {radius*1000:.1f} mm")
    print(f"  - Target force Z: {target_force_z:.1f} N")
    print(f"  - Number of loops: {num_loops}")
    print(f"  - Angular speed: {speed:.2f} rad/s")
    print(f"  - Points per loop: {points_per_loop}")
    print(f"  - Total steps: {total_points}")

    # Generate trajectory (centered on hole)
    x_traj, y_traj, angles = generate_circular_trajectory(
        hole_pos[0], hole_pos[1], radius, num_loops, points_per_loop
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
        'angle': []
    }

    print("\n[INFO]: Running trajectory...")

    # First, move to starting position and establish contact
    print("[INFO]: Moving to start position and establishing contact...")
    start_x, start_y = x_traj[0], y_traj[0]

    # Run a few steps to establish contact before starting trajectory
    for warmup_step in range(50):
        action = compute_action(env, start_x, start_y, target_force_z, configs)
        obs, reward, terminated, truncated, info = env.step(action)

        if warmup_step % 10 == 0:
            current_force = env.unwrapped.robot_force_torque[0, 2].cpu().item()
            print(f"  Warmup step {warmup_step}: Force Z = {current_force:.2f} N")

    print("\n[INFO]: Starting circular trajectory...")

    # Main trajectory loop
    for step, (target_x, target_y, angle) in enumerate(zip(x_traj, y_traj, angles)):
        # Compute and apply action
        action = compute_action(env, target_x, target_y, target_force_z, configs)
        obs, reward, terminated, truncated, info = env.step(action)

        # Collect data
        current_pos = env.unwrapped.fingertip_midpoint_pos[0].cpu().numpy()
        measured_force_z = env.unwrapped.robot_force_torque[0, 2].cpu().item()
        commanded_wrench_z = env.task_wrench[0, 2].cpu().item()

        data['time'].append(step * dt)
        data['measured_force_z'].append(measured_force_z)
        data['target_force_z'].append(-target_force_z)  # Negative for downward
        data['commanded_wrench_z'].append(commanded_wrench_z)
        data['position_x'].append(current_pos[0])
        data['position_y'].append(current_pos[1])
        data['position_z'].append(current_pos[2])
        data['target_x'].append(target_x)
        data['target_y'].append(target_y)
        data['angle'].append(angle)

        # Progress update
        if step % 100 == 0:
            print(f"  Step {step}/{total_points}: "
                  f"Force Z = {measured_force_z:.2f} N (target: {-target_force_z:.2f} N), "
                  f"Wrench Z = {commanded_wrench_z:.2f}")

        if terminated or truncated:
            print(f"[WARNING]: Episode terminated at step {step}")
            break

    print(f"\n[INFO]: Trajectory complete. Collected {len(data['time'])} data points.")
    return data


def plot_results(data, target_force_z):
    """Generate plots of the test results."""
    time = np.array(data['time'])
    measured_force = np.array(data['measured_force_z'])
    target_force = np.array(data['target_force_z'])
    commanded_wrench = np.array(data['commanded_wrench_z'])
    pos_x = np.array(data['position_x'])
    pos_y = np.array(data['position_y'])
    target_x = np.array(data['target_x'])
    target_y = np.array(data['target_y'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Measured Force Z vs Target Force Z
    ax1 = axes[0, 0]
    ax1.plot(time, measured_force, 'b-', label='Measured Force Z', linewidth=1.5)
    ax1.axhline(y=-target_force_z, color='r', linestyle='--', label=f'Target ({-target_force_z:.1f} N)', linewidth=2)
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

    # Plot 3: X/Y Trajectory (top-down view)
    ax3 = axes[1, 0]
    ax3.plot(pos_x * 1000, pos_y * 1000, 'b-', label='Actual Path', linewidth=1.5)
    ax3.plot(target_x * 1000, target_y * 1000, 'r--', label='Target Path', linewidth=1, alpha=0.5)
    ax3.scatter([pos_x[0] * 1000], [pos_y[0] * 1000], c='green', s=100, marker='o', label='Start', zorder=5)
    ax3.scatter([pos_x[-1] * 1000], [pos_y[-1] * 1000], c='red', s=100, marker='x', label='End', zorder=5)
    ax3.set_xlabel('X Position (mm)')
    ax3.set_ylabel('Y Position (mm)')
    ax3.set_title('Trajectory (Top-Down View)')
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

    plt.tight_layout()
    plt.suptitle('Hybrid Force Controller Tuning Test', fontsize=14, y=1.02)

    print("\n[INFO]: Displaying plots...")
    plt.show()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Hybrid Force Controller Test Script")
    print("=" * 60)

    # Create environment
    env, configs = create_test_environment(args_cli.config, args_cli.override)

    try:
        # Run the trajectory test
        data = run_trajectory_test(
            env,
            configs,
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
