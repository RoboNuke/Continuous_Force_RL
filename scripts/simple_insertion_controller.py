#!/usr/bin/env python3
"""
Simple Insertion Controller

A basic controller that moves the peg towards the hole using position control.
If the peg stalls (low velocity but position error remains), the setpoint is
pushed further along the error vector to encourage continued movement.

This is useful for testing whether the environment setup is correct and
the peg can physically be inserted into the hole.

Usage:
    python -m scripts.simple_insertion_controller --config configs/experiments/req_trainning_sets/4_task_noise/2_shape/hex/pose.yaml
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AppLauncher BEFORE other Isaac Lab imports
try:
    from isaaclab.app import AppLauncher
except ImportError:
    from omni.isaac.lab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Simple Insertion Controller")
parser.add_argument(
    "--config",
    type=str,
    default="configs/experiments/req_trainning_sets/4_task_noise/2_shape/hex/pose.yaml",
    help="Path to experiment config"
)
parser.add_argument(
    "--num-steps",
    type=int,
    default=2000,
    help="Number of steps to run"
)
parser.add_argument(
    "--velocity-threshold",
    type=float,
    default=0.002,
    help="Velocity threshold for stall detection (m/s)"
)
parser.add_argument(
    "--stall-window",
    type=int,
    default=50,
    help="Number of steps below velocity threshold to detect stall"
)
parser.add_argument(
    "--setpoint-increment",
    type=float,
    default=0.005,
    help="How much to push setpoint when stalled (meters)"
)
parser.add_argument(
    "--max-setpoint-offset",
    type=float,
    default=0.05,
    help="Maximum setpoint offset beyond goal (meters)"
)
parser.add_argument(
    "--action-gain",
    type=float,
    default=2.0,
    help="Gain for converting position error to action"
)
parser.add_argument(
    "--print-interval",
    type=int,
    default=25,
    help="How often to print status"
)
parser.add_argument("--override", action="append", help="Override config values: key=value")

# Add AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args_cli, hydra_args = parser.parse_known_args()

# Set visualization options
args_cli.headless = False
args_cli.enable_cameras = True

# Clear sys.argv for any downstream argparse
sys.argv = [sys.argv[0]] + hydra_args

# Launch the app BEFORE importing other Isaac Lab modules
print("\n[INFO]: Launching Isaac Sim...")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[INFO]: Isaac Sim launched successfully")

# Enable PhysX contact processing
import carb
settings = carb.settings.get_settings()
settings.set("/physics/disableContactProcessing", False)

# NOW we can import torch and other modules
import torch
import random
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


class SimpleInsertionController:
    """
    Simple controller that moves peg towards hole with stall detection.

    Two-phase approach:
    1. XY alignment phase: Move to align XY with hole (keep Z constant)
    2. Descent phase: Move downward into the hole

    When the peg stalls (low velocity but error remains), the setpoint
    is pushed further along the error vector to encourage movement.
    """

    def __init__(
        self,
        device,
        velocity_threshold=0.001,
        stall_window=50,
        setpoint_increment=0.005,
        max_setpoint_offset=0.05,
        action_gain=2.0,
        xy_alignment_threshold=0.001  # 1mm XY alignment threshold
    ):
        self.device = device
        self.velocity_threshold = velocity_threshold
        self.stall_window = stall_window
        self.setpoint_increment = setpoint_increment
        self.max_setpoint_offset = max_setpoint_offset
        self.action_gain = action_gain
        self.xy_alignment_threshold = xy_alignment_threshold

        # State tracking
        self.low_velocity_count = 0
        self.setpoint_offset = 0.0
        self.prev_peg_pos = None
        self.phase = "xy_align"  # "xy_align" or "descent"

    def reset(self):
        """Reset controller state."""
        self.low_velocity_count = 0
        self.setpoint_offset = 0.0
        self.prev_peg_pos = None
        self.phase = "xy_align"

    def compute_action(self, peg_pos, hole_pos, peg_quat, hole_quat, peg_vel=None):
        """
        Compute position and rotation action to move peg towards hole.

        Phase 1 (xy_align): Move XY to align with hole, align Z rotation, keep Z constant
        Phase 2 (descent): Move downward into hole

        Args:
            peg_pos: Current peg position [3]
            hole_pos: Target hole position [3]
            peg_quat: Current peg quaternion [4] (w, x, y, z)
            hole_quat: Target hole quaternion [4] (w, x, y, z)
            peg_vel: Current peg velocity [3] (optional, computed from position if None)

        Returns:
            pos_action: Position action [3] in range [-1, 1]
            rot_action: Rotation action [3] in range [-1, 1]
            info: Dict with debug information
        """
        # Compute velocity from position change if not provided
        if peg_vel is None and self.prev_peg_pos is not None:
            peg_vel = peg_pos - self.prev_peg_pos
        self.prev_peg_pos = peg_pos.clone()

        # Compute XY error
        xy_error = hole_pos[0:2] - peg_pos[0:2]
        xy_error_magnitude = torch.norm(xy_error)

        # Compute Z error
        z_error = hole_pos[2] - peg_pos[2]

        # Compute total error
        error = hole_pos - peg_pos
        error_magnitude = torch.norm(error)

        # Compute yaw (Z rotation) error from quaternions
        # Extract yaw from quaternions using atan2
        # For quaternion (w, x, y, z), yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        peg_yaw = torch.atan2(
            2.0 * (peg_quat[0] * peg_quat[3] + peg_quat[1] * peg_quat[2]),
            1.0 - 2.0 * (peg_quat[2]**2 + peg_quat[3]**2)
        )
        hole_yaw = torch.atan2(
            2.0 * (hole_quat[0] * hole_quat[3] + hole_quat[1] * hole_quat[2]),
            1.0 - 2.0 * (hole_quat[2]**2 + hole_quat[3]**2)
        )

        # Compute yaw error (wrap to [-pi, pi])
        yaw_error = hole_yaw - peg_yaw
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))  # Wrap to [-pi, pi]
        yaw_error_deg = yaw_error.item() * 180.0 / 3.14159

        # Compute velocity magnitude
        if peg_vel is not None:
            vel_magnitude = torch.norm(peg_vel)
        else:
            vel_magnitude = torch.tensor(float('inf'), device=self.device)

        # Phase transition: switch to descent when XY is aligned AND yaw is aligned
        yaw_threshold = 0.05  # ~3 degrees in radians
        xy_and_yaw_aligned = (xy_error_magnitude < self.xy_alignment_threshold and
                              abs(yaw_error) < yaw_threshold)
        if self.phase == "xy_align" and xy_and_yaw_aligned:
            self.phase = "descent"
            self.low_velocity_count = 0  # Reset stall detection for new phase
            self.setpoint_offset = 0.0

        # Compute target based on current phase
        if self.phase == "xy_align":
            # Phase 1: Only move XY, keep Z constant
            target = torch.tensor([hole_pos[0], hole_pos[1], peg_pos[2]], device=self.device)
            phase_error = xy_error_magnitude
        else:
            # Phase 2: Move towards hole (primarily Z)
            # Target is hole position + any offset along downward direction
            target = hole_pos.clone()
            if self.setpoint_offset > 0:
                target[2] -= self.setpoint_offset  # Push setpoint further down
            phase_error = error_magnitude

        # Stall detection (only in descent phase)
        is_stalled = False
        if self.phase == "descent":
            if vel_magnitude < self.velocity_threshold and phase_error > 0.002:
                self.low_velocity_count += 1
                if self.low_velocity_count >= self.stall_window:
                    is_stalled = True
                    # Increase setpoint offset (push target further down)
                    if self.setpoint_offset < self.max_setpoint_offset:
                        self.setpoint_offset += self.setpoint_increment
                        # Update target with new offset
                        target[2] = hole_pos[2] - self.setpoint_offset
            else:
                self.low_velocity_count = 0

        # Compute position action as scaled error to target
        action_error = target - peg_pos
        pos_action = action_error * self.action_gain

        # Compute rotation action (only Z rotation during xy_align phase)
        rot_action = torch.zeros(3, device=self.device)
        if self.phase == "xy_align":
            # Apply yaw correction (index 2 = Z rotation)
            rot_action[2] = yaw_error * self.action_gain

        # Clip to [-1, 1] range
        pos_action = torch.clamp(pos_action, -1.0, 1.0)
        rot_action = torch.clamp(rot_action, -1.0, 1.0)

        info = {
            'phase': self.phase,
            'xy_error': xy_error_magnitude.item() * 1000,  # mm
            'z_error': z_error.item() * 1000,  # mm
            'yaw_error': yaw_error_deg,  # degrees
            'error_magnitude': error_magnitude.item(),
            'vel_magnitude': vel_magnitude.item() if peg_vel is not None else 0.0,
            'setpoint_offset': self.setpoint_offset,
            'is_stalled': is_stalled,
            'low_velocity_count': self.low_velocity_count,
        }

        return pos_action, rot_action, info


def create_test_environment(config_path, overrides=None):
    """Create and wrap the factory environment for testing."""
    print("\n" + "=" * 60)
    print("Creating Test Environment")
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

    # Disable wrappers we don't need for testing
    #configs['wrappers'].wandb_logging.enabled = False
    #configs['wrappers'].factory_metrics.enabled = False
    #configs['wrappers'].pose_contact_logging.enabled = False
    configs['wrappers'].action_logging.enabled = False
    configs['wrappers'].manual_control.enabled = False

    # Handle camera configuration
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

    # Enable camera light and set viewport camera
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

    # Set up agent configs (required by some wrappers like wandb)
    lUtils.define_agent_configs(configs)

    # Apply wrappers
    print("[INFO]: Applying wrapper stack...")
    env = lUtils.apply_wrappers(env, configs)
    print("[INFO]: Wrappers applied successfully")

    return env, configs


def main():
    print("=" * 80)
    print("Simple Insertion Controller")
    print("=" * 80)

    # Create environment
    env, configs = create_test_environment(args_cli.config, args_cli.override)

    device = env.device

    # Create controller
    controller = SimpleInsertionController(
        device=device,
        velocity_threshold=args_cli.velocity_threshold,
        stall_window=args_cli.stall_window,
        setpoint_increment=args_cli.setpoint_increment,
        max_setpoint_offset=args_cli.max_setpoint_offset,
        action_gain=args_cli.action_gain,
    )

    # Get action dimension
    action_dim = env.action_space.shape[-1]
    print(f"Action dimension: {action_dim}")

    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset()
    controller.reset()

    # Get initial positions
    unwrapped = env.unwrapped

    print("\n" + "=" * 80)
    print("Starting insertion controller loop")
    print("=" * 80)
    print(f"  Velocity threshold: {args_cli.velocity_threshold} m/s")
    print(f"  Stall window: {args_cli.stall_window} steps")
    print(f"  Setpoint increment: {args_cli.setpoint_increment * 1000:.1f} mm")
    print(f"  Max setpoint offset: {args_cli.max_setpoint_offset * 1000:.1f} mm")
    print(f"  Action gain: {args_cli.action_gain}")
    print("=" * 80)

    # Find factory metrics wrapper to track engagement
    factory_metrics_wrapper = None
    current_wrapper = env
    while current_wrapper is not None:
        if current_wrapper.__class__.__name__ == 'FactoryMetricsWrapper':
            factory_metrics_wrapper = current_wrapper
            print(f"[DEBUG] Found FactoryMetricsWrapper!")
            break
        if hasattr(current_wrapper, 'env'):
            current_wrapper = current_wrapper.env
        else:
            break

    if factory_metrics_wrapper is None:
        print("[WARNING] FactoryMetricsWrapper not found - engagement tracking disabled")
    else:
        print(f"[DEBUG] Found FactoryMetricsWrapper")
        # Check if engage_threshold exists - this is required for engagement detection
        if hasattr(unwrapped, 'cfg_task'):
            if hasattr(unwrapped.cfg_task, 'engage_threshold'):
                print(f"[DEBUG] engage_threshold = {unwrapped.cfg_task.engage_threshold}")
            else:
                print(f"[WARNING] cfg_task missing 'engage_threshold' - engagement detection will NOT work!")
            if hasattr(unwrapped.cfg_task, 'success_threshold'):
                print(f"[DEBUG] success_threshold = {unwrapped.cfg_task.success_threshold}")
        else:
            print(f"[WARNING] unwrapped missing 'cfg_task' - engagement detection will NOT work!")

    # State tracking for engagement and success transitions
    was_engaged = False
    was_successful = False

    # Main loop
    for step in range(args_cli.num_steps):
        # Get current peg and hole positions
        # Peg position is fingertip/held asset position
        peg_pos = unwrapped.fingertip_midpoint_pos[0].clone()

        # Hole position is fixed_pos (target hole, corrected by our wrapper)
        hole_pos = unwrapped.fixed_pos[0].clone()

        # Get quaternions
        peg_quat = unwrapped.fingertip_midpoint_quat[0].clone()
        hole_quat = unwrapped.fixed_quat[0].clone()

        # Get velocity if available
        if hasattr(unwrapped, 'fingertip_midpoint_linvel'):
            peg_vel = unwrapped.fingertip_midpoint_linvel[0].clone()
        else:
            peg_vel = None

        # Compute action
        pos_action, rot_action, ctrl_info = controller.compute_action(
            peg_pos, hole_pos, peg_quat, hole_quat, peg_vel
        )

        # Create full action tensor (position + rotation)
        action = torch.zeros(1, action_dim, device=device)
        action[0, 0:3] = pos_action
        action[0, 3:6] = rot_action  # Rotation actions

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Check engagement state from FactoryMetricsWrapper
        is_engaged = False
        if factory_metrics_wrapper is not None and hasattr(factory_metrics_wrapper, 'curr_engaged'):
            is_engaged = factory_metrics_wrapper.curr_engaged[0].item()

        # Debug: print engagement status every 25 steps with actual values used by _get_curr_successes
        if step % 25 == 0 and step > 0:
            # These are the actual values used in _get_curr_successes
            held_base = unwrapped.held_base_pos[0]
            target_held_base = unwrapped.target_held_base_pos[0]
            xy_dist = torch.norm(target_held_base[0:2] - held_base[0:2]).item() * 1000
            z_disp = (held_base[2] - target_held_base[2]).item() * 1000
            print(f"  [DEBUG] Step {step}: xy_dist={xy_dist:.1f}mm (need <2.5), z_disp={z_disp:.1f}mm (need <22.5), engaged={is_engaged}")

        # Detect engagement transition
        if is_engaged and not was_engaged:
            print(f"\n{'='*60}")
            print(f"*** ENGAGED at step {step}! ***")
            print(f"    XY_err={ctrl_info['xy_error']:5.1f}mm, Z_err={ctrl_info['z_error']:+6.1f}mm, Yaw_err={ctrl_info['yaw_error']:+5.1f}deg")
            print(f"{'='*60}\n")
        was_engaged = is_engaged

        # Check for success from FactoryMetricsWrapper (consistent with engagement detection)
        is_success = False
        if factory_metrics_wrapper is not None and hasattr(factory_metrics_wrapper, 'successes'):
            is_success = factory_metrics_wrapper.successes[0].item()

        # Detect success transition
        if is_success and not was_successful:
            print(f"\n{'='*60}")
            print(f"*** SUCCESS at step {step}! ***")
            print(f"    XY_err={ctrl_info['xy_error']:5.1f}mm, Z_err={ctrl_info['z_error']:+6.1f}mm, Yaw_err={ctrl_info['yaw_error']:+5.1f}deg")
            print(f"{'='*60}\n")
        was_successful = is_success

        # Print status
        if step % args_cli.print_interval == 0 or is_success:
            status = "STALLED" if ctrl_info['is_stalled'] else "moving"
            engaged_str = " [ENGAGED]" if is_engaged else ""
            if is_success:
                status = "SUCCESS!"

            phase_str = "XY-ALIGN" if ctrl_info['phase'] == "xy_align" else "DESCENT"

            print(f"Step {step:4d} [{phase_str:8s}]: "
                  f"XY_err={ctrl_info['xy_error']:5.1f}mm, "
                  f"Z_err={ctrl_info['z_error']:+6.1f}mm, "
                  f"Yaw_err={ctrl_info['yaw_error']:+5.1f}deg, "
                  f"vel={ctrl_info['vel_magnitude']*1000:5.2f}mm/s, "
                  f"offset={ctrl_info['setpoint_offset']*1000:5.1f}mm, "
                  f"[{status}]{engaged_str}")

        if is_success:
            print("\n" + "=" * 80)
            print(f"INSERTION SUCCESSFUL at step {step}!")
            print(f"Final: XY_err={ctrl_info['xy_error']:5.1f}mm, Z_err={ctrl_info['z_error']:+6.1f}mm, Yaw_err={ctrl_info['yaw_error']:+5.1f}deg")
            print("=" * 80)
            break

        # Reset if terminated/truncated
        if terminated.any() or truncated.any():
            print(f"\nEpisode ended at step {step}, resetting...")
            obs, info = env.reset()
            controller.reset()
            was_engaged = False  # Reset engagement tracking
            was_successful = False  # Reset success tracking

    print("\nDone.")

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
