#!/usr/bin/env python3
"""
Environment Reset Visualizer

This script allows visualization of environment resets to debug reset behavior.
Can run in two modes:
1. Interactive mode: Watch resets in real-time with viewport
2. Headless mode: Generate GIF showing sequence of resets

Usage:
    # Interactive mode - watch resets in viewport
    python utils/reset_visualizer.py --manual-control

    # Headless mode - generate 100-frame GIF
    python utils/reset_visualizer.py --manual-control --gen-gif --num-resets 100

    # With custom config
    python utils/reset_visualizer.py --config configs/my_config.yaml --gen-gif
"""

import argparse
import sys
import os
from datetime import datetime

try:
    from isaaclab.app import AppLauncher
except:
    from omni.isaac.lab.app import AppLauncher

# Minimal argparse arguments - configuration system handles the rest
parser = argparse.ArgumentParser(description="Visualize environment resets.")

# Essential arguments
parser.add_argument("--config", type=str, default=None, help="Path to configuration file (optional if --manual-control used)")
parser.add_argument("--override", action="append", help="Override config values: key=value")
parser.add_argument("--manual-control", action="store_true", help="Use manual control base config")
parser.add_argument("--gen-gif", action="store_true", help="Run in headless mode to generate GIF")
parser.add_argument("--num-resets", type=int, default=100, help="Number of resets to perform (default: 100)")
parser.add_argument("--output-name", type=str, default=None, help="Custom name for GIF output (default: reset_viz_timestamp.gif)")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Handle manual control config selection
if args_cli.manual_control:
    if args_cli.config is None:
        args_cli.config = "configs/base/manual_control_base.yaml"
        print(f"[INFO]: --manual-control flag set, using default config: {args_cli.config}")
    else:
        print(f"[INFO]: --manual-control flag set, using provided config: {args_cli.config}")
else:
    if args_cli.config is None:
        raise ValueError("--config argument is required when not using --manual-control")

# Set visualization based on gen_gif flag
args_cli.video = False
args_cli.enable_cameras = True  # Always need cameras for capturing
args_cli.headless = args_cli.gen_gif  # Set AppLauncher's headless based on our gen_gif flag

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

print("\n\n\n Calling App Launcher \n\n\n\n")
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Enable PhysX contact processing for ContactSensor filtering to work
import carb
settings = carb.settings.get_settings()
settings.set("/physics/disableContactProcessing", False)

import random
import torch
import numpy as np
from PIL import Image

from skrl.utils import set_seed

try:
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
    )
    import isaaclab_tasks  # noqa: F401
    print("Isaac Lab v2.0.0+ successfully loaded")
except ImportError:
    try:
        from omni.isaac.lab.envs import (
            DirectMARLEnv,
            DirectMARLEnvCfg,
            DirectRLEnvCfg,
            ManagerBasedRLEnvCfg,
        )
        import omni.isaac.lab_tasks  # noqa: F401
        print("Isaac Lab v1.4.1 successfully loaded")
    except ImportError:
        print("ERROR: Could not import Isaac Lab tasks module.")
        print("Please ensure you have either:")
        print("  - Isaac Lab v2.0.0+ (isaaclab_tasks)")
        print("  - Isaac Lab v1.4.1 or earlier (omni.isaac.lab_tasks)")
        sys.exit(1)

import learning.launch_utils_v3 as lUtils
from configs.config_manager_v3 import ConfigManagerV3
from wrappers.skrl.async_critic_isaaclab_wrapper import AsyncCriticIsaacLabWrapper

# Import factory environment class for direct instantiation
try:
    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
except ImportError:
    try:
        from omni.isaac.lab_tasks.direct.factory.factory_env import FactoryEnv
    except ImportError:
        print("ERROR: Could not import FactoryEnv.")
        print("Please ensure Isaac Lab tasks are properly installed.")
        sys.exit(1)

from wrappers.sensors.factory_env_with_sensors import create_sensor_enabled_factory_env



def main(args_cli):
    """Main function to run reset visualization."""
    # ===== STEP 1: LOAD CONFIGURATION =====
    print("="*100)
    print("[INFO]: STEP 1 - Loading Configuration")
    configManager = ConfigManagerV3()
    configs = configManager.process_config(args_cli.config, args_cli.override)

    # Set or generate seed
    if configs['primary'].seed == -1:
        configs['primary'].seed = random.randint(0, 10000)
    print(f"[INFO]: Setting global seed: {configs['primary'].seed}")
    set_seed(configs['primary'].seed)
    configs['environment'].seed = configs['primary'].seed

    # Add wandb tracking tags
    lUtils.add_wandb_tracking_tags(configs)

    # Disable WandB and other logging wrappers for visualization
    # (they require agent configs that we don't need for reset visualization)
    configs['wrappers'].wandb_logging.enabled = False
    configs['wrappers'].factory_metrics.enabled = False
    configs['wrappers'].action_logging.enabled = False
    configs['wrappers'].pose_contact_logging.enabled = False
    print("[INFO]: Disabled logging wrappers (not needed for reset visualization)")

    # Should not matter but removes annoying warning message
    configs["environment"].sim.render_interval = configs["primary"].decimation

    # Setup camera configuration
    print("[INFO]: Setting up camera configuration...")
    try:
        from isaaclab.sensors import TiledCameraCfg, CameraCfg
        import isaaclab.sim as sim_utils
    except:
        from omni.isaac.lab.sensors import TiledCameraCfg, CameraCfg
        import omni.isaac.lab.sim as sim_utils

    configs['environment'].scene.tiled_camera = TiledCameraCfg(
        prim_path="/World/envs/env_0/Camera",  # Single environment camera
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
        width=960,   # Increased from 240 (4x larger)
        height=720,  # Increased from 180 (4x larger)
        debug_vis=False,
    )
    print("[INFO]:   Camera configured: 960x720, RGB")

    print(f"[INFO]: Environment Configured from {args_cli.config}")
    print(f"[INFO]: Task: {configs['experiment'].task_name}")
    print(f"[INFO]: Episode length: {configs['environment'].episode_length_s}s")
    print(f"[INFO]: Decimation: {configs['environment'].decimation}")
    print(f"[INFO]: Mode: {'HEADLESS (GIF Generation)' if args_cli.gen_gif else 'INTERACTIVE (Viewport)'}")
    print(f"[INFO]: Number of resets: {args_cli.num_resets}")
    print("="*100)

    # ===== STEP 2: CREATE ENVIRONMENT =====
    print("[INFO]: Step 2 - Creating environment")

    # Wrap with sensor support if contact sensor config is present AND use_contact_sensor is enabled
    if (hasattr(configs['environment'].task, 'held_fixed_contact_sensor') and
        configs['wrappers'].force_torque_sensor.use_contact_sensor):
        EnvClass = create_sensor_enabled_factory_env(FactoryEnv)
        print("[INFO]:   Using sensor-enabled factory environment")
    else:
        EnvClass = FactoryEnv
        print("[INFO]:   Using standard factory environment")

    # Create environment with appropriate render mode
    render_mode = "rgb_array" if args_cli.gen_gif else None
    env = EnvClass(cfg=configs['environment'], render_mode=render_mode)
    print("[INFO]: Environment created successfully")

    # Enable camera light for better visibility (both modes)
    try:
        # Enable camera light (headlight) using carb settings
        # This is the Omniverse-standard way to enable viewport headlight
        settings.set("/rtx/useViewLightingMode", True)
        print("[INFO]:   Camera light (headlight) enabled via RTX settings")
    except Exception as e:
        print(f"[WARNING]: Could not enable camera light: {e}")

    # Set viewport camera for interactive mode
    if not args_cli.gen_gif:
        try:
            import omni.kit.viewport.utility as vp_utils
            viewport_api = vp_utils.get_active_viewport()
            if viewport_api:
                # Switch viewport to use our configured camera
                camera_path = "/World/envs/env_0/Camera"
                viewport_api.set_active_camera(camera_path)
                print(f"[INFO]:   Viewport camera set to {camera_path}")
        except Exception as e:
            print(f"[WARNING]: Could not configure viewport camera: {e}")
            import traceback
            traceback.print_exc()

    # ===== STEP 3: APPLY WRAPPER STACK =====
    print("="*100)
    print("[INFO]: Step 3 - Applying wrapper stack")
    env = lUtils.apply_wrappers(env, configs)
    print("  - Applying async critic isaac lab wrapper (derived from SKRL isaaclab wrapper)")
    env = AsyncCriticIsaacLabWrapper(env)
    print("[INFO]: Wrappers Applied successfully")
    print("="*100)

    # ===== STEP 4: RUN RESET VISUALIZATION =====
    print("[INFO]: Step 4 - Running reset visualization")

    if args_cli.gen_gif:
        # Headless mode - generate GIF
        print(f"[INFO]: Performing {args_cli.num_resets} resets and capturing images...")
        images = []

        for i in range(args_cli.num_resets):
            # Reset environment
            obs, info = env.reset()

            # IMPORTANT: Take multiple dummy steps to allow simulation/rendering to fully update
            # The first few steps after reset are needed for physics to settle and camera to render
            zero_action = torch.zeros(env.unwrapped.action_space.shape, device=env.device)
            for _ in range(5):  # Take 5 steps to let scene fully initialize and render
                obs, reward, terminated, truncated, info = env.step(zero_action)
                # Force simulation app to update/render (helps ensure camera gets fresh data)
                simulation_app.update()

            # Get camera image from scene
            # The camera data is in the unwrapped environment's scene
            camera_data = env.unwrapped.scene.sensors["tiled_camera"].data.output["rgb"]

            # Extract image for env_0 [H, W, C]
            img = camera_data[0].cpu().numpy()

            # Convert from float [0, 1] to uint8 [0, 255] if needed
            if img.dtype == np.float32:
                img = (img * 255).astype(np.uint8)

            images.append(img)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{args_cli.num_resets} resets complete")

        print(f"[INFO]: Captured {len(images)} images")

        # Create GIF
        print("[INFO]: Creating GIF...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = args_cli.output_name if args_cli.output_name else f"reset_viz_{timestamp}.gif"
        if not output_name.endswith('.gif'):
            output_name += '.gif'

        output_dir = os.path.join(os.path.dirname(__file__), "reset_gifs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)

        # Convert numpy arrays to PIL Images
        pil_images = [Image.fromarray(img) for img in images]

        # Save as GIF with 1 second per frame (1000ms)
        pil_images[0].save(
            output_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=1000,  # 1 second per frame
            loop=0  # Loop forever
        )

        print(f"[INFO]: GIF saved to: {output_path}")
        print(f"[INFO]:   Total frames: {len(pil_images)}")
        print(f"[INFO]:   Frame duration: 1000ms (1 second)")
        print(f"[INFO]:   Total GIF duration: {len(pil_images)} seconds")

    else:
        # Interactive mode - continuous reset loop
        print("[INFO]: Interactive mode - press Ctrl+C to exit")
        print("[INFO]: Environment will reset continuously...")

        try:
            reset_count = 0
            while True:
                obs, info = env.reset()
                reset_count += 1
                print(f"  Reset #{reset_count}")

                # Take several steps to allow simulation/rendering to fully update
                zero_action = torch.zeros(env.unwrapped.action_space.shape, device=env.device)
                for _ in range(5):  # Take 5 steps to let scene fully initialize and render
                    obs, reward, terminated, truncated, info = env.step(zero_action)
                    # Force simulation app to update/render (updates viewport)
                    simulation_app.update()

                # Small delay to see each reset
                import time
                time.sleep(0.5)

        except KeyboardInterrupt:
            print(f"\n[INFO]: Exiting after {reset_count} resets")

    print("="*100)
    print("[INFO]: Reset visualization complete")


if __name__ == "__main__":
    # run the main function
    main(args_cli)

    # close sim app
    simulation_app.close()
