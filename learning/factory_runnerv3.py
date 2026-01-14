import argparse
import sys
import signal
import os

try:
    from isaaclab.app import AppLauncher
except:
    from omni.isaac.lab.app import AppLauncher

# Minimal argparse arguments - configuration system handles the rest
parser = argparse.ArgumentParser(description="Train an RL agent with configuration system.")

# Essential arguments
parser.add_argument("--config", type=str, default=None, help="Path to configuration file (optional if --manual-control used)")
parser.add_argument("--task", type=str, default=None, help="Name of the task (defaults to config value)")
#parser.add_argument("--device", type=str, default=None, help="Device to run on")
parser.add_argument("--seed", type=int, default=-1, help="Random seed for reproducibility (-1 for random)")
parser.add_argument("--override", action="append", help="Override config values: key=value")
parser.add_argument("--manual-control", action="store_true", help="Enable manual control mode with visualization")
parser.add_argument("--test_reset", action="store_true", help="Initializes env and then resets it 20 times allowing space for different checks")
parser.add_argument("--eval_tag", type=str, default=None, help="Evaluation tag for tracking training runs (overrides auto-generated timestamp tag)")
parser.add_argument("--checkpoint_tag", type=str, default=None, help="WandB tag to find checkpoint runs for initialization")
parser.add_argument("--checkpoint_step", type=int, default=None, help="Checkpoint step number to load (required with --checkpoint_tag)")
parser.add_argument("--checkpoint_project", type=str, default="SG_Exps", help="WandB project to search for checkpoint runs (default: SG_Exps)")
parser.add_argument("--checkpoint_entity", type=str, default="hur", help="WandB entity for checkpoint runs (default: hur)")
parser.add_argument("--new_project", type=str, default=None, help="WandB project for new training runs (overrides config value)")

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
    if args_cli.config is None and args_cli.checkpoint_tag is None:
        raise ValueError("--config argument is required when not using --manual-control or --checkpoint_tag")

# Set visualization based on manual control flag
args_cli.video = False
args_cli.enable_cameras = args_cli.manual_control
args_cli.headless = not args_cli.manual_control


# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Store simulation app reference for cleanup handler
_simulation_app = simulation_app


# Enable PhysX contact processing for ContactSensor filtering to work
import carb
settings = carb.settings.get_settings()
settings.set("/physics/disableContactProcessing", False)

import random
import torch
import gymnasium as gym

from skrl.utils import set_seed
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
try:
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
    )
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
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
        from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
        print("Isaac Lab v1.4.1 successfully loaded")
    except ImportError:
        print("ERROR: Could not import Isaac Lab tasks module.")
        print("Please ensure you have either:")
        print("  - Isaac Lab v2.0.0+ (isaaclab_tasks)")
        print("  - Isaac Lab v1.4.1 or earlier (omni.isaac.lab_tasks)")
        sys.exit(1)

from memories.multi_random import MultiRandomMemory
from memories.importance_sampling_memory import ImportanceSamplingMemory
import learning.launch_utils_v3 as lUtils
from configs.config_manager_v3 import ConfigManagerV3
from wrappers.skrl.async_critic_isaaclab_wrapper import AsyncCriticIsaacLabWrapper
from eval.checkpoint_utils import query_runs_by_tag, load_checkpoints_parallel, reconstruct_config_from_wandb

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

print("\n\nImports complete\n\n")


def debug_print_usd_hierarchy(env):
    """
    Print the entire USD prim hierarchy for key assets in the scene.
    Shows every prim and what APIs/properties are attached to them.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    import omni.usd

    print("\n" + "=" * 100)
    print("USD PRIM HIERARCHY DEBUG")
    print("=" * 100)

    # Get the USD stage
    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("ERROR: Could not get USD stage")
        return

    def get_prim_info(prim):
        """Get detailed info about a prim's APIs and properties."""
        info_parts = []

        # Check for physics APIs (with safe hasattr checks for version compatibility)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            info_parts.append("RigidBody")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            info_parts.append("Collision")
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            mass_attr = mass_api.GetMassAttr()
            if mass_attr and mass_attr.HasValue():
                info_parts.append(f"Mass={mass_attr.Get():.4f}")
            else:
                info_parts.append("Mass")
        if hasattr(UsdPhysics, 'ArticulationRootAPI') and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            info_parts.append("ArticulationRoot")
        if hasattr(UsdPhysics, 'MaterialAPI') and prim.HasAPI(UsdPhysics.MaterialAPI):
            info_parts.append("PhysMaterial")

        # Check if it's a joint by type name
        type_name = prim.GetTypeName()
        if "Joint" in type_name:
            info_parts.append(f"Joint({type_name})")

        # Check for geometry
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            if points:
                info_parts.append(f"Mesh({len(points)} verts)")
        elif prim.IsA(UsdGeom.Xform):
            info_parts.append("Xform")
        elif prim.IsA(UsdGeom.Scope):
            info_parts.append("Scope")

        return info_parts

    def print_prim_tree(prim, indent=0):
        """Recursively print prim and its children."""
        prim_type = prim.GetTypeName() or "Prim"
        info = get_prim_info(prim)
        info_str = f" [{', '.join(info)}]" if info else ""

        prefix = "  " * indent
        print(f"{prefix}{prim.GetPath()} - {prim_type}{info_str}")

        for child in prim.GetChildren():
            print_prim_tree(child, indent + 1)

    # Print hierarchy for env_0 assets
    env_path = "/World/envs/env_0"
    env_prim = stage.GetPrimAtPath(env_path)

    if not env_prim or not env_prim.IsValid():
        print(f"ERROR: Could not find prim at {env_path}")
        return

    # Key asset paths to inspect
    asset_names = ["HeldAsset", "FixedAsset", "Robot", "Table"]

    for asset_name in asset_names:
        asset_path = f"{env_path}/{asset_name}"
        asset_prim = stage.GetPrimAtPath(asset_path)

        if asset_prim and asset_prim.IsValid():
            print(f"\n{'─' * 80}")
            print(f"ASSET: {asset_name}")
            print(f"{'─' * 80}")
            print_prim_tree(asset_prim)
        else:
            print(f"\n[SKIP] {asset_name} - not found at {asset_path}")

    # Also print any contact sensors if present
    print(f"\n{'─' * 80}")
    print("CONTACT SENSORS (if any)")
    print(f"{'─' * 80}")

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if "contact" in prim_path.lower() or "sensor" in prim_path.lower():
            if "env_0" in prim_path:
                info = get_prim_info(prim)
                info_str = f" [{', '.join(info)}]" if info else ""
                print(f"  {prim.GetPath()} - {prim.GetTypeName()}{info_str}")

    print("\n" + "=" * 100)
    print("END USD HIERARCHY DEBUG")
    print("=" * 100 + "\n")

# Global references for cleanup handler
_wandb_wrapper = None
_simulation_app = None
_cleanup_completed = False

def print_configs(configs):
    print("Full configuration:")
    for k, v in configs.items():
        print(f"\t{k}:{type(v)}")
        atts = dir(v)
        for att in atts:
            if "__" in att:
                continue
            val = getattr(v,att)
            ty = type(val)
            if callable(val) or "__" in att:
                continue
            if ty in [list, float, int, str, bool]:
                print(f"\t\t{att}({ty.__name__}):{val}")
            elif ty in [dict]:
                print(f"\t\t{att}({ty.__name__})")
                for k2,v2 in val.items():
                    print(f"\t\t\t{k2}({type(v2).__name__}):{v2}")
            else: # not a dict
                print(f"\t\t{att}({ty.__name__})")
                for att2 in dir(val):

                    val2 = getattr(val,att2)
                    if callable(val2) or "__" in att2:
                        continue
                    ty2 = type(val2)
                    if ty2 in [list,float,int,str,bool]:
                        print(f"\t\t\t{att2}({ty2.__name__}):{val2}")
                    else:
                        print(f"\t\t\t{att2}({ty2.__name__})")
    print("=" * 100)

def cleanup_wandb(signum=None, frame=None):
    """Cleanup handler: sync wandb and delete local files.

    Args:
        signum: Signal number if called from signal handler
        frame: Stack frame if called from signal handler
    """
    global _wandb_wrapper, _simulation_app, _cleanup_completed

    # Prevent multiple cleanup calls
    if _cleanup_completed:
        return
    _cleanup_completed = True

    if _wandb_wrapper is not None:
        print("\n" + "="*80)
        print("[INFO]: WandB Cleanup Starting...")
        print("="*80)
        print("[INFO]:   - Syncing data to cloud")
        print("[INFO]:   - Deleting local run directories")

        try:
            _wandb_wrapper.close(delete_local_files=True)
            print("[INFO]: WandB cleanup complete")
        except Exception as e:
            print(f"[ERROR]: WandB cleanup failed: {e}")
            import traceback
            traceback.print_exc()

        print("="*80)

    if signum is not None:
        # Called from signal handler - need to shutdown simulation and exit
        signal_names = {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT"}
        print(f"[INFO]: Exiting due to signal {signal_names.get(signum, signum)}")

        # Close simulation app before exiting
        if _simulation_app is not None:
            print("[INFO]: Closing simulation app...")
            try:
                _simulation_app.close()
            except Exception as e:
                print(f"[WARNING]: Error closing simulation app: {e}")

        # Force exit (bypasses all exit hooks)
        os._exit(0)

def main(
    args_cli
):
    # Track WandB runs if loading from checkpoint (used later for checkpoint loading)
    checkpoint_runs = None

    # Load configuration - either from WandB checkpoint or from local config file
    if args_cli.checkpoint_tag is not None:
        # Validate checkpoint arguments
        if args_cli.checkpoint_step is None:
            raise ValueError("--checkpoint_step is required when using --checkpoint_tag")

        print(f"[INFO]: Loading config from WandB checkpoint tag '{args_cli.checkpoint_tag}'")

        # Query WandB for runs with the specified tag
        checkpoint_runs = query_runs_by_tag(
            tag=args_cli.checkpoint_tag,
            entity=args_cli.checkpoint_entity,  # Can be None, will use default
            project=args_cli.checkpoint_project
        )

        if len(checkpoint_runs) == 0:
            raise ValueError(f"No runs found with tag '{args_cli.checkpoint_tag}'")

        # Download config from the first run (all runs should have same config)
        configs = reconstruct_config_from_wandb(checkpoint_runs[0])

        # Apply any CLI overrides on top of the downloaded config
        if args_cli.override:
            configManager = ConfigManagerV3()
            parsed_overrides = configManager.parse_cli_overrides(args_cli.override)
            configManager.apply_cli_overrides(configs, parsed_overrides)
            # Re-apply primary config to propagate changes (e.g., num_envs_per_agent -> scene.num_envs)
            configManager._apply_primary_config_to_all(configs)
    else:
        # Load config from local file
        configManager = ConfigManagerV3()
        configs = configManager.process_config(args_cli.config, args_cli.override)

    # Override WandB project for new training runs if specified
    if args_cli.new_project is not None:
        configs['experiment'].wandb_project = args_cli.new_project
        print(f"[INFO]: New training runs will be logged to project: {args_cli.new_project}")

    # Validate manual control mode consistency
    if args_cli.manual_control:
        if 'wrappers' not in configs or not configs['wrappers'].manual_control.enabled:
            raise ValueError("--manual-control flag set but manual_control.enabled=False in config")

    print("[INFO]: STEP 1 - Loading Configuration")
    # set or generate seed
    if configs['primary'].seed == -1:
        configs['primary'].seed = random.randint(0, 10000)
    print(f"[INFO]: Setting global seed: {configs['primary'].seed}")
    set_seed(configs['primary'].seed)
    configs['environment'].seed = configs['primary'].seed
    
    # add wandb tracking tags
    lUtils.add_wandb_tracking_tags(configs)
    print(configs['experiment'].tags)

    # create agent specific configs
    lUtils.define_agent_configs(configs, eval_tag=args_cli.eval_tag)

    # Should not matter but removes annoying warning message
    configs["environment"].sim.render_interval = configs["primary"].decimation

    # Setup camera configuration for manual control visualization
    if args_cli.manual_control:
        print("[INFO]: Setting up camera configuration for manual control...")
        try:
            from isaaclab.sensors import TiledCameraCfg, CameraCfg
            import isaaclab.sim as sim_utils
        except:
            from omni.isaac.lab.sensors import TiledCameraCfg, CameraCfg
            import omni.isaac.lab.sim as sim_utils

        configs['environment'].scene.tiled_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.35),#(1.25, 0.0, 0.35),
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
        print("[INFO]:   Camera configured: 240x180, RGB")

    print(f"[INFO]: Environment Configured from {args_cli.config}")
    print(f"[INFO]: Task: {configs['experiment'].task_name}")
    print(f"[INFO]: Episode length: {configs['environment'].episode_length_s}s")
    print(f"[INFO]: Decimation: {configs['environment'].decimation}")

    print("[INFO]: Ckpt Path:", configs['primary'].ckpt_tracker_path)
    print("[INFO]: Configuration fully loaded")

    # Apply asset variant if specified (must happen after all config overrides)
    task_cfg = configs['environment'].task
    if hasattr(task_cfg, 'apply_asset_variant_if_specified'):
        if task_cfg.apply_asset_variant_if_specified():
            print("[INFO]: Asset variant applied successfully")

    print("=" * 100)
    # ===== STEP 2: CREATE ENVIRONMENT =====
    # Environment creation using fully configured objects from Step 1
    print("[INFO]: Step 2 - Creating environment")

    # Wrap with sensor support if contact sensor config is present AND use_contact_sensor is enabled
    if (hasattr(configs['environment'].task, 'held_fixed_contact_sensor') and
        configs['wrappers'].force_torque_sensor.use_contact_sensor):
        EnvClass = create_sensor_enabled_factory_env(FactoryEnv)
        print("[INFO]:   Using sensor-enabled factory environment")
    else:
        EnvClass = FactoryEnv
        print("[INFO]:   Using standard factory environment")

    # Create environment directly
    env = EnvClass(cfg=configs['environment'], render_mode=None)
    print("[INFO]: Environment created successfully")

    # Debug: Print USD prim hierarchy
    #if configs['primary'].debug_mode:
    debug_print_usd_hierarchy(env)

    # Enable camera light and set viewport camera for manual control mode
    if args_cli.manual_control:
        try:
            import omni.kit.viewport.utility as vp_utils
            viewport_api = vp_utils.get_active_viewport()
            if viewport_api:
                # Switch viewport to use our configured camera
                camera_path = "/World/envs/env_0/Camera"
                viewport_api.set_active_camera(camera_path)
                print(f"[INFO]:   Viewport camera set to {camera_path}")

                # Enable camera light (headlight) using carb settings
                # This is the Omniverse-standard way to enable viewport headlight
                settings.set("/rtx/useViewLightingMode", True)
                print("[INFO]:   Camera light (headlight) enabled via RTX settings")
        except Exception as e:
            print(f"[WARNING]: Could not configure viewport camera: {e}")
            import traceback
            traceback.print_exc()

    # ===== STEP 3: APPLY WRAPPER STACK =====
    # Apply wrappers using pre-configured wrapper settings from Step 1
    print("=" * 100)
    print("[INFO]: Step 3 - Applying wrapper stack")
    env = lUtils.apply_wrappers(env, configs)

    # Print yaw initialization config if set
    task_cfg = configs['environment'].task
    if hasattr(task_cfg, 'fixed_asset_init_orn_deg') and hasattr(task_cfg, 'fixed_asset_init_orn_range_deg'):
        hole_min = task_cfg.fixed_asset_init_orn_deg
        hole_max = hole_min + task_cfg.fixed_asset_init_orn_range_deg
        peg_noise_rad = task_cfg.hand_init_orn_noise[2] if hasattr(task_cfg, 'hand_init_orn_noise') else 0
        peg_noise_deg = peg_noise_rad * 180 / 3.14159
        max_diff = (task_cfg.fixed_asset_init_orn_range_deg / 2) + peg_noise_deg
        print(f"[INFO]:   Yaw init: hole=[{hole_min}°, {hole_max}°], peg=±{peg_noise_deg:.1f}°, max_diff=±{max_diff:.1f}°")

    # Apply physics debug wrapper if debug_mode is enabled
    if configs['primary'].debug_mode:
        from wrappers.debug.physics_debug_wrapper import PhysicsDebugWrapper
        print("  - Applying physics debug wrapper (NaN/explosion monitoring)")
        env = PhysicsDebugWrapper(
            env,
            check_every_n_steps=1,
            print_obs_on_nan=True,
            raise_on_nan=True,  # Will stop training on NaN to catch issues immediately
            verbose=True
        )
    print("  - Applying async critic isaac lab wrapper (derived from SKRL isaaclab wrapper)")
    env = AsyncCriticIsaacLabWrapper(env)

    print("[INFO]: Wrappers Applied successfully")
    print("=" * 100)

    # Store wandb wrapper reference for cleanup
    global _wandb_wrapper
    current = env
    while current is not None:
        if current.__class__.__name__ == 'GenericWandbLoggingWrapper':
            _wandb_wrapper = current
            break
        current = getattr(current, 'env', None)

    if _wandb_wrapper is None:
        print("[WARNING]: GenericWandbLoggingWrapper not found - cleanup will not work")
    else:
        print(f"[INFO]: Found wandb wrapper with {len(_wandb_wrapper.trackers)} tracker(s)")

    # Register signal handlers for graceful cleanup
    signal.signal(signal.SIGTERM, cleanup_wandb)
    signal.signal(signal.SIGINT, cleanup_wandb)
    print("[INFO]: Signal handlers registered (SIGTERM, SIGINT)")
    print("=" * 100)

    #print_configs(configs)

    print("[INFO]: Step 4 - Creating learning objects")
    # Define memory
    device = configs['environment'].sim.device

    # Create memory using pre-configured parameters
    print("[INFO]:   Creating memory")
    memory_type = configs['agent'].memory_type

    if memory_type == 'multi_random':
        memory = MultiRandomMemory(
            memory_size=configs['primary'].rollout_steps(configs['environment'].episode_length_s),
            num_envs=env.num_envs,
            device=device,
            replacement=True,
            num_agents=configs['primary'].total_agents
        )
        print(f"[INFO]:   Memory Created: MultiRandomMemory")
    elif memory_type == 'importance_sampling':
        memory = ImportanceSamplingMemory(
            memory_size=configs['primary'].rollout_steps(configs['environment'].episode_length_s),
            num_envs=env.num_envs,
            device=device,
            replacement=True,
            num_agents=configs['primary'].total_agents,
            target_true_ratio=0.5,  # Default value, can be made configurable later
            min_true_percentage=0.05  # Default value, can be made configurable later
        )
        print(f"[INFO]:   Memory Created: ImportanceSamplingMemory")
    else:
        raise ValueError(
            f"Invalid memory_type '{memory_type}'. "
            f"Must be 'multi_random' or 'importance_sampling'"
        )

    # Create models using pre-configured parameters
    print("[INFO]:   Creating models")
    models = lUtils.create_policy_and_value_models(env, configs)
    print("[INFO]:   Models Created")


    # Set up reward shaping function
    print("[INFO]:   Setting up reward shaping")
    lUtils.set_reward_shaping(configs['environment'], configs['agent'])

    # Create agents using pre-configured parameters
    print("[INFO]:   Creating agents")
    agents = lUtils.create_block_ppo_agents(env, configs, models, memory)
    print("[INFO]:   Agents Created")

    # Load from WandB checkpoint if specified
    if checkpoint_runs is not None:
        print(f"[INFO]: Loading checkpoints from WandB at step {args_cli.checkpoint_step}")

        # Verify number of runs matches expected agents
        num_agents = configs['primary'].total_agents
        if len(checkpoint_runs) != num_agents:
            raise ValueError(
                f"Number of WandB runs ({len(checkpoint_runs)}) does not match total_agents ({num_agents}). "
                f"Found runs: {[r.id for r in checkpoint_runs]}"
            )

        # Download and load checkpoints into agent
        download_dirs = load_checkpoints_parallel(
            runs=checkpoint_runs,
            step=args_cli.checkpoint_step,
            env=env,
            agent=agents
        )

        # Clean up temporary download directories
        import shutil
        for d in download_dirs:
            shutil.rmtree(d, ignore_errors=True)

        print(f"[INFO]: Successfully loaded checkpoints from WandB")

    print("[INFO]: Learning objects instanciated")
    print("=" * 100)

    # reset test
    if args_cli.test_reset:
        # ===== STEP 5: PERFORM SEVERAL RESETS =====
        print("\n" + "="*80)
        print("DYNAMICS RANDOMIZATION RESET TEST")
        print("="*80)

        # Find dynamics wrapper
        dynamics_wrapper = None
        current_env = env
        while current_env is not None:
            if current_env.__class__.__name__ == 'DynamicsRandomizationWrapper':
                dynamics_wrapper = current_env
                break
            if hasattr(current_env, 'env'):
                current_env = current_env.env
            else:
                break

        if dynamics_wrapper is None:
            print("ERROR: DynamicsRandomizationWrapper not found in wrapper chain!")
            print("Please enable dynamics_randomization in your config.")
            return

        print("✓ Found DynamicsRandomizationWrapper")

        # Find hybrid wrapper for force parameters
        hybrid_wrapper = dynamics_wrapper._find_hybrid_wrapper()
        has_hybrid = hybrid_wrapper is not None
        if has_hybrid:
            print("✓ Found HybridForcePositionWrapper")

        # Initial reset
        obs, info = env.reset()
        print(f"\n✓ Initial reset complete ({env.num_envs} environments)")

        # Take a few steps to ensure environment is stable
        print("\nTaking 5 warmup steps...")
        for _ in range(5):
            action = torch.zeros((env.num_envs, env.unwrapped.cfg.action_space), device=device)
            obs, reward, terminated, truncated, info = env.step(action)
        print("✓ Warmup complete")

        # Trigger initial dynamics randomization for all environments
        print("\nTriggering initial dynamics randomization...")
        all_env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
        env.unwrapped._reset_idx(all_env_ids)
        print("✓ All environments randomized")

        def get_params_dict():
            """Get current dynamics parameters from env.unwrapped and wrappers (where they're actually used).
            Only collects parameters that are enabled in the dynamics randomization config.
            All parameters are moved to CPU for consistent comparison."""
            params = {}

            # Friction - access from actual asset PhysX material properties
            if dynamics_wrapper.randomize_friction:
                if hasattr(env.unwrapped, '_held_asset') and hasattr(env.unwrapped._held_asset, 'root_physx_view'):
                    try:
                        materials = env.unwrapped._held_asset.root_physx_view.get_material_properties()
                        # Use ellipsis pattern matching base env and dynamics wrapper: materials[..., 0] = static friction
                        friction = materials[..., 0].clone().float()  # Shape: (num_envs, num_shapes)
                        # All shapes should have same friction value (broadcast during setting), take first shape
                        params['friction'] = friction[:, 0].cpu()  # Move to CPU for consistent comparison
                    except Exception as e:
                        print(f"   ⚠️  Could not read friction from asset: {e}")

            # Mass - access from actual asset PhysX mass properties (as scale factor)
            if dynamics_wrapper.randomize_held_mass:
                if hasattr(env.unwrapped, '_held_asset') and hasattr(env.unwrapped._held_asset, 'root_physx_view'):
                    try:
                        masses = env.unwrapped._held_asset.root_physx_view.get_masses()
                        default_masses = env.unwrapped._held_asset.data.default_mass
                        # masses shape: (num_envs, num_bodies), default_masses shape: (num_envs, num_bodies)
                        # Sum across all bodies to get total mass per environment
                        total_mass = masses.sum(dim=1).clone().float()
                        total_default_mass = default_masses.sum(dim=1).clone().float()
                        # Compute scale factor: current_mass / default_mass
                        mass_scale = total_mass / total_default_mass
                        params['held_mass_scale'] = mass_scale.cpu()  # Move to CPU for consistent comparison
                    except Exception as e:
                        print(f"   ⚠️  Could not read mass from asset: {e}")

            # Controller gains (position and rotation) - from base environment
            if dynamics_wrapper.randomize_gains:
                if hasattr(env.unwrapped, 'task_prop_gains'):
                    params['pos_gains'] = env.unwrapped.task_prop_gains[:, 0].clone().float().cpu()  # First pos dim
                    params['rot_gains'] = env.unwrapped.task_prop_gains[:, 3].clone().float().cpu()  # First rot dim

            # Thresholds - from base environment
            if dynamics_wrapper.randomize_pos_threshold:
                if hasattr(env.unwrapped, 'pos_threshold'):
                    params['pos_threshold'] = env.unwrapped.pos_threshold[:, 0].clone().float().cpu()  # First dim

            if dynamics_wrapper.randomize_rot_threshold:
                if hasattr(env.unwrapped, 'rot_threshold'):
                    params['rot_threshold'] = env.unwrapped.rot_threshold[:, 0].clone().float().cpu()  # First dim

            # Force parameters - from hybrid wrapper (if present)
            if has_hybrid:
                if dynamics_wrapper.randomize_force_threshold:
                    if hasattr(hybrid_wrapper, 'force_threshold'):
                        params['force_threshold'] = hybrid_wrapper.force_threshold[:, 0].clone().float().cpu()

                if dynamics_wrapper.randomize_force_gains:
                    if hasattr(hybrid_wrapper, 'kp'):
                        params['force_gains'] = hybrid_wrapper.kp[:, 0].clone().float().cpu()  # First force dim

            return params

        def print_stats(params, title):
            """Print min, max, std for each parameter."""
            print(f"\n{title}")
            print("-" * 80)
            for name, values in params.items():
                print(f"{name:20s}: min={values.min().item():8.4f}, max={values.max().item():8.4f}, "
                      f"std={values.std().item():8.4f}, mean={values.mean().item():8.4f}")

        # Perform multiple reset tests
        num_tests = 10
        print(f"\nPerforming {num_tests} reset tests...")

        # Initialize aggregation tracking
        total_reset_envs = 0
        total_non_reset_envs = 0
        param_reset_success = {}  # Track how many times each param correctly changed
        param_non_reset_success = {}  # Track how many times each param correctly stayed same

        for test_idx in range(num_tests):
            print(f"\n{'='*80}")
            print(f"TEST {test_idx + 1}/{num_tests}")
            print('='*80)

            # Get parameters before reset
            params_before = get_params_dict()
            print_stats(params_before, "📊 PARAMETERS BEFORE RESET")

            # Randomly select environments to reset (between 10% and 50%)
            num_to_reset = torch.randint(
                low=max(1, env.num_envs // 10),
                high=max(2, env.num_envs // 2),
                size=(1,)
            ).item()

            reset_env_ids = torch.randperm(env.num_envs)[:num_to_reset]
            reset_env_ids = reset_env_ids.sort()[0]  # Sort for easier verification

            print(f"\n🔄 Resetting {num_to_reset} environments: {reset_env_ids[:10].tolist()}" +
                  (f"..." if num_to_reset > 10 else ""))

            # Perform reset via the environment's reset mechanism
            # This will trigger dynamics wrapper's _wrapped_reset_idx
            env.unwrapped._reset_idx(reset_env_ids)

            # Get parameters after reset
            params_after = get_params_dict()
            print_stats(params_after, "📊 PARAMETERS AFTER RESET")

            # Aggregate verification results (don't print individual test results)
            total_reset_envs += num_to_reset

            # Track reset environment changes
            for param_name, values_before in params_before.items():
                values_after = params_after[param_name]

                # Initialize tracking for new parameters
                if param_name not in param_reset_success:
                    param_reset_success[param_name] = 0
                    param_non_reset_success[param_name] = 0

                # Check if reset environments changed
                changed_mask = values_before[reset_env_ids] != values_after[reset_env_ids]
                num_changed = changed_mask.sum().item()
                param_reset_success[param_name] += num_changed

            # Track non-reset environments staying the same
            non_reset_mask = torch.ones(env.num_envs, dtype=torch.bool, device='cpu')
            non_reset_mask[reset_env_ids.cpu()] = False
            non_reset_ids = torch.where(non_reset_mask)[0]

            if len(non_reset_ids) > 0:
                total_non_reset_envs += len(non_reset_ids)

                for param_name, values_before in params_before.items():
                    values_after = params_after[param_name]

                    # Check if non-reset environments stayed the same
                    unchanged_mask = values_before[non_reset_ids] == values_after[non_reset_ids]
                    num_unchanged = unchanged_mask.sum().item()
                    param_non_reset_success[param_name] += num_unchanged

            print(f"✓ Test {test_idx + 1} complete (reset {num_to_reset} envs, kept {len(non_reset_ids)} envs)")

            # Take a few steps between tests
            for _ in range(3):
                action = torch.zeros((env.num_envs, env.unwrapped.cfg.action_space), device=device)
                obs, reward, terminated, truncated, info = env.step(action)

        # Print aggregated summary
        print(f"\n{'='*80}")
        print("AGGREGATED TEST RESULTS")
        print("="*80)

        print(f"\nTotal tests: {num_tests}")
        print(f"Total reset environments: {total_reset_envs}")
        print(f"Total non-reset environments: {total_non_reset_envs}")

        print(f"\n🔍 RESET ENVIRONMENTS (should change):")
        print("-" * 80)
        all_reset_passed = True
        for param_name in sorted(param_reset_success.keys()):
            num_success = param_reset_success[param_name]
            pct_success = (num_success / total_reset_envs) * 100 if total_reset_envs > 0 else 0
            status = "✓" if num_success == total_reset_envs else "✗"
            print(f"{status} {param_name:20s}: {num_success}/{total_reset_envs} changed ({pct_success:.1f}%)")
            if num_success < total_reset_envs:
                all_reset_passed = False

        print(f"\n🔍 NON-RESET ENVIRONMENTS (should stay same):")
        print("-" * 80)
        all_non_reset_passed = True
        for param_name in sorted(param_non_reset_success.keys()):
            num_success = param_non_reset_success[param_name]
            pct_success = (num_success / total_non_reset_envs) * 100 if total_non_reset_envs > 0 else 0
            status = "✓" if num_success == total_non_reset_envs else "✗"
            print(f"{status} {param_name:20s}: {num_success}/{total_non_reset_envs} unchanged ({pct_success:.1f}%)")
            if num_success < total_non_reset_envs:
                all_non_reset_passed = False

        print(f"\n{'='*80}")
        if all_reset_passed and all_non_reset_passed:
            print("✅ ALL TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED")
            if not all_reset_passed:
                print("   - Some reset environments did not change")
            if not all_non_reset_passed:
                print("   - Some non-reset environments unexpectedly changed")
        print("="*80)
        return

    elif args_cli.manual_control:
        # ===== STEP 5: MANUAL CONTROL VISUALIZATION =====
        print("\n" + "="*80)
        print("MANUAL CONTROL MODE")
        print("="*80)
        print("[INFO]: Press Ctrl+C to exit")

        # Initial reset
        print("\n[INFO]: Performing initial reset...")
        obs, info = env.reset()
        print(f"[INFO]: Reset complete ({env.num_envs} environments)")

        # Step loop
        print("[INFO]: Starting step loop (zero actions)...")
        step_count = 0
        try:
            while True:
                # Zero action
                action = torch.zeros((env.num_envs, env.unwrapped.cfg.action_space), device=device)
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1

                # Update simulation app for rendering
                simulation_app.update()

                # Print progress every 100 steps
                if step_count % 100 == 0:
                    print(f"[INFO]: Step {step_count}")

        except KeyboardInterrupt:
            print(f"\n[INFO]: Manual control ended after {step_count} steps")

        return

    else:
        # ===== STEP 5: CREATE AND START TRAINER =====
        # Trainer uses pre-configured parameters from Step 1
        print("[INFO]: Step 5 - Creating trainer")

        cfg_trainer = {
            "timesteps": configs['primary'].total_agents * configs['primary'].max_steps // (configs['primary'].total_num_envs),
            "headless": True,
            "close_environment_at_exit": True,
            "disable_progressbar": configs['agent'].disable_progressbar
        }

        trainer = SequentialTrainer(
            cfg=cfg_trainer,
            env=env,
            agents=agents
        )
        print("[INFO]: Trainer created successfully")
        print("=" * 100)

        # Start training
        print("[INFO]: Step 6 - Starting training...")
        torch.autograd.set_detect_anomaly(True)
        trainer.train()

        # Explicit cleanup after successful training
        print("\n[INFO]: Training completed successfully")
        cleanup_wandb()


if __name__ == "__main__":
    # run the main function

    main(args_cli)

    # close sim app
    simulation_app.close()