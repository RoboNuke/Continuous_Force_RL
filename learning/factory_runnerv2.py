import argparse
import sys

try:
    from isaaclab.app import AppLauncher
except:
    from omni.isaac.lab.app import AppLauncher

# Minimal argparse arguments - configuration system handles the rest
parser = argparse.ArgumentParser(description="Train an RL agent with configuration system.")

# Essential arguments
parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
parser.add_argument("--task", type=str, default=None, help="Name of the task (defaults to config value)")
#parser.add_argument("--device", type=str, default=None, help="Device to run on")
parser.add_argument("--seed", type=int, default=-1, help="Random seed for reproducibility (-1 for random)")
parser.add_argument("--override", action="append", help="Override config values: key=value")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

args_cli.video = False
args_cli.enable_cameras = False

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

print("\n\n\n Calling App Launcher \n\n\n\n")
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("\n\n\nApp Launched\n\n\n")

import gymnasium as gym
import random
import torch

from skrl.utils import set_seed
from skrl.trainers.torch import SequentialTrainer
from memories.multi_random import MultiRandomMemory

import envs.factory
import learning.launch_utils_v2 as lUtils

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

# Load configuration system - support both old and new systems
import os
USE_NEW_CONFIG = os.getenv('USE_NEW_CONFIG', 'false').lower() == 'true'

if USE_NEW_CONFIG:
    print("[INFO]: Using NEW configuration system (ConfigManagerV2)")
    from configs.config_manager_v2 import ConfigManagerV2
    from configs.cfg_exts.ctrl_cfg import ExtendedCtrlCfg
else:
    print("[INFO]: Using LEGACY configuration system (ConfigManager)")
    from configs.config_manager import ConfigManager
    from configs.cfg_exts.ctrl_cfg import ExtendedCtrlCfg

# seed for reproducibility - set once globally
if args_cli.seed == -1:
    args_cli.seed = random.randint(0, 10000)
print(f"[INFO]: Setting global seed: {args_cli.seed}")
set_seed(args_cli.seed)

# Load and resolve configuration
print(f"\n\n[INFO]: Loading configuration from {args_cli.config}\n")

if USE_NEW_CONFIG:
    # Use new ConfigManagerV2 system
    config_bundle = ConfigManagerV2.load_defaults_first_config(
        config_path=args_cli.config,
        cli_overrides=args_cli.override or [],
        cli_task=args_cli.task
    )

    # Convert to legacy format for compatibility with existing runner code
    resolved_config = ConfigManagerV2.get_legacy_config_dict(config_bundle)

    # Extract task name from bundle
    if args_cli.task is None:
        args_cli.task = config_bundle.task_name
        print(f"[INFO]: Using task name from config: {args_cli.task}")

else:
    # Use legacy ConfigManager system
    resolved_config = ConfigManager.load_and_resolve_config(args_cli.config, args_cli.override)

    # Use task name from config if not provided as argument
    if args_cli.task is None:
        args_cli.task = resolved_config.get('defaults', {}).get('task_name', "Isaac-Factory-PegInsert-Local-v0")
        print(f"[INFO]: Using task name from config: {args_cli.task}")

# Extract configuration sections for convenience (same structure from both systems)
primary = resolved_config['primary']
derived = resolved_config['derived']
environment = resolved_config.get('environment', {})
agent_config = resolved_config['agent']
model = resolved_config['model']
wrappers_config = resolved_config.get('wrappers', {})
experiment = resolved_config.get('experiment', {})

print(f"[INFO]: Configuration loaded successfully")
print(f"  - Total agents: {derived['total_agents']}")
print(f"  - Environments per agent: {primary['num_envs_per_agent']}")
print(f"  - Total environments: {derived['total_num_envs']}")
print(f"  - Break forces: {primary['break_forces']}")
print(f"  - Rollout steps: {derived['rollout_steps']}")

print("\n\nImports complete\n\n")

def add_force_torque_to_isaaclab_configs():
    """Add force-torque dimensions to IsaacLab configuration dictionaries if force-torque sensor is enabled."""
    if not wrappers_config.get('force_torque_sensor', {}).get('enabled', False):
        print("[INFO]: Force-torque sensor not enabled, skipping dimension config modification")
        return False

    try:
        from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
        print("[INFO]: Imported IsaacLab factory configs from isaaclab_tasks (v2.0.0+)")
    except ImportError:
        try:
            from omni.isaac.lab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
            print("[INFO]: Imported IsaacLab factory configs from omni.isaac.lab_tasks (v1.4.1)")
        except ImportError:
            print("[WARNING]: Could not import IsaacLab factory configs - force-torque sensor may cause dimension mismatch")
            return False

    # Add force-torque dimensions if not already present
    modified = False
    if 'force_torque' not in OBS_DIM_CFG:
        OBS_DIM_CFG['force_torque'] = 6
        modified = True
        print("[INFO]: Added 'force_torque: 6' to OBS_DIM_CFG")

    if 'force_torque' not in STATE_DIM_CFG:
        STATE_DIM_CFG['force_torque'] = 6
        modified = True
        print("[INFO]: Added 'force_torque: 6' to STATE_DIM_CFG")

    if not modified:
        print("[INFO]: Force-torque dimensions already present in IsaacLab configs")

    return True

def main():
    """Main training function using our configuration system instead of Hydra."""

    # Add force-torque dimensions to IsaacLab configs if needed
    add_force_torque_to_isaaclab_configs()

    # Create Isaac Lab environment configuration
    # We'll create a basic ManagerBasedRLEnvCfg and then apply our configuration to it
    try:
        # Try to get the task environment configuration from Isaac Lab
        import gymnasium as gym
        # Tasks module already imported in try/except block above

        print(f"[DEBUG]: Attempting to load task: {args_cli.task}")
        env_spec = gym.spec(args_cli.task)
        print(f"[DEBUG]: Found env_spec: {env_spec}")

        if hasattr(env_spec, 'kwargs') and 'env_cfg_entry_point' in env_spec.kwargs:
            # Use the default configuration from the task
            env_cfg = env_spec.kwargs['env_cfg_entry_point']()

            # Replace the ctrl config with our extended version
            print(f"[DEBUG]: Replacing ctrl config with ExtendedCtrlCfg")
            original_ctrl = env_cfg.ctrl
            extended_ctrl = ExtendedCtrlCfg()

            # Copy all existing ctrl values to our extended config
            for attr_name in dir(original_ctrl):
                if not attr_name.startswith('_') and hasattr(extended_ctrl, attr_name):
                    original_value = getattr(original_ctrl, attr_name)
                    if not callable(original_value):
                        setattr(extended_ctrl, attr_name, original_value)
                        print(f"[DEBUG]:   Copied {attr_name}: {original_value}")

            # Replace the ctrl config
            env_cfg.ctrl = extended_ctrl
            print(f"[DEBUG]: Successfully replaced ctrl with ExtendedCtrlCfg")

            print(f"[DEBUG]: Successfully loaded Isaac Lab config")
            print(f"[DEBUG]: Has obs_order: {hasattr(env_cfg, 'obs_order')}")
            print(f"[DEBUG]: Has state_order: {hasattr(env_cfg, 'state_order')}")
            if hasattr(env_cfg, 'obs_order'):
                print(f"[DEBUG]: obs_order: {env_cfg.obs_order}")
            if hasattr(env_cfg, 'state_order'):
                print(f"[DEBUG]: state_order: {env_cfg.state_order}")
        else:
            print(f"[ERROR]: env_spec has no cfg attribute")
            print(f"[ERROR]: env_spec.kwargs: {getattr(env_spec, 'kwargs', 'None')}")
            raise ValueError(f"Invalid task configuration for {args_cli.task}")

    except Exception as e:
        print(f"[ERROR]: Failed to load Isaac Lab task configuration: {e}")
        print(f"[ERROR]: Task name: {args_cli.task}")
        print(f"[ERROR]: This likely means:")
        print(f"  1. The task name is incorrect")
        print(f"  2. Isaac Lab tasks are not properly registered")
        print(f"  3. Isaac Lab is not properly installed")
        sys.exit(1)

    # Create basic agent configuration structure
    agent_cfg = {
        'agent': {
            'class': 'PPO',
            'disable_progressbar': True,
        },
        'models': {
            'policy': {},
            'value': {}
        }
    }

    # Merge agent configuration from resolved config
    agent_config_from_file = resolved_config.get('agent', {})
    if agent_config_from_file:
        print("[INFO]: Merging agent configuration from config file")
        for key, value in agent_config_from_file.items():
            agent_cfg['agent'][key] = value
            print(f"[CONFIG]: Setting agent.{key} = {value}")

    print("Ckpt Path:", derived.get('ckpt_tracker_path', "/nfs/stak/users/brownhun/ckpt_tracker2.txt"))

    # ===== STEP 1: COMPLETE CONFIGURATION =====
    # All configuration happens here - no other steps should modify config objects
    print("[INFO]: Step 1 - Applying complete configuration")
    ConfigManager.apply_complete_configuration(env_cfg, agent_cfg, resolved_config)

    # Debug: Print configurations
    ConfigManager.print_env_config(env_cfg)
    ConfigManager.print_agent_config(agent_cfg)

    # Validate factory configuration
    print("[INFO]: Step 1.5 - Validating factory configuration")
    lUtils.validate_factory_configuration(env_cfg)

    # Extract derived values calculated in Step 1
    max_rollout_steps = derived['rollout_steps']
    print(f"[INFO]: Configuration complete - rollout steps: {max_rollout_steps}")

    # ===== STEP 2: CREATE ENVIRONMENT =====
    # Environment creation using fully configured objects from Step 1
    print("[INFO]: Step 2 - Creating environment")
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode=None
    )
    print("[INFO]: Environment created successfully")

    # ===== STEP 3: APPLY WRAPPER STACK =====
    # Apply wrappers using pre-configured wrapper settings from Step 1
    print("[INFO]: Step 3 - Applying wrapper stack")
    if wrappers_config.get('fragile_objects', {}).get('enabled', False):
        print("  - Applying FragileObjectWrapper")
        env = lUtils.apply_fragile_object_wrapper(env, wrappers_config['fragile_objects'], primary, derived)

    if wrappers_config.get('force_torque_sensor', {}).get('enabled', False):
        print("  - Applying ForceTorqueWrapper")
        env = lUtils.apply_force_torque_wrapper(env, wrappers_config['force_torque_sensor'])

    if wrappers_config.get('observation_manager', {}).get('enabled', False):
        print("  - Applying ObservationManagerWrapper")
        env = lUtils.apply_observation_manager_wrapper(env, wrappers_config['observation_manager'])

    if wrappers_config.get('observation_noise', {}).get('enabled', False):
        print("  - Applying ObservationNoiseWrapper")
        env = lUtils.apply_observation_noise_wrapper(env, wrappers_config['observation_noise'])

    if wrappers_config.get('hybrid_control', {}).get('enabled', False):
        print("  - Applying HybridForcePositionWrapper")
        env = lUtils.apply_hybrid_control_wrapper(env, wrappers_config['hybrid_control'])

    if wrappers_config.get('wandb_logging', {}).get('enabled', False):
        print("  - Applying GenericWandbLoggingWrapper")
        env = lUtils.apply_wandb_logging_wrapper(env, wrappers_config['wandb_logging'], derived, agent_cfg, env_cfg, resolved_config)

    if wrappers_config.get('factory_metrics', {}).get('enabled', False):
        print("  - Applying FactoryMetricsWrapper")
        env = lUtils.apply_factory_metrics_wrapper(env, derived)

    if wrappers_config.get('action_logging', {}).get('enabled', False):
        print("  - Applying EnhancedActionLoggingWrapper")
        env = lUtils.apply_enhanced_action_logging_wrapper(env, wrappers_config['action_logging'])

    # ===== STEP 4: WRAP FOR SKRL WITH ASYNC CRITIC SUPPORT =====
    print("[INFO]: Step 4 - Wrapping environment for SKRL with AsyncCriticIsaacLabWrapper")
    from wrappers.skrl.async_critic_isaaclab_wrapper import AsyncCriticIsaacLabWrapper
    env = AsyncCriticIsaacLabWrapper(env)

    print("[INFO]: Environment wrapped successfully")
    print(f"[INFO]: Observation Space Size: {env.cfg.observation_space}")
    print(f"[INFO]: State Space Size: {env.cfg.state_space}")
    print(f"[INFO]: Action Space Size: {env.action_space}")

    # ===== STEP 5: CREATE AGENTS AND TRAINING COMPONENTS =====
    # All components use pre-configured objects from Step 1
    print("[INFO]: Step 5 - Creating training components")

    device = env_cfg.sim.device

    # Create memory using pre-configured parameters
    print("[INFO]:   Creating memory")
    memory = MultiRandomMemory(
        memory_size=derived['rollout_steps'],
        num_envs=env.num_envs,
        device=device,
        replacement=True,
        num_agents=derived['total_agents']
    )

    # Create models using pre-configured parameters
    print("[INFO]:   Creating models")
    models = lUtils.create_policy_and_value_models(env_cfg, agent_cfg, env, model, wrappers_config, derived)

    # Create agents using pre-configured parameters
    print("[INFO]:   Creating agents")
    agents = lUtils.create_block_ppo_agents(env_cfg, agent_cfg, env, models, memory, derived)

    # ===== STEP 6: CREATE AND START TRAINER =====
    # Trainer uses pre-configured parameters from Step 1
    print("[INFO]: Step 6 - Creating and starting trainer")

    cfg_trainer = {
        "timesteps": derived['max_steps'] // (derived['total_num_envs']),
        "headless": True,
        "close_environment_at_exit": True,
        "disable_progressbar": agent_cfg['agent']['disable_progressbar']
    }

    trainer = SequentialTrainer(
        cfg=cfg_trainer,
        env=env,
        agents=agents
    )
    print("[INFO]: Trainer created successfully")

    # Start training
    print("[INFO]: Starting training...")
    torch.autograd.set_detect_anomaly(True)
    trainer.train()


if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()