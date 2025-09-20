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
    print("isaaclab successfully loaded")
except:
    print("Isaaclab not successfully loaded")
    from omni.isaac.lab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
    )
    import omni.isaac.lab_tasks  # noqa: F401
    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
    print("Fallback worked!")

# Load configuration system
from configs.config_manager import ConfigManager

# seed for reproducibility - set once globally
if args_cli.seed == -1:
    args_cli.seed = random.randint(0, 10000)
print(f"[INFO]: Setting global seed: {args_cli.seed}")
set_seed(args_cli.seed)

# Load and resolve configuration
print(f"\n\n[INFO]: Loading configuration from {args_cli.config}\n")
resolved_config = ConfigManager.load_and_resolve_config(args_cli.config, args_cli.override)

# Use task name from config if not provided as argument
if args_cli.task is None:
    args_cli.task = resolved_config.get('defaults', {}).get('task_name', "Isaac-Factory-PegInsert-Local-v0")
    print(f"[INFO]: Using task name from config: {args_cli.task}")

# Extract configuration sections for convenience
primary = resolved_config['primary']
derived = resolved_config['derived']
environment = resolved_config.get('environment', {})
learning = resolved_config['learning']
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

def main():
    """Main training function using our configuration system instead of Hydra."""

    # Create Isaac Lab environment configuration
    # We'll create a basic ManagerBasedRLEnvCfg and then apply our configuration to it
    try:
        # Try to get the task environment configuration from Isaac Lab
        import gymnasium as gym
        import omni.isaac.lab_tasks  # This registers the tasks

        # Get the environment configuration class for the task
        env_spec = gym.spec(args_cli.task)
        if hasattr(env_spec, 'kwargs') and 'cfg' in env_spec.kwargs:
            # Use the default configuration from the task
            env_cfg = env_spec.kwargs['cfg']()
        else:
            # Fallback to a basic configuration
            env_cfg = ManagerBasedRLEnvCfg()

    except Exception as e:
        print(f"[INFO]: Could not get default environment config, using basic config: {e}")
        env_cfg = ManagerBasedRLEnvCfg()

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

    print("Ckpt Path:", derived.get('ckpt_tracker_path', "/nfs/stak/users/brownhun/ckpt_tracker2.txt"))

    # Step 1: Apply basic configuration to Isaac Lab configs
    print("[INFO]: Step 1 - Applying basic configuration")
    ConfigManager.apply_to_isaac_lab(env_cfg, agent_cfg, resolved_config)

    # Set device from configuration or default to cuda if available
    if hasattr(env_cfg, 'sim') and hasattr(env_cfg.sim, 'device'):
        if env_cfg.sim.device is None:
            env_cfg.sim.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        # If sim config doesn't exist, create it
        env_cfg.sim = type('SimConfig', (), {'device': 'cuda' if torch.cuda.is_available() else 'cpu'})()

    # Step 2: Set break forces and agent distribution
    print("[INFO]: Step 2 - Setting break forces and agent distribution")
    env_cfg.break_force = primary['break_forces']
    agent_cfg['agent']['break_force'] = primary['break_forces']

    # Step 3: Apply easy mode if debug enabled
    if primary.get('debug_mode', False):
        print("[INFO]: Step 3 - Applying easy mode for debug")
        lUtils.apply_easy_mode(env_cfg, agent_cfg)

    # Step 4: Set environment scene configuration
    print("[INFO]: Step 4 - Setting environment scene configuration")
    lUtils.configure_environment_scene(env_cfg, primary, derived)

    # Step 5: Configure force sensor if needed
    if wrappers_config.get('force_torque_sensor', {}).get('enabled', False):
        print("[INFO]: Step 5 - Enabling force-torque sensor")
        lUtils.enable_force_sensor(env_cfg)

    # Step 6: Apply environment parameter overrides
    print("[INFO]: Step 6 - Applying environment parameter overrides")
    lUtils.apply_environment_overrides(env_cfg, environment)

    # Step 7: Calculate rollout parameters
    max_rollout_steps = derived['rollout_steps']
    print(f"[INFO]: Step 7 - Calculated rollout steps: {max_rollout_steps}")

    # Step 8: Update agent configuration with learning parameters
    print("[INFO]: Step 8 - Updating agent configuration")
    lUtils.apply_learning_config(agent_cfg, learning, max_rollout_steps)

    # Step 9: Apply model configuration
    print("[INFO]: Step 9 - Applying model configuration")
    lUtils.apply_model_config(agent_cfg, model)

    # Step 10: Set up experiment and logging paths
    print("[INFO]: Step 10 - Setting up experiment and logging")
    lUtils.setup_experiment_logging(env_cfg, agent_cfg, resolved_config)

    print("[INFO]: Creating Env...")
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode=None
    )

    print("[INFO]: Env Built!")

    # Step 11: Apply wrapper stack
    print("[INFO]: Step 11 - Applying wrapper stack")
    if wrappers_config.get('fragile_objects', {}).get('enabled', False):
        print("  - Applying FragileObjectWrapper")
        env = lUtils.apply_fragile_object_wrapper(env, wrappers_config['fragile_objects'], primary)

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

    if wrappers_config.get('factory_metrics', {}).get('enabled', False):
        print("  - Applying FactoryMetricsWrapper")
        env = lUtils.apply_factory_metrics_wrapper(env, primary)

    if wrappers_config.get('wandb_logging', {}).get('enabled', False):
        print("  - Applying GenericWandbLoggingWrapper")
        env = lUtils.apply_wandb_logging_wrapper(env, wrappers_config['wandb_logging'], derived, agent_cfg, env_cfg, resolved_config)

    if wrappers_config.get('action_logging', {}).get('enabled', False):
        print("  - Applying EnhancedActionLoggingWrapper")
        env = lUtils.apply_enhanced_action_logging_wrapper(env, wrappers_config['action_logging'])

    # Step 12: Wrap for SKRL
    print("[INFO]: Step 12 - Wrapping environment for SKRL")
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # Step 13: Set preprocessors
    print("[INFO]: Step 13 - Setting up preprocessors")
    ## ORIGINAL CODE - SHARED PREPROCESSORS (BROKEN FOR MULTI-AGENT) ##
    # lUtils.setup_preprocessors(env_cfg, agent_cfg, env, learning) # TODO MAKE THIS MULTI AGENT

    ## NEW CODE - PER-AGENT PREPROCESSORS HANDLED IN create_block_ppo_agents ##
    print("  - Preprocessor setup moved to create_block_ppo_agents for per-agent independence")

    print("[INFO]: Observation Space Size:", env.cfg.observation_space)
    print("[INFO]: State Space Size:", env.cfg.state_space)
    print("[INFO]: Action Space Size:", env.action_space)

    device = env_cfg.sim.device

    # Step 14: Create memory
    print("[INFO]: Step 14 - Creating memory")
    memory = MultiRandomMemory(
        memory_size=derived['rollout_steps'],
        num_envs=env.num_envs,
        device=device,
        replacement=True,
        num_agents=derived['total_agents']
    )

    # Step 15: Create models
    print("[INFO]: Step 15 - Creating models")
    models = lUtils.create_policy_and_value_models(env_cfg, agent_cfg, env, model, wrappers_config, derived)

    # Step 16: Create agents
    print("[INFO]: Step 16 - Creating agents and optimizer")
    ## ORIGINAL CODE - USING BLOCKWANDBLOGGERPPO ##
    # agents = lUtils.create_block_wandb_agents(env_cfg, agent_cfg, env, models, memory, derived, learning)

    ## NEW CODE - USING BLOCKPPO WITH PER-AGENT PREPROCESSORS ##
    agents = lUtils.create_block_ppo_agents(env_cfg, agent_cfg, env, models, memory, derived, learning)

    print(agents)

    # configure and instantiate the RL trainer
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
    print("Trainer Created")

    # Training loop
    torch.autograd.set_detect_anomaly(True)
    trainer.train()


if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()