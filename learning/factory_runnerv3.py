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

args_cli.video = True #False
args_cli.enable_cameras = True #False
args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

print("\n\n\n Calling App Launcher \n\n\n\n")
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("\n\n\nApp Launched\n\n\n")


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
import learning.launch_utils_v3 as lUtils
from configs.config_manager_v3 import ConfigManagerV3
from wrappers.skrl.async_critic_isaaclab_wrapper import AsyncCriticIsaacLabWrapper
print("\n\nImports complete\n\n")

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

def main(
    args_cli
):
    # unpack configuration files
    configManager = ConfigManagerV3()

    configs = configManager.process_config(args_cli.config, args_cli.override)
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
    lUtils.define_agent_configs(configs)

    # Should not matter but removes annoying warning message    
    configs["environment"].sim.render_interval = configs["primary"].decimation


    print(f"[INFO]: Environment Configured from {args_cli.config}")
    print(f"[INFO]: Task: {configs['experiment'].task_name}")
    print(f"[INFO]: Episode length: {configs['environment'].episode_length_s}s")
    print(f"[INFO]: Decimation: {configs['environment'].decimation}")

    print("[INFO]: Ckpt Path:", configs['primary'].ckpt_tracker_path)
    print("[INFO]: Configuration fully loaded")
    print("=" * 100)
    # ===== STEP 2: CREATE ENVIRONMENT =====
    # Environment creation using fully configured objects from Step 1
    print("[INFO]: Step 2 - Creating environment")
    env = gym.make(
        id=configs['experiment'].task_name,
        cfg=configs['environment'],
        render_mode=None
    )
    print("[INFO]: Environment created successfully")
    
    # ===== STEP 3: APPLY WRAPPER STACK =====
    # Apply wrappers using pre-configured wrapper settings from Step 1
    print("=" * 100)
    print("[INFO]: Step 3 - Applying wrapper stack")
    env = lUtils.apply_wrappers(env, configs)
    print("  - Applying async critic isaac lab wrapper (derived from SKRL isaaclab wrapper)")
    env = AsyncCriticIsaacLabWrapper(env)
    print("[INFO]: Wrappers Applied successfully")
    print("=" * 100)
    
    #print_configs(configs)
    
    print("[INFO]: Step 4 - Creating learning objects")
    # Define memory
    device = configs['environment'].sim.device

    # Create memory using pre-configured parameters
    print("[INFO]:   Creating memory")
    memory = MultiRandomMemory(
        memory_size=configs['primary'].rollout_steps(configs['environment'].episode_length_s),
        num_envs=env.num_envs,
        device=device,
        replacement=True,
        num_agents=configs['primary'].total_agents
    )
    print("[INFO]:   Memory Created")

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
    print("[INFO]: Learning objects instanciated")
    print("=" * 100)

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

    
if __name__ == "__main__":
    # run the main function

    main(args_cli)

    # close sim app
    simulation_app.close()