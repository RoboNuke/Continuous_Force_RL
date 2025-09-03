


import argparse
import sys

try:
    from isaaclab.app import AppLauncher
except:
    from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")

# exp
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Local-v0", help="Name of the task.")
parser.add_argument("--log_ckpt_data", type=int, default=0, help="Value of 1 turns on logging checkpoint data for replay")
parser.add_argument("--ckpt_tracker_path", type=str, default="/nfs/stak/users/brownhun/ckpt_tracker2.txt", help="Path the ckpt recording data")
parser.add_argument("--easy_mode", action="store_true", default=False, help="Limits the intialization to simplify problem")
parser.add_argument("--use_obs_noise", action="store_true", default=False, help="Adds Gaussian noise specificed by env cfg to observations")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=-1, help="Seed used for the environment")
parser.add_argument("--max_steps", type=int, default=10240000, help="RL Policy training iterations.")
parser.add_argument("--force_encoding", type=str, default=None, help="Which type of force encoding to use if force is included")
parser.add_argument("--ckpt_path", type=str, default=None, help="Absolute path to cp file to begin run from")
parser.add_argument("--num_agents", type=int, default=1, help="How many agents to train in parallel")
parser.add_argument("--dmp_obs", default=False, action="store_true", help="Should we use dmps for the observation space")
parser.add_argument("--init_eval", default=True, action="store_false", help="When added, we will not perform an eval before any training has happened")
parser.add_argument("--decimation", type=int, default=16, help="How many simulation steps between policy observations")
parser.add_argument("--history_sample_size", type=int, default=8, help="How many samples to keep from sim steps, spread evenly from zero to decimation-1")
parser.add_argument("--policy_hz", type=int, default=15, help="Rate in hz that the policy should get new observations")
parser.add_argument("--use_ft_sensor", type=int, default=0, help="Adds force sensor data to the observation space")
parser.add_argument("--break_force", type=str, default="-1.0", help="Force at which the held object breaks (peg, gear or nut)")
parser.add_argument("--exp_tag", type=str, default="debug", help="Tag to apply to exp in wandb")
parser.add_argument("--wandb_group_prefix", type=str, default="", help="Prefix of wandb group to add this to")

# controller/agent params
parser.add_argument("--parallel_control", type=int, default=0, help="Switches to parallel force position control as the action space")
parser.add_argument("--parallel_agent", type=int, default=0, help="Switches to parallel force position agent using calculated log probs based on controller")
parser.add_argument("--hybrid_control", type=int, default=0, help="Switches to hybrid force/position control as the action space")
parser.add_argument("--hybrid_agent", type=int, default=0, help="Switches to hybrid force/position agent using calculated log probs based on controller")
parser.add_argument("--hybrid_selection_reward", type=str, default="simp", help="Allows different rewards on the force/position selection: options are [simp, dirs, delta]")
parser.add_argument("--control_torques", type=int, default=0, help="Allows hybrid control to effect torques not just forces")
parser.add_argument("--impedance_control", type=int, default=0, help="Switches to impedance control as the action space")
parser.add_argument("--impedance_agent", type=int, default=0, help="Switches to impedance agent using calculated log probs based on controller")
parser.add_argument("--control_damping", type=int, default=0, help="Allows Impedance Controller Policy to predict the damping constants, else calculated with kd=2*sqrt(kp) (critically damped)")
parser.add_argument("--lr_scheduler_type", type=str, default="cfg", help="Sets the learning rate scheduler type possible values: 'none', 'KLAdaptiveLR', 'LinearWarmup' ** parameters set in cfg file")
# logging
parser.add_argument("--exp_name", type=str, default=None, help="What to name the experiment on WandB")
parser.add_argument("--exp_dir", type=str, default=None, help="Directory to store the experiment in")
parser.add_argument("--dump_yaml", action="store_true", default=False, help="Store config files in yaml format")
parser.add_argument("--log_smoothness_metrics", action="store_true", default=False, help="Log the sum squared velocity, jerk and force metrics")
parser.add_argument("--no_vids", action="store_true", default=False, help="Set up sim environment to support cameras")
parser.add_argument("--debug_mode", action="store_true", default=False, help="Enables extra information for debugging")

# wandb
parser.add_argument("--no_log_wandb", action="store_false", default=True, help="Disables the wandb logger")
parser.add_argument("--wandb_entity", type=str, default="hur", help="Name of wandb entity")
parser.add_argument("--wandb_api_key", type=str, default="-1", help="API key for WandB")
parser.add_argument("--wandb_project", type=str, default="Continuous_Force_RL", help="Wandb project to save logging to")

parser.add_argument("--sel_adjs", type=str, default="none", help="A comma seperated list from the following ('init_bias', 'zero_weights', 'force_add_zout', 'scale_zout'")

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

import itertools
#from wrappers.video_recoder_wrapper import ExtRecordVideo
import os
import random
from datetime import datetime

import skrl
from packaging import version

from skrl.utils import set_seed
from skrl.memories.torch import RandomMemory
from memories.multi_random import MultiRandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

from wrappers.smoothness_obs_wrapper import SmoothnessObservationWrapper
from learning.ext_sequential_trainer import ExtSequentialTrainer, EXT_SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from wrappers.close_gripper_action_wrapper import GripperCloseEnv
from models.default_mixin import Shared
from envs.factory.dmp_obs_factory_env import DMPObsFactoryEnv
from agents.agent_list import AgentList

#import envs.FPiH.config.franka
import envs.factory
from torch.optim.lr_scheduler import LinearLR

try:
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
    )
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config
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
    from omni.isaac.lab.utils.dict import print_dict

    import omni.isaac.lab_tasks  # noqa: F401
    from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
    print("Fallback worked!")

#from wrappers.info_video_recorder_wrapper import InfoRecordVideo
import copy
import torch
import learning.launch_utils as lUtils
# seed for reproducibility
#set_seed(args_cli.seed)  # e.g. `set_seed(42)` for fixed seed
if args_cli.seed == -1:
    args_cli.seed = random.randint(0, 10000)
set_seed(args_cli.seed)
#set_seed(7378)


if args_cli.debug_mode:
    agent_cfg_entry_point = f"SimBaNet_debug_entry_point"
else:
    agent_cfg_entry_point = f"SimBaNet_ppo_cfg_entry_point"

evaluating = False
print("\n\nImports complete\n\n")
@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, 
    agent_cfg: dict
):
    global evaluating
    global mp_agent
    env_cfg.filter_collisions = True
    print("Ckpt Path:", args_cli.ckpt_tracker_path)

    """ Set up fragileness """
    print("Break Force:", args_cli.break_force.split(","))
    if len(args_cli.break_force.split(",")) > 1:
        forces = [ float(val)  for val in args_cli.break_force.split(",")]
        env_cfg.break_force = forces
    else:
        env_cfg.break_force = float(args_cli.break_force)
    agent_cfg['agent']['break_force'] = env_cfg.break_force
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    lUtils.set_use_force(# conditions we need force torque sensor
        env_cfg,
        agent_cfg,
        args_cli,
        args_cli.parallel_control==1 or
        args_cli.hybrid_control==1 or
        args_cli.use_ft_sensor==1
    )

    if agent_cfg['agent']['rewards_shaper_scale'] > 0.0:
        def scale_reward(rew, timestep, timesteps, scale=agent_cfg['agent']['rewards_shaper_scale']):
            return rew * scale
        agent_cfg['agent']['rewards_shaper'] = scale_reward
    # set initialization params
    lUtils.set_easy_mode(env_cfg, agent_cfg, args_cli.easy_mode)
    lUtils.set_time_params(env_cfg, args_cli)
    lUtils.set_use_obs_noise(env_cfg, args_cli.use_obs_noise)
    
    max_rollout_steps = int((1/env_cfg.sim.dt) / env_cfg.decimation * env_cfg.episode_length_s)#TODO
    print("[INFO]: Maximum Steps ", max_rollout_steps)
    
    lUtils.set_controller_tagging(env_cfg, agent_cfg, args_cli, max_rollout_steps)
    lUtils.set_selection_adjustments(env_cfg, agent_cfg, args_cli)
    lUtils.set_wandb_env_data(env_cfg, agent_cfg, args_cli)
    
    env_per_agent = args_cli.num_envs // args_cli.num_agents
    args_cli.max_steps += max_rollout_steps * env_per_agent - (args_cli.max_steps % (max_rollout_steps * env_per_agent))
    lUtils.set_learn_rate_scheduler(env_cfg, agent_cfg, args_cli)
    lUtils.set_individual_agent_log_paths(env_cfg, agent_cfg, args_cli)
    print("[INFO]: Creating Env...")
    env = gym.make(
        args_cli.task, 
        cfg=env_cfg, 
        render_mode=None
    )
    # load and wrap the Isaac Lab environment
    print("[INFO]: Env Built!")
    lUtils.set_controller_wrapper(env_cfg, agent_cfg, args_cli, env)
    
    if args_cli.log_smoothness_metrics:
        print("\n\n[INFO]: Recording Smoothness Metrics in info.\n\n")
        env = SmoothnessObservationWrapper(env)
        
    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(
        env, 
        ml_framework="torch"
    )      
    lUtils.set_preprocessors(
        env_cfg, 
        agent_cfg, 
        env, 
        state=agent_cfg['agent']['state_preprocessor'], 
        value=agent_cfg['agent']['value_preprocessor']
    )
    print("[INFO]: Observation Space Size:", env.cfg.observation_space)
    print("[INFO]: State Space Size:", env.cfg.state_space)
    print("[INFO]: Action Space Size:", env.action_space)
    
    device = "cuda" #env.device

    memory = MultiRandomMemory( #AdaptiveRandomMemory( #
        memory_size=agent_cfg['agent']["rollouts"], 
        num_envs=env.num_envs, 
        device=device,
        replacement=True#,
        #num_agents=args_cli.num_agents
    )
    models = lUtils.set_block_models(env_cfg, agent_cfg, args_cli, env) 
    #models = lUtils.set_models(env_cfg, agent_cfg, args_cli, env)
    #print(models)
    #agents = lUtils.set_agent(env_cfg, agent_cfg, args_cli, models, memory, env)
    agents = lUtils.set_block_agent(env_cfg, agent_cfg, args_cli, models, memory, env)
    print(agents)
    
    # configure and instantiate the RL trainer
    cfg_trainer = {
        "timesteps": args_cli.max_steps // (args_cli.num_envs * args_cli.num_agents), 
        "headless": True,
        "close_environment_at_exit": True,
        "disable_progressbar": agent_cfg['agent']['disable_progressbar']
    }
    
    trainer = ExtSequentialTrainer(
        cfg = cfg_trainer,
        env = env,
        agents = agents
    )
    print("Trainer Created")

    # our actual learning loop
    ckpt_int = agent_cfg["agent"]["experiment"]["checkpoint_interval"]
    num_evals = max(1,args_cli.max_steps // (ckpt_int * env_per_agent))
    evaluating = False

    vid_env = None
    
    torch.autograd.set_detect_anomaly(True)
    trainer.train(args_cli.max_steps // env_per_agent, vid_env)

    
if __name__ == "__main__":
    # run the main function

    main()

    # close sim app
    simulation_app.close()
