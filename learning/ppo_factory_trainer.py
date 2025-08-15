import argparse
import sys

#try:
#    from isaaclab.app import AppLauncher
#except:
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")

# exp
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Local-v0", help="Name of the task.")
parser.add_argument("--log_ckpt_data", type=int, default=0, help="Value of 1 turns on logging checkpoint data for replay")
parser.add_argument("--ckpt_tracker_path", type=str, default="/nfs/stak/users/brownhun/ckpt_tracker2.txt", help="Path the ckpt recording data")
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
parser.add_argument("--break_force", type=float, default=-1.0, help="Force at which the held object breaks (peg, gear or nut)")
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

if not args_cli.no_vids:  
    args_cli.video = True
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args


# launch our threads before simulation app
from agents.mp_agent import MPAgent
import torch.multiprocessing as mp
mp_agent = None
if args_cli.num_agents > 1:
    n = args_cli.num_envs // args_cli.num_agents
    agents_scope = [[i * n, (i+1) * n] for i in range(args_cli.num_agents)]

    #mp.set_start_method('forkserver', force=True) #"spawn")
    mp_agent = MPAgent(args_cli.num_agents, agents_scope=agents_scope )

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


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
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR

from learning.ext_sequential_trainer import ExtSequentialTrainer, EXT_SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from wrappers.smoothness_obs_wrapper import SmoothnessObservationWrapper
from wrappers.close_gripper_action_wrapper import GripperCloseEnv
from models.default_mixin import Shared
from models.bro_model import BroAgent, BroActor, BroCritic
from models.SimBa import SimBaAgent, SimBaActor, SimBaCritic
from models.SimBa_parallel_control import ParallelControlSimBaActor
from models.SimBa_hybrid_control import HybridControlSimBaActor
from envs.factory.dmp_obs_factory_env import DMPObsFactoryEnv
from agents.agent_list import AgentList
#import envs.FPiH.config.franka
import envs.factory
from torch.optim.lr_scheduler import LinearLR
"""
try:
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
    )
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_pickle, dump_yaml

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
except:
"""
from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper


#from wrappers.info_video_recorder_wrapper import InfoRecordVideo
from wrappers.parallel_force_pos_action_wrapper import ParallelForcePosActionWrapper
from wrappers.hybrid_control_action_wrapper import HybridForcePosActionWrapper
from agents.wandb_logger_ppo_agent import WandbLoggerPPO
from agents.mp_agent import MPAgent
import copy
import torch

# seed for reproducibility
#set_seed(args_cli.seed)  # e.g. `set_seed(42)` for fixed seed
if args_cli.seed == -1:
    args_cli.seed = random.randint(0, 10000)
set_seed(args_cli.seed)
#set_seed(6454)

agent_cfg_entry_point = f"SimBaNet_ppo_cfg_entry_point"
evaluating = False

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
    env_cfg.break_force = args_cli.break_force

    env_cfg.use_force_sensor = False or args_cli.parallel_control==1 or args_cli.hybrid_control==1
    if args_cli.use_ft_sensor > 0:
        env_cfg.use_force_sensor = True
        env_cfg.obs_order.append("force_torque")
        env_cfg.state_order.append("force_torque")
    #env_cfg.episode_length_s = 1.5

    """ set up time scales """
    env_cfg.decimation = args_cli.decimation
    env_cfg.history_samples = args_cli.history_sample_size
    env_cfg.sim.dt = (1/args_cli.policy_hz) / args_cli.decimation
    env_cfg.sim.render_interval=args_cli.decimation
    print(f"Time scale config parameters\n\tDec: {env_cfg.decimation}\n\tSim_dt:{1/env_cfg.sim.dt}\n\tPolicy_Hz:{args_cli.policy_hz}")
    
    """Train with skrl agent."""
    #max_rollout_steps = agent_cfg['agent']['rollouts']
    max_rollout_steps = int((1/env_cfg.sim.dt) / env_cfg.decimation * env_cfg.episode_length_s)#TODO
    agent_cfg['agent']['rollouts'] = max_rollout_steps
    agent_cfg['agent']['experiment']['write_interval'] = max_rollout_steps
    agent_cfg['agent']['experiment']['checkpoint_interval'] = max_rollout_steps * 10
    env_cfg.episode_length_s = float(max_rollout_steps * env_cfg.sim.dt * env_cfg.decimation)
    # things below are just important to have in wandb config file
    agent_cfg['agent']['experiment']['tags'].append(env_cfg.task_name)
    agent_cfg['agent']['force_boundry_enforce'] = True
    agent_cfg['agent']['ctrl_torque'] = args_cli.control_torques
    if args_cli.hybrid_control:
        agent_cfg['agent']['experiment']['tags'].append('hybrid_ctrl')
        agent_cfg['agent']['experiment']['tags'].append('hybrid_agent' if args_cli.hybrid_agent==1 else 'baseline_agent')
        agent_cfg['agent']['experiment']['tags'].append('ctrl_torque' if args_cli.control_torques else 'no_torque')
        agent_cfg['agent']['rew_type'] = args_cli.hybrid_selection_reward
        agent_cfg['agent']['agent_type'] = 'hybrid_agent' if args_cli.hybrid_agent==1 else 'baseline_agent'
        agent_cfg['agent']['ctrl_type'] = 'hybrid_ctrl'
    elif args_cli.parallel_control:
        agent_cfg['agent']['experiment']['tags'].append('parallel_ctrl')
        agent_cfg['agent']['experiment']['tags'].append('parallel_agent' if args_cli.parallel_agent==1 else 'baseline_agent')
        agent_cfg['agent']['agent_type'] = 'parallel_agent' if args_cli.parallel_agent==1 else 'baseline_agent'
        agent_cfg['agent']['ctrl_type'] = 'parallel_ctrl'
    elif args_cli.impedance_control:
        agent_cfg['agent']['experiment']['tags'].append('impedance_ctrl')
        agent_cfg['agent']['ctrl_type'] = 'impedance_ctrl'
        agent_cfg['agent']['experiment']['tags'].append('impedance_agent' if args_cli.impedance_agent==1 else 'baseline_agent')
        agent_cfg['agent']['agent_type'] = 'impedance_agent' if args_cli.impedance_agent==1 else 'baseline_agent'
        agent_cfg['agent']['experiment']['tags'].append('ctrl_damping' if args_cli.control_damping else 'crit_damped')    
    else:
        agent_cfg['agent']['experiment']['tags'].append('pose_ctrl')
        agent_cfg['agent']['ctrl_type'] = 'pose_ctrl'
        agent_cfg['agent']['experiment']['tags'].append('baseline_agent')
        agent_cfg['agent']['agent_type'] = 'baseline_agent'


    agent_cfg['agent']['sel_adjs'] = args_cli.sel_adjs.split(",")
    sel_adj_types = args_cli.sel_adjs.split(',')
    agent_cfg['agent']['init_bias'] = "init_bias" in sel_adj_types
    agent_cfg['agent']['force_add'] = "force_add_zout" in sel_adj_types
    agent_cfg['agent']['zero_weights'] = "zero_weights" in sel_adj_types
    agent_cfg['agent']['scale_z'] = "scale_zout" in sel_adj_types
    
    #if args_cli.force_bias_sel == 1:
    #    agent_cfg['agent']['force_bias_sel'] = True
    #    agent_cfg['agent']['experiment']['tags'].append('force_bias_sel')
    #else:
    #    agent_cfg['agent']['force_bias_sel'] = Fals
    
    agent_cfg['agent']['experiment']['project'] = args_cli.wandb_project
    agent_cfg['agent']['experiment']['tags'].append(args_cli.exp_tag)
    if args_cli.use_ft_sensor > 0:
        agent_cfg['agent']['experiment']['tags'].append("force")
        agent_cfg['agent']['use_ft_sensor'] = True
    else:
        agent_cfg['agent']['use_ft_sensor'] = False
        agent_cfg['agent']['experiment']['tags'].append("no-force")
        
    agent_cfg['agent']['break_force'] = args_cli.break_force
    obs_type = args_cli.task.split("-")[3]
    agent_cfg['agent']['obs_type'] = obs_type
    agent_cfg['agent']['experiment']['tags'].append(obs_type)
    task_type = args_cli.task.split("-")[2]
    agent_cfg['agent']['experiment']['group'] += args_cli.wandb_group_prefix + "_" + task_type + "_" + obs_type + "_" + str(args_cli.break_force) + "_" + str(args_cli.history_sample_size) +  "_" + args_cli.hybrid_selection_reward
    agent_cfg['agent']['history_sample_size'] = args_cli.history_sample_size
    agent_cfg['agent']['decimation'] = args_cli.decimation
    agent_cfg['agent']['sim_hz'] = 1 / env_cfg.sim.dt
    agent_cfg['agent']['policy_hz'] = args_cli.policy_hz

    agent_cfg['agent']['model_params'] = agent_cfg['models']
    agent_cfg['agent']['num_envs'] = args_cli.num_envs

    agent_cfg['agent']['track_ckpts'] = (args_cli.log_ckpt_data==1)
    agent_cfg['agent']['ckpt_tracker_path'] = args_cli.ckpt_tracker_path
    print("max rollout steps:", max_rollout_steps)
    assert args_cli.num_envs % args_cli.num_agents == 0, f'Number of agents {args_cli.num_agents} does not even divide into number of envs {args_cli.num_envs}'
    env_per_agent = args_cli.num_envs // args_cli.num_agents
    
    args_cli.max_steps += max_rollout_steps * env_per_agent - (args_cli.max_steps % (max_rollout_steps * env_per_agent))

    # check inputs
    assert args_cli.max_steps % env_per_agent == 0, f'Iterations must be a multiple of num_envs: {env_per_agent}'
    assert args_cli.max_steps % max_rollout_steps == 0, f'Iterations must be multiple of max_rollout_steps {args_cli.max_steps % max_rollout_steps}'
    
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs 
    #env_cfg.scene.replicate_physics = True
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.num_agents = args_cli.num_agents
    
    agent_cfg['seed'] = args_cli.seed

    print("Seed:", agent_cfg['seed'], args_cli.seed)
    #print(env_cfg)
    # random sample some parameters

    if args_cli.lr_scheduler_type == 'cfg':
        if agent_cfg['agent']['learning_rate_scheduler'] == "KLAdaptiveLR":
            # yaml doesn't read it as a class, but as a string idk
            agent_cfg['agent']['learning_rate_scheduler'] = KLAdaptiveLR
            agent_cfg['agent']['learning_rate_scheduler_kwargs'] = agent_cfg['agent']['klAdaptive_lr_scheduler_kwargs']
            agent_cfg['agent']['experiment']['tags'].append("KLAdaptiveLR")
        elif agent_cfg['agent']['learning_rate_scheduler'] == 'LinearWarmup':
            agent_cfg['agent']['learning_rate_scheduler'] = LinearLR
            agent_cfg['agent']['learning_rate_scheduler_kwargs'] = agent_cfg['agent']['linear_warmup_learning_rate_scheduler_kwargs']
            agent_cfg['agent']['experiment']['tags'].append("LinearWarmup")
        else:
            print("\n\n[Info]: Not using a learning rate scheduler\n\n")
    elif args_cli.lr_scheduler_type == 'KLAdaptiveLR':
        agent_cfg['agent']['learning_rate_scheduler'] = KLAdaptiveLR
        agent_cfg['agent']['learning_rate_scheduler_kwargs'] = agent_cfg['agent']['klAdaptive_lr_scheduler_kwargs']
        agent_cfg['agent']['experiment']['tags'].append("KLAdaptiveLR")
    elif args_cli.lr_scheduler_type == "LinearWarmup":
        agent_cfg['agent']['learning_rate_scheduler'] = LinearLR
        agent_cfg['agent']['learning_rate_scheduler_kwargs'] = agent_cfg['agent']['linear_warmup_learning_rate_scheduler_kwargs']
        agent_cfg['agent']['experiment']['tags'].append("LinearWarmup")
    else:
        print("\n\n[Info]: Not using a learning rate scheduler\n\n")
            
        

    #print("Decimation:", dec)
    #agent_cfg['agent']['env_cfg'] = env_cfg
    agent_cfgs = [copy.deepcopy(agent_cfg) for _ in range(args_cli.num_agents)]
    # randomly sample a seed if seed = -1
    for agent_idx, a_cfg in enumerate(agent_cfgs):
        if agent_idx > 0:
            a_cfg['seed'] =  random.randint(0, 10000)
        print(f"Agent {agent_idx} seed: {a_cfg['seed']}")
        # specify directory for logging experiments
        if args_cli.exp_dir is None:
            log_root_path = os.path.join("logs", a_cfg["agent"]["experiment"]["directory"])
            if agent_idx > 0:
                log_root_path = os.path.join("logs", a_cfg["agent"]["experiment"]["directory"] + f"_{agent_idx}")
        else:
            log_root_path = os.path.join("logs", args_cli.exp_dir)
            if agent_idx > 0:
                log_root_path = os.path.join("logs", args_cli.exp_dir + f"_{agent_idx}")

        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")

        # specify directory for logging runs: {time-stamp}_{run_name}
        if args_cli.exp_name is None:
            if a_cfg["agent"]["experiment"]["experiment_name"] == "":
                log_dir = args_cli.task
            else:
                log_dir = f'{a_cfg["agent"]["experiment"]["experiment_name"]}'
        else:
            log_dir = f"{args_cli.exp_name}"

        log_dir += f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        
        # set directory into agent config
        a_cfg["agent"]["experiment"]["directory"] = log_root_path
        a_cfg["agent"]["experiment"]["experiment_name"] = log_dir

        # update log_dir
        log_dir = os.path.join(log_root_path, log_dir)

        # agent configuration

        # dump the configuration into log-directory
        if args_cli.dump_yaml:
            dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
            dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), a_cfg)

        # determine video kwargs
        vid = not args_cli.no_vids
        
        if vid:
            cfg = a_cfg['video_tracking']
            vid_interval = cfg['train_video_interval']
            vid_len = cfg['video_length']
            eval_vid = cfg['record_evals']
            train_vid = cfg['record_training']
        else:
            if agent_idx == 0:
                print("\n\nNo Videos will be recorded\n\n")
                print("Video not implemented")
                eval_vid=False
                train_vid=False


    env = gym.make(
        args_cli.task, 
        cfg=env_cfg, 
        render_mode="rgb_array" if vid else None
    )

    #if args_cli.dmp_obs:
    #    env = DMPObservationWrapper(env)

    if vid:
        vid_fps = int(1.0 / (env.cfg.sim.dt * env.cfg.sim.render_interval ))

        print(f"\n*******Video Kwargs*******:\n\tvid:{vid}\n\tinterval:{vid_interval}")
        print(f"\teval:{eval_vid}\n\ttrain:{train_vid}\n\tlength:{vid_len}")
        print(f"\tFPS:{vid_fps}")
        print("***************************")
        
        def check_record(step):
            global evaluating
            if not evaluating:
                return step % vid_interval == 0
            return evaluating
        
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": check_record, 
            "video_length": vid_len,
            "disable_logger": True,
            "fps": vid_fps
        }
        if eval_vid:
            os.makedirs(os.path.join(log_dir, "videos/evals"))
        if train_vid:
            os.makedirs(os.path.join(log_dir, "videos/training"))

        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        #env = ExtRecordVideo(env, **video_kwargs)
        env = InfoRecordVideo(env, **video_kwargs)
        vid_env = env
    else:
        vid_env = None

    if args_cli.parallel_control==1:
        print("\n\n[INFO]: Using Parallel Control Wrapper.\n\n")
        env = ParallelForcePosActionWrapper(env)
    elif args_cli.hybrid_control==1:
        print("\n\n[INFO]: Using Hybrid Control Wrapper.\n\n")
        env = HybridForcePosActionWrapper(
            env,
            reward_type=args_cli.hybrid_selection_reward,
            ctrl_torque = args_cli.control_torques
        )
    elif args_cli.impedance_control==1:
        print("\n\n[Info]: Using Impedance Control Wrapper.\n\n")
        env = ImpedanceActionWrapper(
            env,
            ctrl_damping=args_cli.control_damping
        )
        #env._pre_physics_step(None)
        #assert 1 == 0

    if args_cli.log_smoothness_metrics:
        print("\n\n[INFO] Recording Smoothness Metrics in info.\n\n")
        env = SmoothnessObservationWrapper(env)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(
        env, 
        ml_framework="torch"
    )  # same as: `wrap_env(env, wrapper="auto")

    #if args_cli.hybrid_control==1:
    #    env._unwrapped = env._env
    
    
    print("Obs space:", env.cfg.observation_space)
    print("State Space:", env.cfg.state_space)
    print("Action Space:", env.action_space)
    #assert 1 == 0
    #env._reset_once = False
    # default factory env sets gripper to zero on line 445
    #env = GripperCloseEnv(env)
    
    device = env.device

    memory = RandomMemory( #AdaptiveRandomMemory( #
        memory_size=agent_cfg['agent']["rollouts"], 
        num_envs=env.num_envs // args_cli.num_agents, 
        device=device,
        replacement=True
    )
    
    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    models = [{} for i in range(args_cli.num_agents)]
    print(models)
    #models["policy"] = Shared(env.observation_space, env.action_space, device)
    
    agent_list = None
    # set wandb parameters
    for a_idx, a_cfg in enumerate(agent_cfgs):
        #a_cfg['agent']['env_cfg'] = env_cfg
        a_cfg['agent']['experiment']['wandb'] = args_cli.no_log_wandb
        wandb_kwargs = {
            "project":a_cfg['agent']['experiment']['project'], #args_cli.wandb_project,
            "entity":args_cli.wandb_entity,
            "api_key":args_cli.wandb_api_key,
            "tags":a_cfg['agent']['experiment']['tags'],
            "group":a_cfg['agent']['experiment']['group'],
            #"tags":args_cli.wandb_tags,
            #"group":args_cli.wandb_group,
            "run_name":a_cfg["agent"]["experiment"]["experiment_name"] + f"_{a_idx}"
        }

        a_cfg["agent"]["experiment"]["wandb_kwargs"] = wandb_kwargs

        if args_cli.parallel_agent==1:

            models[a_idx]['policy'] = ParallelControlSimBaActor( 
            observation_space=env.cfg.observation_space, 
                action_space=env.action_space,
                #action_gain=0.05,
                device=device,
                act_init_std = agent_cfg['models']['act_init_std'],
                actor_n = agent_cfg['models']['actor']['n'],
                actor_latent = agent_cfg['models']['actor']['latent_size'],
                force_scale= env_cfg.ctrl.default_task_force_gains[0] * env_cfg.ctrl.force_action_threshold[0]
            )
        elif args_cli.hybrid_agent==1:
            
            if a_cfg['agent']['hybrid_agent']['unit_std_init']:
                import math
                a_cfg['agent']['hybrid_agent']['pos_init_std'] = (1 / (env_cfg.ctrl.default_task_prop_gains[0] * env_cfg.ctrl.pos_action_threshold[0])) ** 2
                a_cfg['agent']['hybrid_agent']['rot_init_std'] = (1 / (env_cfg.ctrl.default_task_prop_gains[-1] * env_cfg.ctrl.rot_action_threshold[0]))**2
                a_cfg['agent']['hybrid_agent']['force_init_std'] = (1 / (env_cfg.ctrl.default_task_force_gains[0] * env_cfg.ctrl.force_action_threshold[0]))**2

            a_cfg['agent']['hybrid_agent']['pos_scale'] = env_cfg.ctrl.default_task_prop_gains[0] * env_cfg.ctrl.pos_action_threshold[0]     # 2    => 4
            a_cfg['agent']['hybrid_agent']['rot_scale'] = env_cfg.ctrl.default_task_prop_gains[-1] * env_cfg.ctrl.rot_action_threshold[0]     # 2.91 => 8.4681
            a_cfg['agent']['hybrid_agent']['force_scale'] = env_cfg.ctrl.default_task_force_gains[0] * env_cfg.ctrl.force_action_threshold[0] # 1    => 1
            a_cfg['agent']['hybrid_agent']['torque_scale'] = env_cfg.ctrl.default_task_force_gains[-1] * env_cfg.ctrl.torque_action_bounds[0]
            a_cfg['agent']['hybrid_agent']['ctrl_torque'] = args_cli.control_torques
            a_cfg['agent']['hybrid_agent']['selection_adjustment_types'] = args_cli.sel_adjs.split(",")

            # define the actual model
            models[a_idx]['policy'] = HybridControlSimBaActor(
                observation_space=env.cfg.observation_space, 
                action_space=env.action_space,
                device=device,
                hybrid_agent_parameters=a_cfg['agent']['hybrid_agent'],
                actor_n = agent_cfg['models']['actor']['n'],
                actor_latent = agent_cfg['models']['actor']['latent_size'],
            )
        elif args_cli.impedance_agent==1:
            models[a_idx]['policy'] = ImpedanceControlSimBaActor(
                observation_space=env.cfg.observation_space, 
                action_space=env.action_space,
                #action_gain=0.05,
                device=device,
                act_init_std = agent_cfg['models']['act_init_std'],
                actor_n = agent_cfg['models']['actor']['n'],
                actor_latent = agent_cfg['models']['actor']['latent_size'],
                pos_scale = env_cfg.ctrl.pos_action_threshold[0],  
                rot_scale = env_cfg.ctrl.rot_action_threshold[0],
                prop_scale = env_cfg.ctrl.vic_prop_action_threshold[0],
                damp_scale = env_cfg.ctrl.vic_damp_action_threshold[0],
                ctrl_damping=args_cli.control_damping
            )
        else:
            if args_cli.hybrid_control or args_cli.parallel_control:
                sigma_idx = 6 if args_cli.control_torques else 3
            else:
                sigma_idx = 0
            models[a_idx]['policy'] = SimBaActor( #BroAgent(
                observation_space=env.cfg.observation_space, 
                action_space=env.action_space,
                #action_gain=0.05,
                device=device,
                act_init_std = agent_cfg['models']['act_init_std'],
                actor_n = agent_cfg['models']['actor']['n'],
                actor_latent = agent_cfg['models']['actor']['latent_size'],
                sigma_idx = sigma_idx
            ) 

        models[a_idx]["value"] = SimBaCritic( #BroAgent(
            state_space_size=env.cfg.state_space, 
            device=device,
            critic_output_init_mean = agent_cfg['models']['critic_output_init_mean'],
            critic_n = agent_cfg['models']['critic']['n'],
            critic_latent = agent_cfg['models']['critic']['latent_size']
        ) 

    # create the agent
    agent_list = [
        WandbLoggerPPO(
            models=models[i], #copy.deepcopy(models),
            memory=copy.deepcopy(memory),
            cfg=agent_cfgs[i]['agent'],
            observation_space=env.observation_space,
            action_space=env.action_space,
            num_envs=args_cli.num_envs // args_cli.num_agents,
            state_size=env.cfg.observation_space+env.cfg.state_space,
            device=device,
            task = args_cli.task
        ) for i in range(args_cli.num_agents)
    ]
    for i, agent in enumerate(agent_list):
        agent.optimizer = torch.optim.Adam(
            itertools.chain(agent.policy.parameters(), agent.value.parameters()), 
            lr=agent_cfgs[i]['agent']['learning_rate'],
            betas=(0.999, 0.999)
        )
    print("Agents generated")
    agents = None
    if args_cli.num_agents > 1:
        mp_agent.set_agents(agent_list)
        agents = mp_agent
    else:
        agents = agent_list[0]
        if args_cli.ckpt_path != None:
            print("Loading: ", args_cli.ckpt_path)
            agents.load(args_cli.ckpt_path)
    print("Agents set")

    #TODO undo for vids later   
    if vid:
        vid_env.set_agent(agents)

    # configure and instantiate the RL trainer
    cfg_trainer = {
        "timesteps": args_cli.max_steps // (args_cli.num_envs * args_cli.num_agents), 
        "headless": True,
        "close_environment_at_exit": True,
        "disable_progressbar": agent_cfgs[0]['agent']['disable_progressbar']
    }
    
    trainer = ExtSequentialTrainer(
        cfg = cfg_trainer,
        env = env,
        agents = agents
    )

    env.cfg.recording = True #vid # True
    #check_logprob_consistant_loop(trainer, vid_env)
    #return

    
    # our actual learning loop
    ckpt_int = agent_cfg["agent"]["experiment"]["checkpoint_interval"]
    num_evals = max(1,args_cli.max_steps // (ckpt_int * env_per_agent))
    evaluating = True
    
    if eval_vid:   
       vid_env.set_video_name(f"evals/eval_0")
    
    if args_cli.init_eval:
        trainer.eval(0, vid_env)
        
    for i in range(num_evals):
        print(f"Beginning epoch {i+1}/{num_evals}")
        print("\tTraining")
        evaluating = False
        if train_vid:     
            vid_env.set_video_name(f"training/train_STEP_NUM")
        trainer.train(ckpt_int, vid_env)
        #if args_cli.init_eval:
        evaluating = True
        if eval_vid:
            vid_env.set_video_name(f"evals/eval_{i+1}")

        print("\tEvaluating")
        trainer.eval(ckpt_int*(i+1), vid_env)
            



def check_logprob_consistant_loop(trainer, vid_env):
    import copy
    steps=150
    num_envs=256
    # get copy of starting policy
    old_agent = copy.deepcopy(trainer.abs_agent)

    # step forward 1 rollout
    trainer.train(steps, vid_env)
    #assert 1==0
    # get observation and log probs from memory
    states = trainer.agents.memory.get_tensor_by_name("states")
    actions = trainer.agents.memory.get_tensor_by_name("actions")
    stored_log_probs = trainer.agents.memory.get_tensor_by_name("log_prob").flatten()
    print("Stored Log Probs NaNs:", torch.isnan(stored_log_probs).sum().item())
    
    recomputed_log_probs = torch.zeros(((steps-1)*num_envs,1), device = old_agent.agents.policy.device)
    new_log_probs = torch.zeros_like(recomputed_log_probs)
    
    for i in range(steps-1):
        inputs = {'states':states[i,:,:], 'taken_actions':actions[i,:,:]}
        #a,b,c = old_agent.agents.policy.act(inputs, 'policy')
        #print(b.size(), recomputed_log_probs[i*256:(i+1)*256,:].size())
        _, recomputed_log_probs[i*num_envs:(i+1)*num_envs,:], _ = old_agent.agents.policy.act(inputs, 'policy')
        _, new_log_probs[i*num_envs:(i+1)*num_envs,:], _ = trainer.abs_agent.agents.policy.act(inputs, 'policy')


    print("recomputed nans:", torch.isnan(recomputed_log_probs).sum().item())
    recomputed_log_probs = torch.nan_to_num(recomputed_log_probs, nan=0)

    print("New NaNs:", torch.isnan(new_log_probs).sum().item())
    new_log_probs = torch.nan_to_num(new_log_probs, nan=0)
    
    diff_stored_vs_recomputed = (stored_log_probs - recomputed_log_probs).abs().mean()
    diff_old_vs_new = (recomputed_log_probs - new_log_probs).abs().mean().item()

    print(f"Diff(stored vs recomputed old log_probs): {diff_stored_vs_recomputed:.6f}")
    print(f"Diff(old vs new log_probs): {diff_old_vs_new:.6f}")

    # KL between old and new
    with torch.no_grad():
        kl = (recomputed_log_probs - new_log_probs).mean().item()
    print(f"Mean KL estimate: {kl:.6f}")

    if diff_stored_vs_recomputed > 1e-5:
        print("⚠ Stored log_probs mismatch! Likely preprocessing or policy snapshot bug.")
    elif diff_old_vs_new > 0.5:  # heuristic threshold
        print("⚠ Large policy update detected. Possibly high LR or too many epochs.")
    else:
        print("✅ Log-prob math seems consistent.")
    
if __name__ == "__main__":
    # run the main function

    main()

    # close sim app
    simulation_app.close()
