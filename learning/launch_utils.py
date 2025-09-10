

from skrl.resources.schedulers.torch import KLAdaptiveLR
from wrappers.smoothness_obs_wrapper import SmoothnessObservationWrapper
from models.bro_model import BroAgent, BroActor, BroCritic
from models.SimBa import SimBaAgent, SimBaActor, SimBaCritic

from models.SimBa_parallel_control import ParallelControlSimBaActor
from models.SimBa_hybrid_control import HybridControlSimBaActor, HybridControlBlockSimBaActor

try:
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_pickle, dump_yaml
except:
    from omni.isaac.lab.utils.noise import GaussianNoiseCfg, NoiseModelCfg
    from omni.isaac.lab.utils.dict import print_dict
    from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
    

from wrappers.parallel_force_pos_action_wrapper import ParallelForcePosActionWrapper
from wrappers.hybrid_control_action_wrapper import HybridForcePosActionWrapper
from agents.wandb_logger_ppo_agent import WandbLoggerPPO
from agents.MultiWandbLoggerPPO import MultiWandbLoggerPPO

import os
import random
from datetime import datetime

import torch
from torch.optim.lr_scheduler import LinearLR
import gymnasium as gym

import itertools

import skrl
from packaging import version

from skrl.utils import set_seed
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from models.block_simba import BlockSimBaCritic, BlockSimBaActor
from agents.block_wandb_logger_PPO import BlockWandbLoggerPPO

from models.block_simba import export_policies, make_agent_optimizer


def set_preprocessors(env_cfg, agent_cfg, env, state, value):
    # import the preprocessor class 
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    if state:
        agent_cfg['agent']["state_preprocessor"] = RunningStandardScaler
        agent_cfg['agent']["state_preprocessor_kwargs"] = {"size": env.cfg.observation_space + env.cfg.state_space, "device": env_cfg.sim.device}
    
    if value:
        agent_cfg['agent']["value_preprocessor"] = RunningStandardScaler
        agent_cfg['agent']["value_preprocessor_kwargs"] = {"size": 1, "device": env_cfg.sim.device}


def set_reward_shaping(env_cfg, agent_cfg):

    if agent_cfg['agent']['reward_shaper_type'] == 'const_scale':
        def scale_reward(rew, timestep, timesteps, scale=agent_cfg['agent']['rewards_shaper_scale']):
            return rew * scale
        agent_cfg['agent']['rewards_shaper'] = scale_reward
    elif agent_cfg['agent']['reward_shaper_type'] == 'running_scalar':
        agent_cfg['agent']['rewards_shaper'] = RunningStandardScaler( **{"size": 1, "device":env_cfg.sim.device})
    
def set_easy_mode(env_cfg, agent_cfg, easy_mode):
    agent_cfg['agent']['easy_mode'] = easy_mode
    if not easy_mode:
        return
    
    env_cfg.episode_length_s = 10.0
    # robot hand start relative to fixed asset
    env_cfg.task.duration_s = env_cfg.episode_length_s
    env_cfg.task.hand_init_pos = [0.0, 0.0, 0.035]  # Relative to fixed asset tip.
    env_cfg.task.hand_init_pos_noise = [0.0025, 0.0025, 0.00]
    env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    
    # Fixed Asset (applies to all tasks)
    env_cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.fixed_asset_init_orn_deg = 0.0
    env_cfg.task.fixed_asset_init_orn_range_deg = 0.0
    
    # Held Asset (applies to all tasks)
    env_cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]  # noise level of the held asset in gripper
    env_cfg.task.held_asset_rot_init = 0.0
    print("\n\n[INFO]:Easy Mode Selected\n\n")


def set_use_force(env_cfg, agent_cfg, args_cli, use_ft_sensor):
    env_cfg.use_force_sensor = use_ft_sensor
    if use_ft_sensor:
        env_cfg.obs_order.append("force_torque")
        env_cfg.state_order.append("force_torque")
    if args_cli.use_ft_sensor > 0:
        agent_cfg['agent']['experiment']['tags'].append("force")
        agent_cfg['agent']['use_ft_sensor'] = True
    else:
        agent_cfg['agent']['use_ft_sensor'] = False
        agent_cfg['agent']['experiment']['tags'].append("no-force")
    print("\n[INFO]:Using Force Torque Sensor" if use_ft_sensor else "\n[INFO]:Not Using Force Torque Sensor")

    
def set_time_params(env_cfg, args_cli):
    env_cfg.decimation = args_cli.decimation
    env_cfg.history_samples = args_cli.history_sample_size
    env_cfg.sim.dt = (1/args_cli.policy_hz) / args_cli.decimation
    env_cfg.sim.render_interval=args_cli.decimation
    print(f"Time scale config parameters\n\tDec: {env_cfg.decimation}\n\tSim_dt:{1/env_cfg.sim.dt}\n\tPolicy_Hz:{args_cli.policy_hz}")


def set_use_obs_noise(env_cfg, use_noise):
    env_cfg.use_obs_noise = use_noise
    if not use_noise:
        return
    
    obsCfg = NoiseModelCfg()
    gaussCfg = GaussianNoiseCfg()
    means = []
    stds = []
        
    for obs_type in env_cfg.obs_order:
        for x in env_cfg.obs_noise_mean[obs_type]:
            means.append(x)
        for x in env_cfg.obs_noise_std[obs_type]:
            stds.append(x)
        
    gaussCfg.mean = torch.tensor(means, device=env_cfg.sim.device)
    gaussCfg.std = torch.tensor(stds, device=env_cfg.sim.device)
    obsCfg.noise_cfg = gaussCfg
    env_cfg.observation_noise_model = obsCfg
    print("\n\n[INFO]:Applying Noise to Observations\n\n")

def set_controller_tagging(env_cfg, agent_cfg, args_cli, max_rollout_steps):
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


def set_selection_adjustments(env_cfg, agent_cfg, args_cli):
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
    agent_cfg['agent']['hybrid_agent']['selection_adjustment_types'] = args_cli.sel_adjs.split(",")
    
def set_wandb_env_data(env_cfg, agent_cfg, args_cli):
    
    agent_cfg['agent']['experiment']['project'] = args_cli.wandb_project
    agent_cfg['agent']['experiment']['tags'].append(args_cli.exp_tag)
    
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
    
    # override configurations with non-hydra CLI arguments
    env_cfg.agent_groups =  (len(env_cfg.break_force) if type(env_cfg.break_force)==list else 1) 
    env_cfg.scene.num_envs = args_cli.num_envs * env_cfg.agent_groups
    print(f"Scene set to {env_cfg.scene.replicate_physics}")
    env_cfg.scene.replicate_physics = True

    env_cfg.num_agents = args_cli.num_agents * env_cfg.agent_groups
    
    agent_cfg['agent']['seed'] = args_cli.seed

    # get control params
    agent_cfg['agent']['ctrl_params'] = {}
    for key, item in vars(env_cfg.ctrl).items():
        agent_cfg['agent']['ctrl_params'][key] = item

    # get env params
    agent_cfg['agent']['env_cfg'] = {}
    #keys = ['force_tanh_scale','decimation', 'use_force_sensor','episode_length_s']
    keys = ['sim','ctrl','task','scene','robot','viewer','is_finite_horizon','rerender_on_reset','wait_for_textures','xr','ui_window_class_type','seed','event']
    for key, item in vars(env_cfg).items():
        if not key in keys:
            agent_cfg['agent']['env_cfg'][key] = item

    # get task params
    ignore_keys = ['fixed_asset', 'held_asset']
    agent_cfg['agent']['task_cfg'] = {}
    for key, item in vars(env_cfg.task).items():
        if not key in ignore_keys:
            agent_cfg['agent']['task_cfg'][key] = item
    


def set_learn_rate_scheduler(env_cfg, agent_cfg, args_cli):

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


def set_individual_agent_log_paths(env_cfg, agent_cfg, args_cli):
    break_forces = agent_cfg['agent']['break_force']
    if not type(break_forces) == list:
        break_forces = [break_forces]
    
    agent_idx = 0 
    for break_force in break_forces:
        for i in range(args_cli.num_agents):
            agent_cfg['agent'][f'agent_{agent_idx}'] = {}
            agent_cfg['agent'][f'agent_{agent_idx}']['break_force'] = break_force
            agent_cfg['agent'][f'agent_{agent_idx}']['experiment'] = {}
            if agent_idx > 0:
                agent_cfg['agent'][f'agent_{agent_idx}']['seed'] =  random.randint(0, 10000)
            
                print(f"[INFO]:\tAgent {agent_idx} seed: {agent_cfg['agent'][f'agent_{agent_idx}']['seed']}")
            else:
                print(f"[INFO]:\tAgent {agent_idx} seed: {agent_cfg['agent']['seed']}")
        
            #agent_cfg['agent']['agent_{agent_idx}']['experiment']['directory'] = agent_cfg['agent']['experiment']['directory'] + f"_{agent_idx}"
            # specify directory for logging experiments
            if args_cli.exp_dir is None:
                #log_root_path = os.path.join("logs", agent_cfg["agent"][f'agent_{agent_idx}']["experiment"]["directory"])
                #if agent_idx > 0:
                log_root_path = os.path.join("logs", agent_cfg["agent"]["experiment"]["directory"] + f"_f({break_force})_{agent_idx}")
            else:
                #agent_cfg['agent']['agent_{agent_idx}']['experiment']['directory'] = args_cli.exp_dir + f"_{agent_idx}"
                log_root_path = os.path.join("logs", args_cli.exp_dir + f"_f({break_force})_{agent_idx}")
                #if agent_idx > 0:
                #    log_root_path = os.path.join("logs", args_cli.exp_dir + f"_{agent_idx}")

            log_root_path = os.path.abspath(log_root_path)
            print(f"[INFO] Logging Agent {agent_idx} experiment in directory: {log_root_path}")

            # specify directory for logging runs: {time-stamp}_{run_name}
            #agent_cfg['agent'][f'agent_{agent_idx}']['experiment']['experiment_name'] = agent_cfg['agent']['experiment']['experiment_name'] + f"_{agent_idx}"
            if args_cli.exp_name is None:
                if agent_cfg["agent"]["experiment"]["experiment_name"] == "":
                    log_dir = args_cli.task
                else:
                    log_dir = agent_cfg["agent"]["experiment"]["experiment_name"] + f"_f({break_force})_{agent_idx}"
            else:
                log_dir = f"{args_cli.exp_name}_f({break_force})_{agent_idx}"

            #log_dir += f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        
            # set directory into agent config
            agent_cfg["agent"][f'agent_{agent_idx}']["experiment"]["directory"] = log_root_path
            agent_cfg["agent"][f'agent_{agent_idx}']["experiment"]["experiment_name"] = log_dir

            # update log_dir
            log_dir = os.path.join(log_root_path, log_dir)
            #os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
            # dump the configuration into log-directory
            if args_cli.dump_yaml:
                dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
                dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        
            agent_cfg['agent'][f'agent_{agent_idx}']['experiment']['wandb'] = args_cli.no_log_wandb
            wandb_kwargs = {
                "project":agent_cfg['agent']['experiment']['project'], #args_cli.wandb_project,
                "entity":args_cli.wandb_entity,
                "api_key":args_cli.wandb_api_key,
                "tags":agent_cfg['agent']['experiment']['tags'],
                "group":agent_cfg['agent']['experiment']['group'] + f"_f({break_force})",
                #"tags":args_cli.wandb_tags,
                #"group":args_cli.wandb_group,
                "run_name":agent_cfg["agent"][f"agent_{agent_idx}"]["experiment"]["experiment_name"] + f"_f({break_force})_{agent_idx}"
            }

            print(f"Wandb Args {agent_idx}:", wandb_kwargs)
            agent_cfg["agent"][f'agent_{agent_idx}']["experiment"]["wandb_kwargs"] = wandb_kwargs
            agent_idx += 1
            
def set_video(env_cfg, agent_cfg, args_cli, env):
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
    return vid_env

def set_controller_wrapper(env_cfg, agent_cfg, args_cli, env):
    
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
    
def set_models(env_cfg, agent_cfg, args_cli, env):
    
    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    
    models = {}
    if args_cli.parallel_agent==1:
        ############ TODO: Make Multi Version ##########################
        models['policy'] = ParallelControlSimBaActor( 
            observation_space=env.cfg.observation_space, 
            action_space=env.action_space,
            #action_gain=0.05,
            device=env.device,
            act_init_std = agent_cfg['models']['act_init_std'],
            actor_n = agent_cfg['models']['actor']['n'],
            actor_latent = agent_cfg['models']['actor']['latent_size'],
            force_scale= env_cfg.ctrl.default_task_force_gains[0] * env_cfg.ctrl.force_action_threshold[0]
        )
    elif args_cli.hybrid_agent==1:
        
        set_hybrid_agent_init_stds(env_cfg, agent_cfg, args_cli)

        models['policy'] = HybridControlSimBaActor(
            observation_space=env.cfg.observation_space, 
            action_space=env.action_space,
            device=env.device,
            hybrid_agent_parameters=agent_cfg['agent']['hybrid_agent'],
            actor_n = agent_cfg['models']['actor']['n'],
            actor_latent = agent_cfg['models']['actor']['latent_size'],
            num_agents = env_cfg.num_agents, #args_cli.num_agents
        )
    elif args_cli.impedance_agent==1:
        ############ TODO: Make Multi Version ##########################
        models['policy'] = ImpedanceControlSimBaActor(
            observation_space=env.cfg.observation_space, 
            action_space=env.action_space,
            #action_gain=0.05,
            device=env.device,
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
        models['policy'] = torch.compile(SimBaActor( #BroAgent(
            observation_space=env.cfg.observation_space, 
            action_space=env.action_space,
            #action_gain=0.05,
            device=env.device,
            act_init_std = agent_cfg['models']['act_init_std'],
            actor_n = agent_cfg['models']['actor']['n'],
            actor_latent = agent_cfg['models']['actor']['latent_size'],
            sigma_idx = sigma_idx,
            num_agents = env_cfg.num_agents, #args_cli.num_agents,
            last_layer_scale=agent_cfg['models']['last_layer_scale']
        ) )

    models["value"] = torch.compile(SimBaCritic( #BroAgent(
        state_space_size=env.cfg.state_space, 
        device=env.device,
        critic_output_init_mean = agent_cfg['models']['critic_output_init_mean'],
        critic_n = agent_cfg['models']['critic']['n'],
        critic_latent = agent_cfg['models']['critic']['latent_size'],
        num_agents = env_cfg.num_agents #args_cli.num_agents
    ))
    print("[INFO]: Models created")
    return models


def set_hybrid_agent_init_stds(env_cfg, agent_cfg, args_cli):
            
    if agent_cfg['agent']['hybrid_agent']['unit_std_init']:
        import math
        agent_cfg['agent']['hybrid_agent']['pos_init_std'] = (1 / (env_cfg.ctrl.default_task_prop_gains[0] * env_cfg.ctrl.pos_action_threshold[0])) ** 2
        agent_cfg['agent']['hybrid_agent']['rot_init_std'] = (1 / (env_cfg.ctrl.default_task_prop_gains[-1] * env_cfg.ctrl.rot_action_threshold[0]))**2
        agent_cfg['agent']['hybrid_agent']['force_init_std'] = (1 / (env_cfg.ctrl.default_task_force_gains[0] * env_cfg.ctrl.force_action_threshold[0]))**2

    agent_cfg['agent']['hybrid_agent']['pos_scale'] = env_cfg.ctrl.default_task_prop_gains[0] * env_cfg.ctrl.pos_action_threshold[0]     # 2    => 4
    agent_cfg['agent']['hybrid_agent']['rot_scale'] = env_cfg.ctrl.default_task_prop_gains[-1] * env_cfg.ctrl.rot_action_threshold[0]     # 2.91 => 8.4681
    agent_cfg['agent']['hybrid_agent']['force_scale'] = env_cfg.ctrl.default_task_force_gains[0] * env_cfg.ctrl.force_action_threshold[0] # 1    => 1
    agent_cfg['agent']['hybrid_agent']['torque_scale'] = env_cfg.ctrl.default_task_force_gains[-1] * env_cfg.ctrl.torque_action_bounds[0]
    agent_cfg['agent']['hybrid_agent']['ctrl_torque'] = args_cli.control_torques


def set_block_models(env_cfg, agent_cfg, args_cli, env):
    models = {}

    if args_cli.hybrid_agent==1:
        set_hybrid_agent_init_stds(env_cfg, agent_cfg, args_cli)
        models['policy'] = HybridControlBlockSimBaActor(
            observation_space=env.cfg.observation_space, 
            action_space=env.action_space,
            device=env.device,
            hybrid_agent_parameters=agent_cfg['agent']['hybrid_agent'],
            actor_n = agent_cfg['models']['actor']['n'],
            actor_latent = agent_cfg['models']['actor']['latent_size'],
            num_agents = env_cfg.num_agents, #args_cli.num_agents
        )
    else:
        if args_cli.hybrid_control or args_cli.parallel_control:
            sigma_idx = 6 if args_cli.control_torques else 3
        else:
            sigma_idx = 0
        models['policy'] = BlockSimBaActor( #BroAgent(
            observation_space=env.cfg.observation_space, 
            action_space=env.action_space,
            #action_gain=0.05,
            device=env.device,
            act_init_std = agent_cfg['models']['act_init_std'],
            actor_n = agent_cfg['models']['actor']['n'],
            actor_latent = agent_cfg['models']['actor']['latent_size'],
            sigma_idx = sigma_idx,
            num_agents = env_cfg.num_agents, #args_cli.num_agents,
            last_layer_scale=agent_cfg['models']['last_layer_scale']
        )

    models["value"] = BlockSimBaCritic( #BroAgent(
        state_space_size=env.cfg.state_space, 
        device=env.device,
        critic_output_init_mean = agent_cfg['models']['critic_output_init_mean'] * agent_cfg['agent']['rewards_shaper_scale'],
        critic_n = agent_cfg['models']['critic']['n'],
        critic_latent = agent_cfg['models']['critic']['latent_size'],
        num_agents = env_cfg.num_agents #args_cli.num_agents
    )
    print("[INFO]: Block models created")
    return models
    
def set_block_agent(env_cfg, agent_cfg, args_cli, models, memory, env=None):
    for key, item in agent_cfg['agent']['agent_0']['experiment'].items():
        agent_cfg['agent']['experiment'][key] = item
    agent = BlockWandbLoggerPPO(
        models=models, #copy.deepcopy(models),
        memory=memory,
        cfg=agent_cfg['agent'],
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_envs=env_cfg.scene.num_envs, #args_cli.num_envs,
        state_size=env.cfg.observation_space+env.cfg.state_space,
        device=env.device,
        task = args_cli.task,
        task_cfg = env_cfg,
        num_agents= env_cfg.num_agents #args_cli.num_agents
    ) 

    agent.optimizer = make_agent_optimizer(
        models['policy'],
        models['value'],
        #policy_lr = agent_cfg['agent']['learning_rate'],
        #critic_lr = agent_cfg['agent']['learning_rate'],
        policy_lr = agent_cfg['agent']['policy_learning_rate'],
        critic_lr = agent_cfg['agent']['critic_learning_rate'],
        betas=(0.999, 0.999),
        eps=1e-8,
        weight_decay=0,
        debug=args_cli.debug_mode
    )
    if agent_cfg['agent']['value_update_ratio'] > 1:
        print(f"[INFO]: Agent using value update ratio of {agent_cfg['agent']['value_update_ratio']}")
    """
    print("Named parameters in the optimizer:")
    for i, param_group in enumerate(agent.optimizer.param_groups):
        print(f"\nParameter Group {i+1}:")
        for j, param in enumerate(param_group['params']):
            # Find the name of the parameter by iterating through model.named_parameters()
            # This is necessary because the optimizer only stores the parameter tensors, not their names
            param_name = None
            for k, model in enumerate(models):
                for name, model_param in models[model].named_parameters():
                    if model_param is param:
                        param_name = name
                        break
                    
                if param_name:
                    print(f"    Parameter {j+1} (Name: {param_name}, Shape: {param.shape})")
                else:
                    print(f"    Parameter {j+1} (Shape: {param.shape})")
    #print(agent.optimizer.param_groups)
    assert 1==0
    """
    print("[INFO]: Block Agents and optimizer generated")
    return agent
    
    

def set_agent(env_cfg, agent_cfg, args_cli, models, memory, env=None):
    # temp until we redo the logger
    for key, item in agent_cfg['agent']['agent_0']['experiment'].items():
        agent_cfg['agent']['experiment'][key] = item
    """
    # create the agent
    agent = WandbLoggerPPO(
        models=models, #copy.deepcopy(models),
        memory=memory,
        cfg=agent_cfg['agent'],
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_envs=args_cli.num_envs,
        state_size=env.cfg.observation_space+env.cfg.state_space,
        device=env.device,
        task = args_cli.task
    ) 
    """
    agent = MultiWandbLoggerPPO(
        models=models, #copy.deepcopy(models),
        memory=memory,
        cfg=agent_cfg['agent'],
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_envs=env_cfg.scene.num_envs, #args_cli.num_envs,
        state_size=env.cfg.observation_space+env.cfg.state_space,
        device=env.device,
        task = args_cli.task,
        num_agents= env_cfg.num_agents #args_cli.num_agents
    ) 

    agent.optimizer = torch.optim.Adam(
        itertools.chain(models['policy'].parameters(), models['value'].parameters()), 
        lr=agent_cfg['agent']['learning_rate'],
        betas=(0.999, 0.999)
    )

    #for name, param in agent.optimizer.named_parameters():
    #    print(f"Name: {name}, Shape: {param.size()}")
    """
    print("Named parameters in the optimizer:")
    for i, param_group in enumerate(agent.optimizer.param_groups):
        print(f"\nParameter Group {i+1}:")
        for j, param in enumerate(param_group['params']):
            # Find the name of the parameter by iterating through model.named_parameters()
            # This is necessary because the optimizer only stores the parameter tensors, not their names
            param_name = None
            for model in models:
                for name, model_param in models[model].named_parameters():
                    if model_param is param:
                        param_name = name
                        break
                    
                if param_name:
                    print(f"    Parameter {j+1} (Name: {param_name}, Shape: {param.shape})")
                else:
                    print(f"    Parameter {j+1} (Shape: {param.shape})")
    #print(agent.optimizer.param_groups)
    assert 1==0
    """
    print("[INFO]: Agent(s) and optimizer generated")
    return agent

def print_warn(warning):
    print(f'\033[1;33;40m [WARN]: {warning} \033[0m')

def attach_grad_debug_hooks(model, role='policy'):
    """
    Internal helper: attach gradient debug hooks to parameters.
    Prints warning if grad is None or zero.
    """
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        #pname = getattr(p, "_name", "unnamed")
        def make_hook(pname=pname, role=role):
            def hook(grad):
                if grad is None:
                    print_warn("f{role} {pname} → NO GRAD")
                elif grad.norm().item() == 0:
                    print_warn(f"{role} {pname} → ZERO GRAD")
                return grad
            return hook
        p.register_hook(make_hook())

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
