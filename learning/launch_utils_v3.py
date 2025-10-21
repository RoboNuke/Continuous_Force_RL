import os
from wrappers.mechanics.fragile_object_wrapper import FragileObjectWrapper
from wrappers.mechanics.force_reward_wrapper import ForceRewardWrapper
from wrappers.mechanics.efficient_reset_wrapper import EfficientResetWrapper
from wrappers.mechanics.close_gripper_action_wrapper import GripperCloseEnv
from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper
from wrappers.observations.observation_manager_wrapper import ObservationManagerWrapper
from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
from wrappers.logging.wandb_wrapper import GenericWandbLoggingWrapper
from wrappers.logging.enhanced_action_logging_wrapper import EnhancedActionLoggingWrapper
from wrappers.logging.pose_contact_logging_wrapper import PoseContactLoggingWrapper
from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

from models.SimBa_hybrid_control import HybridControlBlockSimBaActor
from models.block_simba import BlockSimBaActor, BlockSimBaCritic, make_agent_optimizer
from agents.block_ppo import BlockPPO

from dataclasses import asdict
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.resources.preprocessors.torch import RunningStandardScaler

def set_reward_shaping(env_cfg, agent_cfg):
    """Set up reward shaping function based on config."""
    if agent_cfg.reward_shaper_type == 'const_scale':
        scale = agent_cfg.rewards_shaper_scale
        def scale_reward(rew, timestep, timesteps):
            return rew * scale
        agent_cfg.rewards_shaper = scale_reward
    elif agent_cfg.reward_shaper_type == 'running_scalar':
        agent_cfg.rewards_shaper = RunningStandardScaler(**{"size": 1, "device": env_cfg.sim.device})


def apply_wrappers(env, configs):
    wrappers_config = configs['wrappers']
    if wrappers_config.fragile_objects.enabled:
        print(f"  - Applying Fragile Object Wrapper with break forces: {configs['primary'].break_forces}")
        env = FragileObjectWrapper(
            env,
            break_force=configs['primary'].break_forces,
            num_agents=configs['primary'].total_agents,
            config=wrappers_config.fragile_objects
        )

    if wrappers_config.efficient_reset_enabled:
        print("  - Applying Efficient Reset Wrapper")
        env = EfficientResetWrapper(env)

    if wrappers_config.force_torque_sensor.enabled:
        print("  - Applying Force Torque Wrapper")

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

    if wrappers_config.force_reward.enabled:
        print("  - Applying Force Reward Wrapper")
        raise NotImplementedError

    if wrappers_config.observation_manager.enabled:
        print("  - Applying Observation Manager Wrapper")
        env = ObservationManagerWrapper(
            env,
            merge_strategy=wrappers_config.observation_manager.merge_strategy
        )

    if wrappers_config.observation_noise.enabled:
        print("  - Applying Observation Noise Wrapper")
        raise NotImplementedError

    if wrappers_config.hybrid_control.enabled:
        print("  - Applying Hybrid Force Position Wrapper")
        env = HybridForcePositionWrapper(
            env,
            ctrl_torque=configs['primary'].ctrl_torque,
            reward_type=wrappers_config.hybrid_control.reward_type,
            ctrl_cfg=configs['environment'].ctrl,
            task_cfg=configs['environment'].hybrid_task,
            num_agents=configs['primary'].total_agents
        )

    # Apply WandbWrapper BEFORE FactoryMetricsWrapper so factory metrics can send data to wandb
    if wrappers_config.wandb_logging.enabled:
        print("  - Applying Generic Wandb Logging Wrapper")
        env = GenericWandbLoggingWrapper(
            env,
            num_agents=configs['primary'].total_agents,
            all_configs=configs
        )

    if wrappers_config.factory_metrics.enabled:
        print("  - Applying Factory Metrics Wrapper")
        env = FactoryMetricsWrapper(
            env,
            num_agents=configs['primary'].total_agents,
            publish_to_wandb=wrappers_config.factory_metrics.publish_to_wandb
        )

    if wrappers_config.pose_contact_logging.enabled:
        print("  - Applying Pose Contact Logging Wrapper")
        env = PoseContactLoggingWrapper(
            env,
            num_agents=configs['primary'].total_agents
        )

    if wrappers_config.action_logging.enabled:
        print("  - Applying Enhanced Action Logging Wrapper")
        env = EnhancedActionLoggingWrapper(
            env,
            track_selection=wrappers_config.action_logging.track_selection,
            track_pos=wrappers_config.action_logging.track_pos,
            track_rot=wrappers_config.action_logging.track_rot,
            track_force=wrappers_config.action_logging.track_force,
            track_torque=wrappers_config.action_logging.track_torque,
            force_size=wrappers_config.action_logging.force_size,
            logging_frequency=wrappers_config.action_logging.logging_frequency
        )
    return env
def add_wandb_tracking_tags(configs):
    env_cfg = configs['environment']
    wrappers_config = configs['wrappers']


    env_cfg.use_ft_sensor = wrappers_config.force_torque_sensor.enabled
    if env_cfg.use_ft_sensor:
        configs['experiment'].tags.append('force')
    else:
        configs['experiment'].tags.append('no-force')

    env_cfg.obs_type = 'local'
    configs['experiment'].tags.append('local-obs')

    if wrappers_config.hybrid_control.enabled:
        env_cfg.ctrl_type = 'hybrid'
        configs['experiment'].tags.append('hybrid-ctrl')
    else:
        env_cfg.ctrl_type = 'pose'
        configs['experiment'].tags.append('pose-ctrl')
    
    if configs['model'].use_hybrid_agent:
        env_cfg.agent_type = 'CLoP'
        configs['experiment'].tags.append('CLoP-agent')
    else:
        env_cfg.agent_type = 'basic'
        configs['experiment'].tags.append('basic-agent')


def define_agent_configs(configs):
    configs['agent'].write_interval = configs['primary'].rollout_steps(configs['environment'].episode_length_s) #max rollouts
    configs['agent'].checkpoint_interval = configs['agent'].checkpoint_interval_multiplier * configs['agent'].write_interval

    print(f"  -Write Interval:{configs['agent'].write_interval}")
    print(f"  -Checkpoint Interval:{configs['agent'].checkpoint_interval}")

    agent_idx = 0
    configs['agent'].agent_exp_cfgs = []
    experiment = configs['experiment']
    for break_force in configs['primary'].break_forces:
        for i in range(configs['primary'].agents_per_break_force):
            configs['agent'].agent_exp_cfgs.append({})
            configs['agent'].agent_exp_cfgs[agent_idx]['break_force'] = break_force
            configs['agent'].agent_exp_cfgs[agent_idx]['experiment'] = {}

            # Set up logging directory
            exp_name = experiment.name
            log_root_path = os.path.join("logs", f"{exp_name}_f({break_force})_{agent_idx}")
            log_root_path = os.path.abspath(log_root_path)

            # current format
            # name_FT({on/off})_{exp_tag}_{task_type}_{obs_type}_Hyb-Ctrl({0/1})_Hyb-Agent({0/1})_{date_time}_f({break_force})
            log_dir = f"{exp_name}_f({break_force})_{agent_idx}"

            configs['agent'].agent_exp_cfgs[agent_idx]['experiment']["directory"] = log_root_path
            configs['agent'].agent_exp_cfgs[agent_idx]['experiment']["experiment_name"] = log_dir

            # Set up wandb configuration
            configs['agent'].agent_exp_cfgs[agent_idx]['experiment']['wandb'] = True
            wandb_kwargs = {
                "project": configs['experiment'].wandb_project,
                "entity": configs['experiment'].wandb_entity,
                "api_key": '-1',
                "tags": configs['experiment'].tags,
                "group": configs['experiment'].group + f"_f({break_force})",
                "run_name": log_dir
            }

            configs['agent'].agent_exp_cfgs[agent_idx]['experiment']["wandb_kwargs"] = wandb_kwargs
            agent_idx += 1
            

# ===== MODEL CREATION =====

def create_policy_and_value_models(env, configs):
    """Create policy and value models based on configuration."""
    models = {}

    # Determine model type and parameters
    use_hybrid_agent = configs['model'].use_hybrid_agent
    use_hybrid_control = configs['wrappers'].hybrid_control.enabled

    if use_hybrid_agent:
        print("  - Creating hybrid control agent models")
        models['policy'] = _create_hybrid_policy_model(env, configs)
    else:
        print("  - Creating standard SimBa agent models")
        models['policy'] = _create_standard_policy_model(env, configs)

    print("  - Creating value model")
    models["value"] = _create_value_model(env, configs)

    return models


def _create_hybrid_policy_model(env, configs):
    """Create hybrid control policy model."""
    # Set hybrid agent initialization parameters
    _configure_hybrid_agent_parameters(configs)
    return HybridControlBlockSimBaActor(
        observation_space=env.cfg.observation_space,
        action_space=env.cfg.action_space,
        device=env.device,
        hybrid_agent_parameters=asdict(configs['model'].hybrid_agent),
        actor_n=configs['model'].actor.n,
        actor_latent=configs['model'].actor.latent_size,
        num_agents=configs['primary'].total_agents
    )

def _configure_hybrid_agent_parameters(configs):
    """Configure hybrid agent initialization parameters."""
    env_cfg = configs['environment']

    # Check if unit_std_init is enabled
    unit_std_init = configs['model'].hybrid_agent.unit_std_init
    unit_factor_std_init = configs['model'].hybrid_agent.unit_factor_std_init
    if unit_std_init:
        configs['model'].hybrid_agent.pos_init_std = (unit_factor_std_init / (env_cfg.ctrl.default_task_prop_gains[0] * env_cfg.ctrl.pos_action_threshold[0])) 
        configs['model'].hybrid_agent.rot_init_std = (unit_factor_std_init / (env_cfg.ctrl.default_task_prop_gains[-1] * env_cfg.ctrl.rot_action_threshold[0]))
        configs['model'].hybrid_agent.force_init_std = (unit_factor_std_init / (env_cfg.ctrl.default_task_force_gains[0] * env_cfg.ctrl.force_action_threshold[0]))

    configs['model'].hybrid_agent.pos_scale = env_cfg.ctrl.default_task_prop_gains[0] * env_cfg.ctrl.pos_action_threshold[0]
    configs['model'].hybrid_agent.rot_scale = env_cfg.ctrl.default_task_prop_gains[-1] * env_cfg.ctrl.rot_action_threshold[0]
    configs['model'].hybrid_agent.force_scale = env_cfg.ctrl.default_task_force_gains[0] * env_cfg.ctrl.force_action_threshold[0]
    configs['model'].hybrid_agent.torque_scale = env_cfg.ctrl.default_task_force_gains[-1] * env_cfg.ctrl.torque_action_bounds[0]

    # Extract ctrl_torque from wrappers_config
    configs['model'].hybrid_agent.ctrl_torque = configs['primary'].ctrl_torque

    print("  - Hybrid agent parameters configured")

def _create_standard_policy_model(env, configs):
    """Create standard SimBa policy model."""
    sigma_idx = 0
    if configs['wrappers'].hybrid_control.enabled:
        ctrl_torque = configs['primary'].ctrl_torque
        sigma_idx = 6 if ctrl_torque else 3

    return BlockSimBaActor(
        observation_space=env.cfg.observation_space,
        action_space=env.cfg.action_space,
        device=env.device,
        act_init_std=configs['model'].act_init_std,
        actor_n=configs['model'].actor.n,
        actor_latent=configs['model'].actor.latent_size,
        sigma_idx=sigma_idx,
        init_sel_bias = configs['model'].hybrid_agent.init_bias,
        num_agents=configs['primary'].total_agents,
        last_layer_scale=configs['model'].last_layer_scale
    )


def _create_value_model(env, configs):
    """Create value model."""
    return BlockSimBaCritic(
        state_space_size=env.unwrapped.cfg.state_space,
        device=env.device,
        critic_output_init_mean=configs['model'].critic_output_init_mean * configs['agent'].rewards_shaper_scale,
        critic_n=configs['model'].critic.n,
        critic_latent=configs['model'].critic.latent_size,
        num_agents=configs['primary'].total_agents
    )


# ===== AGENT CREATION =====

def create_block_ppo_agents(env, configs, models, memory):
    """Create BlockPPO agents with per-agent independent preprocessors and optimizer.
    """
    print("  - Creating BlockPPO agents with per-agent preprocessors")

    # Set up preprocessor configurations (BlockPPO will create independent instances)
    setup_per_agent_preprocessors(
        env, configs
    )
    print(configs['agent'].state_preprocessor_kwargs)
    # Update agent_cfg with preprocessor configs
    #configs['agent'].preprocessor_configs = preprocessor_configs

    # Resolve learning rate scheduler from string to actual class
    if configs['agent'].learning_rate_scheduler is not None:
        scheduler_name = configs['agent'].learning_rate_scheduler
        if scheduler_name == "KLAdaptiveLR":
            configs['agent'].learning_rate_scheduler = KLAdaptiveLR
            print(f"  - Resolved learning_rate_scheduler: {scheduler_name} -> {KLAdaptiveLR}")
        else:
            print(f"  - WARNING: Unknown learning_rate_scheduler: {scheduler_name}")

    # Create BlockPPO agent

    agent = BlockPPO(
        models=models,
        memory=memory,
        cfg=configs['agent'].to_skrl_dict(configs['environment'].episode_length_s),
        observation_space=env.observation_space,
        action_space=env.unwrapped.cfg.action_space,
        num_envs=configs['environment'].scene.num_envs,
        state_size=env.unwrapped.cfg.observation_space + env.unwrapped.cfg.state_space,
        device=env.device,
        task=configs['environment'].task_name,
        num_agents=configs['primary'].total_agents,
        env=env 
    )

    # Create optimizer
    agent.optimizer = make_agent_optimizer(
        models['policy'],
        models['value'],
        policy_lr=configs['agent'].policy_learning_rate,
        critic_lr=configs['agent'].critic_learning_rate,
        betas=tuple(configs['agent'].optimizer_betas),  
        eps=configs['agent'].optimizer_eps,             # From config instead of hardcoded
        weight_decay=configs['agent'].optimizer_weight_decay,  # From config instead of hardcoded
        debug=configs['primary'].debug_mode
    )

    print(f"  - Created BlockPPO with {configs['primary'].total_agents} independent agents")
    

    return agent

def setup_per_agent_preprocessors(env, configs):
    """Set up preprocessor configurations for BlockPPO per-agent system.

    Instead of creating complex per-agent configs, this just sets up the standard
    preprocessor config that BlockPPO will use to create independent instances.
    """
    #preprocessor_configs = {}

    if configs['agent'].state_preprocessor:
        configs['agent'].state_preprocessor = RunningStandardScaler
        configs['agent'].state_preprocessor_kwargs = {
            "size": env.unwrapped.cfg.observation_space + env.unwrapped.cfg.state_space,
            "device": configs['environment'].sim.device
        }
        print(f"  - State preprocessor config set (will create {configs['primary'].total_agents} independent instances with input size {env.unwrapped.cfg.observation_space + env.unwrapped.cfg.state_space})")
    print(configs['agent'].state_preprocessor_kwargs)
    if configs['agent'].value_preprocessor:
        configs['agent'].value_preprocessor = RunningStandardScaler
        configs['agent'].value_preprocessor_kwargs = {
            "size": 1,
            "device": configs['environment'].sim.device
        }
        print(f"  - Value preprocessor config set (will create {configs['primary'].total_agents} independent instances wihnt input size 1)")

    #return preprocessor_configs