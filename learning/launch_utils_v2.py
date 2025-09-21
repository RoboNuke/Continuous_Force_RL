"""
Launch utilities for configuration-based factory training.

This module provides step-by-step configuration application with clear logic flow.
Each function has a single, clear responsibility.
"""

import sys

from skrl.resources.schedulers.torch import KLAdaptiveLR
from models.SimBa_hybrid_control import HybridControlBlockSimBaActor
from configs.config_manager import ConfigManager

# Isaac Lab utilities are imported but not used in this file
# They may be needed for future functionality

# Import wrappers
from wrappers.mechanics.fragile_object_wrapper import FragileObjectWrapper
from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper
from wrappers.observations.observation_manager_wrapper import ObservationManagerWrapper
from wrappers.observations.observation_noise_wrapper import ObservationNoiseWrapper, ObservationNoiseConfig, NoiseGroupConfig
from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper
from wrappers.logging.factory_metrics_wrapper import FactoryMetricsWrapper
from wrappers.logging.wandb_wrapper import GenericWandbLoggingWrapper

# Removed LoggingConfigPresets - using simplified wandb wrapper

from agents.block_ppo import BlockPPO
try:
    from agents.wandb_logger_ppo_agent import BlockWandbLoggerPPO
except ImportError:
    # BlockWandbLoggerPPO not available - may have been removed in refactor
    BlockWandbLoggerPPO = None

import os
import torch
import gymnasium as gym

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from models.block_simba import BlockSimBaCritic, BlockSimBaActor, export_policies, make_agent_optimizer


# ===== ENVIRONMENT CONFIGURATION FUNCTIONS =====

def apply_easy_mode(env_cfg, agent_cfg):
    """
    Apply easy mode configuration to environment and agent settings.

    Configures the environment for easier training by modifying control gains,
    action thresholds, and break forces. Easy mode reduces task difficulty by
    increasing position/rotation gains and decreasing force requirements.

    Args:
        env_cfg: Environment configuration object with control and task settings
        agent_cfg: Agent configuration dictionary containing training parameters

    Side Effects:
        - Modifies env_cfg.ctrl properties (gains, thresholds, bounds)
        - Updates agent_cfg['agent']['break_force'] for easier object manipulation
        - Prints confirmation message when easy mode is applied
    """
    """Apply easy mode settings to environment for debugging."""
    agent_cfg['agent']['easy_mode'] = True

    # Set easy environment parameters
    env_cfg.task.duration_s = env_cfg.episode_length_s
    env_cfg.task.hand_init_pos = [0.0, 0.0, 0.035]  # Relative to fixed asset tip
    env_cfg.task.hand_init_pos_noise = [0.0025, 0.0025, 0.00]
    env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]

    # Fixed Asset (applies to all tasks)
    env_cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.fixed_asset_init_orn_deg = 0.0
    env_cfg.task.fixed_asset_init_orn_range_deg = 0.0

    # Held Asset (applies to all tasks)
    env_cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.held_asset_rot_init = 0.0

    print("  - Easy mode environment settings applied")


def configure_environment_scene(env_cfg, primary, derived):
    """
    Configure environment scene parameters for multi-agent factory training.

    Sets up the Isaac Lab environment scene configuration including number of
    environments, physics replication settings, and derived environment counts
    based on primary configuration parameters.

    Args:
        env_cfg: Environment configuration object with scene settings
        primary: Primary configuration dictionary containing base parameters
        derived: Derived configuration dictionary with calculated values

    Side Effects:
        - Sets env_cfg.scene.num_envs from derived['total_num_envs']
        - Configures env_cfg.scene.replicate_physics based on environment count
        - Prints scene configuration summary
    """
    """Configure Isaac Lab environment scene settings."""
    env_cfg.scene.num_envs = derived['total_num_envs']
    env_cfg.scene.replicate_physics = True
    env_cfg.num_agents = derived['total_agents']

    print(f"  - Scene configured: {derived['total_num_envs']} envs, {derived['total_agents']} agents")


def validate_factory_configuration(env_cfg):
    """Validate that environment configuration has required factory settings."""

    required_attributes = ['obs_order', 'state_order']
    missing_attributes = []

    for attr in required_attributes:
        if not hasattr(env_cfg, attr):
            missing_attributes.append(attr)

    if missing_attributes:
        print(f"[ERROR]: Missing critical factory configuration attributes: {missing_attributes}")
        print(f"[ERROR]: This indicates Isaac Lab factory configuration was not loaded properly.")
        print(f"[ERROR]: Required attributes from Isaac Lab factory config:")
        print(f"  - obs_order: Defines observation component ordering for policy")
        print(f"  - state_order: Defines state component ordering for critic")
        print(f"[ERROR]: Check that:")
        print(f"  1. Task name is correct: {getattr(env_cfg, 'task_name', 'UNKNOWN')}")
        print(f"  2. Isaac Lab factory tasks are properly registered")
        print(f"  3. Isaac Lab version supports the specified task")
        sys.exit(1)

    # Validate the attributes are not empty
    for attr in required_attributes:
        value = getattr(env_cfg, attr)
        if not value or len(value) == 0:
            print(f"[ERROR]: Factory configuration attribute '{attr}' is empty: {value}")
            print(f"[ERROR]: This indicates a problem with Isaac Lab factory configuration.")
            sys.exit(1)

    print(f"[INFO]: Factory configuration validation passed")
    print(f"  - obs_order: {env_cfg.obs_order}")
    print(f"  - state_order: {env_cfg.state_order}")


def enable_force_sensor(env_cfg):
    """
    Enable force sensor functionality in the environment configuration.

    Adds 'force_torque' to the environment's observation and state orders if not
    already present. This enables force-torque sensor data collection and
    inclusion in the observation pipeline for force-aware training.

    Args:
        env_cfg: Environment configuration object with obs_order and state_order lists

    Side Effects:
        - Appends 'force_torque' to env_cfg.obs_order if not present
        - Appends 'force_torque' to env_cfg.state_order if not present
        - Prints confirmation when force sensor is enabled
    """
    """Enable force-torque sensor in environment configuration."""
    env_cfg.use_force_sensor = True
    if 'force_torque' not in env_cfg.obs_order:
        env_cfg.obs_order.append("force_torque")
    if 'force_torque' not in env_cfg.state_order:
        env_cfg.state_order.append("force_torque")

    print("  - Force-torque sensor enabled in observation and state")


def apply_environment_overrides(env_cfg, environment_config):
    """
    Apply environment configuration overrides using dot notation.

    Allows dynamic modification of environment configuration parameters using
    nested attribute paths. Supports both object attributes and dictionary keys,
    enabling flexible environment customization without code changes.

    Args:
        env_cfg: Environment configuration object to modify
        environment_config: Dictionary mapping attribute paths to new values
                          (e.g., {'task.duration_s': 15.0, 'ctrl.gains': [1000, 1000]})

    Side Effects:
        - Modifies env_cfg attributes according to environment_config mappings
        - Uses _set_nested_attr helper to handle nested attribute setting
        - Prints applied overrides for debugging

    Example:
        environment_config = {
            'task.duration_s': 20.0,
            'ctrl.default_task_prop_gains': [2000, 2000, 2000, 200, 200, 200]
        }
        apply_environment_overrides(env_cfg, environment_config)
    """
    """Apply environment parameter overrides from configuration."""
    for key, value in environment_config.items():
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, value)
            print(f"  - Set env_cfg.{key} = {value}")
        elif '.' in key:
            _set_nested_attr(env_cfg, key, value)
            print(f"  - Set nested env_cfg.{key} = {value}")


# ===== AGENT CONFIGURATION FUNCTIONS =====

def apply_learning_config(agent_cfg, learning_config, max_rollout_steps):
    """
    Apply learning configuration parameters to agent settings.

    Updates agent configuration with learning-specific parameters such as
    rollout steps, learning rate, batch size, and entropy coefficient.
    Handles both direct parameter setting and calculated values.

    Args:
        agent_cfg: Agent configuration dictionary to modify
        learning_config: Learning configuration dictionary with training parameters
        max_rollout_steps: Maximum number of rollout steps per episode

    Side Effects:
        - Sets agent_cfg['agent']['rollouts'] from max_rollout_steps
        - Applies learning parameters from learning_config to agent_cfg['agent']
        - Only updates parameters that exist in learning_config
        - Prints applied learning configuration summary
    """
    """Apply learning configuration to agent."""
    # Set rollout parameters
    agent_cfg['agent']['rollouts'] = max_rollout_steps

    # Ensure experiment section exists
    if 'experiment' not in agent_cfg['agent']:
        agent_cfg['agent']['experiment'] = {}

    agent_cfg['agent']['experiment']['write_interval'] = max_rollout_steps
    agent_cfg['agent']['experiment']['checkpoint_interval'] = max_rollout_steps * 10

    # Apply learning parameters - both existing and known new parameters
    known_learning_params = {
        'learning_rate', 'batch_size', 'entropy_coeff', 'learning_epochs',
        'mini_batches', 'policy_learning_rate', 'critic_learning_rate',
        'value_update_ratio', 'use_huber_value_loss', 'state_preprocessor',
        'value_preprocessor', 'rollouts'
    }

    for key, value in learning_config.items():
        # Apply if it's already in config or if it's a known learning parameter
        if key in agent_cfg['agent'] or key in known_learning_params:
            agent_cfg['agent'][key] = value
            print(f"  - Set agent learning param: {key} = {value}")
        else:
            print(f"  - Skipping unknown learning param: {key}")

    print(f"  - Rollout steps set to: {max_rollout_steps}")


def apply_model_config(agent_cfg, model_config):
    """
    Apply model configuration parameters to agent model settings.

    Updates agent model configuration using dot notation for nested parameter
    setting. Supports flexible model architecture configuration without
    requiring code changes for different model structures.

    Args:
        agent_cfg: Agent configuration dictionary containing 'models' section
        model_config: Dictionary mapping model parameter paths to values
                     (e.g., {'network.layers': [64, 32], 'activation': 'relu'})

    Side Effects:
        - Modifies agent_cfg['models'] according to model_config mappings
        - Uses _set_nested_attr helper for nested parameter setting
        - Supports both policy and value model configuration

    Example:
        model_config = {
            'network.layers': [128, 64, 32],
            'activation': 'tanh',
            'policy.learning_rate': 0.0001
        }
        apply_model_config(agent_cfg, model_config)
    """
    """Apply model configuration to agent."""

    # Debug: Print model configuration being applied
    ConfigManager.print_model_config(model_config)

    # Ensure models section exists
    if 'models' not in agent_cfg:
        agent_cfg['models'] = {}

    # Ensure policy and value sections exist
    if 'policy' not in agent_cfg['models']:
        agent_cfg['models']['policy'] = {}
    if 'value' not in agent_cfg['models']:
        agent_cfg['models']['value'] = {}

    for key, value in model_config.items():
        if key == 'actor' and isinstance(value, dict):
            # Map actor config to policy
            for subkey, subvalue in value.items():
                agent_cfg['models']['policy'][subkey] = subvalue
                print(f"  - Set policy param: {subkey} = {subvalue}")
        elif key == 'critic' and isinstance(value, dict):
            # Map critic config to value
            for subkey, subvalue in value.items():
                agent_cfg['models']['value'][subkey] = subvalue
                print(f"  - Set value param: {subkey} = {subvalue}")
        elif key in agent_cfg['models']:
            agent_cfg['models'][key] = value
            print(f"  - Set model param: {key} = {value}")
        elif '.' in key:
            _set_nested_attr(agent_cfg['models'], key, value)
            print(f"  - Set nested model param: {key} = {value}")
        else:
            # Apply to both policy and value if no specific target
            agent_cfg['models']['policy'][key] = value
            agent_cfg['models']['value'][key] = value
            print(f"  - Set general model param: {key} = {value}")


def setup_experiment_logging(env_cfg, agent_cfg, resolved_config):
    """
    Set up experiment logging configuration for multi-agent training.

    Configures wandb experiment parameters, agent-specific logging paths,
    and individual agent configurations for distributed training with
    proper experiment tracking and organization.

    Args:
        env_cfg: Environment configuration object with task information
        agent_cfg: Agent configuration dictionary to update with logging settings
        resolved_config: Resolved configuration containing experiment and agent details

    Side Effects:
        - Sets agent_cfg['agent']['experiment'] with project, tags, and group
        - Adds task name to experiment tags automatically
        - Configures agent-specific logging paths and wandb settings
        - Creates individual agent configurations for multi-agent scenarios
        - Prints experiment configuration summary

    Resolved Config Structure:
        resolved_config = {
            'primary': {'break_forces': [...], 'agents_per_break_force': int},
            'derived': {'total_agents': int, 'total_num_envs': int},
            'experiment': {'wandb_project': str, 'name': str, 'tags': list, 'group': str}
        }
    """
    """Set up experiment and logging configuration."""
    primary = resolved_config['primary']
    experiment = resolved_config.get('experiment', {})

    # Ensure experiment section exists
    if 'experiment' not in agent_cfg['agent']:
        agent_cfg['agent']['experiment'] = {}

    # Set basic experiment parameters
    agent_cfg['agent']['experiment']['project'] = experiment.get('wandb_project', 'Continuous_Force_RL')
    agent_cfg['agent']['experiment']['tags'] = experiment.get('tags', [])
    agent_cfg['agent']['experiment']['group'] = experiment.get('group', '')

    # Set agent-specific data
    agent_cfg['agent']['break_force'] = primary['break_forces']
    agent_cfg['agent']['num_envs'] = resolved_config['derived']['total_num_envs']

    # Add task tags
    task_name = env_cfg.task_name if hasattr(env_cfg, 'task_name') else 'factory'
    agent_cfg['agent']['experiment']['tags'].append(task_name)

    # Set up individual agent logging paths
    _setup_individual_agent_logging(agent_cfg, resolved_config)

    print(f"  - Experiment '{experiment.get('name', 'unnamed')}' configured for {resolved_config['derived']['total_agents']} agents")


# ===== WRAPPER APPLICATION FUNCTIONS =====

def apply_fragile_object_wrapper(env, wrapper_config, primary, derived):
    """
    Apply fragile object wrapper to enable breakable object functionality.

    Wraps the environment with FragileObjectWrapper to simulate fragile objects
    that break when excessive force is applied. Supports per-agent break force
    configuration for multi-agent training scenarios.

    Args:
        env: Base environment to wrap
        wrapper_config: Wrapper configuration dictionary (may contain 'break_force')
        primary: Primary configuration with 'break_forces' fallback values
        derived: Derived configuration with 'total_agents' count

    Returns:
        FragileObjectWrapper: Wrapped environment with fragile object functionality

    Side Effects:
        - Prints fragile object wrapper configuration details
        - Uses wrapper_config['break_force'] if available, else primary['break_forces']

    Break Force Configuration:
        - Single float: Same threshold for all agents
        - List of floats: Different threshold per agent (length must match total_agents)
        - -1 value: Unbreakable objects (very high threshold)
    """
    """Apply fragile object wrapper to environment."""
    env = FragileObjectWrapper(
        env,
        break_force=wrapper_config.get('break_force', primary['break_forces']),
        num_agents=derived['total_agents']
    )
    return env


def apply_force_torque_wrapper(env, wrapper_config):
    """
    Apply force-torque sensor wrapper for contact force measurement.

    Wraps the environment with ForceTorqueWrapper to enable 6-DOF force-torque
    sensor functionality at the robot end-effector. Supports optional tanh
    scaling for bounded sensor readings.

    Args:
        env: Base environment to wrap
        wrapper_config: Configuration dictionary with optional parameters:
                       - 'use_tanh_scaling': bool, apply tanh scaling to readings
                       - 'tanh_scale': float, scaling factor for tanh transformation

    Returns:
        ForceTorqueWrapper: Wrapped environment with force-torque sensor capability

    Side Effects:
        - Prints force-torque wrapper configuration details
        - Adds force_torque observations to environment observation space
        - Initializes Isaac Sim RobotView for sensor data collection

    Default Configuration:
        - use_tanh_scaling: False
        - tanh_scale: 0.03
    """
    """Apply force-torque wrapper to environment."""
    env = ForceTorqueWrapper(
        env,
        use_tanh_scaling=wrapper_config.get('use_tanh_scaling', False),
        tanh_scale=wrapper_config.get('tanh_scale', 0.03)
    )
    return env


def apply_observation_manager_wrapper(env, wrapper_config):
    """
    Apply observation manager wrapper for flexible observation processing.

    Wraps the environment with ObservationManagerWrapper to enable dynamic
    observation space management, component selection, and observation
    preprocessing pipelines.

    Args:
        env: Base environment to wrap
        wrapper_config: Configuration dictionary with observation management settings

    Returns:
        ObservationManagerWrapper: Wrapped environment with observation management

    Side Effects:
        - Prints observation manager wrapper configuration
        - Configures observation component selection and processing
        - May modify observation space dimensions and structure
    """
    """Apply observation manager wrapper to environment."""
    env = ObservationManagerWrapper(
        env,
        merge_strategy=wrapper_config.get('merge_strategy', 'concatenate')
    )
    return env


def apply_observation_noise_wrapper(env, wrapper_config):
    """
    Apply observation noise wrapper for robust policy training.

    Wraps the environment with ObservationNoiseWrapper to inject configurable
    noise into observations during training. Supports per-component noise
    configuration with different noise types and parameters.

    Args:
        env: Base environment to wrap
        wrapper_config: Noise configuration dictionary with:
                       - 'global_noise_scale': float, global scaling factor
                       - 'enabled': bool, enable/disable noise injection
                       - 'apply_to_critic': bool, apply noise to critic observations
                       - 'noise_groups': dict, per-component noise configurations

    Returns:
        ObservationNoiseWrapper: Wrapped environment with noise injection capability

    Side Effects:
        - Prints observation noise configuration summary
        - Configures noise groups and application policies
        - Creates separate noise policies for actor and critic if specified

    Noise Group Configuration:
        noise_groups = {
            'component_name': {
                'noise_type': 'gaussian' | 'uniform',
                'std': float,         # for gaussian noise
                'mean': float,        # for gaussian noise
                'enabled': bool,
                'timing': 'step' | 'reset'
            }
        }
    """
    """Apply observation noise wrapper to environment."""
    noise_config = ObservationNoiseConfig(
        global_noise_scale=wrapper_config.get('global_noise_scale', 1.0),
        enabled=wrapper_config.get('enabled', True),
        apply_to_critic=wrapper_config.get('apply_to_critic', True)
    )

    # Add noise groups
    noise_groups = wrapper_config.get('noise_groups', {})
    for group_name, group_config in noise_groups.items():
        noise_group = NoiseGroupConfig(
            group_name=group_name,
            noise_type=group_config.get('noise_type', 'gaussian'),
            std=group_config.get('std', 0.01),
            mean=group_config.get('mean', 0.0),
            enabled=group_config.get('enabled', True),
            timing=group_config.get('timing', 'step')
        )
        noise_config.add_group_noise(noise_group)

    env = ObservationNoiseWrapper(env, noise_config)
    return env


def apply_hybrid_control_wrapper(env, wrapper_config):
    """
    Apply hybrid control wrapper for force-position control integration.

    Wraps the environment with HybridForcePositionWrapper to enable hybrid
    control modes combining position and force control. Supports selective
    control component activation and action space modification.

    Args:
        env: Base environment to wrap
        wrapper_config: Hybrid control configuration with:
                       - 'ctrl_force': bool, enable force control
                       - 'ctrl_torque': bool, enable torque control
                       - Additional control parameters

    Returns:
        HybridForcePositionWrapper: Wrapped environment with hybrid control capability

    Side Effects:
        - Prints hybrid control configuration details
        - Modifies action space to include force/torque components
        - Updates environment control configuration
        - Adds 'prev_actions' to observation components for action history

    Control Modes:
        - Position-only: Traditional position/rotation control
        - Force-only: Pure force/torque control
        - Hybrid: Combined position and force control
    """
    """Apply hybrid force-position control wrapper to environment."""
    from wrappers.control.hybrid_control_cfg import HybridCtrlCfg, HybridTaskCfg

    ctrl_torque = wrapper_config.get('ctrl_torque', False)

    # Create configuration objects with environment defaults and wrapper overrides
    env_cfg_ctrl = getattr(env.unwrapped.cfg, 'ctrl', {})

    # Helper function to get attribute from object or dictionary
    def get_ctrl_attr(obj, attr_name, default=None):
        if hasattr(obj, attr_name):
            return getattr(obj, attr_name)
        elif isinstance(obj, dict):
            return obj.get(attr_name, default)
        else:
            return default

    ctrl_cfg = HybridCtrlCfg(
        ema_factor=wrapper_config.get('ema_factor', 0.2),
        no_sel_ema=wrapper_config.get('no_sel_ema', True),
        target_init_mode=wrapper_config.get('target_init_mode', 'zero'),
        default_task_force_gains=wrapper_config.get('default_task_force_gains',
                                                   get_ctrl_attr(env_cfg_ctrl, 'default_task_force_gains', None)),
        force_action_bounds=get_ctrl_attr(env_cfg_ctrl, 'force_action_bounds', None),
        torque_action_bounds=get_ctrl_attr(env_cfg_ctrl, 'torque_action_bounds', None),
        force_action_threshold=get_ctrl_attr(env_cfg_ctrl, 'force_action_threshold', None),
        torque_action_threshold=get_ctrl_attr(env_cfg_ctrl, 'torque_action_threshold', None),
        pos_action_bounds=get_ctrl_attr(env_cfg_ctrl, 'pos_action_bounds', None),
        rot_action_bounds=get_ctrl_attr(env_cfg_ctrl, 'rot_action_bounds', None)
    )

    task_cfg = HybridTaskCfg(
        force_active_threshold=wrapper_config.get('force_active_threshold', 0.1),
        torque_active_threshold=wrapper_config.get('torque_active_threshold', 0.01),
        good_force_cmd_rew=wrapper_config.get('good_force_cmd_rew', 0.1),
        bad_force_cmd_rew=wrapper_config.get('bad_force_cmd_rew', -0.1),
        wrench_norm_scale=wrapper_config.get('wrench_norm_scale', 0.01)
    )

    env = HybridForcePositionWrapper(
        env,
        ctrl_torque=ctrl_torque,
        reward_type=wrapper_config.get('reward_type', 'simp'),
        ctrl_cfg=ctrl_cfg,
        task_cfg=task_cfg
    )

    # Update prev_actions dimension to match the new action space
    # HybridForcePositionWrapper sets action_space_size based on ctrl_torque:
    # - Base action space: 6 (position + rotation)
    # - Hybrid adds: 2*force_size where force_size = 6 if ctrl_torque else 3
    # - Total: 6 + 6 = 12 (force only) or 6 + 12 = 18 (force + torque)
    if hasattr(env.unwrapped.cfg, 'component_dims'):
        if ctrl_torque:
            new_action_size = 18  # 6 (pos+rot) + 12 (force+torque: 2*6)
        else:
            new_action_size = 12  # 6 (pos+rot) + 6 (force only: 2*3)

        env.unwrapped.cfg.component_dims['prev_actions'] = new_action_size
        print(f"  - Updated prev_actions dimension to {new_action_size} for hybrid control (ctrl_torque={ctrl_torque})")

    return env


def apply_factory_metrics_wrapper(env, derived):
    """
    Apply factory metrics wrapper for training performance tracking.

    Wraps the environment with FactoryMetricsWrapper to collect and track
    factory-specific training metrics such as success rates, contact forces,
    and task completion statistics across multiple agents.

    Args:
        env: Base environment to wrap
        derived: Derived configuration dictionary containing:
                - 'total_agents': int, number of agents for metric aggregation

    Returns:
        FactoryMetricsWrapper: Wrapped environment with metrics collection

    Side Effects:
        - Prints factory metrics wrapper initialization
        - Validates environment count divisibility by agent count
        - Sets up per-agent metric collection and aggregation

    Collected Metrics:
        - Task success rates per agent
        - Contact force statistics
        - Episode completion metrics
        - Performance benchmarking data
    """
    """Apply factory metrics wrapper to environment."""
    env = FactoryMetricsWrapper(env, num_agents=derived['total_agents'])
    return env


def apply_wandb_logging_wrapper(env, wrapper_config, derived, agent_cfg, env_cfg, resolved_config, logging_config=None):
    """
    Apply Weights & Biases logging wrapper for experiment tracking.

    Wraps the environment with WandbWrapper to enable comprehensive experiment
    tracking, metric logging, and visualization through Weights & Biases.
    Supports multi-agent logging with agent-specific configurations.

    Args:
        env: Base environment to wrap
        wrapper_config: Wandb wrapper configuration dictionary
        derived: Derived configuration with agent and environment counts
        agent_cfg: Agent configuration with experiment settings
        env_cfg: Environment configuration with task information
        resolved_config: Resolved configuration with agent-specific settings
        logging_config: Optional additional logging configuration

    Returns:
        WandbWrapper: Wrapped environment with wandb logging capability

    Side Effects:
        - Prints wandb logging configuration summary
        - Initializes wandb experiment tracking
        - Configures agent-specific logging paths and parameters
        - Sets up metric collection and visualization

    Required Agent Config:
        agent_cfg['agent'] must contain:
        - 'experiment': {'project': str, 'tags': list, 'group': str}
        - Agent-specific configurations for multi-agent scenarios
    """
    """Apply simplified wandb logging wrapper to environment."""
    print("  - Using wandb wrapper with per-agent configuration")

    # Get pre-built agent configs from resolved_config (created in _setup_individual_agent_logging)
    agent_specific_configs = resolved_config.get('agent_specific_configs', {})
    if not agent_specific_configs:
        raise ValueError("agent_specific_configs not found in resolved_config. Make sure _setup_individual_agent_logging was called.")

    # Add agent configs to env_cfg for the wrapper to extract
    import copy
    env_cfg_with_agents = copy.deepcopy(env_cfg)
    env_cfg_with_agents.agent_configs = agent_specific_configs

    env = GenericWandbLoggingWrapper(env, num_agents=derived['total_agents'], env_cfg=env_cfg_with_agents)
    return env


def apply_enhanced_action_logging_wrapper(env, wrapper_config):
    """
    Apply enhanced action logging wrapper for detailed action analysis.

    Wraps the environment with EnhancedActionLoggingWrapper to provide
    comprehensive action tracking and analysis. Requires WandbWrapper
    to be present in the wrapper chain for metric reporting.

    Args:
        env: Base environment to wrap (must have wandb logging capability)
        wrapper_config: Action logging configuration with:
                       - 'track_selection': bool, track selection components
                       - 'track_pos': bool, track position components
                       - 'track_rot': bool, track rotation components
                       - 'track_force': bool, track force components
                       - 'track_torque': bool, track torque components
                       - 'force_size': int, force vector size (0, 3, or 6)
                       - 'logging_frequency': int, steps between metric collection

    Returns:
        EnhancedActionLoggingWrapper: Wrapped environment with enhanced action logging

    Side Effects:
        - Prints action logging configuration and component summary
        - Validates wandb wrapper presence in environment chain
        - Configures action component tracking and analysis
        - Sets up periodic metric collection and reporting

    Raises:
        ValueError: If wandb wrapper is not found in environment chain
    """
    """Apply enhanced action logging wrapper to environment."""
    from wrappers.logging.enhanced_action_logging_wrapper import EnhancedActionLoggingWrapper

    env = EnhancedActionLoggingWrapper(
        env,
        track_selection=wrapper_config.get('track_selection', True),
        track_pos=wrapper_config.get('track_pos', True),
        track_rot=wrapper_config.get('track_rot', True),
        track_force=wrapper_config.get('track_force', True),
        track_torque=wrapper_config.get('track_torque', True),
        force_size=wrapper_config.get('force_size', 6),
        logging_frequency=wrapper_config.get('logging_frequency', 100)
    )
    return env


# ===== PREPROCESSOR SETUP =====

def setup_preprocessors(env_cfg, agent_cfg, env, learning_config):
    """
    Set up SKRL preprocessors for observation and state processing.

    Configures and initializes SKRL preprocessor instances for policy and value
    function inputs. Handles both single-agent and multi-agent preprocessor
    configurations based on learning settings.

    Args:
        env_cfg: Environment configuration with observation specifications
        agent_cfg: Agent configuration dictionary to update with preprocessor settings
        env: Environment instance for observation space information
        learning_config: Learning configuration with preprocessor parameters

    Side Effects:
        - Adds preprocessor instances to agent_cfg['agent']
        - Configures state_preprocessor and value_preprocessor for SKRL
        - Sets up preprocessor_kwargs for both policy and value networks
        - Prints preprocessor configuration summary

    Preprocessor Types:
        - State preprocessor: For policy network observations
        - Value preprocessor: For value function state inputs
        - Supports normalization, scaling, and other preprocessing transforms
    """
    """Set up state and value preprocessors."""
    ## ORIGINAL SHARED PREPROCESSOR CODE (BROKEN FOR MULTI-AGENT) ##
    # if learning_config.get('state_preprocessor', True):
    #     agent_cfg['agent']["state_preprocessor"] = RunningStandardScaler
    #     agent_cfg['agent']["state_preprocessor_kwargs"] = {
    #         "size": env.cfg.observation_space + env.cfg.state_space,
    #         "device": env_cfg.sim.device
    #     }
    #     print("  - State preprocessor enabled")
    #
    # if learning_config.get('value_preprocessor', True):
    #     agent_cfg['agent']["value_preprocessor"] = RunningStandardScaler
    #     agent_cfg['agent']["value_preprocessor_kwargs"] = {"size": 1, "device": env_cfg.sim.device}
    #     print("  - Value preprocessor enabled")

    ## NEW PER-AGENT INDEPENDENT PREPROCESSOR SETUP ##
    # Note: This function is now deprecated for BlockPPO.
    # Per-agent preprocessors are handled directly in create_block_ppo_agents function.
    # Keeping this function for backward compatibility with other agent types.

    if learning_config.get('state_preprocessor', True):
        agent_cfg['agent']["state_preprocessor"] = RunningStandardScaler
        agent_cfg['agent']["state_preprocessor_kwargs"] = {
            "size": env.cfg.observation_space + env.cfg.state_space,
            "device": env_cfg.sim.device
        }
        print("  - State preprocessor enabled (shared - legacy mode)")

    if learning_config.get('value_preprocessor', True):
        agent_cfg['agent']["value_preprocessor"] = RunningStandardScaler
        agent_cfg['agent']["value_preprocessor_kwargs"] = {"size": 1, "device": env_cfg.sim.device}
        print("  - Value preprocessor enabled (shared - legacy mode)")


def setup_per_agent_preprocessors(env_cfg, env, agent_config, num_agents):
    """
    Set up per-agent preprocessors for multi-agent training scenarios.

    Creates individual preprocessor instances for each agent in multi-agent
    training setups. Each agent gets dedicated preprocessor configuration
    for independent observation and state processing.

    Args:
        env_cfg: Environment configuration with observation specifications
        env: Environment instance for observation space information
        agent_config: Agent configuration with preprocessor parameters
        num_agents: Number of agents requiring individual preprocessors

    Returns:
        dict: Dictionary mapping agent indices to preprocessor configurations

    Side Effects:
        - Creates separate preprocessor instances for each agent
        - Prints per-agent preprocessor setup summary

    Agent Preprocessor Structure:
        {
            'agent_0': {'state_preprocessor': ..., 'value_preprocessor': ...},
            'agent_1': {'state_preprocessor': ..., 'value_preprocessor': ...},
            ...
        }
    """
    """Set up preprocessor configurations for BlockPPO per-agent system.

    Instead of creating complex per-agent configs, this just sets up the standard
    preprocessor config that BlockPPO will use to create independent instances.
    """
    preprocessor_configs = {}

    if agent_config.get('state_preprocessor', True):
        preprocessor_configs["state_preprocessor"] = RunningStandardScaler
        preprocessor_configs["state_preprocessor_kwargs"] = {
            "size": env.cfg.observation_space + env.cfg.state_space,
            "device": env_cfg.sim.device
        }
        print(f"  - State preprocessor config set (will create {num_agents} independent instances)")

    if agent_config.get('value_preprocessor', True):
        preprocessor_configs["value_preprocessor"] = RunningStandardScaler
        preprocessor_configs["value_preprocessor_kwargs"] = {
            "size": 1,
            "device": env_cfg.sim.device
        }
        print(f"  - Value preprocessor config set (will create {num_agents} independent instances)")

    return preprocessor_configs


# ===== MODEL CREATION =====

def create_policy_and_value_models(env_cfg, agent_cfg, env, model_config, wrappers_config, derived):
    """
    Create policy and value function models for reinforcement learning.

    Creates appropriate model architectures based on agent configuration.
    Supports both standard policy models and hybrid control policy models
    with automatic architecture selection based on environment capabilities.

    Args:
        env_cfg: Environment configuration with control and observation settings
        agent_cfg: Agent configuration dictionary with model parameters
        env: Environment instance for action/observation space information
        model_config: Model architecture configuration dictionary
        wrappers_config: Wrapper configuration for model architecture decisions
        derived: Derived configuration with calculated parameters

    Returns:
        dict: Dictionary containing 'policy' and 'value' model instances

    Side Effects:
        - Configures hybrid agent parameters if hybrid control is enabled
        - Prints model creation summary with architecture details
        - Calls appropriate model creation functions based on configuration

    Model Selection:
        - Hybrid policy: For environments with force/torque control capabilities
        - Standard policy: For traditional position/orientation control
        - Value model: Always created for value function approximation
    """
    """Create policy and value models based on configuration."""
    models = {}

    # Determine model type and parameters
    use_hybrid_agent = model_config.get('use_hybrid_agent', False)
    use_hybrid_control = wrappers_config.get('hybrid_control', {}).get('enabled', False)

    if use_hybrid_agent:
        print("  - Creating hybrid control agent models")
        models['policy'] = _create_hybrid_policy_model(env_cfg, agent_cfg, env, model_config, derived, wrappers_config)
    else:
        print("  - Creating standard SimBa agent models")
        models['policy'] = _create_standard_policy_model(env, model_config, wrappers_config, derived)

    print("  - Creating value model")
    models["value"] = _create_value_model(env, model_config, agent_cfg, derived)

    return models


def _create_hybrid_policy_model(env_cfg, agent_cfg, env, model_config, derived, wrappers_config):
    """
    Create hybrid policy model for force-position control integration.

    Creates a specialized policy model architecture designed for hybrid control
    scenarios where the agent must learn to coordinate position and force control.
    Includes automatic parameter calculation and model configuration.

    Args:
        env_cfg: Environment configuration with control specifications
        agent_cfg: Agent configuration with hybrid agent parameters
        env: Environment instance for action space information
        model_config: Model architecture configuration
        derived: Derived configuration with calculated parameters

    Returns:
        torch.nn.Module: Hybrid policy model instance

    Side Effects:
        - Configures hybrid agent parameters automatically
        - Prints hybrid policy model creation details
        - Sets up model architecture for multi-modal control

    Hybrid Control Features:
        - Separate networks for position and force control
        - Coordinated action selection mechanisms
        - Adaptive control mode switching
    """
    """Create hybrid control policy model."""
    # Set hybrid agent initialization parameters
    _configure_hybrid_agent_parameters(env_cfg, agent_cfg, model_config, wrappers_config)

    return HybridControlBlockSimBaActor(
        observation_space=env.cfg.observation_space,
        action_space=env.action_space,
        device=env.device,
        hybrid_agent_parameters=agent_cfg['agent']['hybrid_agent'],
        actor_n=model_config['actor']['n'],
        actor_latent=model_config['actor']['latent_size'],
        num_agents=derived['total_agents']
    )


def _create_standard_policy_model(env, model_config, wrappers_config, derived):
    """
    Create standard policy model for traditional control scenarios.

    Creates a conventional policy model architecture for standard reinforcement
    learning control tasks. Supports configurable network architectures and
    activation functions based on model configuration.

    Args:
        env: Environment instance for action/observation space information
        model_config: Model architecture configuration dictionary
        wrappers_config: Wrapper configuration for architecture decisions
        derived: Derived configuration with calculated parameters

    Returns:
        torch.nn.Module: Standard policy model instance

    Side Effects:
        - Prints standard policy model creation details
        - Configures model architecture based on environment specifications
        - Sets up appropriate input/output dimensions

    Model Features:
        - Configurable hidden layer sizes
        - Flexible activation function selection
        - Standard actor-critic architecture compatibility
    """
    """Create standard SimBa policy model."""
    sigma_idx = 0
    if wrappers_config.get('hybrid_control', {}).get('enabled', False):
        ctrl_torque = wrappers_config.get('hybrid_control', {}).get('ctrl_torque', False)
        sigma_idx = 6 if ctrl_torque else 3

    return BlockSimBaActor(
        observation_space=env.cfg.observation_space,
        action_space=env.action_space,
        device=env.device,
        act_init_std=model_config['act_init_std'],
        actor_n=model_config['actor']['n'],
        actor_latent=model_config['actor']['latent_size'],
        sigma_idx=sigma_idx,
        num_agents=derived['total_agents'],
        last_layer_scale=model_config['last_layer_scale']
    )


def _create_value_model(env, model_config, agent_cfg, derived):
    """
    Create value function model for state value estimation.

    Creates a value function network for estimating state values in reinforcement
    learning. Supports both standard and hybrid agent configurations with
    appropriate network architectures.

    Args:
        env: Environment instance for state space information
        model_config: Model architecture configuration dictionary
        agent_cfg: Agent configuration with value function parameters
        derived: Derived configuration with calculated parameters

    Returns:
        torch.nn.Module: Value function model instance

    Side Effects:
        - Prints value model creation details
        - Configures network architecture for state value estimation
        - Sets up appropriate input dimensions for state processing

    Value Function Features:
        - State-dependent value estimation
        - Configurable network architecture
        - Compatible with both on-policy and off-policy algorithms
    """
    """Create value model."""
    return BlockSimBaCritic(
        state_space_size=env.cfg.state_space,
        device=env.device,
        critic_output_init_mean=model_config['critic_output_init_mean'] * agent_cfg['agent']['rewards_shaper_scale'],
        critic_n=model_config['critic']['n'],
        critic_latent=model_config['critic']['latent_size'],
        num_agents=derived['total_agents']
    )


# ===== AGENT CREATION =====

def create_block_wandb_agents(env_cfg, agent_cfg, env, models, memory, derived, learning_config):
    """
    Create BlockWandbLoggerPPO agents for training with wandb integration.

    Creates specialized PPO agents with integrated Weights & Biases logging
    capabilities. Handles both single-agent and multi-agent configurations
    with automatic agent distribution and wandb setup.

    Args:
        env_cfg: Environment configuration with training specifications
        agent_cfg: Agent configuration with algorithm and logging parameters
        env: Environment instance for agent initialization
        models: Dictionary containing 'policy' and 'value' model instances
        memory: Memory/replay buffer instance for experience storage
        derived: Derived configuration with agent and environment counts
        learning_config: Learning configuration with algorithm parameters

    Returns:
        list: List of BlockWandbLoggerPPO agent instances

    Side Effects:
        - Creates individual agent instances for multi-agent training
        - Configures wandb logging for each agent
        - Sets up agent-specific memory and model configurations
        - Prints agent creation summary

    Agent Features:
        - Integrated wandb experiment tracking
        - Multi-agent support with agent-specific configurations
        - PPO algorithm implementation with custom logging
    """
    """Create BlockWandbLoggerPPO agents with optimizer."""
    if BlockWandbLoggerPPO is None:
        raise ImportError("BlockWandbLoggerPPO is not available. It may have been removed during refactoring.")

    # Copy first agent experiment config to main config (required by MultiWandbLoggerPPO)
    if 'agent_0' in agent_cfg['agent']:
        for key, item in agent_cfg['agent']['agent_0']['experiment'].items():
            agent_cfg['agent']['experiment'][key] = item

    agent = BlockWandbLoggerPPO(
        models=models,
        memory=memory,
        cfg=agent_cfg['agent'],
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_envs=env_cfg.scene.num_envs,
        state_size=env.cfg.observation_space + env.cfg.state_space,
        device=env.device,
        task=env_cfg.task_name if hasattr(env_cfg, 'task_name') else 'factory',
        task_cfg=env_cfg,
        num_agents=derived['total_agents']
    )

    # Create optimizer
    agent.optimizer = make_agent_optimizer(
        models['policy'],
        models['value'],
        policy_lr=learning_config['policy_learning_rate'],
        critic_lr=learning_config['critic_learning_rate'],
        betas=tuple(learning_config['optimizer']['betas']),  # From config instead of hardcoded
        eps=learning_config['optimizer']['eps'],             # From config instead of hardcoded
        weight_decay=learning_config['optimizer']['weight_decay'],  # From config instead of hardcoded
        debug=primary.get('debug_mode', False)
    )

    print(f"  - Created {derived['total_agents']} BlockWandB agents with optimizer")
    return agent


## NEW FUNCTION: CREATE BLOCKPPO AGENTS WITH PER-AGENT PREPROCESSORS ##

def create_block_ppo_agents(env_cfg, agent_cfg, env, models, memory, derived):
    """
    Create standard BlockPPO agents for training without wandb integration.

    Creates basic PPO agents for reinforcement learning training without
    additional logging overhead. Supports multi-agent configurations with
    standard PPO algorithm implementation.

    Args:
        env_cfg: Environment configuration with training specifications
        agent_cfg: Agent configuration with algorithm parameters (includes SKRL PPO params)
        env: Environment instance for agent initialization
        models: Dictionary containing 'policy' and 'value' model instances
        memory: Memory/replay buffer instance for experience storage
        derived: Derived configuration with agent and environment counts

    Returns:
        list: List of BlockPPO agent instances

    Side Effects:
        - Creates individual agent instances for multi-agent training
        - Configures agent-specific memory and model associations
        - Sets up standard PPO algorithm parameters
        - Prints agent creation summary

    Agent Features:
        - Standard PPO algorithm implementation
        - Multi-agent support without logging overhead
        - Efficient training for performance-critical scenarios
    """
    """Create BlockPPO agents with per-agent independent preprocessors and optimizer.

    This replaces create_block_wandb_agents for the new BlockPPO system.
    Key differences:
    - Uses BlockPPO instead of BlockWandbLoggerPPO
    - Sets up per-agent independent preprocessors
    - Integrates with wrapper system for logging instead of direct wandb
    """
    print("  - Creating BlockPPO agents with per-agent preprocessors")

    # Set up preprocessor configurations (BlockPPO will create independent instances)
    preprocessor_configs = setup_per_agent_preprocessors(
        env_cfg, env, agent_cfg['agent'], derived['total_agents']
    )

    # Update agent_cfg with preprocessor configs
    agent_cfg['agent'].update(preprocessor_configs)

    # Copy experiment configs for checkpoint tracking (needed by BlockPPO)
    agent_exp_cfgs = []
    if 'agent_0' in agent_cfg['agent']:
        for i in range(derived['total_agents']):
            if f'agent_{i}' in agent_cfg['agent']:
                agent_exp_cfgs.append(agent_cfg['agent'][f'agent_{i}'])
            else:
                # Fallback to agent_0 config
                agent_exp_cfgs.append(agent_cfg['agent']['agent_0'])

    # Create BlockPPO agent
    agent = BlockPPO(
        models=models,
        memory=memory,
        cfg=agent_cfg['agent'],
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_envs=env_cfg.scene.num_envs,
        state_size=env.cfg.observation_space + env.cfg.state_space,
        device=env.device,
        task=env_cfg.task_name if hasattr(env_cfg, 'task_name') else 'factory',
        num_agents=derived['total_agents'],
        env=env 
    )

    # Store experiment configs for checkpoint saving
    agent.agent_exp_cfgs = agent_exp_cfgs

    # Create optimizer
    agent.optimizer = make_agent_optimizer(
        models['policy'],
        models['value'],
        policy_lr=agent_cfg['agent']['policy_learning_rate'],
        critic_lr=agent_cfg['agent']['critic_learning_rate'],
        betas=tuple(agent_cfg['agent']['optimizer']['betas']),  # From config instead of hardcoded
        eps=agent_cfg['agent']['optimizer']['eps'],             # From config instead of hardcoded
        weight_decay=agent_cfg['agent']['optimizer']['weight_decay'],  # From config instead of hardcoded
        debug=agent_cfg['agent'].get('debug_mode', False)
    )

    print(f"  - Created BlockPPO with {derived['total_agents']} independent agents")
    print(f"  - Preprocessor configs set: {len(preprocessor_configs)} types")

    return agent


# ===== HELPER FUNCTIONS =====

def _set_nested_attr(obj, key, value):
    """
    Set nested attribute or dictionary key using dot notation.

    Utility function for setting nested attributes in objects or dictionary keys
    using dot-separated paths. Supports both object attribute access and dictionary
    key access, automatically detecting the appropriate access method.

    Args:
        obj: Object or dictionary to modify
        key: Dot-separated path to the target attribute/key (e.g., 'task.duration_s')
        value: Value to set at the specified path

    Side Effects:
        - Modifies obj by setting the specified nested attribute/key to value
        - Handles both object.attribute and dict['key'] access patterns
        - Traverses nested structures following the dot-separated path

    Examples:
        _set_nested_attr(config, 'task.duration_s', 20.0)
        _set_nested_attr(agent_cfg, 'models.network.layers', [64, 32])
        _set_nested_attr(settings, 'env.scene.num_envs', 16)

    Access Pattern Detection:
        - Uses isinstance(obj, dict) to determine access method
        - Supports mixed object/dictionary structures in nested paths
    """
    """Set nested attribute using dot notation."""
    keys = key.split('.')
    current = obj
    for k in keys[:-1]:
        if isinstance(current, dict):
            current = current[k]
        else:
            current = getattr(current, k)

    if isinstance(current, dict):
        current[keys[-1]] = value
    else:
        setattr(current, keys[-1], value)


def _setup_individual_agent_logging(agent_cfg, resolved_config):
    """
    Set up individual agent logging configurations for multi-agent training.

    Creates agent-specific logging paths, wandb configurations, and experiment
    tracking settings for distributed multi-agent training scenarios. Each agent
    gets unique logging directories and wandb run configurations.

    Args:
        agent_cfg: Agent configuration dictionary to update with logging settings
        resolved_config: Resolved configuration containing:
                        - 'primary': {'break_forces': list, 'agents_per_break_force': int}
                        - 'derived': {'total_agents': int}
                        - 'experiment': wandb experiment configuration

    Side Effects:
        - Adds agent-specific configurations to agent_cfg['agent']
        - Creates unique logging directories for each agent
        - Configures wandb kwargs for individual agent tracking
        - Updates resolved_config with agent_specific_configs
        - Prints individual agent logging setup summary

    Agent Configuration Structure:
        agent_cfg['agent']['agent_i'] = {
            'break_force': float,
            'experiment': {
                'directory': str,
                'wandb_kwargs': dict
            }
        }
    """
    """Set up individual agent logging paths and wandb configuration."""
    primary = resolved_config['primary']
    derived = resolved_config['derived']
    experiment = resolved_config.get('experiment', {})
    break_forces = primary['break_forces']

    if not isinstance(break_forces, list):
        break_forces = [break_forces]

    # Create agent-specific configurations for wandb wrapper
    agent_specific_configs = {}

    agent_idx = 0
    for break_force in break_forces:
        for i in range(primary['agents_per_break_force']):
            agent_cfg['agent'][f'agent_{agent_idx}'] = {}
            agent_cfg['agent'][f'agent_{agent_idx}']['break_force'] = break_force
            agent_cfg['agent'][f'agent_{agent_idx}']['experiment'] = {}

            # Agent seeding is handled globally by factory runner

            # Set up logging directory
            exp_name = experiment.get('name', 'default_experiment')
            log_root_path = os.path.join("logs", f"{exp_name}_f({break_force})_{agent_idx}")
            log_root_path = os.path.abspath(log_root_path)

            log_dir = f"{exp_name}_f({break_force})_{agent_idx}"

            agent_cfg["agent"][f'agent_{agent_idx}']["experiment"]["directory"] = log_root_path
            agent_cfg["agent"][f'agent_{agent_idx}']["experiment"]["experiment_name"] = log_dir

            # Set up wandb configuration
            agent_cfg['agent'][f'agent_{agent_idx}']['experiment']['wandb'] = True
            wandb_kwargs = {
                "project": agent_cfg['agent']['experiment']['project'],
                "entity": experiment.get('wandb_entity', 'hur'),
                "api_key": '-1',
                "tags": agent_cfg['agent']['experiment']['tags'],
                "group": agent_cfg['agent']['experiment']['group'] + f"_f({break_force})",
                "run_name": f"{log_dir}_f({break_force})_{agent_idx}"
            }

            agent_cfg["agent"][f'agent_{agent_idx}']["experiment"]["wandb_kwargs"] = wandb_kwargs

            # Create simplified config for wandb wrapper
            agent_config = {
                'agent_index': agent_idx,
                'break_force': break_force,
                'wandb_entity': experiment.get('wandb_entity', 'hur'),
                'wandb_project': agent_cfg['agent']['experiment']['project'],
                'wandb_name': f"{log_dir}_f({break_force})_{agent_idx}",
                'wandb_group': agent_cfg['agent']['experiment']['group'] + f"_f({break_force})",
                'wandb_tags': agent_cfg['agent']['experiment']['tags'],
                'api_key': '-1'
            }
            agent_specific_configs[f'agent_{agent_idx}'] = agent_config

            agent_idx += 1

    # Store agent configs in resolved_config for wandb wrapper to use
    resolved_config['agent_specific_configs'] = agent_specific_configs

    print(f"  - Set up logging for {agent_idx} agents with unique paths and wandb runs")


def _configure_hybrid_agent_parameters(env_cfg, agent_cfg, model_config, wrappers_config):
    """
    Configure hybrid agent parameters for force-position control integration.

    Automatically calculates and sets hybrid agent parameters based on environment
    control configuration. Handles both unit standard deviation initialization
    and scale factor calculation for position, rotation, force, and torque control.

    Args:
        env_cfg: Environment configuration with control gains and bounds
        agent_cfg: Agent configuration to update with hybrid parameters
        model_config: Model configuration for parameter calculation

    Side Effects:
        - Updates agent_cfg['agent']['hybrid_agent'] with calculated parameters
        - Sets init_std values when unit_std_init is enabled
        - Calculates scale factors based on environment control configuration
        - Prints hybrid agent parameter configuration summary

    Calculated Parameters:
        - pos_init_std, rot_init_std, force_init_std: Initialization standard deviations
        - pos_scale, rot_scale, force_scale, torque_scale: Action scaling factors

    Parameter Calculation:
        - Scale factors derived from environment control gains and bounds
        - Standard deviations based on action space characteristics
        - Unit standard deviation mode for normalized initialization
    """
    """Configure hybrid agent initialization parameters."""

    # Copy hybrid agent configuration from model config to agent config
    if 'hybrid_agent' in model_config:
        for key, value in model_config['hybrid_agent'].items():
            agent_cfg['agent']['hybrid_agent'][key] = value

    # Check if unit_std_init is enabled
    unit_std_init = agent_cfg['agent']['hybrid_agent'].get('unit_std_init', False)
    if unit_std_init:
        agent_cfg['agent']['hybrid_agent']['pos_init_std'] = (1 / (env_cfg.ctrl.default_task_prop_gains[0] * env_cfg.ctrl.pos_action_threshold[0])) ** 2
        agent_cfg['agent']['hybrid_agent']['rot_init_std'] = (1 / (env_cfg.ctrl.default_task_prop_gains[-1] * env_cfg.ctrl.rot_action_threshold[0]))**2
        agent_cfg['agent']['hybrid_agent']['force_init_std'] = (1 / (env_cfg.ctrl.default_task_force_gains[0] * env_cfg.ctrl.force_action_threshold[0]))**2

    agent_cfg['agent']['hybrid_agent']['pos_scale'] = env_cfg.ctrl.default_task_prop_gains[0] * env_cfg.ctrl.pos_action_threshold[0]
    agent_cfg['agent']['hybrid_agent']['rot_scale'] = env_cfg.ctrl.default_task_prop_gains[-1] * env_cfg.ctrl.rot_action_threshold[0]
    agent_cfg['agent']['hybrid_agent']['force_scale'] = env_cfg.ctrl.default_task_force_gains[0] * env_cfg.ctrl.force_action_threshold[0]
    agent_cfg['agent']['hybrid_agent']['torque_scale'] = env_cfg.ctrl.default_task_force_gains[-1] * env_cfg.ctrl.torque_action_bounds[0]

    # Extract ctrl_torque from wrappers_config
    agent_cfg['agent']['hybrid_agent']['ctrl_torque'] = wrappers_config.get('hybrid_control', {}).get('ctrl_torque', False)

    # Set uniform sampling rate (typically a constant)
    agent_cfg['agent']['hybrid_agent']['uniform_sampling_rate'] = 0.1  # Default value

    # Set additional hybrid model parameters with defaults
    agent_cfg['agent']['hybrid_agent']['selection_adjustment_types'] = ['l2_norm']  # Default adjustment type
    agent_cfg['agent']['hybrid_agent']['init_scale_last_layer'] = True
    agent_cfg['agent']['hybrid_agent']['init_layer_scale'] = 0.1
    agent_cfg['agent']['hybrid_agent']['init_scale_weights_factor'] = 0.01
    agent_cfg['agent']['hybrid_agent']['init_bias'] = -1.1
    agent_cfg['agent']['hybrid_agent']['pre_layer_scale_factor'] = 0.01

    print("  - Hybrid agent parameters configured")