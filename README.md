# Continuous Force RL

This repository contains reinforcement learning implementations for continuous force control in robotic manipulation tasks.

## Classes

### BlockPPO

**Purpose:** A multi-agent Proximal Policy Optimization (PPO) implementation that extends SKRL's PPO agent to support block-parallel processing for multiple agents. This class is designed for factory automation tasks where multiple agents need to be trained simultaneously with shared parameters but independent execution.

**Location:** `agents/block_ppo.py`

#### Public Methods

##### `__init__(models, memory, observation_space, action_space, device, cfg, state_size=-1, track_ckpt_paths=False, task="Isaac-Factory-PegInsert-Local-v0", num_agents=1, num_envs=256, env=None)`

**Purpose:** Initialize the BlockPPO agent with multi-agent support and wrapper integration.

**Parameters:**
- `models` (Mapping[str, Model]): Dictionary containing 'policy' and 'value' models
- `memory` (Optional[Union[Memory, Tuple[Memory]]]): Memory buffer(s) for storing transitions
- `observation_space` (Optional[Union[int, Tuple[int], gymnasium.Space]]): Environment observation space
- `action_space` (Optional[Union[int, Tuple[int], gymnasium.Space]]): Environment action space
- `device` (Optional[Union[str, torch.device]]): Computation device (CPU/GPU)
- `cfg` (Optional[dict]): Configuration dictionary with agent parameters
- `state_size` (int): Size of state representation (-1 for auto-detection)
- `track_ckpt_paths` (bool): Whether to track checkpoint file paths
- `task` (str): Name of the task environment
- `num_agents` (int): Number of agents to train
- `num_envs` (int): Total number of environments
- `env`: Environment instance with wrapper support (required)

**Returns:** None

**Raises:**
- `ValueError`: If env is None or doesn't have required wrapper methods

---

##### `init(trainer_cfg=None)`

**Purpose:** Initialize the agent for training by setting up intervals, creating checkpoint directories, setting up preprocessors, and creating memory tensors.

**Parameters:**
- `trainer_cfg` (Optional[Mapping[str, Any]]): Training configuration dictionary

**Returns:** None

---

##### `write_checkpoint(timestep, timesteps)`

**Purpose:** Save model checkpoints for all agents including policy networks, critic networks, and preprocessor states.

**Parameters:**
- `timestep` (int): Current training timestep
- `timesteps` (int): Total number of training timesteps

**Returns:** None

---

##### `load(path, **kwargs)`

**Purpose:** Load agent checkpoint from file including policy, critic, and preprocessor states.

**Parameters:**
- `path` (str): Path to the checkpoint file
- `**kwargs`: Additional keyword arguments

**Returns:** None

---

##### `record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)`

**Purpose:** Record a single environment transition in memory with optional reward shaping and time-limit bootstrapping.

**Parameters:**
- `states` (torch.Tensor): Current environment states
- `actions` (torch.Tensor): Actions taken by agents
- `rewards` (torch.Tensor): Rewards received
- `next_states` (torch.Tensor): Next environment states
- `terminated` (torch.Tensor): Episode termination flags
- `truncated` (torch.Tensor): Episode truncation flags
- `infos` (Any): Additional environment information
- `timestep` (int): Current timestep
- `timesteps` (int): Total timesteps

**Returns:** None

---

##### `post_interaction(timestep, timesteps)`

**Purpose:** Handle post-interaction logic including policy updates, checkpoint writing, and tracking data logging.

**Parameters:**
- `timestep` (int): Current timestep
- `timesteps` (int): Total timesteps

**Returns:** None

---

##### `add_sample_to_memory(**tensors)`

**Purpose:** Add samples to both primary and secondary memory buffers.

**Parameters:**
- `**tensors` (torch.Tensor): Named tensors to add to memory

**Returns:** None

---

##### `update_nets(loss)`

**Purpose:** Update neural networks using the optimizer with gradient clipping and mixed precision support.

**Parameters:**
- `loss` (torch.Tensor): Loss tensor for backpropagation

**Returns:** None

---

##### `calc_value_loss(sampled_states, sampled_values, sampled_returns, keep_mask, sample_size)`

**Purpose:** Calculate value function loss using either MSE or Huber loss with per-agent masking.

**Parameters:**
- `sampled_states` (torch.Tensor): Sampled state tensors
- `sampled_values` (torch.Tensor): Sampled value predictions
- `sampled_returns` (torch.Tensor): Target return values
- `keep_mask` (torch.Tensor): Boolean mask for agents to include in loss
- `sample_size` (int): Size of the mini-batch sample

**Returns:**
- `value_loss` (torch.Tensor): Scalar loss value
- `value_losses` (torch.Tensor): Per-element losses
- `predicted_values` (torch.Tensor): Current value predictions

---

##### `adaptive_huber_delta(predicted, sampled, k=1.35)`

**Purpose:** Calculate adaptive delta parameter for Huber loss based on median absolute deviation.

**Parameters:**
- `predicted` (torch.Tensor): Predicted values
- `sampled` (torch.Tensor): Target values
- `k` (float): Scaling factor for delta calculation

**Returns:**
- `delta` (float): Adaptive delta parameter

---

#### Private/Protected Methods

##### `_setup_per_agent_preprocessors()`

**Purpose:** Initialize independent preprocessors for each agent using the same configuration.

**Returns:** None

---

##### `_apply_per_agent_preprocessing(tensor_input, preprocessor_list, train=False, inverse=False)`

**Purpose:** Apply per-agent preprocessing to input tensors with support for 2D and 3D inputs.

**Parameters:**
- `tensor_input` (torch.Tensor): Input tensor to preprocess
- `preprocessor_list` (List): List of preprocessors, one per agent
- `train` (bool): Whether to update preprocessor statistics
- `inverse` (bool): Whether to apply inverse transformation

**Returns:**
- `torch.Tensor`: Preprocessed tensor with same shape as input

---

##### `_validate_wrapper_integration()`

**Purpose:** Validate that the environment has the required wrapper system for logging.

**Returns:**
- `bool`: True if wrapper integration is valid

---

##### `_get_logging_wrapper()`

**Purpose:** Get the logging wrapper for metrics reporting.

**Returns:**
- `object`: Wrapper object if found, None otherwise

---

##### `_log_minibatch_update(**kwargs)`

**Purpose:** Log training metrics for each mini-batch update through the wrapper system.

**Parameters:**
- `**kwargs`: Various metric tensors (returns, values, advantages, log_probs, etc.)

**Returns:** None

---

##### `_get_network_state(agent_idx)`

**Purpose:** Extract optimizer and network state information for a specific agent.

**Parameters:**
- `agent_idx` (int): Index of the agent

**Returns:**
- `dict`: Dictionary containing policy and critic network states

---

##### `_update(timestep, timesteps)`

**Purpose:** Main PPO update step including GAE computation, policy loss, value loss, and network updates.

**Parameters:**
- `timestep` (int): Current timestep
- `timesteps` (int): Total timesteps

**Returns:** None

---

### ConfigManager

**Purpose:** A static class that provides comprehensive configuration management for hierarchical YAML-based configuration systems. Handles loading, resolving base inheritance, applying CLI overrides, and computing derived parameters for the training system.

**Location:** `configs/config_manager.py`

#### Public Methods

##### `load_and_resolve_config(config_path, overrides=None)`

**Purpose:** Load and resolve configuration with base inheritance and CLI overrides.

**Parameters:**
- `config_path` (str): Path to the configuration file
- `overrides` (Optional[List[str]]): List of override strings in format "key=value"

**Returns:**
- `Dict[str, Any]`: Fully resolved configuration dictionary

---

##### `apply_to_isaac_lab(env_cfg, agent_cfg, resolved_config)`

**Purpose:** Apply configuration to Isaac Lab environment and agent configuration objects.

**Parameters:**
- `env_cfg`: Environment configuration object
- `agent_cfg`: Agent configuration object
- `resolved_config` (Dict[str, Any]): Resolved configuration dictionary

**Returns:** None

---

#### Private/Protected Methods

##### `_load_yaml_file(file_path)`

**Purpose:** Load YAML configuration file.

**Parameters:**
- `file_path` (str): Path to YAML file

**Returns:**
- `Dict[str, Any]`: Loaded configuration dictionary

---

##### `_resolve_config_path(config_path, current_file)`

**Purpose:** Resolve configuration path relative to configs directory.

**Parameters:**
- `config_path` (str): Configuration path to resolve
- `current_file` (str): Path to current configuration file

**Returns:**
- `str`: Resolved absolute path

---

##### `_merge_configs(base_config, override_config)`

**Purpose:** Merge two configuration dictionaries with deep merging.

**Parameters:**
- `base_config` (Dict): Base configuration dictionary
- `override_config` (Dict): Override configuration dictionary

**Returns:**
- `Dict`: Merged configuration dictionary

---

##### `_apply_override(config, override_str)`

**Purpose:** Apply a CLI override to configuration.

**Parameters:**
- `config` (Dict): Configuration dictionary to modify
- `override_str` (str): Override string in format "key=value"

**Returns:** None

---

##### `_set_nested_key(config, key, value)` / `_set_nested_attr(obj, key, value)`

**Purpose:** Set nested keys in dictionaries or attributes on objects using dot notation.

**Parameters:**
- `config/obj`: Target dictionary or object
- `key` (str): Dot-separated key path
- `value` (Any): Value to set

**Returns:** None

---

##### `_resolve_config(config)`

**Purpose:** Resolve configuration and calculate derived parameters.

**Parameters:**
- `config` (Dict): Configuration dictionary

**Returns:**
- `Dict`: Resolved configuration with derived parameters

---

##### `_calculate_derived_params(primary)`

**Purpose:** Calculate derived parameters from primary configuration.

**Parameters:**
- `primary` (Dict): Primary configuration section

**Returns:**
- `Dict`: Dictionary of derived parameters

---

##### `_resolve_references(config, context=None)`

**Purpose:** Resolve ${section.key} references in configuration.

**Parameters:**
- `config` (Dict): Configuration to process
- `context` (Optional[Dict]): Reference context (defaults to config)

**Returns:**
- `Dict`: Configuration with resolved references

---

##### `_get_nested_value(config, key_path)`

**Purpose:** Get nested value from configuration using dot notation with list indexing support.

**Parameters:**
- `config` (Dict): Configuration dictionary
- `key_path` (str): Dot-separated key path (supports list indices)

**Returns:**
- `Any`: Retrieved value

---

### MetricConfig

**Purpose:** A dataclass that defines configuration for a single tracked metric in the logging system.

**Location:** `configs/config_manager.py`

#### Attributes

- `name` (str): Name of the metric in the info dictionary
- `default_value` (Union[float, int, torch.Tensor]): Default value if metric is not present (default: 0.0)
- `metric_type` (str): Type of metric: 'scalar', 'boolean', 'tensor', 'histogram' (default: "scalar")
- `aggregation` (str): Aggregation method: 'mean', 'sum', 'max', 'min', 'median' (default: "mean")
- `wandb_name` (Optional[str]): Custom name for Wandb logging (default: None)
- `enabled` (bool): Whether to track this metric (default: True)
- `normalize_by_episode_length` (bool): Whether to normalize by episode length (default: False)

---

### LoggingConfig

**Purpose:** A dataclass that provides main configuration for logging wrappers with Wandb integration and metric tracking.

**Location:** `configs/config_manager.py`

#### Attributes

- `wandb_entity` (Optional[str]): Wandb entity name
- `wandb_project` (Optional[str]): Wandb project name
- `wandb_name` (Optional[str]): Wandb run name
- `wandb_group` (Optional[str]): Wandb group name
- `wandb_tags` (Optional[List[str]]): Wandb tags
- `tracked_metrics` (Dict[str, MetricConfig]): Dictionary of tracked metrics
- `track_learning_metrics` (bool): Enable learning metrics tracking (default: True)
- `track_action_metrics` (bool): Enable action metrics tracking (default: True)
- `track_episodes` (bool): Enable episode tracking (default: True)
- `track_rewards` (bool): Enable reward tracking (default: True)
- `track_episode_length` (bool): Enable episode length tracking (default: True)
- `track_terminations` (bool): Enable termination tracking (default: True)
- `clip_eps` (float): PPO clipping epsilon (default: 0.2)
- `num_agents` (int): Number of agents (default: 1)

#### Public Methods

##### `add_metric(metric)`

**Purpose:** Add a metric to the tracking configuration.

**Parameters:**
- `metric` (MetricConfig): Metric configuration to add

**Returns:** None

---

##### `remove_metric(metric_name)`

**Purpose:** Remove a metric from tracking.

**Parameters:**
- `metric_name` (str): Name of metric to remove

**Returns:** None

---

##### `enable_metric(metric_name, enabled=True)`

**Purpose:** Enable or disable a specific metric.

**Parameters:**
- `metric_name` (str): Name of metric to enable/disable
- `enabled` (bool): Whether to enable the metric (default: True)

**Returns:** None

---

##### `to_wandb_config()`

**Purpose:** Convert to Wandb configuration dictionary.

**Returns:**
- `Dict[str, Any]`: Wandb-compatible configuration dictionary

---

### LoggingConfigPresets

**Purpose:** A static class providing predefined logging configurations for common use cases. Marked as deprecated in favor of simplified wrapper approach.

**Location:** `configs/config_manager.py`

#### Public Methods

##### `basic_config()`

**Purpose:** Create basic configuration with minimal tracking.

**Returns:**
- `LoggingConfig`: Basic logging configuration

---

##### `factory_config()`

**Purpose:** Create configuration optimized for factory manipulation tasks.

**Returns:**
- `LoggingConfig`: Factory-optimized logging configuration

---

##### `from_dict(config_dict)`

**Purpose:** Create logging config from dictionary.

**Parameters:**
- `config_dict` (Dict[str, Any]): Configuration dictionary

**Returns:**
- `LoggingConfig`: Created logging configuration

---

##### `create_from_config(config_dict, preset="factory")`

**Purpose:** Create logging config from configuration with preset base and overrides.

**Parameters:**
- `config_dict` (Dict[str, Any]): Override configuration dictionary
- `preset` (str): Preset name ('factory', 'basic', or other) (default: "factory")

**Returns:**
- `LoggingConfig`: Merged logging configuration

---

### load_config_from_file

**Purpose:** Standalone function to load logging configuration from YAML or JSON files.

**Location:** `configs/config_manager.py`

#### Parameters

- `file_path` (str): Path to configuration file (.yaml, .yml, or .json)

#### Returns

- `LoggingConfig`: Loaded logging configuration

#### Raises

- `FileNotFoundError`: If configuration file doesn't exist
- `ImportError`: If PyYAML is required but not available
- `ValueError`: If file format is unsupported

---

## Key Features

- **Multi-Agent Support**: Trains multiple agents simultaneously with shared parameters
- **Block-Parallel Processing**: Efficient parallel computation for multiple agents
- **Wrapper Integration**: Compatible with logging and metrics collection wrappers
- **Per-Agent Preprocessing**: Independent preprocessing for each agent
- **Flexible Memory**: Support for multiple memory buffers
- **Comprehensive Logging**: Detailed metrics tracking and checkpoint management
- **Adaptive Huber Loss**: Dynamic loss scaling for improved training stability
- **Hierarchical Configuration**: YAML-based configuration system with inheritance and overrides
- **Reference Resolution**: Support for ${section.key} references with list indexing
- **Derived Parameters**: Automatic calculation of training parameters from primary config
- **CLI Override Support**: Command-line configuration overrides with type inference

---

### HybridForcePositionWrapper

**Purpose:** A gymnasium wrapper that implements hybrid force-position control for factory environments. This wrapper enables robots to switch between position control and force control on different axes based on a selection matrix, allowing for complex manipulation tasks requiring both precise positioning and force regulation.

**Location:** `wrappers/control/hybrid_force_position_wrapper.py`

#### Public Methods

##### `__init__(env, ctrl_torque=False, reward_type="simp", ctrl_cfg=None, task_cfg=None)`

**Purpose:** Initialize hybrid force-position control wrapper with configuration validation and action space management.

**Parameters:**
- `env`: Base environment to wrap (must have force-torque sensor data)
- `ctrl_torque` (bool): Whether to control torques (6DOF) or just forces (3DOF) (default: False)
- `reward_type` (str): Reward computation strategy: "simp", "dirs", "delta", "base", "pos_simp", "wrench_norm" (default: "simp")
- `ctrl_cfg` (HybridCtrlCfg): Control configuration with EMA parameters and thresholds (required)
- `task_cfg` (HybridTaskCfg): Task configuration with reward parameters and thresholds (required)

**Returns:** None

**Raises:**
- `ValueError`: If ctrl_cfg or task_cfg is None, or if required configurations are missing
- `ImportError`: If Isaac Sim torch utilities are not available

---

##### `step(action)`

**Purpose:** Step environment with hybrid control and ensure wrapper initialization.

**Parameters:**
- `action` (torch.Tensor): Action tensor with selection matrix + pose actions + force actions

**Returns:**
- `obs` (torch.Tensor): Environment observations
- `reward` (torch.Tensor): Rewards including hybrid control components
- `terminated` (torch.Tensor): Episode termination flags
- `truncated` (torch.Tensor): Episode truncation flags
- `info` (Dict): Additional environment information

---

##### `reset(**kwargs)`

**Purpose:** Reset environment and ensure wrapper initialization.

**Parameters:**
- `**kwargs`: Additional reset arguments

**Returns:**
- `obs` (torch.Tensor): Initial observations
- `info` (Dict): Reset information

---

#### Private/Protected Methods

##### `_initialize_wrapper()`

**Purpose:** Initialize wrapper by overriding environment methods and validating dependencies.

**Returns:** None

**Raises:**
- `ValueError`: If force-torque sensor data is not available

---

##### `_extract_goals_from_action(action)`

**Purpose:** Extract pose, force, and selection goals from action tensor.

**Parameters:**
- `action` (torch.Tensor): Combined action tensor

**Returns:** None

---

##### `_calc_pose_goal()`

**Purpose:** Calculate pose goal from action with position bounds enforcement and quaternion conversion.

**Returns:** None

**Raises:**
- `ValueError`: If position action bounds are not configured

---

##### `_calc_force_goal()`

**Purpose:** Calculate force goal from action with bounds checking and torque handling.

**Returns:** None

**Raises:**
- `ValueError`: If force/torque action thresholds or bounds are not configured

---

##### `_update_targets_with_ema()`

**Purpose:** Update targets using EMA filtering of goals with configurable selection matrix filtering.

**Returns:** None

---

##### `_set_control_targets_from_targets()`

**Purpose:** Set environment control targets from filtered targets with threshold application.

**Returns:** None

---

##### `_reset_targets(env_ids)`

**Purpose:** Reset targets for specific environment IDs based on initialization mode.

**Parameters:**
- `env_ids` (torch.Tensor): Environment indices to reset

**Returns:** None

---

##### `_get_target_out_of_bounds()`

**Purpose:** Check if fingertip target is out of position bounds for safety.

**Returns:**
- `torch.Tensor`: Boolean tensor indicating out-of-bounds status per environment and axis

---

##### `_wrapped_pre_physics_step(action)`

**Purpose:** Process actions through goal→target→control flow with reset handling.

**Parameters:**
- `action` (torch.Tensor): Action tensor

**Returns:** None

---

##### `_wrapped_apply_action()`

**Purpose:** Apply hybrid force-position control using filtered targets with wrench computation and safety checks.

**Returns:** None

---

##### `_wrapped_update_rew_buf(curr_successes)`

**Purpose:** Update reward buffer with hybrid control rewards based on reward type.

**Parameters:**
- `curr_successes` (torch.Tensor): Current success flags

**Returns:**
- `torch.Tensor`: Updated reward buffer

---

##### `_simple_force_reward()` / `_directional_force_reward()` / `_delta_selection_reward()` / `_position_simple_reward()` / `_low_wrench_reward()`

**Purpose:** Various reward computation strategies for hybrid control evaluation.

**Returns:**
- `torch.Tensor`: Reward contributions based on force activity and selection matrix usage

---

### HybridCtrlCfg

**Purpose:** Configuration dataclass for hybrid force-position control parameters including EMA filtering, target initialization, and action bounds.

**Location:** `wrappers/control/hybrid_control_cfg.py`

#### Attributes

- `ema_factor` (float): Factor for EMA filtering of targets (default: 0.2)
- `no_sel_ema` (bool): If True, selection matrix is not filtered with EMA (default: True)
- `target_init_mode` (str): Target initialization strategy: "zero" or "first_goal" (default: "zero")
- `default_task_force_gains` (List[float]): Force task gains [fx, fy, fz, tx, ty, tz] (default: None)
- `force_action_bounds` (List[float]): Force action bounds [fx, fy, fz] (default: None)
- `torque_action_bounds` (List[float]): Torque action bounds [tx, ty, tz] (default: None)
- `force_action_threshold` (List[float]): Force action scaling threshold (default: None)
- `torque_action_threshold` (List[float]): Torque action scaling threshold (default: None)
- `pos_action_bounds` (List[float]): Position action bounds [x, y, z] (default: None)
- `rot_action_bounds` (List[float]): Rotation action bounds [rx, ry, rz] (default: None)

#### Public Methods

##### `__post_init__()`

**Purpose:** Validate configuration parameters after initialization.

**Returns:** None

**Raises:**
- `ValueError`: If ema_factor is not between 0 and 1, or target_init_mode is invalid

---

### HybridTaskCfg

**Purpose:** Configuration dataclass for hybrid control task-specific parameters including reward structure and force activity thresholds.

**Location:** `wrappers/control/hybrid_control_cfg.py`

#### Attributes

- `force_active_threshold` (float): Minimum force magnitude for active force detection (default: 0.1)
- `torque_active_threshold` (float): Minimum torque magnitude for active torque detection (default: 0.01)
- `good_force_cmd_rew` (float): Reward for using force control when force is active (default: 0.1)
- `bad_force_cmd_rew` (float): Penalty for using force control when no force is active (default: -0.1)
- `wrench_norm_scale` (float): Scale factor for wrench norm penalty (default: 0.01)

---

### Factory Control Utilities

**Purpose:** Control functions extracted from factory environments for operational space control and hybrid force-position control computations.

**Location:** `wrappers/control/factory_control_utils.py`

#### Public Functions

##### `compute_pose_task_wrench(cfg, dof_pos, fingertip_midpoint_pos, fingertip_midpoint_quat, fingertip_midpoint_linvel, fingertip_midpoint_angvel, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, task_prop_gains, task_deriv_gains, device)`

**Purpose:** Compute task-space wrench for pose control using PD control law.

**Parameters:**
- `cfg`: Environment configuration
- `dof_pos` (torch.Tensor): Joint positions
- `fingertip_midpoint_pos` (torch.Tensor): Current fingertip position
- `fingertip_midpoint_quat` (torch.Tensor): Current fingertip quaternion
- `fingertip_midpoint_linvel` (torch.Tensor): Current linear velocity
- `fingertip_midpoint_angvel` (torch.Tensor): Current angular velocity
- `ctrl_target_fingertip_midpoint_pos` (torch.Tensor): Target fingertip position
- `ctrl_target_fingertip_midpoint_quat` (torch.Tensor): Target fingertip quaternion
- `task_prop_gains` (torch.Tensor): Proportional gains
- `task_deriv_gains` (torch.Tensor): Derivative gains
- `device` (torch.device): Computation device

**Returns:**
- `torch.Tensor`: Task-space wrench for pose control

---

##### `compute_force_task_wrench(cfg, dof_pos, eef_force, ctrl_target_force, task_gains, device)`

**Purpose:** Compute task-space wrench for force control using proportional control.

**Parameters:**
- `cfg`: Environment configuration
- `dof_pos` (torch.Tensor): Joint positions
- `eef_force` (torch.Tensor): Current end-effector force
- `ctrl_target_force` (torch.Tensor): Target force
- `task_gains` (torch.Tensor): Force control gains
- `device` (torch.device): Computation device

**Returns:**
- `torch.Tensor`: Task-space wrench for force control

---

##### `compute_dof_torque_from_wrench(cfg, dof_pos, dof_vel, task_wrench, jacobian, arm_mass_matrix, device)`

**Purpose:** Compute joint torques for given task wrench with null space compensation.

**Parameters:**
- `cfg`: Environment configuration
- `dof_pos` (torch.Tensor): Joint positions
- `dof_vel` (torch.Tensor): Joint velocities
- `task_wrench` (torch.Tensor): Desired task-space wrench
- `jacobian` (torch.Tensor): End-effector Jacobian
- `arm_mass_matrix` (torch.Tensor): Arm mass matrix
- `device` (torch.device): Computation device

**Returns:**
- `dof_torque` (torch.Tensor): Joint torques (clamped to [-100, 100])
- `task_wrench` (torch.Tensor): Resulting task wrench

---

##### `get_pose_error(fingertip_midpoint_pos, fingertip_midpoint_quat, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, jacobian_type, rot_error_type)`

**Purpose:** Compute task-space error between target and current fingertip pose.

**Parameters:**
- `fingertip_midpoint_pos` (torch.Tensor): Current position
- `fingertip_midpoint_quat` (torch.Tensor): Current quaternion
- `ctrl_target_fingertip_midpoint_pos` (torch.Tensor): Target position
- `ctrl_target_fingertip_midpoint_quat` (torch.Tensor): Target quaternion
- `jacobian_type` (str): Jacobian type ("geometric")
- `rot_error_type` (str): Rotation error type ("axis_angle" or "quat")

**Returns:**
- `pos_error` (torch.Tensor): Position error
- `rot_error` (torch.Tensor): Rotation error (axis-angle or quaternion)

**Raises:**
- `ImportError`: If required torch utilities or math functions are not available

---

## Key Features (Updated)

- **Multi-Agent Support**: Trains multiple agents simultaneously with shared parameters
- **Block-Parallel Processing**: Efficient parallel computation for multiple agents
- **Wrapper Integration**: Compatible with logging and metrics collection wrappers
- **Per-Agent Preprocessing**: Independent preprocessing for each agent
- **Flexible Memory**: Support for multiple memory buffers
- **Comprehensive Logging**: Detailed metrics tracking and checkpoint management
- **Adaptive Huber Loss**: Dynamic loss scaling for improved training stability
- **Hierarchical Configuration**: YAML-based configuration system with inheritance and overrides
- **Reference Resolution**: Support for ${section.key} references with list indexing
- **Derived Parameters**: Automatic calculation of training parameters from primary config
- **CLI Override Support**: Command-line configuration overrides with type inference
- **Hybrid Force-Position Control**: Selection matrix-based switching between force and position control
- **EMA Filtering**: Exponential moving average filtering for smooth target transitions
- **Safety Bounds Checking**: Position and force bounds enforcement for safe operation
- **Multiple Reward Strategies**: Configurable reward computation for hybrid control evaluation
- **Operational Space Control**: Task-space control with null space compensation