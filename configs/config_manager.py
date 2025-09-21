"""
Configuration Manager for hierarchical configuration system.

Handles loading, resolving, and applying configurations for the factory training system.
Includes integrated logging configuration management.
"""

import yaml
import os
from typing import Dict, Any, List, Optional, Union
import copy
from dataclasses import dataclass, field
import torch

# Import for individual agent logging setup
def _import_setup_individual_agent_logging():
    """Lazy import to avoid circular imports."""
    from learning.launch_utils_v2 import _setup_individual_agent_logging
    return _setup_individual_agent_logging


class ConfigManager:
    """Manages configuration loading and resolution."""

    # Class-level source tracking
    _config_sources = {}

    @staticmethod
    def load_and_resolve_config(config_path: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load and resolve configuration with base inheritance and CLI overrides.

        Args:
            config_path: Path to the configuration file
            overrides: List of override strings in format "key=value"

        Returns:
            Fully resolved configuration dictionary
        """
        # Clear previous source tracking
        ConfigManager._config_sources = {}

        # Load main config
        config = ConfigManager._load_yaml_file(config_path)

        # Load base configuration if specified
        if 'base' in config:
            base_path = ConfigManager._resolve_config_path(config['base'], config_path)
            base_config = ConfigManager._load_yaml_file(base_path)

            # Track base config sources
            ConfigManager._track_config_sources(base_config, 'base')

            # Track local override sources (only for keys that override base)
            ConfigManager._track_config_sources(config, 'local_override', base_config)

            config = ConfigManager._merge_configs(base_config, config)
        else:
            # No base config, all current values are base values
            ConfigManager._track_config_sources(config, 'base')

        # Apply CLI overrides
        if overrides:
            for override in overrides:
                key, _ = override.split('=', 1)
                ConfigManager._config_sources[key] = 'cli_override'
                ConfigManager._apply_override(config, override)

        # Resolve configuration
        resolved_config = ConfigManager._resolve_config(config)

        return resolved_config

    @staticmethod
    def apply_complete_configuration(env_cfg, agent_cfg, resolved_config):
        """
        Apply complete configuration to Isaac Lab environment and agent configs.

        This is the single point of configuration that handles ALL configuration
        aspects in Step 1. No other steps should modify configuration objects.

        Args:
            env_cfg: Isaac Lab environment configuration object
            agent_cfg: Agent configuration dictionary
            resolved_config: Fully resolved configuration from ConfigManager
        """
        print("[CONFIG]: Starting complete configuration application")

        # Extract configuration sections
        environment = resolved_config.get('environment', {})
        primary = resolved_config.get('primary', {})
        derived = resolved_config.get('derived', {})
        model = resolved_config.get('model', {})
        wrappers_config = resolved_config.get('wrappers', {})
        experiment = resolved_config.get('experiment', {})

        # 1. Apply basic Isaac Lab configuration (task, ctrl, and other basic settings)
        ConfigManager._apply_isaac_lab_basic_config(env_cfg, environment)

        # 2. Apply device configuration
        ConfigManager._apply_device_config(env_cfg)

        # 3. Apply break forces and agent distribution
        ConfigManager._apply_break_force_config(env_cfg, agent_cfg, primary)

        # 4. Apply easy mode if enabled
        if primary.get('debug_mode', False):
            ConfigManager._apply_easy_mode_config(env_cfg, agent_cfg)

        # 5. Apply environment scene configuration
        ConfigManager._apply_scene_config(env_cfg, primary, derived)

        # 6. Apply force sensor configuration if enabled
        if wrappers_config.get('force_torque_sensor', {}).get('enabled', False):
            ConfigManager._apply_force_sensor_config(env_cfg)

        # 7. Apply model configuration
        ConfigManager._apply_model_config(agent_cfg, model)

        # 8. Apply experiment and logging configuration
        ConfigManager._apply_experiment_config(env_cfg, agent_cfg, resolved_config)

        print("[CONFIG]: Complete configuration application finished")

    @staticmethod
    def apply_to_isaac_lab(env_cfg, agent_cfg, resolved_config):
        """
        DEPRECATED: Use apply_complete_configuration instead.

        This method is kept for backward compatibility but should not be used
        in new code. Use apply_complete_configuration for the clean architecture.
        """
        print("[CONFIG]: WARNING - Using deprecated apply_to_isaac_lab. Use apply_complete_configuration instead.")
        ConfigManager._apply_isaac_lab_basic_config(env_cfg, resolved_config.get('environment', {}))

    @staticmethod
    def _apply_isaac_lab_basic_config(env_cfg, environment):
        """Apply basic Isaac Lab configuration (task, ctrl, and other settings)."""
        # Isaac Lab attribute name mapping
        isaac_lab_attr_mapping = {
            'task': 'task',  # Our 'task' config maps to Isaac Lab's 'task' attribute
            'ctrl': 'ctrl'   # Our 'ctrl' config maps to Isaac Lab's 'ctrl' attribute
        }

        # Validate critical Isaac Lab configuration objects before applying overrides
        ConfigManager._validate_isaac_lab_config_objects(env_cfg, environment, isaac_lab_attr_mapping)

        # Apply environment overrides
        for key, value in environment.items():
            # Handle critical Isaac Lab configuration objects specially
            if key in ['task', 'ctrl'] and isinstance(value, dict):
                # Map to correct Isaac Lab attribute name
                isaac_lab_attr = isaac_lab_attr_mapping.get(key, key)
                existing_config = getattr(env_cfg, isaac_lab_attr, None)

                if existing_config is not None:
                    print(f"[CONFIG]: Merging {key} parameters into existing Isaac Lab {isaac_lab_attr} object")
                    for prop_key, prop_value in value.items():
                        if hasattr(existing_config, prop_key):
                            old_value = getattr(existing_config, prop_key)
                            print(f"[CONFIG]:   Updating {isaac_lab_attr}.{prop_key}: {old_value} -> {prop_value}")
                            setattr(existing_config, prop_key, prop_value)
                        else:
                            print(f"[CONFIG]:   WARNING: {isaac_lab_attr}.{prop_key} not found in ExtendedCtrlCfg - skipping")
                            # With ExtendedCtrlCfg, all attributes should exist, so this is unexpected
                else:
                    print(f"[CONFIG]: WARNING - Isaac Lab {isaac_lab_attr} object not found, cannot merge {key} parameters")
                continue

            # Handle other environment configuration
            existing_value = getattr(env_cfg, key, "NOT_SET") if hasattr(env_cfg, key) else "NOT_SET"
            print(f"[CONFIG]: Applying {key}: {existing_value} -> {value}")

            if '.' in key:
                ConfigManager._set_nested_attr(env_cfg, key, value)
            else:
                # Handle dictionary configurations that need to be merged with existing Isaac Lab configs
                if isinstance(value, dict):
                    existing_config = getattr(env_cfg, key, None)
                    if existing_config is not None:
                        # Check if existing config is an object (with attributes) or a dictionary
                        if isinstance(existing_config, dict):
                            # Existing config is a dictionary, merge directly
                            print(f"[CONFIG]: Merging {key} properties into existing dictionary configuration")
                            for prop_key, prop_value in value.items():
                                print(f"[CONFIG]:   Setting {key}[{prop_key}] = {prop_value}")
                                existing_config[prop_key] = prop_value
                        else:
                            # Existing config is an object, merge using setattr
                            print(f"[CONFIG]: Merging {key} properties into existing Isaac Lab configuration object")
                            for prop_key, prop_value in value.items():
                                print(f"[CONFIG]:   Setting {key}.{prop_key} = {prop_value}")
                                setattr(existing_config, prop_key, prop_value)
                    else:
                        # Special handling for configurations that need to remain as dictionaries
                        # for dynamic access (component_attr_map for Isaac Lab attribute mapping)
                        if key in ['component_attr_map']:
                            print(f"[CONFIG]: Creating new {key} dictionary configuration")
                            setattr(env_cfg, key, value.copy())
                        else:
                            # No existing config, create a new object from our dictionary
                            print(f"[CONFIG]: Creating new {key} configuration object")
                            config_obj = type(f'{key.title()}Config', (), value)()
                            setattr(env_cfg, key, config_obj)
                else:
                    # Set attribute directly (whether it exists or not)
                    setattr(env_cfg, key, value)

    @staticmethod
    def _validate_isaac_lab_config_objects(env_cfg, environment_overrides, isaac_lab_attr_mapping):
        """
        Validate that Isaac Lab environment has required configuration objects.

        Ensures that critical configuration objects (task, ctrl) exist and are proper
        Isaac Lab objects (not dictionaries) before we attempt to merge custom
        configurations. This prevents accidentally creating new objects that are
        missing Isaac Lab's required attributes.

        Args:
            env_cfg: Isaac Lab environment configuration object
            environment_overrides: Dictionary of environment configuration overrides
            isaac_lab_attr_mapping: Mapping from config key to Isaac Lab attribute name

        Raises:
            ValueError: If required configuration objects are missing or invalid
        """
        # List of critical configuration keys that should be objects, not dicts
        critical_config_keys = ['task', 'ctrl']

        # Only validate keys that we're trying to override
        keys_to_validate = [key for key in critical_config_keys
                           if key in environment_overrides and isinstance(environment_overrides[key], dict)]

        for key in keys_to_validate:
            # Map to correct Isaac Lab attribute name
            isaac_lab_attr = isaac_lab_attr_mapping.get(key, key)

            if not hasattr(env_cfg, isaac_lab_attr):
                raise ValueError(
                    f"Isaac Lab environment configuration is missing required '{isaac_lab_attr}' attribute "
                    f"(mapped from config key '{key}'). This suggests the environment was not properly "
                    f"initialized with Isaac Lab defaults. Cannot safely merge custom configuration."
                )

            existing_obj = getattr(env_cfg, isaac_lab_attr)

            # Check that it's not a plain dictionary (should be a proper config object)
            if isinstance(existing_obj, dict):
                raise ValueError(
                    f"Isaac Lab environment '{isaac_lab_attr}' configuration is a dictionary instead of a proper "
                    f"configuration object. This suggests Isaac Lab was not properly loaded. "
                    f"Expected a configclass or similar object with attributes, got: {type(existing_obj)}"
                )

            # Additional check: ensure it has some expected Isaac Lab attributes
            if key == 'task' and not (hasattr(existing_obj, 'duration_s') or hasattr(existing_obj, 'fixed_asset') or hasattr(existing_obj, 'held_asset')):
                raise ValueError(
                    f"Isaac Lab task configuration object does not have expected Isaac Lab attributes "
                    f"(duration_s, fixed_asset, held_asset). This suggests the task object is not a "
                    f"proper Isaac Lab configuration. Got object: {existing_obj}"
                )

            if key == 'ctrl' and not (hasattr(existing_obj, 'default_task_prop_gains') or hasattr(existing_obj, 'default_task_force_gains')):
                raise ValueError(
                    f"Isaac Lab ctrl configuration object does not have expected Isaac Lab attributes "
                    f"(default_task_prop_gains, default_task_force_gains). This suggests the ctrl object is not a "
                    f"proper Isaac Lab configuration. Got object: {existing_obj}"
                )

        print(f"[CONFIG]: Validation passed - Isaac Lab configuration objects are properly initialized")

    @staticmethod
    def _apply_device_config(env_cfg):
        """Apply device configuration."""
        print("[CONFIG]: Applying device configuration")
        import torch

        if hasattr(env_cfg, 'sim') and hasattr(env_cfg.sim, 'device'):
            if env_cfg.sim.device is None:
                env_cfg.sim.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                print(f"[CONFIG]:   Set sim.device = {env_cfg.sim.device}")
        else:
            # If sim config doesn't exist, create it
            env_cfg.sim = type('SimConfig', (), {'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'})()
            print(f"[CONFIG]:   Created sim config with device = {env_cfg.sim.device}")

    @staticmethod
    def _apply_break_force_config(env_cfg, agent_cfg, primary):
        """Apply break force and agent distribution configuration."""
        print("[CONFIG]: Applying break force configuration")

        break_forces = primary['break_forces']
        env_cfg.break_force = break_forces
        agent_cfg['agent']['break_force'] = break_forces

        print(f"[CONFIG]:   Set break_force = {break_forces}")

    @staticmethod
    def _apply_easy_mode_config(env_cfg, agent_cfg):
        """Apply easy mode configuration."""
        print("[CONFIG]: Applying easy mode configuration")

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

        print("[CONFIG]:   Easy mode environment settings applied")

    @staticmethod
    def _apply_scene_config(env_cfg, primary, derived):
        """Apply environment scene configuration."""
        print("[CONFIG]: Applying environment scene configuration")

        env_cfg.scene.num_envs = derived['total_num_envs']
        env_cfg.scene.replicate_physics = True
        env_cfg.num_agents = derived['total_agents']

        print(f"[CONFIG]:   Scene configured: {derived['total_num_envs']} envs, {derived['total_agents']} agents")

    @staticmethod
    def _apply_force_sensor_config(env_cfg):
        """Apply force sensor configuration."""
        print("[CONFIG]: Applying force sensor configuration")

        # Add force-torque to observation and state orders if not already present
        if hasattr(env_cfg, 'obs_order') and 'force_torque' not in env_cfg.obs_order:
            env_cfg.obs_order.append('force_torque')
            print("[CONFIG]:   Added 'force_torque' to obs_order")

        if hasattr(env_cfg, 'state_order') and 'force_torque' not in env_cfg.state_order:
            env_cfg.state_order.append('force_torque')
            print("[CONFIG]:   Added 'force_torque' to state_order")

        print("[CONFIG]:   Force-torque sensor enabled in observation and state")

    @staticmethod
    def _apply_model_config(agent_cfg, model_config):
        """Apply model configuration."""
        print("[CONFIG]: Applying model configuration")

        # Ensure models section exists
        if 'models' not in agent_cfg:
            agent_cfg['models'] = {'policy': {}, 'value': {}}

        # Apply model configuration
        for key, value in model_config.items():
            if key in ['actor', 'policy']:
                # Policy model configuration
                for param_key, param_value in value.items():
                    agent_cfg['models']['policy'][param_key] = param_value
                    print(f"[CONFIG]:   Set policy param: {param_key} = {param_value}")
            elif key in ['critic', 'value']:
                # Value model configuration
                for param_key, param_value in value.items():
                    agent_cfg['models']['value'][param_key] = param_value
                    print(f"[CONFIG]:   Set value param: {param_key} = {param_value}")
            else:
                # General model configuration
                agent_cfg['models'][key] = value
                print(f"[CONFIG]:   Set general model param: {key} = {value}")

    @staticmethod
    def _apply_experiment_config(env_cfg, agent_cfg, resolved_config):
        """Apply experiment and logging configuration."""
        print("[CONFIG]: Applying experiment and logging configuration")

        experiment = resolved_config.get('experiment', {})
        derived = resolved_config.get('derived', {})

        # Ensure experiment section exists
        if 'experiment' not in agent_cfg['agent']:
            agent_cfg['agent']['experiment'] = {}

        # Apply experiment configuration
        agent_cfg['agent']['experiment']['project'] = experiment.get('wandb_project', 'Continuous_Force_RL')
        agent_cfg['agent']['experiment']['tags'] = experiment.get('tags', [])
        agent_cfg['agent']['experiment']['group'] = experiment.get('group', '')
        agent_cfg['agent']['experiment']['name'] = experiment.get('name', 'unnamed')

        # Set agent-specific data
        agent_cfg['agent']['break_force'] = resolved_config.get('primary', {}).get('break_forces', -1)
        agent_cfg['agent']['num_envs'] = derived.get('total_num_envs', 128)

        # Add task tags
        task_name = env_cfg.task_name if hasattr(env_cfg, 'task_name') else 'factory'
        agent_cfg['agent']['experiment']['tags'].append(task_name)

        # Set up individual agent logging configurations
        setup_individual_agent_logging = _import_setup_individual_agent_logging()
        setup_individual_agent_logging(agent_cfg, resolved_config)

        print(f"[CONFIG]:   Set up logging for {derived.get('total_agents', 1)} agents with unique paths and wandb runs")
        print(f"[CONFIG]:   Experiment '{experiment.get('name', 'unnamed')}' configured for {derived.get('total_agents', 1)} agents")

    @staticmethod
    def _load_yaml_file(file_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _resolve_config_path(config_path: str, current_file: str) -> str:
        """Resolve configuration path relative to configs directory."""
        if os.path.isabs(config_path):
            return config_path

        # Get configs directory (parent of current file's directory)
        configs_dir = os.path.dirname(os.path.dirname(current_file))
        if not config_path.endswith('.yaml'):
            config_path += '.yaml'

        return os.path.join(configs_dir, config_path)

    @staticmethod
    def _merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """Merge two configuration dictionaries."""
        merged = copy.deepcopy(base_config)

        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    @staticmethod
    def _apply_override(config: Dict, override_str: str):
        """Apply a CLI override to configuration."""
        if '=' not in override_str:
            raise ValueError(f"Invalid override format: {override_str}. Expected key=value")

        key, value = override_str.split('=', 1)

        # Parse value
        try:
            # Try to parse as Python literal (for lists, numbers, booleans)
            import ast
            value = ast.literal_eval(value)
        except:
            # Handle common string representations
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'none':
                value = None
            # Otherwise keep as string if literal_eval fails

        # Set nested key
        ConfigManager._set_nested_key(config, key, value)

    @staticmethod
    def _set_nested_key(config: Dict, key: str, value: Any):
        """Set nested key in configuration dictionary."""
        keys = key.split('.')
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    @staticmethod
    def _set_nested_attr(obj, key: str, value: Any):
        """Set nested attribute using dot notation."""
        keys = key.split('.')
        current = obj

        for k in keys[:-1]:
            current = getattr(current, k)

        setattr(current, keys[-1], value)

    @staticmethod
    def _resolve_config(config: Dict) -> Dict:
        """Resolve configuration and calculate derived parameters."""
        resolved = copy.deepcopy(config)

        # Calculate derived parameters
        derived = ConfigManager._calculate_derived_params(resolved.get('primary', {}))
        resolved['derived'] = derived

        # Resolve parameter references
        resolved = ConfigManager._resolve_references(resolved)

        return resolved

    @staticmethod
    def _calculate_derived_params(primary: Dict) -> Dict:
        """Calculate derived parameters from primary configuration."""
        derived = {}

        # Basic calculations using new structure
        agents_per_break_force = primary.get('agents_per_break_force', 2)
        num_envs_per_agent = primary.get('num_envs_per_agent', 256)
        break_forces = primary.get('break_forces', -1)

        # Calculate total agents and environments
        # total_agents = number_of_break_forces * agents_per_break_force
        # total_envs = total_agents * num_envs_per_agent
        num_break_forces = len(break_forces) if isinstance(break_forces, list) else 1
        derived['total_agents'] = num_break_forces * agents_per_break_force
        derived['total_num_envs'] = derived['total_agents'] * num_envs_per_agent

        # Calculate rollout steps from timing parameters
        episode_length_s = primary.get('episode_length_s', 12.0)
        decimation = primary.get('decimation', 4)
        policy_hz = primary.get('policy_hz', 15)
        sim_dt = (1/policy_hz) / decimation
        derived['rollout_steps'] = int((1/sim_dt) / decimation * episode_length_s)

        # Training parameters
        max_steps = primary.get('max_steps', 10240000)
        derived['max_steps'] = max_steps

        # Other derived values
        derived['sim_dt'] = sim_dt
        derived['ckpt_tracker_path'] = primary.get('ckpt_tracker_path', '/nfs/stak/users/brownhun/ckpt_tracker2.txt')
        derived['seed'] = primary.get('seed', -1)

        return derived

    @staticmethod
    def _resolve_references(config: Dict, context: Optional[Dict] = None) -> Dict:
        """Resolve ${section.key} references in configuration."""
        if context is None:
            context = config

        if isinstance(config, dict):
            resolved = {}
            for key, value in config.items():
                resolved[key] = ConfigManager._resolve_references(value, context)
            return resolved
        elif isinstance(config, list):
            return [ConfigManager._resolve_references(item, context) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            # Resolve reference
            ref_path = config[2:-1]  # Remove ${ and }
            return ConfigManager._get_nested_value(context, ref_path)
        else:
            return config

    @staticmethod
    def _track_config_sources(config: Dict, source_type: str, base_config: Optional[Dict] = None, path_prefix: str = ""):
        """
        Recursively track configuration sources for all keys.

        Args:
            config: Configuration dictionary to track
            source_type: Type of source ('base', 'local_override', 'cli_override')
            base_config: Base configuration to compare against (for determining actual overrides)
            path_prefix: Current path prefix for nested keys
        """
        for key, value in config.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key

            # For local overrides, only mark as override if it actually overrides base
            if source_type == 'local_override' and base_config is not None:
                # Skip 'base' key itself and section-level keys if they're dictionaries
                if key == 'base':
                    continue  # Don't track the base reference itself

                # For dictionary values, we'll track their children, not the section itself
                if isinstance(value, dict):
                    # Don't mark the section as override, let children determine their own status
                    pass
                else:
                    # For non-dict values, check if they actually override base
                    base_value = ConfigManager._get_nested_value_safe(base_config, current_path)
                    if base_value is None:
                        # This is a true override - new key not in base
                        ConfigManager._config_sources[current_path] = source_type
                    elif base_value != value:
                        # This is a true override - different value from base
                        ConfigManager._config_sources[current_path] = source_type
                    # If same as base, don't track (keep base marking)
            else:
                # Track this key's source (base or cli_override)
                ConfigManager._config_sources[current_path] = source_type

            # Recursively track nested dictionaries
            if isinstance(value, dict):
                base_nested = ConfigManager._get_nested_value_safe(base_config, current_path) if base_config else None
                ConfigManager._track_config_sources(value, source_type, base_nested, current_path)

    @staticmethod
    def _get_nested_value(config: Dict, key_path: str) -> Any:
        """Get nested value from configuration using dot notation."""
        keys = key_path.split('.')
        current = config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and key.isdigit():
                # Handle list indexing
                index = int(key)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    raise ValueError(f"Reference not found: {key_path} (list index {index} out of range)")
            else:
                raise ValueError(f"Reference not found: {key_path}")

        return current

    @staticmethod
    def _get_nested_value_safe(config: Dict, key_path: str) -> Any:
        """Get nested value from configuration using dot notation, return None if not found."""
        if not config or not key_path:
            return None

        try:
            return ConfigManager._get_nested_value(config, key_path)
        except (ValueError, KeyError, TypeError):
            return None

    @staticmethod
    def print_env_config(env_cfg: Any) -> None:
        """Print environment configuration with color-coded source tracking."""
        from learning.config_printer import print_config
        ConfigManager._print_color_legend()
        print_config("Environment Configuration", env_cfg, color_map=ConfigManager._config_sources, path_prefix="environment")

    @staticmethod
    def print_agent_config(agent_cfg: Any) -> None:
        """Print agent configuration with color-coded source tracking."""
        from learning.config_printer import print_config
        print_config("Agent Configuration", agent_cfg, color_map=ConfigManager._config_sources, path_prefix="agent")

    @staticmethod
    def print_model_config(model_cfg: Any) -> None:
        """Print model configuration with color-coded source tracking."""
        from learning.config_printer import print_config
        print_config("Model Configuration", model_cfg, color_map=ConfigManager._config_sources, path_prefix="primary")

    @staticmethod
    def print_task_config(task_cfg: Any) -> None:
        """Print task configuration with color-coded source tracking."""
        from learning.config_printer import print_config
        print_config("Task Configuration", task_cfg, color_map=ConfigManager._config_sources)

    @staticmethod
    def print_control_config(ctrl_cfg: Any) -> None:
        """Print control configuration with color-coded source tracking."""
        from learning.config_printer import print_config
        print_config("Control Configuration", ctrl_cfg, color_map=ConfigManager._config_sources)

    @staticmethod
    def print_config_sources() -> None:
        """Print all tracked configuration sources for debugging."""
        print("\n🔍 Configuration Source Tracking:")
        print("=" * 50)
        if not ConfigManager._config_sources:
            print("No source tracking data available.")
            return

        from learning.config_printer import Colors
        for path, source in sorted(ConfigManager._config_sources.items()):
            if source == 'local_override':
                print(f"{Colors.BLUE}{path}{Colors.RESET}: {source}")
            elif source == 'cli_override':
                print(f"{Colors.GREEN}{path}{Colors.RESET}: {source}")
            else:
                print(f"{path}: {source}")
        print("=" * 50)

    @staticmethod
    def _print_color_legend():
        """Print color legend for configuration source tracking."""
        from learning.config_printer import Colors
        print(f"\n🎨 Configuration Colors: {Colors.BLUE}Local overrides{Colors.RESET} | {Colors.GREEN}CLI overrides{Colors.RESET} | Default (base values)")


# ===== INTEGRATED LOGGING CONFIGURATION SYSTEM =====

@dataclass
class MetricConfig:
    """Configuration for a single tracked metric."""

    name: str
    """Name of the metric in the info dictionary."""

    default_value: Union[float, int, torch.Tensor] = 0.0
    """Default value if metric is not present in info."""

    metric_type: str = "scalar"
    """Type of metric: 'scalar', 'boolean', 'tensor', 'histogram'."""

    aggregation: str = "mean"
    """How to aggregate across environments: 'mean', 'sum', 'max', 'min', 'median'."""

    wandb_name: Optional[str] = None
    """Custom name for Wandb logging. If None, uses the metric name."""

    enabled: bool = True
    """Whether to track this metric."""

    normalize_by_episode_length: bool = False
    """Whether to normalize by episode length for episode metrics."""


@dataclass
class LoggingConfig:
    """Main configuration for logging wrappers."""

    # Wandb configuration
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[List[str]] = None

    # Tracking configuration
    tracked_metrics: Dict[str, MetricConfig] = field(default_factory=dict)
    track_learning_metrics: bool = True
    track_action_metrics: bool = True

    # Episode tracking
    track_episodes: bool = True
    track_rewards: bool = True
    track_episode_length: bool = True
    track_terminations: bool = True

    # PPO-specific settings
    clip_eps: float = 0.2

    # Multi-agent settings
    num_agents: int = 1

    def add_metric(self, metric: MetricConfig):
        """Add a metric to the tracking configuration."""
        self.tracked_metrics[metric.name] = metric

    def remove_metric(self, metric_name: str):
        """Remove a metric from tracking."""
        if metric_name in self.tracked_metrics:
            del self.tracked_metrics[metric_name]

    def enable_metric(self, metric_name: str, enabled: bool = True):
        """Enable or disable a specific metric."""
        if metric_name in self.tracked_metrics:
            self.tracked_metrics[metric_name].enabled = enabled

    def to_wandb_config(self) -> Dict[str, Any]:
        """Convert to Wandb configuration dictionary."""
        return {
            'entity': self.wandb_entity,
            'project': self.wandb_project,
            'name': self.wandb_name,
            'group': self.wandb_group,
            'tags': self.wandb_tags or [],
            'num_agents': self.num_agents,
            'tracked_metrics': [metric.name for metric in self.tracked_metrics.values() if metric.enabled]
        }


class LoggingConfigPresets:
    """
    DEPRECATED: Predefined logging configurations.

    This class is obsolete as the logging system has been simplified to use
    direct function calls instead of configuration files. The new system
    uses simple boolean parameters for wrapper configuration.

    See GenericWandbLoggingWrapper and EnhancedActionLoggingWrapper for the
    new simplified approach.
    """

    @staticmethod
    def basic_config() -> LoggingConfig:
        """Basic configuration with minimal tracking."""
        config = LoggingConfig()

        # Only track basic episode metrics
        config.add_metric(MetricConfig(
            name="reward",
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Episode / Reward"
        ))

        return config

    @staticmethod
    def factory_config() -> LoggingConfig:
        """Configuration optimized for factory manipulation tasks."""
        config = LoggingConfig()

        # Core episode metrics
        config.add_metric(MetricConfig(
            name="reward",
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Episode / Reward"
        ))

        # Factory-specific metrics
        config.add_metric(MetricConfig(
            name="current_engagements",
            default_value=0.0,
            metric_type="boolean",
            aggregation="mean",
            wandb_name="Engagement / Engaged Rate"
        ))

        config.add_metric(MetricConfig(
            name="current_successes",
            default_value=0.0,
            metric_type="boolean",
            aggregation="mean",
            wandb_name="Success / Success Rate"
        ))

        # Smoothness metrics
        config.add_metric(MetricConfig(
            name="sum_squared_velocity",
            default_value=0.0,
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Smoothness / Sum Squared Velocity",
            normalize_by_episode_length=True
        ))

        # Force/torque metrics (if available)
        config.add_metric(MetricConfig(
            name="max_force",
            default_value=0.0,
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Force / Max Force"
        ))

        config.add_metric(MetricConfig(
            name="max_torque",
            default_value=0.0,
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Force / Max Torque"
        ))

        config.add_metric(MetricConfig(
            name="avg_force",
            default_value=0.0,
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Force / Avg Force"
        ))

        config.add_metric(MetricConfig(
            name="avg_torque",
            default_value=0.0,
            metric_type="scalar",
            aggregation="mean",
            wandb_name="Force / Avg Torque"
        ))

        # Reward components (will be detected dynamically)
        for component in ["reach_reward", "grasp_reward", "lift_reward", "align_reward"]:
            config.add_metric(MetricConfig(
                name=f"Reward / {component}",
                default_value=0.0,
                metric_type="scalar",
                aggregation="mean",
                wandb_name=f"Reward / {component.replace('_', ' ').title()}"
            ))

        return config

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> LoggingConfig:
        """Create logging config from dictionary (integrated with ConfigManager YAML loading)."""
        config = LoggingConfig()

        # Set basic config
        config.wandb_entity = config_dict.get('wandb_entity')
        config.wandb_project = config_dict.get('wandb_project')
        config.wandb_name = config_dict.get('wandb_name')
        config.wandb_group = config_dict.get('wandb_group')
        config.wandb_tags = config_dict.get('wandb_tags', [])
        config.num_agents = config_dict.get('num_agents', 1)
        config.clip_eps = config_dict.get('clip_eps', 0.2)

        # Set tracking flags
        config.track_learning_metrics = config_dict.get('track_learning_metrics', True)
        config.track_action_metrics = config_dict.get('track_action_metrics', True)
        config.track_episodes = config_dict.get('track_episodes', True)
        config.track_rewards = config_dict.get('track_rewards', True)
        config.track_episode_length = config_dict.get('track_episode_length', True)
        config.track_terminations = config_dict.get('track_terminations', True)

        # Add tracked metrics
        for metric_name, metric_dict in config_dict.get('tracked_metrics', {}).items():
            metric = MetricConfig(
                name=metric_name,
                default_value=metric_dict.get('default_value', 0.0),
                metric_type=metric_dict.get('metric_type', 'scalar'),
                aggregation=metric_dict.get('aggregation', 'mean'),
                wandb_name=metric_dict.get('wandb_name'),
                enabled=metric_dict.get('enabled', True),
                normalize_by_episode_length=metric_dict.get('normalize_by_episode_length', False)
            )
            config.add_metric(metric)

        return config

    @staticmethod
    def create_from_config(config_dict: Dict[str, Any], preset: str = "factory") -> LoggingConfig:
        """Create logging config from configuration with preset base and overrides."""
        # Start with preset
        if preset == "factory":
            config = LoggingConfigPresets.factory_config()
        elif preset == "basic":
            config = LoggingConfigPresets.basic_config()
        else:
            config = LoggingConfig()

        # Apply overrides from config_dict
        if config_dict:
            override_config = LoggingConfigPresets.from_dict(config_dict)
            # Merge configurations (prioritize override values)
            for key, value in override_config.__dict__.items():
                if key == 'tracked_metrics':
                    # Only merge if override has metrics to add
                    if value:
                        config.tracked_metrics.update(value)
                elif value is not None:
                    # Override any existing value with the new one
                    setattr(config, key, value)

        return config


def load_config_from_file(file_path: str) -> LoggingConfig:
    """Load logging configuration from YAML or JSON file."""
    from pathlib import Path
    import json

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    # Load file based on extension
    if file_path.suffix.lower() in ['.yaml', '.yml']:
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configuration files")

        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {file_path.suffix}. Use .yaml, .yml, or .json")

    return LoggingConfigPresets.from_dict(config_dict)