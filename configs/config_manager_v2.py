"""
Configuration Manager V2 - Defaults-First Architecture

This is the refactored configuration system that implements a defaults-first
loading approach. It replaces the reference resolution system with direct
configuration application and computed properties.
"""

import yaml
import os
from typing import Dict, Any, List, Optional, Union, Tuple, Type
import copy
from dataclasses import dataclass

# Import our extended configurations
from .cfg_exts.primary_cfg import PrimaryConfig
from .cfg_exts.extended_factory_env_cfg import ExtendedFactoryEnvCfg
from .cfg_exts.extended_peg_insert_cfg import ExtendedFactoryTaskPegInsertCfg
from .cfg_exts.extended_gear_mesh_cfg import ExtendedFactoryTaskGearMeshCfg
from .cfg_exts.extended_nut_thread_cfg import ExtendedFactoryTaskNutThreadCfg
from .cfg_exts.extended_model_cfg import ExtendedModelConfig, ExtendedHybridAgentConfig
from .cfg_exts.extended_wrapper_cfg import ExtendedWrapperConfig
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.extended_ppo_cfg import ExtendedPPOConfig


@dataclass
class ConfigBundle:
    """
    Complete configuration bundle containing all configuration objects.

    This replaces the old dictionary-based approach with typed configuration objects.
    """
    # Core configurations
    env_cfg: ExtendedFactoryEnvCfg
    """Isaac Lab environment configuration"""

    agent_cfg: ExtendedPPOConfig
    """SKRL PPO agent configuration"""

    primary_cfg: PrimaryConfig
    """Primary shared configuration"""

    model_cfg: ExtendedModelConfig
    """Model architecture configuration"""

    wrapper_cfg: ExtendedWrapperConfig
    """Wrapper configuration"""

    hybrid_cfg: Optional[ExtendedHybridAgentConfig] = None
    """Hybrid agent configuration (if enabled)"""

    # Metadata
    task_name: str = ""
    """Selected task name"""

    config_source_path: str = ""
    """Path to source configuration file"""


class ConfigManagerV2:
    """
    Configuration Manager V2 with defaults-first architecture.

    This manager loads Isaac Lab and SKRL defaults first, then applies
    overrides in a clear hierarchy: Primary -> YAML -> CLI arguments.
    """

    # Task name to extended config class mapping
    TASK_CONFIG_MAP = {
        "Isaac-Factory-PegInsert-Direct-v0": ExtendedFactoryTaskPegInsertCfg,
        "Isaac-Factory-PegInsert-Local-v0": ExtendedFactoryTaskPegInsertCfg,
        "peg_insert": ExtendedFactoryTaskPegInsertCfg,

        "Isaac-Factory-GearMesh-Direct-v0": ExtendedFactoryTaskGearMeshCfg,
        "Isaac-Factory-GearMesh-Local-v0": ExtendedFactoryTaskGearMeshCfg,
        "gear_mesh": ExtendedFactoryTaskGearMeshCfg,

        "Isaac-Factory-NutThread-Direct-v0": ExtendedFactoryTaskNutThreadCfg,
        "Isaac-Factory-NutThread-Local-v0": ExtendedFactoryTaskNutThreadCfg,
        "nut_thread": ExtendedFactoryTaskNutThreadCfg,
    }

    @staticmethod
    def load_defaults_first_config(
        config_path: str,
        cli_overrides: Optional[List[str]] = None,
        cli_task: Optional[str] = None
    ) -> ConfigBundle:
        """
        Load configuration using defaults-first architecture.

        Args:
            config_path: Path to the YAML configuration file
            cli_overrides: List of CLI override strings in "key=value" format
            cli_task: Task name from CLI (takes precedence over YAML)

        Returns:
            Complete configuration bundle with all configs properly initialized

        Raises:
            ValueError: If task name is missing, unknown, or configuration is invalid
            FileNotFoundError: If configuration file doesn't exist
        """
        print(f"[CONFIG V2]: Loading configuration from {config_path}")

        # 1. Load and validate YAML file
        yaml_config = ConfigManagerV2._load_yaml_file(config_path)

        # 2. Resolve task name with proper error handling
        task_name = ConfigManagerV2._resolve_task_name(yaml_config, cli_task)
        print(f"[CONFIG V2]: Resolved task name: {task_name}")

        # 3. Load extended configs with Isaac Lab/SKRL defaults
        env_cfg = ConfigManagerV2._load_extended_env_config(task_name)
        agent_cfg = ConfigManagerV2._load_extended_agent_config()

        # 4. Create and apply primary configuration
        primary_cfg = ConfigManagerV2._create_primary_config(yaml_config)
        env_cfg.apply_primary_cfg(primary_cfg)
        agent_cfg.apply_primary_cfg(primary_cfg)

        # 5. Create additional configuration objects
        model_cfg = ConfigManagerV2._create_model_config(yaml_config, primary_cfg)
        wrapper_cfg = ConfigManagerV2._create_wrapper_config(yaml_config, primary_cfg)

        # 6. Create hybrid config if needed
        hybrid_cfg = None
        if model_cfg.use_hybrid_agent:
            hybrid_cfg = ConfigManagerV2._create_hybrid_config(yaml_config)

        # 7. Apply YAML overrides
        ConfigManagerV2._apply_yaml_overrides(env_cfg, yaml_config.get('environment', {}))
        ConfigManagerV2._apply_yaml_overrides(model_cfg, yaml_config.get('model', {}))
        ConfigManagerV2._apply_yaml_overrides(wrapper_cfg, yaml_config.get('wrappers', {}))
        ConfigManagerV2._apply_yaml_overrides(agent_cfg, yaml_config.get('agent', {}))

        if hybrid_cfg and 'model' in yaml_config and 'hybrid_agent' in yaml_config['model']:
            ConfigManagerV2._apply_yaml_overrides(hybrid_cfg, yaml_config['model']['hybrid_agent'])

        # 8. Apply CLI overrides
        if cli_overrides:
            ConfigManagerV2._apply_cli_overrides(env_cfg, agent_cfg, model_cfg, wrapper_cfg, cli_overrides)

        # 9. Validate final configuration
        config_bundle = ConfigBundle(
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            primary_cfg=primary_cfg,
            model_cfg=model_cfg,
            wrapper_cfg=wrapper_cfg,
            hybrid_cfg=hybrid_cfg,
            task_name=task_name,
            config_source_path=config_path
        )

        ConfigManagerV2._validate_config_bundle(config_bundle)

        print(f"[CONFIG V2]: Successfully loaded configuration for {task_name}")
        print(f"[CONFIG V2]: Total agents: {primary_cfg.total_agents}, Total envs: {primary_cfg.total_num_envs}")

        return config_bundle

    @staticmethod
    def _resolve_task_name(yaml_config: Dict[str, Any], cli_task: Optional[str]) -> str:
        """
        Resolve task name with proper error handling.

        Priority: CLI argument > YAML task_name > Error

        Args:
            yaml_config: Loaded YAML configuration
            cli_task: Task name from CLI

        Returns:
            Resolved task name

        Raises:
            ValueError: If no task name found or task name is unknown
        """
        # Priority 1: CLI argument
        if cli_task:
            task_name = cli_task
            source = "CLI argument"
        # Priority 2: YAML task_name field
        elif 'task_name' in yaml_config:
            task_name = yaml_config['task_name']
            source = "YAML configuration"
        # Priority 3: YAML defaults section (fallback)
        elif 'defaults' in yaml_config and 'task_name' in yaml_config['defaults']:
            task_name = yaml_config['defaults']['task_name']
            source = "YAML defaults section"
        else:
            raise ValueError(
                "No task name specified. Provide task name via:\n"
                "  1. CLI argument: --task <task_name>\n"
                "  2. YAML field: task_name: <task_name>\n"
                "  3. YAML defaults: defaults.task_name: <task_name>"
            )

        # Validate task name is known
        if task_name not in ConfigManagerV2.TASK_CONFIG_MAP:
            available_tasks = list(ConfigManagerV2.TASK_CONFIG_MAP.keys())
            raise ValueError(
                f"Unknown task name: '{task_name}' (from {source})\n"
                f"Available tasks: {available_tasks}"
            )

        return task_name

    @staticmethod
    def _load_extended_env_config(task_name: str) -> ExtendedFactoryEnvCfg:
        """
        Load extended environment configuration for the specified task.

        Args:
            task_name: Name of the task

        Returns:
            Extended environment configuration with Isaac Lab defaults

        Raises:
            ValueError: If task name is unknown
        """
        if task_name not in ConfigManagerV2.TASK_CONFIG_MAP:
            raise ValueError(f"Unknown task name: {task_name}")

        config_class = ConfigManagerV2.TASK_CONFIG_MAP[task_name]

        try:
            env_cfg = config_class()
            print(f"[CONFIG V2]: Loaded {config_class.__name__} with Isaac Lab defaults")
            return env_cfg
        except Exception as e:
            raise ValueError(f"Failed to load extended environment config for {task_name}: {e}")

    @staticmethod
    def _load_extended_agent_config() -> ExtendedPPOConfig:
        """
        Load extended agent configuration with SKRL defaults.

        Returns:
            Extended PPO configuration with SKRL defaults
        """
        try:
            agent_cfg = ExtendedPPOConfig()
            print(f"[CONFIG V2]: Loaded ExtendedPPOConfig with SKRL defaults")
            return agent_cfg
        except Exception as e:
            raise ValueError(f"Failed to load extended agent config: {e}")

    @staticmethod
    def _create_primary_config(yaml_config: Dict[str, Any]) -> PrimaryConfig:
        """
        Create primary configuration from YAML, applying overrides to defaults.

        Args:
            yaml_config: Loaded YAML configuration

        Returns:
            Primary configuration with overrides applied
        """
        # Start with defaults
        primary_cfg = PrimaryConfig()

        # Apply YAML overrides if primary section exists
        if 'primary' in yaml_config:
            primary_overrides = yaml_config['primary']
            ConfigManagerV2._apply_yaml_overrides(primary_cfg, primary_overrides)
            print(f"[CONFIG V2]: Applied primary config overrides: {list(primary_overrides.keys())}")
        else:
            print(f"[CONFIG V2]: Using primary config defaults (no primary section found)")

        return primary_cfg

    @staticmethod
    def _create_model_config(yaml_config: Dict[str, Any], primary_cfg: PrimaryConfig) -> ExtendedModelConfig:
        """
        Create model configuration from YAML.

        Args:
            yaml_config: Loaded YAML configuration
            primary_cfg: Primary configuration for reference

        Returns:
            Model configuration with overrides applied
        """
        model_cfg = ExtendedModelConfig()

        if 'model' in yaml_config:
            # Handle nested actor/critic configuration
            model_overrides = yaml_config['model'].copy()

            # Extract and handle actor configuration
            if 'actor' in model_overrides:
                actor_config = model_overrides.pop('actor')
                if 'n' in actor_config:
                    model_cfg.actor_n = actor_config['n']
                    print(f"[CONFIG V2]:   Override actor_n: {model_cfg.actor_n} -> {actor_config['n']}")
                if 'latent_size' in actor_config:
                    old_value = model_cfg.actor_latent_size
                    model_cfg.actor_latent_size = actor_config['latent_size']
                    print(f"[CONFIG V2]:   Override actor_latent_size: {old_value} -> {actor_config['latent_size']}")

            # Extract and handle critic configuration
            if 'critic' in model_overrides:
                critic_config = model_overrides.pop('critic')
                if 'n' in critic_config:
                    model_cfg.critic_n = critic_config['n']
                    print(f"[CONFIG V2]:   Override critic_n: {model_cfg.critic_n} -> {critic_config['n']}")
                if 'latent_size' in critic_config:
                    old_value = model_cfg.critic_latent_size
                    model_cfg.critic_latent_size = critic_config['latent_size']
                    print(f"[CONFIG V2]:   Override critic_latent_size: {old_value} -> {critic_config['latent_size']}")

            # Apply remaining overrides
            ConfigManagerV2._apply_yaml_overrides(model_cfg, model_overrides)
            print(f"[CONFIG V2]: Applied model config overrides")

        return model_cfg

    @staticmethod
    def _create_wrapper_config(yaml_config: Dict[str, Any], primary_cfg: PrimaryConfig) -> ExtendedWrapperConfig:
        """
        Create wrapper configuration from YAML.

        Args:
            yaml_config: Loaded YAML configuration
            primary_cfg: Primary configuration for reference

        Returns:
            Wrapper configuration with overrides applied
        """
        wrapper_cfg = ExtendedWrapperConfig()
        wrapper_cfg.apply_primary_cfg(primary_cfg)

        if 'wrappers' in yaml_config:
            # Handle nested wrapper configuration
            wrapper_overrides = yaml_config['wrappers']
            ConfigManagerV2._apply_nested_wrapper_overrides(wrapper_cfg, wrapper_overrides)
            print(f"[CONFIG V2]: Applied wrapper config overrides")

        return wrapper_cfg

    @staticmethod
    def _create_hybrid_config(yaml_config: Dict[str, Any]) -> ExtendedHybridAgentConfig:
        """
        Create hybrid agent configuration from YAML.

        Args:
            yaml_config: Loaded YAML configuration

        Returns:
            Hybrid agent configuration with overrides applied
        """
        hybrid_cfg = ExtendedHybridAgentConfig()

        if 'model' in yaml_config and 'hybrid_agent' in yaml_config['model']:
            hybrid_overrides = yaml_config['model']['hybrid_agent']
            ConfigManagerV2._apply_yaml_overrides(hybrid_cfg, hybrid_overrides)
            print(f"[CONFIG V2]: Applied hybrid agent config overrides")

        return hybrid_cfg

    @staticmethod
    def _load_yaml_file(file_path: str) -> Dict[str, Any]:
        """
        Load YAML configuration file with error handling.

        Args:
            file_path: Path to YAML file

        Returns:
            Loaded YAML configuration

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load {file_path}: {e}")

    @staticmethod
    def _apply_yaml_overrides(target_obj: Any, yaml_overrides: Dict[str, Any]) -> None:
        """
        Apply YAML overrides to target configuration object.

        Args:
            target_obj: Configuration object to modify
            yaml_overrides: Dictionary of override values
        """
        for key, value in yaml_overrides.items():
            if hasattr(target_obj, key):
                old_value = getattr(target_obj, key)

                # Special handling for ctrl configuration - merge instead of replace
                if key == 'ctrl' and isinstance(value, dict):
                    ctrl_obj = getattr(target_obj, 'ctrl')
                    if ctrl_obj is not None:
                        # Apply ctrl parameters to existing ExtendedCtrlCfg object
                        for ctrl_param, ctrl_value in value.items():
                            if hasattr(ctrl_obj, ctrl_param):
                                old_ctrl_value = getattr(ctrl_obj, ctrl_param)
                                setattr(ctrl_obj, ctrl_param, ctrl_value)
                                print(f"[CONFIG V2]:   Override ctrl.{ctrl_param}: {old_ctrl_value} -> {ctrl_value}")
                            else:
                                print(f"[CONFIG V2]:   Warning: ctrl has no attribute '{ctrl_param}' (skipping)")
                        print(f"[CONFIG V2]:   Applied ctrl configuration overrides")
                        continue  # Skip the normal assignment

                setattr(target_obj, key, value)
                print(f"[CONFIG V2]:   Override {key}: {old_value} -> {value}")
            else:
                print(f"[CONFIG V2]:   Warning: {type(target_obj).__name__} has no attribute '{key}' (skipping)")

    @staticmethod
    def _apply_nested_wrapper_overrides(wrapper_cfg: ExtendedWrapperConfig, wrapper_overrides: Dict[str, Any]) -> None:
        """
        Apply nested wrapper configuration overrides.

        Args:
            wrapper_cfg: Wrapper configuration object
            wrapper_overrides: Nested wrapper override configuration
        """
        # Handle each wrapper section with correct attribute mapping
        wrapper_mappings = {
            'fragile_objects': {
                'enabled': 'fragile_objects_enabled'
            },
            'force_torque_sensor': {
                'enabled': 'force_torque_sensor_enabled',
                'use_tanh_scaling': 'force_torque_use_tanh_scaling',
                'tanh_scale': 'force_torque_tanh_scale'
            },
            'observation_noise': {
                'enabled': 'observation_noise_enabled',
                'global_scale': 'observation_noise_global_scale',
                'apply_to_critic': 'observation_noise_apply_to_critic',
                'seed': 'observation_noise_seed'
            },
            'hybrid_control': {
                'enabled': 'hybrid_control_enabled',
                'reward_type': 'hybrid_control_reward_type'
            },
            'wandb_logging': {
                'enabled': 'wandb_logging_enabled',
                'wandb_project': 'wandb_project',
                'wandb_entity': 'wandb_entity',
                'wandb_name': 'wandb_name',
                'wandb_group': 'wandb_group',
                'wandb_tags': 'wandb_tags'
            },
            'action_logging': {
                'enabled': 'action_logging_enabled',
                'track_selection': 'action_logging_track_selection',
                'track_pos': 'action_logging_track_pos',
                'track_rot': 'action_logging_track_rot',
                'track_force': 'action_logging_track_force',
                'track_torque': 'action_logging_track_torque',
                'force_size': 'action_logging_force_size',
                'logging_frequency': 'action_logging_frequency',
                'track_action_histograms': 'action_logging_track_histograms',
                'track_observation_histograms': 'action_logging_track_observation_histograms'
            }
        }

        for section_name, param_mapping in wrapper_mappings.items():
            if section_name in wrapper_overrides:
                section_config = wrapper_overrides[section_name]

                # Apply each parameter with proper mapping
                for yaml_param, attr_name in param_mapping.items():
                    if yaml_param in section_config and hasattr(wrapper_cfg, attr_name):
                        old_value = getattr(wrapper_cfg, attr_name)
                        setattr(wrapper_cfg, attr_name, section_config[yaml_param])
                        print(f"[CONFIG V2]:   Override {attr_name}: {old_value} -> {section_config[yaml_param]}")

    @staticmethod
    def _apply_cli_overrides(
        env_cfg: ExtendedFactoryEnvCfg,
        agent_cfg: ExtendedPPOConfig,
        model_cfg: ExtendedModelConfig,
        wrapper_cfg: ExtendedWrapperConfig,
        cli_overrides: List[str]
    ) -> None:
        """
        Apply CLI overrides to configuration objects.

        Args:
            env_cfg: Environment configuration
            agent_cfg: Agent configuration
            model_cfg: Model configuration
            wrapper_cfg: Wrapper configuration
            cli_overrides: List of "key=value" override strings
        """
        config_objects = {
            'environment': env_cfg,
            'agent': agent_cfg,
            'model': model_cfg,
            'wrappers': wrapper_cfg
        }

        for override_str in cli_overrides:
            if '=' not in override_str:
                raise ValueError(f"Invalid CLI override format: '{override_str}'. Expected 'key=value'")

            key, value_str = override_str.split('=', 1)

            # Parse value
            try:
                import ast
                value = ast.literal_eval(value_str)
            except:
                # Handle common string representations
                if value_str.lower() == 'true':
                    value = True
                elif value_str.lower() == 'false':
                    value = False
                elif value_str.lower() == 'none':
                    value = None
                else:
                    value = value_str

            # Apply override to appropriate config object
            if '.' in key:
                section, param = key.split('.', 1)
                if section in config_objects:
                    target_obj = config_objects[section]
                    if hasattr(target_obj, param):
                        setattr(target_obj, param, value)
                        print(f"[CONFIG V2]: CLI override {key} = {value}")
                    else:
                        print(f"[CONFIG V2]: Warning: CLI override {key} - no such attribute (skipping)")
                else:
                    print(f"[CONFIG V2]: Warning: CLI override {key} - unknown section '{section}' (skipping)")
            else:
                print(f"[CONFIG V2]: Warning: CLI override {key} - no section specified (skipping)")

    @staticmethod
    def _validate_config_bundle(config_bundle: ConfigBundle) -> None:
        """
        Validate the complete configuration bundle.

        Args:
            config_bundle: Complete configuration bundle

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate individual configs
        config_bundle.env_cfg.validate_configuration()
        config_bundle.primary_cfg.__post_init__()  # Re-run validation

        # Validate consistency between configs
        if config_bundle.env_cfg.num_agents != config_bundle.primary_cfg.total_agents:
            raise ValueError(
                f"Inconsistent agent count: env_cfg.num_agents={config_bundle.env_cfg.num_agents}, "
                f"primary_cfg.total_agents={config_bundle.primary_cfg.total_agents}"
            )

        # Validate hybrid config consistency
        if config_bundle.model_cfg.use_hybrid_agent and config_bundle.hybrid_cfg is None:
            raise ValueError("Model config specifies use_hybrid_agent=True but no hybrid config provided")

        if not config_bundle.model_cfg.use_hybrid_agent and config_bundle.hybrid_cfg is not None:
            raise ValueError("Hybrid config provided but model config has use_hybrid_agent=False")

        print(f"[CONFIG V2]: Configuration validation passed")

    @staticmethod
    def get_legacy_config_dict(config_bundle: ConfigBundle) -> Dict[str, Any]:
        """
        Convert configuration bundle to legacy dictionary format for compatibility.

        This method provides backward compatibility with code that expects
        the old dictionary-based configuration format.

        Args:
            config_bundle: Configuration bundle to convert

        Returns:
            Dictionary in legacy format
        """
        return {
            'primary': {
                'agents_per_break_force': config_bundle.primary_cfg.agents_per_break_force,
                'num_envs_per_agent': config_bundle.primary_cfg.num_envs_per_agent,
                'break_forces': config_bundle.primary_cfg.break_forces,
                'decimation': config_bundle.primary_cfg.decimation,
                'policy_hz': config_bundle.primary_cfg.policy_hz,
                'max_steps': config_bundle.primary_cfg.max_steps,
                'debug_mode': config_bundle.primary_cfg.debug_mode,
                'seed': config_bundle.primary_cfg.seed,
                'ckpt_tracker_path': config_bundle.primary_cfg.ckpt_tracker_path,
                'ctrl_torque': config_bundle.primary_cfg.ctrl_torque
            },
            'derived': {
                'total_agents': config_bundle.primary_cfg.total_agents,
                'total_num_envs': config_bundle.primary_cfg.total_num_envs,
                'rollout_steps': config_bundle.env_cfg.get_rollout_steps(),
                'max_steps': config_bundle.primary_cfg.max_steps,
                'sim_dt': config_bundle.primary_cfg.sim_dt,
                'ckpt_tracker_path': config_bundle.primary_cfg.ckpt_tracker_path,
                'seed': config_bundle.primary_cfg.seed
            },
            'environment': {
                'decimation': config_bundle.env_cfg.decimation,
                'episode_length_s': config_bundle.env_cfg.episode_length_s
            },
            'model': config_bundle.model_cfg.to_dict(),
            'wrappers': config_bundle.wrapper_cfg.to_dict(),
            'agent': config_bundle.agent_cfg.to_skrl_dict(config_bundle.env_cfg.episode_length_s),
            'experiment': {
                'name': config_bundle.agent_cfg.experiment_name,
                'tags': config_bundle.agent_cfg.wandb_tags,
                'group': config_bundle.agent_cfg.wandb_group,
                'wandb_project': config_bundle.agent_cfg.wandb_project
            }
        }