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
from .cfg_exts.actor_cfg import ActorConfig
from .cfg_exts.critic_cfg import CriticConfig
from .cfg_exts.wrapper_sub_configs import (
    ForceTorqueSensorConfig, HybridControlConfig, ObservationNoiseConfig,
    WandbLoggingConfig, ActionLoggingConfig, ForceRewardConfig
)
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

    # Backward compatibility properties for tests
    @property
    def primary(self):
        """Backward compatibility alias for primary_cfg."""
        return self.primary_cfg

    @property
    def env(self):
        """Backward compatibility alias for env_cfg."""
        return self.env_cfg

    @property
    def agent(self):
        """Backward compatibility alias for agent_cfg."""
        return self.agent_cfg

    @property
    def model(self):
        """Backward compatibility alias for model_cfg."""
        return self.model_cfg

    @property
    def wrapper(self):
        """Backward compatibility alias for wrapper_cfg."""
        return self.wrapper_cfg

    @property
    def hybrid(self):
        """Backward compatibility alias for hybrid_cfg."""
        return self.hybrid_cfg

    @property
    def environment(self):
        """Backward compatibility alias for env_cfg."""
        return self.env_cfg


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

        # 1.5. Load and merge base configuration if specified
        if 'base_config' in yaml_config:
            base_config_path = yaml_config['base_config']
            print(f"[CONFIG V2]: Loading base configuration from {base_config_path}")
            base_config = ConfigManagerV2._load_yaml_file(base_config_path)
            # Merge: base config gets overridden by experiment config
            yaml_config = ConfigManagerV2._merge_configs(base_config, yaml_config)
            print(f"[CONFIG V2]: Merged base configuration with experiment configuration")

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

        # 7. Apply environment and agent YAML overrides (model and wrapper overrides already applied)
        ConfigManagerV2._apply_yaml_overrides(env_cfg, yaml_config.get('environment', {}))
        ConfigManagerV2._apply_yaml_overrides(agent_cfg, yaml_config.get('agent', {}))

        # Apply experiment configuration to agent config
        experiment_config = yaml_config.get('experiment', {})
        if experiment_config:
            if 'name' in experiment_config:
                agent_cfg.experiment_name = experiment_config['name']
                print(f"[CONFIG V2]: Applied experiment.name: {agent_cfg.experiment_name}")
            if 'tags' in experiment_config:
                agent_cfg.wandb_tags = experiment_config['tags']
                print(f"[CONFIG V2]: Applied experiment.tags: {agent_cfg.wandb_tags}")
            if 'group' in experiment_config:
                agent_cfg.wandb_group = experiment_config['group']
                print(f"[CONFIG V2]: Applied experiment.group: {agent_cfg.wandb_group}")
            if 'wandb_project' in experiment_config:
                agent_cfg.wandb_project = experiment_config['wandb_project']
                print(f"[CONFIG V2]: Applied experiment.wandb_project: {agent_cfg.wandb_project}")

        # 8. Create configuration bundle for CLI overrides
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

        # Apply CLI overrides
        if cli_overrides:
            ConfigManagerV2._apply_cli_overrides(config_bundle, cli_overrides)

            # Re-apply primary config to propagate any CLI overrides to primary config
            # This ensures environment and agent configs stay synchronized with primary
            config_bundle.env_cfg.apply_primary_cfg(config_bundle.primary_cfg)
            config_bundle.agent_cfg.apply_primary_cfg(config_bundle.primary_cfg)
            config_bundle.wrapper_cfg.apply_primary_cfg(config_bundle.primary_cfg)

        # 9. Validate final configuration

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
        Create model configuration with nested structure support.

        Args:
            yaml_config: Loaded YAML configuration
            primary_cfg: Primary configuration for reference
        Returns:
            Model configuration with overrides applied
        """
        # Start with Isaac Lab defaults
        model_cfg = ExtendedModelConfig()

        if 'model' in yaml_config:
            model_data = yaml_config['model']

            # Handle nested actor configuration
            if 'actor' in model_data:
                actor_data = model_data['actor']
                model_cfg.actor = ActorConfig(
                    n=actor_data.get('n', model_cfg.actor.n),
                    latent_size=actor_data.get('latent_size', model_cfg.actor.latent_size)
                )
                print(f"[CONFIG V2]: Applied actor config: n={model_cfg.actor.n}, latent_size={model_cfg.actor.latent_size}")

            # Handle nested critic configuration
            if 'critic' in model_data:
                critic_data = model_data['critic']
                model_cfg.critic = CriticConfig(
                    n=critic_data.get('n', model_cfg.critic.n),
                    latent_size=critic_data.get('latent_size', model_cfg.critic.latent_size)
                )
                print(f"[CONFIG V2]: Applied critic config: n={model_cfg.critic.n}, latent_size={model_cfg.critic.latent_size}")

            # Handle nested hybrid_agent configuration
            if 'hybrid_agent' in model_data:
                hybrid_data = model_data['hybrid_agent']
                if model_cfg.hybrid_agent is None:
                    model_cfg.hybrid_agent = ExtendedHybridAgentConfig()

                # Apply hybrid agent overrides using existing logic
                for key, value in hybrid_data.items():
                    if hasattr(model_cfg.hybrid_agent, key):
                        setattr(model_cfg.hybrid_agent, key, value)
                        print(f"[CONFIG V2]: Applied hybrid_agent.{key} = {value}")

            # Handle flat model configuration
            flat_configs = {k: v for k, v in model_data.items()
                           if k not in ['actor', 'critic', 'hybrid_agent']}

            for key, value in flat_configs.items():
                if hasattr(model_cfg, key):
                    old_value = getattr(model_cfg, key)
                    setattr(model_cfg, key, value)
                    print(f"[CONFIG V2]: Applied model.{key}: {old_value} -> {value}")
                else:
                    print(f"\033[93m[CONFIG V2]: Warning: ExtendedModelConfig has no attribute '{key}' (skipping)\033[0m")

        # Validate configuration
        model_cfg.validate()

        return model_cfg

    @staticmethod
    def _create_wrapper_config(yaml_config: Dict[str, Any], primary_cfg: PrimaryConfig) -> ExtendedWrapperConfig:
        """
        Create wrapper configuration with nested structure support.

        Args:
            yaml_config: Loaded YAML configuration
            primary_cfg: Primary configuration for reference
        Returns:
            Wrapper configuration with overrides applied
        """
        # Start with Isaac Lab defaults
        wrapper_cfg = ExtendedWrapperConfig()
        wrapper_cfg.apply_primary_cfg(primary_cfg)

        if 'wrappers' in yaml_config:
            wrapper_data = yaml_config['wrappers']

            # Handle nested wrapper configurations
            if 'force_torque_sensor' in wrapper_data:
                fts_data = wrapper_data['force_torque_sensor']
                wrapper_cfg.force_torque_sensor = ForceTorqueSensorConfig(
                    enabled=fts_data.get('enabled', wrapper_cfg.force_torque_sensor.enabled),
                    use_tanh_scaling=fts_data.get('use_tanh_scaling', wrapper_cfg.force_torque_sensor.use_tanh_scaling),
                    tanh_scale=fts_data.get('tanh_scale', wrapper_cfg.force_torque_sensor.tanh_scale),
                    add_force_obs=fts_data.get('add_force_obs', wrapper_cfg.force_torque_sensor.add_force_obs)
                )
                print(f"[CONFIG V2]: Applied force_torque_sensor config: enabled={wrapper_cfg.force_torque_sensor.enabled}")

            if 'hybrid_control' in wrapper_data:
                hc_data = wrapper_data['hybrid_control']
                wrapper_cfg.hybrid_control = HybridControlConfig(
                    enabled=hc_data.get('enabled', wrapper_cfg.hybrid_control.enabled),
                    reward_type=hc_data.get('reward_type', wrapper_cfg.hybrid_control.reward_type)
                )
                print(f"[CONFIG V2]: Applied hybrid_control config: enabled={wrapper_cfg.hybrid_control.enabled}")

            if 'observation_noise' in wrapper_data:
                on_data = wrapper_data['observation_noise']
                wrapper_cfg.observation_noise = ObservationNoiseConfig(
                    enabled=on_data.get('enabled', wrapper_cfg.observation_noise.enabled),
                    global_scale=on_data.get('global_scale', wrapper_cfg.observation_noise.global_scale),
                    apply_to_critic=on_data.get('apply_to_critic', wrapper_cfg.observation_noise.apply_to_critic),
                    seed=on_data.get('seed', wrapper_cfg.observation_noise.seed)
                )
                print(f"[CONFIG V2]: Applied observation_noise config: enabled={wrapper_cfg.observation_noise.enabled}")

            if 'wandb_logging' in wrapper_data:
                wl_data = wrapper_data['wandb_logging']
                wrapper_cfg.wandb_logging = WandbLoggingConfig(
                    enabled=wl_data.get('enabled', wrapper_cfg.wandb_logging.enabled),
                    wandb_project=wl_data.get('wandb_project', wrapper_cfg.wandb_logging.wandb_project),
                    wandb_entity=wl_data.get('wandb_entity', wrapper_cfg.wandb_logging.wandb_entity),
                    wandb_name=wl_data.get('wandb_name', wrapper_cfg.wandb_logging.wandb_name),
                    wandb_group=wl_data.get('wandb_group', wrapper_cfg.wandb_logging.wandb_group),
                    wandb_tags=wl_data.get('wandb_tags', wrapper_cfg.wandb_logging.wandb_tags)
                )
                print(f"[CONFIG V2]: Applied wandb_logging config: enabled={wrapper_cfg.wandb_logging.enabled}")

            if 'action_logging' in wrapper_data:
                al_data = wrapper_data['action_logging']
                wrapper_cfg.action_logging = ActionLoggingConfig(
                    enabled=al_data.get('enabled', wrapper_cfg.action_logging.enabled),
                    track_selection=al_data.get('track_selection', wrapper_cfg.action_logging.track_selection),
                    track_pos=al_data.get('track_pos', wrapper_cfg.action_logging.track_pos),
                    track_rot=al_data.get('track_rot', wrapper_cfg.action_logging.track_rot),
                    track_force=al_data.get('track_force', wrapper_cfg.action_logging.track_force),
                    track_torque=al_data.get('track_torque', wrapper_cfg.action_logging.track_torque),
                    force_size=al_data.get('force_size', wrapper_cfg.action_logging.force_size),
                    logging_frequency=al_data.get('logging_frequency', wrapper_cfg.action_logging.logging_frequency)
                )
                print(f"[CONFIG V2]: Applied action_logging config: enabled={wrapper_cfg.action_logging.enabled}")

            if 'force_reward' in wrapper_data:
                fr_data = wrapper_data['force_reward']
                wrapper_cfg.force_reward = ForceRewardConfig(
                    enabled=fr_data.get('enabled', wrapper_cfg.force_reward.enabled),
                    contact_force_threshold=fr_data.get('contact_force_threshold', wrapper_cfg.force_reward.contact_force_threshold),
                    contact_window_size=fr_data.get('contact_window_size', wrapper_cfg.force_reward.contact_window_size),
                    enable_force_magnitude_reward=fr_data.get('enable_force_magnitude_reward', wrapper_cfg.force_reward.enable_force_magnitude_reward),
                    force_magnitude_reward_weight=fr_data.get('force_magnitude_reward_weight', wrapper_cfg.force_reward.force_magnitude_reward_weight),
                    force_magnitude_base_force=fr_data.get('force_magnitude_base_force', wrapper_cfg.force_reward.force_magnitude_base_force),
                    force_magnitude_keep_sign=fr_data.get('force_magnitude_keep_sign', wrapper_cfg.force_reward.force_magnitude_keep_sign),
                    enable_alignment_award=fr_data.get('enable_alignment_award', wrapper_cfg.force_reward.enable_alignment_award),
                    alignment_award_reward_weight=fr_data.get('alignment_award_reward_weight', wrapper_cfg.force_reward.alignment_award_reward_weight),
                    alignment_goal_orientation=fr_data.get('alignment_goal_orientation', wrapper_cfg.force_reward.alignment_goal_orientation),
                    enable_force_action_error=fr_data.get('enable_force_action_error', wrapper_cfg.force_reward.enable_force_action_error),
                    force_action_error_reward_weight=fr_data.get('force_action_error_reward_weight', wrapper_cfg.force_reward.force_action_error_reward_weight),
                    enable_contact_consistency=fr_data.get('enable_contact_consistency', wrapper_cfg.force_reward.enable_contact_consistency),
                    contact_consistency_reward_weight=fr_data.get('contact_consistency_reward_weight', wrapper_cfg.force_reward.contact_consistency_reward_weight),
                    contact_consistency_beta=fr_data.get('contact_consistency_beta', wrapper_cfg.force_reward.contact_consistency_beta),
                    contact_consistency_use_ema=fr_data.get('contact_consistency_use_ema', wrapper_cfg.force_reward.contact_consistency_use_ema),
                    contact_consistency_ema_alpha=fr_data.get('contact_consistency_ema_alpha', wrapper_cfg.force_reward.contact_consistency_ema_alpha),
                    enable_oscillation_penalty=fr_data.get('enable_oscillation_penalty', wrapper_cfg.force_reward.enable_oscillation_penalty),
                    oscillation_penalty_reward_weight=fr_data.get('oscillation_penalty_reward_weight', wrapper_cfg.force_reward.oscillation_penalty_reward_weight),
                    oscillation_penalty_window_size=fr_data.get('oscillation_penalty_window_size', wrapper_cfg.force_reward.oscillation_penalty_window_size),
                    enable_contact_transition_reward=fr_data.get('enable_contact_transition_reward', wrapper_cfg.force_reward.enable_contact_transition_reward),
                    contact_transition_reward_weight=fr_data.get('contact_transition_reward_weight', wrapper_cfg.force_reward.contact_transition_reward_weight),
                    enable_efficiency=fr_data.get('enable_efficiency', wrapper_cfg.force_reward.enable_efficiency),
                    efficiency_reward_weight=fr_data.get('efficiency_reward_weight', wrapper_cfg.force_reward.efficiency_reward_weight),
                    enable_force_ratio=fr_data.get('enable_force_ratio', wrapper_cfg.force_reward.enable_force_ratio),
                    force_ratio_reward_weight=fr_data.get('force_ratio_reward_weight', wrapper_cfg.force_reward.force_ratio_reward_weight)
                )
                print(f"[CONFIG V2]: Applied force_reward config: enabled={wrapper_cfg.force_reward.enabled}")

            # Handle flat wrapper configurations
            flat_configs = {k: v for k, v in wrapper_data.items()
                           if k not in ['force_torque_sensor', 'hybrid_control', 'observation_noise',
                                       'wandb_logging', 'action_logging', 'force_reward']}

            for key, value in flat_configs.items():
                flat_attr_map = {
                    'fragile_objects': 'fragile_objects_enabled',
                    'efficient_reset': 'efficient_reset_enabled',
                    'observation_manager': 'observation_manager_enabled',
                    'factory_metrics': 'factory_metrics_enabled'
                }

                if key in flat_attr_map:
                    attr_name = flat_attr_map[key]
                    if isinstance(value, dict) and 'enabled' in value:
                        setattr(wrapper_cfg, attr_name, value['enabled'])
                        print(f"[CONFIG V2]: Applied {key}.enabled = {value['enabled']}")
                else:
                    print(f"\033[93m[CONFIG V2]: Warning: Unknown wrapper config '{key}' (skipping)\033[0m")

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
                                print(f"\033[93m[CONFIG V2]:   Warning: ctrl has no attribute '{ctrl_param}' (skipping)\033[0m")
                        print(f"[CONFIG V2]:   Applied ctrl configuration overrides")
                        continue  # Skip the normal assignment

                setattr(target_obj, key, value)
                print(f"[CONFIG V2]:   Override {key}: {old_value} -> {value}")
            else:
                print(f"\033[93m[CONFIG V2]:   Warning: {type(target_obj).__name__} has no attribute '{key}' (skipping)\033[0m")

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
                'tanh_scale': 'force_torque_tanh_scale',
                'add_force_obs' : 'add_force_obs'
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
    def _apply_cli_overrides(config_bundle: ConfigBundle, cli_overrides: List[str]) -> None:
        """
        Apply CLI overrides with support for nested configurations.

        Args:
            config_bundle: Complete configuration bundle
            cli_overrides: List of "key=value" override strings
        """
        for override in cli_overrides:
            if '=' not in override:
                print(f"\033[93m[CONFIG V2]: Warning: Invalid override format '{override}' (expected key=value)\033[0m")
                continue

            key, value = override.split('=', 1)
            parsed_value = ConfigManagerV2._parse_override_value(value)

            # Handle nested model overrides
            if key.startswith('model.actor.'):
                param = key.replace('model.actor.', '')
                if hasattr(config_bundle.model_cfg.actor, param):
                    setattr(config_bundle.model_cfg.actor, param, parsed_value)
                    print(f"[CONFIG V2]: CLI override model.actor.{param} = {parsed_value}")
                else:
                    print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no such attribute\033[0m")

            elif key.startswith('model.critic.'):
                param = key.replace('model.critic.', '')
                if hasattr(config_bundle.model_cfg.critic, param):
                    setattr(config_bundle.model_cfg.critic, param, parsed_value)
                    print(f"[CONFIG V2]: CLI override model.critic.{param} = {parsed_value}")
                else:
                    print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no such attribute\033[0m")

            elif key.startswith('model.hybrid_agent.'):
                param = key.replace('model.hybrid_agent.', '')
                if config_bundle.model_cfg.hybrid_agent is None:
                    config_bundle.model_cfg.hybrid_agent = ExtendedHybridAgentConfig()
                if hasattr(config_bundle.model_cfg.hybrid_agent, param):
                    setattr(config_bundle.model_cfg.hybrid_agent, param, parsed_value)
                    print(f"[CONFIG V2]: CLI override model.hybrid_agent.{param} = {parsed_value}")
                else:
                    print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no such attribute\033[0m")

            elif key.startswith('wrappers.force_torque_sensor.'):
                param = key.replace('wrappers.force_torque_sensor.', '')
                if hasattr(config_bundle.wrapper_cfg.force_torque_sensor, param):
                    setattr(config_bundle.wrapper_cfg.force_torque_sensor, param, parsed_value)
                    print(f"[CONFIG V2]: CLI override wrappers.force_torque_sensor.{param} = {parsed_value}")
                else:
                    print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no such attribute\033[0m")

            elif key.startswith('wrappers.hybrid_control.'):
                param = key.replace('wrappers.hybrid_control.', '')
                if hasattr(config_bundle.wrapper_cfg.hybrid_control, param):
                    setattr(config_bundle.wrapper_cfg.hybrid_control, param, parsed_value)
                    print(f"[CONFIG V2]: CLI override wrappers.hybrid_control.{param} = {parsed_value}")
                else:
                    print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no such attribute\033[0m")

            elif key.startswith('wrappers.observation_noise.'):
                param = key.replace('wrappers.observation_noise.', '')
                if hasattr(config_bundle.wrapper_cfg.observation_noise, param):
                    setattr(config_bundle.wrapper_cfg.observation_noise, param, parsed_value)
                    print(f"[CONFIG V2]: CLI override wrappers.observation_noise.{param} = {parsed_value}")
                else:
                    print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no such attribute\033[0m")

            elif key.startswith('wrappers.wandb_logging.'):
                param = key.replace('wrappers.wandb_logging.', '')
                if hasattr(config_bundle.wrapper_cfg.wandb_logging, param):
                    setattr(config_bundle.wrapper_cfg.wandb_logging, param, parsed_value)
                    print(f"[CONFIG V2]: CLI override wrappers.wandb_logging.{param} = {parsed_value}")
                else:
                    print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no such attribute\033[0m")

            elif key.startswith('wrappers.action_logging.'):
                param = key.replace('wrappers.action_logging.', '')
                if hasattr(config_bundle.wrapper_cfg.action_logging, param):
                    setattr(config_bundle.wrapper_cfg.action_logging, param, parsed_value)
                    print(f"[CONFIG V2]: CLI override wrappers.action_logging.{param} = {parsed_value}")
                else:
                    print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no such attribute\033[0m")

            elif key.startswith('wrappers.force_reward.'):
                param = key.replace('wrappers.force_reward.', '')
                if hasattr(config_bundle.wrapper_cfg.force_reward, param):
                    setattr(config_bundle.wrapper_cfg.force_reward, param, parsed_value)
                    print(f"[CONFIG V2]: CLI override wrappers.force_reward.{param} = {parsed_value}")
                else:
                    print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no such attribute\033[0m")

            elif key == 'experiment.tags':
                # Special handling for experiment tags - merge with existing
                if isinstance(parsed_value, list):
                    cli_tags = parsed_value
                else:
                    # Handle single tag or comma-separated string
                    cli_tags = [tag.strip() for tag in str(parsed_value).split(',')]

                # Store for later merging after launch_utils processes experiment config
                config_bundle._cli_experiment_tags = cli_tags
                print(f"[CONFIG V2]: CLI experiment tags to merge: {cli_tags}")

            # Handle existing flat overrides (primary, model flat attrs, etc.)
            else:
                # Use existing logic for flat overrides
                ConfigManagerV2._apply_flat_cli_override(config_bundle, key, parsed_value)

    @staticmethod
    def _parse_override_value(value_str: str):
        """Parse CLI override value string to appropriate Python type."""
        try:
            import ast
            return ast.literal_eval(value_str)
        except:
            # Handle common string representations
            if value_str.lower() == 'true':
                return True
            elif value_str.lower() == 'false':
                return False
            elif value_str.lower() == 'none':
                return None
            else:
                return value_str

    @staticmethod
    def _apply_flat_cli_override(config_bundle: ConfigBundle, key: str, parsed_value) -> None:
        """Apply flat CLI overrides to config objects."""
        if '.' in key:
            section, param = key.split('.', 1)
            config_objects = {
                'environment': config_bundle.env_cfg,
                'agent': config_bundle.agent_cfg,
                'model': config_bundle.model_cfg,
                'wrappers': config_bundle.wrapper_cfg,
                'primary': config_bundle.primary_cfg
            }

            if section in config_objects:
                target_obj = config_objects[section]
                if hasattr(target_obj, param):
                    setattr(target_obj, param, parsed_value)
                    print(f"[CONFIG V2]: CLI override {key} = {parsed_value}")
                else:
                    print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no such attribute (skipping)\033[0m")
            else:
                print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - unknown section '{section}' (skipping)\033[0m")
        else:
            print(f"\033[93m[CONFIG V2]: Warning: CLI override {key} - no section specified (skipping)\033[0m")

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
                'episode_length_s': config_bundle.env_cfg.episode_length_s,
                'ctrl': config_bundle.env_cfg.ctrl
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

    @staticmethod
    def _merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge base configuration with override configuration.

        The override config takes precedence over base config values.
        Performs deep merge for nested dictionaries.

        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        import copy

        # Start with a deep copy of base config
        merged = copy.deepcopy(base_config)

        # Recursively merge override config
        def _deep_merge(base_dict, override_dict):
            for key, value in override_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    # Special handling for experiment tags - merge instead of replace
                    if key == 'experiment' and 'tags' in base_dict[key] and 'tags' in value:
                        # Merge tags, preserve other experiment fields normally
                        merged_experiment = copy.deepcopy(base_dict[key])
                        for exp_key, exp_value in value.items():
                            if exp_key == 'tags':
                                # Merge tags: base + experiment (remove duplicates, preserve order)
                                merged_tags = list(base_dict[key]['tags'])
                                for tag in exp_value:
                                    if tag not in merged_tags:
                                        merged_tags.append(tag)
                                merged_experiment['tags'] = merged_tags
                            else:
                                # Normal override for other experiment fields
                                merged_experiment[exp_key] = copy.deepcopy(exp_value)
                        base_dict[key] = merged_experiment
                    else:
                        # Recursively merge nested dictionaries
                        _deep_merge(base_dict[key], value)
                else:
                    # Override value (including replacing entire nested dicts)
                    base_dict[key] = copy.deepcopy(value)

        _deep_merge(merged, override_config)
        return merged