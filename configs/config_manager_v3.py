"""
Configuration Manager V3

A simplified, maintainable configuration system that eliminates complexity
while maintaining all required functionality.

Single class approach that handles:
- YAML file loading and validation
- Config class discovery and importing
- Section-to-class mapping
- Configuration bundle creation
"""

import os
import yaml
from typing import Dict, Any, Tuple, Type, List, Optional

# Import required config classes
from configs.cfg_exts.primary_cfg import PrimaryConfig
from configs.cfg_exts.extended_factory_env_cfg import ExtendedFactoryEnvCfg
from configs.cfg_exts.extended_peg_insert_cfg import ExtendedFactoryTaskPegInsertCfg
from configs.cfg_exts.extended_model_cfg import ExtendedModelConfig, ExtendedHybridAgentConfig
from configs.cfg_exts.extended_wrapper_cfg import ExtendedWrapperConfig
from configs.cfg_exts.experiment_cfg import ExperimentConfig
from configs.cfg_exts.extended_ppo_cfg import ExtendedPPOConfig
from configs.cfg_exts.ctrl_cfg import ExtendedCtrlCfg
from configs.cfg_exts.hybrid_task_cfg import HybridTaskCfg

from configs.cfg_exts.actor_cfg import ActorConfig
from configs.cfg_exts.critic_cfg import CriticConfig
from configs.cfg_exts.wrapper_sub_configs import ActionLoggingConfig, ForceRewardConfig, ForceTorqueSensorConfig, FragileObjectConfig, ObsManagerConfig, ObservationNoiseConfig
from configs.cfg_exts.wrapper_sub_configs import HybridControlConfig
# Default section-to-class mapping - single source of truth
SECTION_MAPPING = {
    'primary': PrimaryConfig,
    'environment': ExtendedFactoryTaskPegInsertCfg,
    'model': ExtendedModelConfig,
    'wrappers': ExtendedWrapperConfig,
    'experiment': ExperimentConfig,
    'agent': ExtendedPPOConfig,
    'ctrl': ExtendedCtrlCfg,
    'hybrid_agent': ExtendedHybridAgentConfig,
    'actor': ActorConfig,
    'critic': CriticConfig,
    'action_logging': ActionLoggingConfig,
    'force_reward': ForceRewardConfig,
    'force_torque_sensor': ForceTorqueSensorConfig,
    'fragile_objects': FragileObjectConfig,
    'hybrid_control': HybridControlConfig,
    'observation_manager': ObsManagerConfig,
    'observation_noise': ObservationNoiseConfig,
    'hybrid_task': HybridTaskCfg
}



class ConfigManagerV3:
    """
    Main configuration manager that handles all configuration operations.

    This single class replaces the complex multi-class system from v2
    with a simple, easy-to-maintain approach.
    """

    def __init__(self):
        """Initialize the configuration manager."""
        self.section_mapping = SECTION_MAPPING.copy()

    # =============================================================================
    # YAML Loading Methods (Feature 1)
    # =============================================================================

    def load_yaml(self, file_path: str) -> Tuple[Dict[str, Any], str]:
        """
        Load a YAML configuration file and determine its type.

        Args:
            file_path: Path to the YAML configuration file

        Returns:
            Tuple of (config_data, config_type) where:
            - config_data: Parsed YAML content as dictionary
            - config_type: Either 'base' or 'experiment'

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        # Validate file exists
        self._validate_file_path(file_path)

        # Load YAML content
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Determine config type
        config_type = self._get_config_type(config_data)

        return config_data, config_type

    def _get_config_type(self, config_data: Dict[str, Any]) -> str:
        """
        Determine if config is a base config or experiment config.

        Args:
            config_data: Parsed YAML configuration data

        Returns:
            'base' if no base_config attribute, 'experiment' if base_config exists
        """
        return 'experiment' if 'base_config' in config_data else 'base'

    def _validate_file_path(self, file_path: str) -> None:
        """
        Validate that a file path exists.

        Args:
            file_path: Path to validate

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")


    # =============================================================================
    # Section-to-Class Mapping Methods (Feature 3)
    # =============================================================================

    def get_class_from_section(self, section: str) -> Type:
        """
        Get the config class for a specific YAML section.

        Args:
            section: Name of the YAML section

        Returns:
            Config class for the section

        Raises:
            KeyError: If section is not in mapping
        """
        if section not in self.section_mapping:
            raise KeyError(f"Section '{section}' not found in mapping")

        return self.section_mapping[section]

    # =============================================================================
    # Config Processing Methods (Features 4 & 5)
    # =============================================================================

    def process_config(self, config_path: str, cli_overrides: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a configuration file (base or experiment).

        Automatically detects config type and processes accordingly:
        - Base configs: Creates instances with YAML overrides
        - Experiment configs: Loads base config first, then applies experiment overrides
        - CLI overrides: Applied last with highest priority

        Args:
            config_path: Path to config YAML file (base or experiment)
            cli_overrides: Optional list of CLI override strings in dot notation

        Returns:
            Dictionary containing config instances and metadata
        """
        # Load YAML to determine config type
        yaml_data, config_type = self.load_yaml(config_path)

        if config_type == 'base':
            configs = self._process_base_config(config_path, yaml_data)
        else:  # experiment
            configs = self._process_experiment_config(config_path, yaml_data)

        # Apply primary config to all other configs (propagate shared values)
        self._apply_primary_config_to_all(configs)

        # Apply CLI overrides if provided (highest priority)
        if cli_overrides:
            parsed_cli_overrides = self.parse_cli_overrides(cli_overrides)
            self.apply_cli_overrides(configs, parsed_cli_overrides)
            print("CLI overrides applied")

            # Re-apply primary config after CLI overrides to ensure consistency
            self._apply_primary_config_to_all(configs)

        # Add CLI overrides to metadata
        configs['cli_overrides'] = cli_overrides if cli_overrides else []

        return configs

    def process_base_config(self, base_config_path: str) -> Dict[str, Any]:
        """
        Process a base configuration file and create config instances.

        Deprecated: Use process_config() instead for automatic type detection.
        """
        yaml_data, _ = self.load_yaml(base_config_path)
        return self._process_base_config(base_config_path, yaml_data)

    def _process_base_config(self, base_config_path: str, yaml_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal method to process a base configuration file.

        Uses hybrid approach:
        1. Create config instances with defaults
        2. Apply YAML overrides to instances (including nested objects)

        Args:
            base_config_path: Path to base config YAML file
            yaml_data: Parsed YAML data

        Returns:
            Dictionary containing config instances and metadata
        """
        print(f"Base config overrides from: {base_config_path}")

        # Create config instances
        configs = {}

        # Create instance only for sections present in YAML
        for section in yaml_data.keys():
            # Validate section exists in mapping
            if section not in self.section_mapping:
                raise KeyError(f"Unknown section '{section}' in config file '{base_config_path}'. "
                             f"Available sections: {list(self.section_mapping.keys())}")

            config_class = self.section_mapping[section]

            # Create instance with defaults
            config_instance = config_class()

            # Apply YAML overrides
            print(f"\t{section}")
            self._apply_yaml_overrides(config_instance, yaml_data[section], indent_level=2)

            configs[section] = config_instance

        # Add metadata
        configs['config_paths'] = {'base': base_config_path}

        return configs

    def _process_experiment_config(self, experiment_config_path: str, yaml_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal method to process an experiment configuration file.

        Loads base config first, then applies experiment overrides.

        Args:
            experiment_config_path: Path to experiment config YAML file
            yaml_data: Parsed experiment YAML data

        Returns:
            Dictionary containing config instances and metadata
        """
        # Get base config path from experiment YAML
        if 'base_config' not in yaml_data:
            raise ValueError(f"Experiment config '{experiment_config_path}' missing 'base_config' attribute")

        base_config_path = yaml_data['base_config']

        # Assume path is relative to project root (current working directory)

        # Check if base config exists
        if not os.path.exists(base_config_path):
            raise FileNotFoundError(
                f"Experiment config '{experiment_config_path}' references base config "
                f"'{base_config_path}' which was not found (path assumed relative to project root)"
            )

        # Load base config first
        base_yaml_data, _ = self.load_yaml(base_config_path)
        configs = self._process_base_config(base_config_path, base_yaml_data)

        # Apply experiment overrides
        print(f"Experiment config overrides from: {experiment_config_path}")

        for section, section_overrides in yaml_data.items():
            # Skip base_config reference
            if section == 'base_config':
                continue

            # Validate section exists in mapping
            if section not in self.section_mapping:
                raise KeyError(f"Unknown section '{section}' in experiment config '{experiment_config_path}'. "
                             f"Available sections: {list(self.section_mapping.keys())}")

            # Apply overrides to existing config instance if it exists
            if section in configs:
                print(f"\t{section}")
                self._apply_yaml_overrides(configs[section], section_overrides, indent_level=2)
            else:
                print(f"\t{section}: creating new config instance for experiment override")
                config_class = self.section_mapping[section]
                config_instance = config_class()
                self._apply_yaml_overrides(config_instance, section_overrides, indent_level=2)
                configs[section] = config_instance

        # Update metadata to include both config paths
        configs['config_paths'] = {
            'base': base_config_path,
            'exp': experiment_config_path
        }

        return configs

    def _apply_yaml_overrides(self, config_instance: Any, yaml_overrides: Dict[str, Any], indent_level: int = 0) -> None:
        """
        Apply YAML override values to a config instance.

        Handles both simple attributes and nested objects using setattr.
        Creates new attributes if they don't exist.

        Args:
            config_instance: Config instance to apply overrides to
            yaml_overrides: Dictionary of override values from YAML
            indent_level: Indentation level for debug output
        """
        indent = "\t" * indent_level

        for key, value in yaml_overrides.items():
            if hasattr(config_instance, key):
                current_attr = getattr(config_instance, key)

                # Check for additive behavior (tags field)
                if self._apply_additive_override(config_instance, key, value, current_attr):
                    print(f"{indent}{key}: combining tags - {current_attr} + {value} → {getattr(config_instance, key)}")
                # If the current attribute is a dict and the override is a dict, merge them
                elif isinstance(current_attr, dict) and isinstance(value, dict):
                    print(f"{indent}{key}: merging dict with {len(value)} new entries")
                    current_attr.update(value)
                # If the current attribute is an object and the override is a dict, apply recursively
                elif hasattr(current_attr, '__dict__') and isinstance(value, dict):
                    print(f"{indent}{key}: applying overrides to nested object")
                    self._apply_yaml_overrides(current_attr, value, indent_level + 1)
                # If current attribute is None and override is dict, check if we should instantiate a nested object
                elif current_attr is None and isinstance(value, dict) and key in self.section_mapping:
                    nested_class = self.section_mapping[key]
                    nested_instance = nested_class()
                    print(f"{indent}{key}: instantiating nested {nested_class.__name__} object")
                    self._apply_yaml_overrides(nested_instance, value, indent_level + 1)
                    setattr(config_instance, key, nested_instance)
                # Otherwise, directly set the value
                else:
                    print(f"{indent}{key}: {current_attr} → {value}")
                    setattr(config_instance, key, value)
            else:
                # Attribute doesn't exist, create it
                print(f"{indent}{key}: <new> → {value}")
                setattr(config_instance, key, value)

    # =============================================================================
    # CLI Override Methods (Feature 6)
    # =============================================================================

    def parse_cli_overrides(self, cli_args: List[str]) -> Dict[str, Any]:
        """
        Parse CLI arguments in dot notation format into nested dictionary.

        Args:
            cli_args: List of CLI arguments like ["model.actor.n=2", "primary.seed=42"]

        Returns:
            Nested dictionary structure representing the overrides

        Raises:
            ValueError: If CLI argument format is invalid
        """
        result = {}

        for arg in cli_args:
            if '=' not in arg:
                raise ValueError(f"CLI argument '{arg}' has invalid format (missing '=')")

            key_path, value_str = arg.split('=', 1)  # Split only on first '='

            if not key_path:
                raise ValueError(f"CLI argument '{arg}' has invalid format (empty key)")

            # Infer type and parse dot notation
            typed_value = self._infer_type(value_str)
            parsed_override = self._parse_dot_notation(key_path, typed_value)

            # Merge into result dictionary
            self._merge_dicts(result, parsed_override)

        return result

    def apply_cli_overrides(self, configs: Dict[str, Any], cli_overrides: Dict[str, Any]) -> None:
        """
        Apply CLI overrides to existing config instances.

        Args:
            configs: Dictionary of config instances from process_config()
            cli_overrides: Parsed CLI overrides from parse_cli_overrides()
        """
        for section, section_overrides in cli_overrides.items():
            # Only apply to sections that exist in configs
            if section in configs and section in self.section_mapping:
                print(f"\tCLI overrides for {section}")
                self._apply_yaml_overrides(configs[section], section_overrides, indent_level=2)

    def _infer_type(self, value_str: str) -> Any:
        """
        Infer the appropriate Python type from a string value.

        Args:
            value_str: String value from CLI argument

        Returns:
            Converted value with appropriate type
        """
        # Handle empty string
        if value_str == "":
            return ""

        # Check for comma-separated list
        if ',' in value_str:
            return [item.strip() for item in value_str.split(',')]

        # Check for boolean values
        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true'

        # Try to convert to int
        try:
            return int(value_str)
        except ValueError:
            pass

        # Try to convert to float
        try:
            return float(value_str)
        except ValueError:
            pass

        # Default to string
        return value_str

    def _parse_dot_notation(self, key_path: str, value: Any) -> Dict[str, Any]:
        """
        Convert dot notation key path into nested dictionary structure.

        Args:
            key_path: Dot-separated key path like "model.actor.n"
            value: Value to set at the key path

        Returns:
            Nested dictionary structure
        """
        keys = key_path.split('.')
        result = {}
        current = result

        # Build nested structure
        for key in keys[:-1]:
            current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

        return result

    def _merge_dicts(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Merge source dictionary into target dictionary recursively.

        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                self._merge_dicts(target[key], value)
            else:
                # Set or override value
                target[key] = value

    # =============================================================================
    # Tags Additive Behavior Methods (Feature 7)
    # =============================================================================

    def _combine_tags(self, existing_tags, new_tags):
        """
        Combine two tag lists additively, removing duplicates while preserving order.

        Args:
            existing_tags: Current tags list (can be None, list, or other type)
            new_tags: New tags to add (can be None, list, or other type)

        Returns:
            List of combined tags with duplicates removed, preserving order
        """
        # Convert to lists, handling None and non-list values
        existing_list = []
        if isinstance(existing_tags, list):
            existing_list = existing_tags.copy()
        elif existing_tags is not None:
            # Handle non-list existing values gracefully
            pass

        new_list = []
        if isinstance(new_tags, list):
            new_list = new_tags.copy()
        elif new_tags is not None:
            # Handle non-list new values gracefully
            pass

        # Combine while preserving order and removing duplicates
        result = existing_list.copy()
        for tag in new_list:
            if tag not in result:
                result.append(tag)

        return result

    def _apply_additive_override(self, config_instance, key, value, current_attr):
        """
        Apply additive behavior for tags field, normal override for others.

        Args:
            config_instance: Config instance to modify
            key: Attribute name
            value: New value to apply
            current_attr: Current attribute value

        Returns:
            True if additive behavior was applied, False if normal override should be used
        """
        # Only apply additive behavior to 'tags' field
        if key == 'tags' and isinstance(current_attr, list) and isinstance(value, list):
            combined_tags = self._combine_tags(current_attr, value)
            setattr(config_instance, key, combined_tags)
            return True

        # For all other fields, use normal override behavior
        return False

    # =============================================================================
    # Primary Config Application Methods
    # =============================================================================

    def _apply_primary_config_to_all(self, configs: Dict[str, Any]) -> None:
        """
        Apply primary config to all other config instances.

        This propagates shared values from the primary config to dependent configs,
        similar to ConfigManagerV2's approach.

        Args:
            configs: Dictionary of config instances from process_config()
        """
        if 'primary' not in configs:
            return

        primary_cfg = configs['primary']
        print("Applying primary config to dependent configs")

        # Apply primary config to configs that have apply_primary_cfg method
        for section_name, config_instance in configs.items():
            # Skip non-config entries and primary itself
            if section_name in ['primary', 'config_paths', 'cli_overrides']:
                continue

            # Apply primary config if method exists
            if hasattr(config_instance, 'apply_primary_cfg'):
                print(f"\tApplying to {section_name}")
                config_instance.apply_primary_cfg(primary_cfg)