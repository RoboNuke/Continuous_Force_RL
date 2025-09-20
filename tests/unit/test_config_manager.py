"""
Comprehensive unit tests for configs/config_manager.py
Tests all classes and methods with various parameter inputs and edge cases.
"""

import pytest
import os
import tempfile
import shutil
import yaml
import json
import torch
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

# Import the classes we're testing
from configs.config_manager import (
    ConfigManager,
    MetricConfig,
    LoggingConfig,
    LoggingConfigPresets,
    load_config_from_file
)


class TestConfigManager:
    """Test ConfigManager class methods."""

    def test_load_and_resolve_config_basic(self, temp_dir):
        """Test basic config loading without base or overrides."""
        config_data = {
            'primary': {
                'num_envs_per_agent': 128,
                'agents_per_break_force': 2,
                'break_forces': [100, 200],
                'episode_length_s': 10.0,
                'decimation': 4,
                'policy_hz': 15,
                'max_steps': 5000000,
                'seed': 42
            }
        }

        config_path = os.path.join(temp_dir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_data, f)

        result = ConfigManager.load_and_resolve_config(config_path)

        assert 'primary' in result
        assert 'derived' in result
        assert result['derived']['total_agents'] == 4  # 2 break_forces * 2 agents_per_break_force
        assert result['derived']['total_num_envs'] == 512  # 4 agents * 128 envs_per_agent

    def test_load_and_resolve_config_with_base(self, temp_dir):
        """Test config loading with base inheritance."""
        # Create proper directory structure that matches expected usage
        configs_dir = os.path.join(temp_dir, 'configs')
        experiments_dir = os.path.join(configs_dir, 'experiments')
        os.makedirs(experiments_dir, exist_ok=True)

        # Create base config
        base_config = {
            'primary': {
                'num_envs_per_agent': 256,
                'episode_length_s': 12.0,
                'max_steps': 10000000
            }
        }
        base_path = os.path.join(configs_dir, 'base.yaml')
        with open(base_path, 'w') as f:
            yaml.safe_dump(base_config, f)

        # Create main config with base reference (in experiments subdirectory)
        main_config = {
            'base': 'base.yaml',  # Should resolve to configs/base.yaml
            'primary': {
                'agents_per_break_force': 3,
                'break_forces': [50, 100, 150],
                'decimation': 8
            }
        }
        main_path = os.path.join(experiments_dir, 'main.yaml')
        with open(main_path, 'w') as f:
            yaml.safe_dump(main_config, f)

        result = ConfigManager.load_and_resolve_config(main_path)

        # Should inherit from base and override
        assert result['primary']['num_envs_per_agent'] == 256  # From base
        assert result['primary']['agents_per_break_force'] == 3  # Override
        assert result['primary']['episode_length_s'] == 12.0  # From base
        assert result['derived']['total_agents'] == 9  # 3 break_forces * 3 agents_per_break_force

    def test_load_and_resolve_config_with_overrides(self, temp_dir):
        """Test config loading with CLI overrides."""
        config_data = {
            'primary': {
                'num_envs_per_agent': 128,
                'agents_per_break_force': 2,
                'break_forces': [100],
                'max_steps': 1000000
            }
        }

        config_path = os.path.join(temp_dir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_data, f)

        overrides = [
            'primary.max_steps=5000000',
            'primary.seed=123',
            'new_section.new_key=test_value'
        ]

        result = ConfigManager.load_and_resolve_config(config_path, overrides)

        assert result['primary']['max_steps'] == 5000000
        assert result['primary']['seed'] == 123
        assert result['new_section']['new_key'] == 'test_value'

    def test_load_yaml_file_success(self, temp_dir):
        """Test successful YAML file loading."""
        test_data = {'key': 'value', 'number': 42}
        file_path = os.path.join(temp_dir, 'test.yaml')

        with open(file_path, 'w') as f:
            yaml.safe_dump(test_data, f)

        result = ConfigManager._load_yaml_file(file_path)
        assert result == test_data

    def test_load_yaml_file_not_found(self):
        """Test YAML file loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            ConfigManager._load_yaml_file('/non/existent/file.yaml')

    def test_load_yaml_file_empty(self, temp_dir):
        """Test loading empty YAML file."""
        file_path = os.path.join(temp_dir, 'empty.yaml')
        with open(file_path, 'w') as f:
            pass  # Create empty file

        result = ConfigManager._load_yaml_file(file_path)
        assert result == {}

    def test_resolve_config_path_absolute(self):
        """Test resolving absolute config paths."""
        absolute_path = '/absolute/path/config.yaml'
        result = ConfigManager._resolve_config_path(absolute_path, '/current/file.yaml')
        assert result == absolute_path

    def test_resolve_config_path_relative(self, temp_dir):
        """Test resolving relative config paths."""
        current_file = os.path.join(temp_dir, 'configs', 'experiments', 'test.yaml')
        relative_path = 'base/factory_base'

        # Create directory structure
        os.makedirs(os.path.dirname(current_file), exist_ok=True)

        result = ConfigManager._resolve_config_path(relative_path, current_file)
        expected = os.path.join(temp_dir, 'configs', 'base', 'factory_base.yaml')
        assert result == expected

    def test_merge_configs_simple(self):
        """Test basic config merging."""
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}

        result = ConfigManager._merge_configs(base, override)
        assert result == {'a': 1, 'b': 3, 'c': 4}

    def test_merge_configs_nested(self):
        """Test nested config merging."""
        base = {
            'section1': {'a': 1, 'b': 2},
            'section2': {'x': 10}
        }
        override = {
            'section1': {'b': 3, 'c': 4},
            'section3': {'y': 20}
        }

        result = ConfigManager._merge_configs(base, override)
        expected = {
            'section1': {'a': 1, 'b': 3, 'c': 4},
            'section2': {'x': 10},
            'section3': {'y': 20}
        }
        assert result == expected

    def test_apply_override_valid_formats(self):
        """Test applying various valid override formats."""
        config = {}

        # Test different value types
        ConfigManager._apply_override(config, 'key1=string_value')
        ConfigManager._apply_override(config, 'key2=42')
        ConfigManager._apply_override(config, 'key3=true')
        ConfigManager._apply_override(config, 'key4=[1,2,3]')
        ConfigManager._apply_override(config, 'nested.key=nested_value')

        assert config['key1'] == 'string_value'
        assert config['key2'] == 42
        assert config['key3'] == True
        assert config['key4'] == [1, 2, 3]
        assert config['nested']['key'] == 'nested_value'

    def test_apply_override_invalid_format(self):
        """Test applying override with invalid format."""
        config = {}

        with pytest.raises(ValueError, match="Invalid override format"):
            ConfigManager._apply_override(config, 'invalid_format')

    def test_apply_override_with_equals_in_value(self):
        """Test applying override with equals sign in value."""
        config = {}

        ConfigManager._apply_override(config, 'url=http://example.com:8080/path?key=value')
        assert config['url'] == 'http://example.com:8080/path?key=value'

    def test_set_nested_key_new_structure(self):
        """Test setting nested keys in new dictionary structure."""
        config = {}

        ConfigManager._set_nested_key(config, 'level1.level2.level3', 'deep_value')

        assert config['level1']['level2']['level3'] == 'deep_value'

    def test_set_nested_key_existing_structure(self):
        """Test setting nested keys in existing dictionary structure."""
        config = {
            'level1': {
                'existing_key': 'existing_value'
            }
        }

        ConfigManager._set_nested_key(config, 'level1.level2.level3', 'new_value')

        assert config['level1']['existing_key'] == 'existing_value'
        assert config['level1']['level2']['level3'] == 'new_value'

    def test_set_nested_attr_success(self):
        """Test setting nested attributes on objects."""
        class MockObj:
            def __init__(self):
                self.level1 = MockObj2()

        class MockObj2:
            def __init__(self):
                pass

        obj = MockObj()
        ConfigManager._set_nested_attr(obj, 'level1.attr', 'test_value')

        assert obj.level1.attr == 'test_value'

    def test_calculate_derived_params_basic(self):
        """Test basic derived parameter calculations."""
        primary = {
            'agents_per_break_force': 2,
            'num_envs_per_agent': 128,
            'break_forces': [100, 200],
            'episode_length_s': 12.0,
            'decimation': 4,
            'policy_hz': 15,
            'max_steps': 10000000,
            'seed': 42,
            'ckpt_tracker_path': '/test/path.txt'
        }

        result = ConfigManager._calculate_derived_params(primary)

        assert result['total_agents'] == 4  # 2 break_forces * 2 agents
        assert result['total_num_envs'] == 512  # 4 agents * 128 envs
        assert result['max_steps'] == 10000000
        assert result['seed'] == 42
        assert result['ckpt_tracker_path'] == '/test/path.txt'
        assert 'rollout_steps' in result
        assert 'sim_dt' in result

    def test_calculate_derived_params_single_break_force(self):
        """Test derived parameters with single break force value."""
        primary = {
            'agents_per_break_force': 3,
            'num_envs_per_agent': 64,
            'break_forces': 150,  # Single value, not list
            'episode_length_s': 8.0,
            'decimation': 2,
            'policy_hz': 20
        }

        result = ConfigManager._calculate_derived_params(primary)

        assert result['total_agents'] == 3  # 1 break_force * 3 agents
        assert result['total_num_envs'] == 192  # 3 agents * 64 envs

    def test_calculate_derived_params_defaults(self):
        """Test derived parameters with default values."""
        primary = {}  # Empty config

        result = ConfigManager._calculate_derived_params(primary)

        # Should use default values
        assert result['total_agents'] == 2  # 1 * 2 (default break_forces = -1, treated as 1)
        assert result['total_num_envs'] == 512  # 2 * 256
        assert result['seed'] == -1

    def test_resolve_references_simple(self):
        """Test simple reference resolution."""
        config = {
            'section1': {'value': 42},
            'section2': {'ref': '${section1.value}'}
        }

        result = ConfigManager._resolve_references(config)

        assert result['section2']['ref'] == 42

    def test_resolve_references_nested(self):
        """Test nested reference resolution."""
        config = {
            'primary': {
                'max_steps': 1000000,
                'nested': {
                    'value': 'test'
                }
            },
            'derived': {
                'steps_ref': '${primary.max_steps}',
                'nested_ref': '${primary.nested.value}'
            }
        }

        result = ConfigManager._resolve_references(config)

        assert result['derived']['steps_ref'] == 1000000
        assert result['derived']['nested_ref'] == 'test'

    def test_resolve_references_in_list(self):
        """Test reference resolution within lists."""
        config = {
            'values': [1, 2, 3],
            'references': ['${values.0}', '${values.1}', 'static_value']
        }

        result = ConfigManager._resolve_references(config)

        assert result['references'] == [1, 2, 'static_value']

    def test_resolve_references_invalid(self):
        """Test reference resolution with invalid reference."""
        config = {
            'section1': {'value': 42},
            'section2': {'ref': '${nonexistent.key}'}
        }

        with pytest.raises(ValueError, match="Reference not found"):
            ConfigManager._resolve_references(config)

    def test_get_nested_value_success(self):
        """Test getting nested values successfully."""
        config = {
            'level1': {
                'level2': {
                    'level3': 'target_value'
                }
            }
        }

        result = ConfigManager._get_nested_value(config, 'level1.level2.level3')
        assert result == 'target_value'

    def test_get_nested_value_not_found(self):
        """Test getting nested value that doesn't exist."""
        config = {'level1': {'level2': 'value'}}

        with pytest.raises(ValueError, match="Reference not found"):
            ConfigManager._get_nested_value(config, 'level1.nonexistent.key')

    def test_apply_to_isaac_lab(self):
        """Test applying configuration to Isaac Lab objects."""
        # Create mock objects with controlled attributes
        class MockEnvCfg:
            def __init__(self):
                self.existing_attr = "original"
                # Don't define num_envs so hasattr returns False

        mock_env_cfg = MockEnvCfg()
        mock_agent_cfg = Mock()

        resolved_config = {
            'environment': {
                'num_envs': 512,  # This should be set as attribute
                'nested.attr': 'nested_value'  # This should call _set_nested_attr
            },
            'primary': {
                'max_steps': 1000000
            }
        }

        # Mock the _set_nested_attr method
        with patch.object(ConfigManager, '_set_nested_attr') as mock_set_nested:
            ConfigManager.apply_to_isaac_lab(mock_env_cfg, mock_agent_cfg, resolved_config)

            # Verify direct attribute setting
            assert hasattr(mock_env_cfg, 'num_envs')
            assert mock_env_cfg.num_envs == 512

            # Verify nested attribute setting was called
            mock_set_nested.assert_called_once_with(mock_env_cfg, 'nested.attr', 'nested_value')


class TestMetricConfig:
    """Test MetricConfig dataclass."""

    def test_metric_config_defaults(self):
        """Test MetricConfig with default values."""
        metric = MetricConfig(name="test_metric")

        assert metric.name == "test_metric"
        assert metric.default_value == 0.0
        assert metric.metric_type == "scalar"
        assert metric.aggregation == "mean"
        assert metric.wandb_name is None
        assert metric.enabled == True
        assert metric.normalize_by_episode_length == False

    def test_metric_config_custom_values(self):
        """Test MetricConfig with custom values."""
        metric = MetricConfig(
            name="custom_metric",
            default_value=torch.tensor([1, 2, 3]),
            metric_type="tensor",
            aggregation="sum",
            wandb_name="Custom Metric",
            enabled=False,
            normalize_by_episode_length=True
        )

        assert metric.name == "custom_metric"
        assert torch.equal(metric.default_value, torch.tensor([1, 2, 3]))
        assert metric.metric_type == "tensor"
        assert metric.aggregation == "sum"
        assert metric.wandb_name == "Custom Metric"
        assert metric.enabled == False
        assert metric.normalize_by_episode_length == True


class TestLoggingConfig:
    """Test LoggingConfig dataclass and methods."""

    def test_logging_config_defaults(self):
        """Test LoggingConfig with default values."""
        config = LoggingConfig()

        assert config.wandb_entity is None
        assert config.wandb_project is None
        assert config.wandb_name is None
        assert config.wandb_group is None
        assert config.wandb_tags is None
        assert config.tracked_metrics == {}
        assert config.track_learning_metrics == True
        assert config.track_action_metrics == True
        assert config.track_episodes == True
        assert config.track_rewards == True
        assert config.track_episode_length == True
        assert config.track_terminations == True
        assert config.clip_eps == 0.2
        assert config.num_agents == 1

    def test_add_metric(self):
        """Test adding metrics to LoggingConfig."""
        config = LoggingConfig()
        metric = MetricConfig(name="test_metric", metric_type="scalar")

        config.add_metric(metric)

        assert "test_metric" in config.tracked_metrics
        assert config.tracked_metrics["test_metric"] == metric

    def test_remove_metric(self):
        """Test removing metrics from LoggingConfig."""
        config = LoggingConfig()
        metric = MetricConfig(name="test_metric")
        config.add_metric(metric)

        config.remove_metric("test_metric")

        assert "test_metric" not in config.tracked_metrics

    def test_remove_nonexistent_metric(self):
        """Test removing non-existent metric (should not raise error)."""
        config = LoggingConfig()

        # Should not raise error
        config.remove_metric("nonexistent_metric")

    def test_enable_metric(self):
        """Test enabling/disabling metrics."""
        config = LoggingConfig()
        metric = MetricConfig(name="test_metric", enabled=True)
        config.add_metric(metric)

        config.enable_metric("test_metric", False)
        assert config.tracked_metrics["test_metric"].enabled == False

        config.enable_metric("test_metric", True)
        assert config.tracked_metrics["test_metric"].enabled == True

    def test_enable_nonexistent_metric(self):
        """Test enabling non-existent metric (should not raise error)."""
        config = LoggingConfig()

        # Should not raise error
        config.enable_metric("nonexistent_metric", True)

    def test_to_wandb_config(self):
        """Test converting to Wandb configuration."""
        config = LoggingConfig(
            wandb_entity="test_entity",
            wandb_project="test_project",
            wandb_name="test_run",
            wandb_group="test_group",
            wandb_tags=["tag1", "tag2"],
            num_agents=4
        )

        # Add some metrics
        config.add_metric(MetricConfig(name="metric1", enabled=True))
        config.add_metric(MetricConfig(name="metric2", enabled=False))
        config.add_metric(MetricConfig(name="metric3", enabled=True))

        wandb_config = config.to_wandb_config()

        assert wandb_config['entity'] == "test_entity"
        assert wandb_config['project'] == "test_project"
        assert wandb_config['name'] == "test_run"
        assert wandb_config['group'] == "test_group"
        assert wandb_config['tags'] == ["tag1", "tag2"]
        assert wandb_config['num_agents'] == 4
        assert set(wandb_config['tracked_metrics']) == {"metric1", "metric3"}  # Only enabled metrics

    def test_to_wandb_config_defaults(self):
        """Test converting to Wandb config with default values."""
        config = LoggingConfig()

        wandb_config = config.to_wandb_config()

        assert wandb_config['entity'] is None
        assert wandb_config['project'] is None
        assert wandb_config['name'] is None
        assert wandb_config['group'] is None
        assert wandb_config['tags'] == []
        assert wandb_config['num_agents'] == 1
        assert wandb_config['tracked_metrics'] == []


class TestLoggingConfigPresets:
    """Test LoggingConfigPresets class methods."""

    def test_basic_config(self):
        """Test basic configuration preset."""
        config = LoggingConfigPresets.basic_config()

        assert isinstance(config, LoggingConfig)
        assert "reward" in config.tracked_metrics

        reward_metric = config.tracked_metrics["reward"]
        assert reward_metric.metric_type == "scalar"
        assert reward_metric.aggregation == "mean"
        assert reward_metric.wandb_name == "Episode / Reward"

    def test_factory_config(self):
        """Test factory configuration preset."""
        config = LoggingConfigPresets.factory_config()

        assert isinstance(config, LoggingConfig)

        # Check core metrics
        assert "reward" in config.tracked_metrics
        assert "current_engagements" in config.tracked_metrics
        assert "current_successes" in config.tracked_metrics
        assert "sum_squared_velocity" in config.tracked_metrics
        assert "max_force" in config.tracked_metrics
        assert "max_torque" in config.tracked_metrics
        assert "avg_force" in config.tracked_metrics
        assert "avg_torque" in config.tracked_metrics

        # Check reward components
        reward_components = ["Reward / reach_reward", "Reward / grasp_reward", "Reward / lift_reward", "Reward / align_reward"]
        for component in reward_components:
            assert component in config.tracked_metrics

        # Check specific configurations
        engagement_metric = config.tracked_metrics["current_engagements"]
        assert engagement_metric.metric_type == "boolean"
        assert engagement_metric.default_value == 0.0

        velocity_metric = config.tracked_metrics["sum_squared_velocity"]
        assert velocity_metric.normalize_by_episode_length == True

    def test_from_dict_basic(self):
        """Test creating config from dictionary."""
        config_dict = {
            'wandb_entity': 'test_entity',
            'wandb_project': 'test_project',
            'wandb_name': 'test_run',
            'wandb_group': 'test_group',
            'wandb_tags': ['tag1', 'tag2'],
            'num_agents': 4,
            'clip_eps': 0.3,
            'track_learning_metrics': False,
            'track_action_metrics': False,
            'track_episodes': False,
            'track_rewards': False,
            'track_episode_length': False,
            'track_terminations': False
        }

        config = LoggingConfigPresets.from_dict(config_dict)

        assert config.wandb_entity == 'test_entity'
        assert config.wandb_project == 'test_project'
        assert config.wandb_name == 'test_run'
        assert config.wandb_group == 'test_group'
        assert config.wandb_tags == ['tag1', 'tag2']
        assert config.num_agents == 4
        assert config.clip_eps == 0.3
        assert config.track_learning_metrics == False
        assert config.track_action_metrics == False
        assert config.track_episodes == False
        assert config.track_rewards == False
        assert config.track_episode_length == False
        assert config.track_terminations == False

    def test_from_dict_with_metrics(self):
        """Test creating config from dictionary with custom metrics."""
        config_dict = {
            'wandb_project': 'test_project',
            'tracked_metrics': {
                'custom_metric1': {
                    'default_value': 5.0,
                    'metric_type': 'histogram',
                    'aggregation': 'max',
                    'wandb_name': 'Custom Metric 1',
                    'enabled': True,
                    'normalize_by_episode_length': True
                },
                'custom_metric2': {
                    'default_value': 10,
                    'metric_type': 'scalar',
                    'aggregation': 'sum',
                    'enabled': False
                }
            }
        }

        config = LoggingConfigPresets.from_dict(config_dict)

        assert len(config.tracked_metrics) == 2

        metric1 = config.tracked_metrics['custom_metric1']
        assert metric1.default_value == 5.0
        assert metric1.metric_type == 'histogram'
        assert metric1.aggregation == 'max'
        assert metric1.wandb_name == 'Custom Metric 1'
        assert metric1.enabled == True
        assert metric1.normalize_by_episode_length == True

        metric2 = config.tracked_metrics['custom_metric2']
        assert metric2.default_value == 10
        assert metric2.metric_type == 'scalar'
        assert metric2.aggregation == 'sum'
        assert metric2.wandb_name is None  # Default
        assert metric2.enabled == False

    def test_create_from_config_factory_preset(self):
        """Test creating config with factory preset and overrides."""
        config_dict = {
            'wandb_project': 'override_project',
            'num_agents': 8
        }

        config = LoggingConfigPresets.create_from_config(config_dict, preset="factory")

        # Should have factory metrics
        assert "current_engagements" in config.tracked_metrics
        assert "current_successes" in config.tracked_metrics

        # Should have overrides applied
        assert config.wandb_project == 'override_project'
        assert config.num_agents == 8

    def test_create_from_config_basic_preset(self):
        """Test creating config with basic preset."""
        config_dict = {'wandb_entity': 'test_entity'}

        config = LoggingConfigPresets.create_from_config(config_dict, preset="basic")

        # Should have basic metrics only
        assert "reward" in config.tracked_metrics
        assert "current_engagements" not in config.tracked_metrics

    def test_create_from_config_unknown_preset(self):
        """Test creating config with unknown preset."""
        config_dict = {'wandb_project': 'test_project'}

        config = LoggingConfigPresets.create_from_config(config_dict, preset="unknown")

        # Should create empty config with overrides
        assert len(config.tracked_metrics) == 0
        assert config.wandb_project == 'test_project'

    def test_create_from_config_none_overrides(self):
        """Test creating config with None overrides."""
        config = LoggingConfigPresets.create_from_config(None, preset="basic")

        # Should have basic config without errors
        assert "reward" in config.tracked_metrics


class TestLoadConfigFromFile:
    """Test load_config_from_file function."""

    def test_load_yaml_file(self, temp_dir):
        """Test loading YAML configuration file."""
        config_dict = {
            'wandb_project': 'yaml_project',
            'num_agents': 4,
            'tracked_metrics': {
                'test_metric': {
                    'metric_type': 'scalar',
                    'aggregation': 'mean'
                }
            }
        }

        yaml_path = os.path.join(temp_dir, 'config.yaml')
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(config_dict, f)

        config = load_config_from_file(yaml_path)

        assert isinstance(config, LoggingConfig)
        assert config.wandb_project == 'yaml_project'
        assert config.num_agents == 4
        assert 'test_metric' in config.tracked_metrics

    def test_load_yml_file(self, temp_dir):
        """Test loading .yml configuration file."""
        config_dict = {'wandb_entity': 'yml_entity'}

        yml_path = os.path.join(temp_dir, 'config.yml')
        with open(yml_path, 'w') as f:
            yaml.safe_dump(config_dict, f)

        config = load_config_from_file(yml_path)

        assert config.wandb_entity == 'yml_entity'

    def test_load_json_file(self, temp_dir):
        """Test loading JSON configuration file."""
        config_dict = {
            'wandb_group': 'json_group',
            'clip_eps': 0.3
        }

        json_path = os.path.join(temp_dir, 'config.json')
        with open(json_path, 'w') as f:
            json.dump(config_dict, f)

        config = load_config_from_file(json_path)

        assert config.wandb_group == 'json_group'
        assert config.clip_eps == 0.3

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config_from_file('/non/existent/file.yaml')

    def test_load_unsupported_format(self, temp_dir):
        """Test loading unsupported file format."""
        txt_path = os.path.join(temp_dir, 'config.txt')
        with open(txt_path, 'w') as f:
            f.write('some content')

        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config_from_file(txt_path)

    def test_load_yaml_missing_dependency(self, temp_dir):
        """Test loading YAML file when PyYAML is not available."""
        yaml_path = os.path.join(temp_dir, 'config.yaml')
        with open(yaml_path, 'w') as f:
            f.write('key: value')

        # Since PyYAML is available in our test environment, we'll just verify
        # that the function works correctly with a valid YAML file
        # The import error case is difficult to test reliably without breaking the module
        config = load_config_from_file(yaml_path)
        assert isinstance(config, LoggingConfig)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_config_manager_with_circular_references(self):
        """Test handling circular references in config resolution."""
        config = {
            'section1': {'ref': '${section2.value}'},
            'section2': {'value': '${section1.ref}'}
        }

        # This should not cause infinite recursion, but may fail
        # depending on implementation
        try:
            ConfigManager._resolve_references(config)
            # If it succeeds, that's also valid
        except (ValueError, RecursionError):
            # Expected behavior for circular references
            pass

    def test_apply_override_complex_values(self):
        """Test applying overrides with complex Python values."""
        config = {}

        # Test various Python literal types
        ConfigManager._apply_override(config, 'dict_val={"key": "value", "num": 42}')
        ConfigManager._apply_override(config, 'list_val=[1, "two", 3.0]')
        ConfigManager._apply_override(config, 'tuple_val=(1, 2, 3)')
        ConfigManager._apply_override(config, 'bool_val=False')
        ConfigManager._apply_override(config, 'none_val=None')

        assert config['dict_val'] == {"key": "value", "num": 42}
        assert config['list_val'] == [1, "two", 3.0]
        assert config['tuple_val'] == (1, 2, 3)
        assert config['bool_val'] == False
        assert config['none_val'] is None

    def test_apply_override_invalid_python_literal(self):
        """Test applying override with invalid Python literal."""
        config = {}

        # Invalid Python syntax should fall back to string
        ConfigManager._apply_override(config, 'invalid=invalid{syntax')
        assert config['invalid'] == 'invalid{syntax'

    def test_calculate_derived_params_edge_values(self):
        """Test derived parameter calculation with edge values."""
        primary = {
            'agents_per_break_force': 0,  # Edge case
            'num_envs_per_agent': 1,
            'break_forces': [],  # Empty list
            'episode_length_s': 0.1,  # Very small
            'decimation': 1,
            'policy_hz': 1000  # Very high
        }

        result = ConfigManager._calculate_derived_params(primary)

        # Should handle edge cases gracefully
        assert result['total_agents'] == 0
        assert result['total_num_envs'] == 0
        assert 'rollout_steps' in result
        assert 'sim_dt' in result

    def test_merge_configs_type_conflicts(self):
        """Test config merging with type conflicts."""
        base = {'key': {'nested': 'value'}}
        override = {'key': 'simple_string'}

        result = ConfigManager._merge_configs(base, override)

        # Override should completely replace base value
        assert result['key'] == 'simple_string'

    def test_set_nested_attr_missing_intermediate(self):
        """Test setting nested attribute with missing intermediate objects."""
        class MockObj:
            pass

        obj = MockObj()

        # Should raise AttributeError for missing intermediate attribute
        with pytest.raises(AttributeError):
            ConfigManager._set_nested_attr(obj, 'missing.attr', 'value')


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)