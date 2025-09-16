"""
Unit tests for logging configuration system.
"""

import pytest
import torch
from pathlib import Path
import tempfile
import os

from wrappers.logging.logging_config import (
    MetricConfig, LoggingConfig, LoggingConfigPresets,
    load_config_from_file
)


class TestMetricConfig:
    """Test MetricConfig dataclass."""

    def test_default_values(self):
        """Test default MetricConfig values."""
        config = MetricConfig(name="test_metric")

        assert config.name == "test_metric"
        assert config.default_value == 0.0
        assert config.metric_type == "scalar"
        assert config.aggregation == "mean"
        assert config.wandb_name is None
        assert config.enabled is True
        assert config.normalize_by_episode_length is False

    def test_custom_values(self):
        """Test MetricConfig with custom values."""
        config = MetricConfig(
            name="custom_metric",
            default_value=5.0,
            metric_type="boolean",
            aggregation="sum",
            wandb_name="Custom Metric",
            enabled=False,
            normalize_by_episode_length=True
        )

        assert config.name == "custom_metric"
        assert config.default_value == 5.0
        assert config.metric_type == "boolean"
        assert config.aggregation == "sum"
        assert config.wandb_name == "Custom Metric"
        assert config.enabled is False
        assert config.normalize_by_episode_length is True


class TestLoggingConfig:
    """Test LoggingConfig class."""

    def test_default_values(self):
        """Test default LoggingConfig values."""
        config = LoggingConfig()

        assert config.wandb_entity is None
        assert config.wandb_project is None
        assert config.wandb_name is None
        assert config.wandb_group is None
        assert config.wandb_tags is None
        assert config.tracked_metrics == {}
        assert config.track_learning_metrics is True
        assert config.track_action_metrics is True
        assert config.track_episodes is True
        assert config.track_rewards is True
        assert config.track_episode_length is True
        assert config.track_terminations is True
        assert config.clip_eps == 0.2
        assert config.num_agents == 1

    def test_add_metric(self):
        """Test adding metrics to config."""
        config = LoggingConfig()
        metric = MetricConfig(name="test_metric")

        config.add_metric(metric)

        assert "test_metric" in config.tracked_metrics
        assert config.tracked_metrics["test_metric"] == metric

    def test_remove_metric(self):
        """Test removing metrics from config."""
        config = LoggingConfig()
        metric = MetricConfig(name="test_metric")
        config.add_metric(metric)

        config.remove_metric("test_metric")

        assert "test_metric" not in config.tracked_metrics

    def test_remove_nonexistent_metric(self):
        """Test removing non-existent metric doesn't raise error."""
        config = LoggingConfig()

        # Should not raise error
        config.remove_metric("nonexistent_metric")

    def test_enable_disable_metric(self):
        """Test enabling/disabling metrics."""
        config = LoggingConfig()
        metric = MetricConfig(name="test_metric")
        config.add_metric(metric)

        config.enable_metric("test_metric", False)
        assert config.tracked_metrics["test_metric"].enabled is False

        config.enable_metric("test_metric", True)
        assert config.tracked_metrics["test_metric"].enabled is True

    def test_enable_nonexistent_metric(self):
        """Test enabling non-existent metric doesn't raise error."""
        config = LoggingConfig()

        # Should not raise error
        config.enable_metric("nonexistent_metric", True)

    def test_to_wandb_config(self):
        """Test conversion to wandb config."""
        config = LoggingConfig()
        config.wandb_entity = "test_entity"
        config.wandb_project = "test_project"
        config.wandb_name = "test_name"
        config.wandb_group = "test_group"
        config.wandb_tags = ["tag1", "tag2"]
        config.num_agents = 2

        metric1 = MetricConfig(name="metric1", enabled=True)
        metric2 = MetricConfig(name="metric2", enabled=False)
        config.add_metric(metric1)
        config.add_metric(metric2)

        wandb_config = config.to_wandb_config()

        assert wandb_config['entity'] == "test_entity"
        assert wandb_config['project'] == "test_project"
        assert wandb_config['name'] == "test_name"
        assert wandb_config['group'] == "test_group"
        assert wandb_config['tags'] == ["tag1", "tag2"]
        assert wandb_config['num_agents'] == 2
        assert wandb_config['tracked_metrics'] == ["metric1"]  # Only enabled metrics


class TestLoggingConfigPresets:
    """Test predefined logging configurations."""

    def test_basic_config(self):
        """Test basic configuration preset."""
        config = LoggingConfigPresets.basic_config()

        assert isinstance(config, LoggingConfig)
        assert "reward" in config.tracked_metrics
        assert config.tracked_metrics["reward"].wandb_name == "Episode / Reward"

    def test_factory_config(self):
        """Test factory configuration preset."""
        config = LoggingConfigPresets.factory_config()

        assert isinstance(config, LoggingConfig)

        # Check factory-specific metrics
        expected_metrics = [
            "reward", "current_engagements", "current_successes",
            "sum_squared_velocity", "max_force", "max_torque",
            "avg_force", "avg_torque"
        ]

        for metric in expected_metrics:
            assert metric in config.tracked_metrics
            assert config.tracked_metrics[metric].enabled is True

    def test_locomotion_config(self):
        """Test locomotion configuration preset."""
        config = LoggingConfigPresets.locomotion_config()

        assert isinstance(config, LoggingConfig)

        # Check locomotion-specific metrics
        expected_metrics = [
            "reward", "distance_traveled", "velocity", "energy_consumption"
        ]

        for metric in expected_metrics:
            assert metric in config.tracked_metrics
            assert config.tracked_metrics[metric].enabled is True

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'wandb_entity': 'test_entity',
            'wandb_project': 'test_project',
            'num_agents': 3,
            'clip_eps': 0.3,
            'track_learning_metrics': False,
            'tracked_metrics': {
                'test_metric': {
                    'default_value': 1.0,
                    'metric_type': 'boolean',
                    'aggregation': 'sum',
                    'wandb_name': 'Test Metric',
                    'enabled': False
                }
            }
        }

        config = LoggingConfigPresets.from_dict(config_dict)

        assert config.wandb_entity == 'test_entity'
        assert config.wandb_project == 'test_project'
        assert config.num_agents == 3
        assert config.clip_eps == 0.3
        assert config.track_learning_metrics is False

        assert 'test_metric' in config.tracked_metrics
        metric = config.tracked_metrics['test_metric']
        assert metric.default_value == 1.0
        assert metric.metric_type == 'boolean'
        assert metric.aggregation == 'sum'
        assert metric.wandb_name == 'Test Metric'
        assert metric.enabled is False


class TestConfigFileLoading:
    """Test loading configurations from files."""

    def test_load_yaml_config(self):
        """Test loading YAML configuration file."""
        yaml_content = """
wandb_entity: "test_entity"
wandb_project: "test_project"
num_agents: 2
tracked_metrics:
  test_metric:
    default_value: 2.0
    metric_type: "scalar"
    wandb_name: "Test Metric"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            # Skip if PyYAML not available
            pytest.importorskip("yaml")

            config = load_config_from_file(temp_path)

            assert config.wandb_entity == "test_entity"
            assert config.wandb_project == "test_project"
            assert config.num_agents == 2
            assert "test_metric" in config.tracked_metrics
            assert config.tracked_metrics["test_metric"].default_value == 2.0

        finally:
            os.unlink(temp_path)

    def test_load_json_config(self):
        """Test loading JSON configuration file."""
        json_content = """{
    "wandb_entity": "test_entity",
    "wandb_project": "test_project",
    "num_agents": 2,
    "tracked_metrics": {
        "test_metric": {
            "default_value": 2.0,
            "metric_type": "scalar",
            "wandb_name": "Test Metric"
        }
    }
}"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            config = load_config_from_file(temp_path)

            assert config.wandb_entity == "test_entity"
            assert config.wandb_project == "test_project"
            assert config.num_agents == 2
            assert "test_metric" in config.tracked_metrics
            assert config.tracked_metrics["test_metric"].default_value == 2.0

        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config_from_file("nonexistent_file.yaml")

    def test_load_unsupported_format(self):
        """Test loading unsupported file format raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some content")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                load_config_from_file(temp_path)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])