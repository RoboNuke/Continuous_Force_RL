"""
Unit tests for ConfigManagerV2.

Tests the new defaults-first configuration system including:
- Task resolution
- Configuration loading with defaults
- YAML override application
- CLI override application
- Configuration validation
- Error handling
"""

import pytest
import tempfile
import os
import yaml
from unittest.mock import patch, MagicMock

from configs.config_manager_v2 import ConfigManagerV2, ConfigBundle


class TestConfigManagerV2TaskResolution:
    """Test task name resolution functionality."""

    def test_resolve_task_name_cli_priority(self):
        """Test that CLI task name takes priority over YAML."""
        yaml_config = {'task_name': 'gear_mesh'}
        cli_task = 'peg_insert'

        task_name = ConfigManagerV2._resolve_task_name(yaml_config, cli_task)
        assert task_name == 'peg_insert'

    def test_resolve_task_name_yaml_fallback(self):
        """Test falling back to YAML task name."""
        yaml_config = {'task_name': 'Isaac-Factory-PegInsert-Direct-v0'}
        cli_task = None

        task_name = ConfigManagerV2._resolve_task_name(yaml_config, cli_task)
        assert task_name == 'Isaac-Factory-PegInsert-Direct-v0'

    def test_resolve_task_name_defaults_fallback(self):
        """Test falling back to defaults section."""
        yaml_config = {
            'defaults': {
                'task_name': 'Isaac-Factory-GearMesh-Direct-v0'
            }
        }
        cli_task = None

        task_name = ConfigManagerV2._resolve_task_name(yaml_config, cli_task)
        assert task_name == 'Isaac-Factory-GearMesh-Direct-v0'

    def test_resolve_task_name_no_task_error(self):
        """Test error when no task name is provided."""
        yaml_config = {}
        cli_task = None

        with pytest.raises(ValueError, match="No task name specified"):
            ConfigManagerV2._resolve_task_name(yaml_config, cli_task)

    def test_resolve_task_name_unknown_task_error(self):
        """Test error for unknown task names."""
        yaml_config = {'task_name': 'unknown_task'}
        cli_task = None

        with pytest.raises(ValueError, match="Unknown task name: 'unknown_task'"):
            ConfigManagerV2._resolve_task_name(yaml_config, cli_task)


class TestConfigManagerV2ConfigLoading:
    """Test configuration loading functionality."""

    def test_load_extended_env_config(self):
        """Test loading extended environment configurations."""
        # Test peg insert
        env_cfg = ConfigManagerV2._load_extended_env_config('peg_insert')
        assert env_cfg.task_name == 'peg_insert'
        assert env_cfg.episode_length_s == 10.0

        # Test gear mesh
        env_cfg = ConfigManagerV2._load_extended_env_config('gear_mesh')
        assert env_cfg.task_name == 'gear_mesh'
        assert env_cfg.episode_length_s == 20.0

        # Test nut thread
        env_cfg = ConfigManagerV2._load_extended_env_config('nut_thread')
        assert env_cfg.task_name == 'nut_thread'
        assert env_cfg.episode_length_s == 30.0

    def test_load_extended_env_config_unknown_task(self):
        """Test error for unknown task in env config loading."""
        with pytest.raises(ValueError, match="Unknown task name"):
            ConfigManagerV2._load_extended_env_config('unknown_task')

    def test_load_extended_agent_config(self):
        """Test loading extended agent configuration."""
        agent_cfg = ConfigManagerV2._load_extended_agent_config()

        assert agent_cfg.rollouts == 16  # SKRL default
        assert agent_cfg.learning_rate == 1e-3  # SKRL default
        assert agent_cfg.num_agents == 1  # Extended default

    def test_create_primary_config(self):
        """Test creating primary configuration from YAML."""
        # Test with YAML overrides
        yaml_config = {
            'primary': {
                'agents_per_break_force': 3,
                'num_envs_per_agent': 128,
                'break_forces': [10, 20],
                'decimation': 4
            }
        }

        primary_cfg = ConfigManagerV2._create_primary_config(yaml_config)

        assert primary_cfg.agents_per_break_force == 3
        assert primary_cfg.num_envs_per_agent == 128
        assert primary_cfg.break_forces == [10, 20]
        assert primary_cfg.decimation == 4
        assert primary_cfg.total_agents == 6  # 2 * 3

    def test_create_primary_config_defaults(self):
        """Test creating primary configuration with defaults only."""
        yaml_config = {}  # No primary section

        primary_cfg = ConfigManagerV2._create_primary_config(yaml_config)

        # Should use defaults
        assert primary_cfg.agents_per_break_force == 2
        assert primary_cfg.num_envs_per_agent == 256
        assert primary_cfg.break_forces == -1

    def test_create_model_config(self):
        """Test creating model configuration."""
        yaml_config = {
            'model': {
                'use_hybrid_agent': True,
                'critic_output_init_mean': 60,
                'actor': {
                    'latent_size': 128
                },
                'critic': {
                    'n': 5,
                    'latent_size': 512
                }
            }
        }
        primary_cfg = ConfigManagerV2._create_primary_config({})

        model_cfg = ConfigManagerV2._create_model_config(yaml_config, primary_cfg)

        assert model_cfg.use_hybrid_agent == True
        assert model_cfg.critic_output_init_mean == 60
        assert model_cfg.actor_latent_size == 128
        assert model_cfg.critic_n == 5
        assert model_cfg.critic_latent_size == 512

    def test_create_wrapper_config(self):
        """Test creating wrapper configuration."""
        yaml_config = {
            'wrappers': {
                'force_torque_sensor': {
                    'enabled': True,
                    'tanh_scale': 0.05
                },
                'wandb_logging': {
                    'enabled': True,
                    'wandb_project': 'Test_Project'
                }
            }
        }
        primary_cfg = ConfigManagerV2._create_primary_config({})

        wrapper_cfg = ConfigManagerV2._create_wrapper_config(yaml_config, primary_cfg)

        # Should have applied primary config
        assert hasattr(wrapper_cfg, '_primary_cfg')

    def test_create_hybrid_config(self):
        """Test creating hybrid agent configuration."""
        yaml_config = {
            'model': {
                'hybrid_agent': {
                    'ctrl_torque': True,
                    'pos_scale': 2.0,
                    'force_scale': 1.5
                }
            }
        }

        hybrid_cfg = ConfigManagerV2._create_hybrid_config(yaml_config)

        assert hybrid_cfg.ctrl_torque == True
        assert hybrid_cfg.pos_scale == 2.0
        assert hybrid_cfg.force_scale == 1.5


class TestConfigManagerV2OverrideApplication:
    """Test YAML and CLI override application."""

    def test_apply_yaml_overrides_simple(self):
        """Test applying simple YAML overrides."""
        from configs.cfg_exts.primary_cfg import PrimaryConfig

        target_obj = PrimaryConfig()
        yaml_overrides = {
            'agents_per_break_force': 5,
            'decimation': 16,
            'debug_mode': True
        }

        ConfigManagerV2._apply_yaml_overrides(target_obj, yaml_overrides)

        assert target_obj.agents_per_break_force == 5
        assert target_obj.decimation == 16
        assert target_obj.debug_mode == True

    def test_apply_yaml_overrides_ctrl_special_handling(self):
        """Test special handling for ctrl configuration merging."""
        from configs.cfg_exts.extended_peg_insert_cfg import ExtendedFactoryTaskPegInsertCfg

        env_cfg = ExtendedFactoryTaskPegInsertCfg()
        yaml_overrides = {
            'ctrl': {
                'force_action_bounds': [100.0, 100.0, 100.0],
                'torque_action_bounds': [1.0, 1.0, 1.0]
            }
        }

        # Ctrl should be ExtendedCtrlCfg after creation
        original_ctrl_type = type(env_cfg.ctrl)

        ConfigManagerV2._apply_yaml_overrides(env_cfg, yaml_overrides)

        # Ctrl should still be the same type (not replaced with dict)
        assert type(env_cfg.ctrl) == original_ctrl_type
        assert env_cfg.ctrl.force_action_bounds == [100.0, 100.0, 100.0]
        assert env_cfg.ctrl.torque_action_bounds == [1.0, 1.0, 1.0]

    def test_apply_nested_wrapper_overrides(self):
        """Test applying nested wrapper configuration overrides."""
        from configs.cfg_exts.extended_wrapper_cfg import ExtendedWrapperConfig

        wrapper_cfg = ExtendedWrapperConfig()
        wrapper_overrides = {
            'force_torque_sensor': {
                'enabled': True,
                'use_tanh_scaling': True,
                'tanh_scale': 0.08
            },
            'wandb_logging': {
                'enabled': True,
                'wandb_project': 'New_Project',
                'wandb_entity': 'new_entity'
            }
        }

        ConfigManagerV2._apply_nested_wrapper_overrides(wrapper_cfg, wrapper_overrides)

        assert wrapper_cfg.force_torque_sensor_enabled == True
        assert wrapper_cfg.force_torque_use_tanh_scaling == True
        assert wrapper_cfg.force_torque_tanh_scale == 0.08
        assert wrapper_cfg.wandb_project == 'New_Project'
        assert wrapper_cfg.wandb_entity == 'new_entity'

    def test_apply_cli_overrides(self):
        """Test applying CLI overrides."""
        from configs.cfg_exts.extended_peg_insert_cfg import ExtendedFactoryTaskPegInsertCfg
        from agents.extended_ppo_cfg import ExtendedPPOConfig
        from configs.cfg_exts.extended_model_cfg import ExtendedModelConfig
        from configs.cfg_exts.extended_wrapper_cfg import ExtendedWrapperConfig

        env_cfg = ExtendedFactoryTaskPegInsertCfg()
        agent_cfg = ExtendedPPOConfig()
        model_cfg = ExtendedModelConfig()
        wrapper_cfg = ExtendedWrapperConfig()

        cli_overrides = [
            'environment.decimation=16',
            'agent.learning_rate=1e-5',
            'model.use_hybrid_agent=true',
            'wrappers.force_torque_sensor_enabled=true'
        ]

        ConfigManagerV2._apply_cli_overrides(env_cfg, agent_cfg, model_cfg, wrapper_cfg, cli_overrides)

        assert env_cfg.decimation == 16
        assert agent_cfg.learning_rate == 1e-5
        assert model_cfg.use_hybrid_agent == True
        assert wrapper_cfg.force_torque_sensor_enabled == True

    def test_apply_cli_overrides_value_parsing(self):
        """Test CLI override value parsing."""
        from configs.cfg_exts.primary_cfg import PrimaryConfig

        target_obj = PrimaryConfig()
        cli_overrides = [
            'agent.debug_mode=true',  # Boolean
            'agent.learning_rate=1e-5',  # Float
            'agent.rollouts=32',  # Integer
            'agent.break_forces=[10,20,30]',  # List
            'agent.optimizer_name=Adam'  # String
        ]

        # Create mock configs for testing
        agent_cfg = MagicMock()
        agent_cfg.debug_mode = False
        agent_cfg.learning_rate = 1e-3
        agent_cfg.rollouts = 16
        agent_cfg.break_forces = []
        agent_cfg.optimizer_name = 'SGD'

        ConfigManagerV2._apply_cli_overrides(target_obj, agent_cfg, target_obj, target_obj, cli_overrides)

        # Check that values were parsed correctly
        assert agent_cfg.debug_mode == True
        assert agent_cfg.learning_rate == 1e-5
        assert agent_cfg.rollouts == 32
        assert agent_cfg.break_forces == [10, 20, 30]
        assert agent_cfg.optimizer_name == 'Adam'


class TestConfigManagerV2Integration:
    """Test complete configuration loading integration."""

    def test_load_yaml_file(self):
        """Test YAML file loading."""
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'task_name': 'peg_insert',
                'primary': {
                    'agents_per_break_force': 3
                }
            }, f)
            temp_path = f.name

        try:
            config = ConfigManagerV2._load_yaml_file(temp_path)
            assert config['task_name'] == 'peg_insert'
            assert config['primary']['agents_per_break_force'] == 3
        finally:
            os.unlink(temp_path)

    def test_load_yaml_file_not_found(self):
        """Test error for missing YAML file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ConfigManagerV2._load_yaml_file('/nonexistent/file.yaml')

    def test_validate_config_bundle(self):
        """Test configuration bundle validation."""
        # Create valid config bundle
        from configs.cfg_exts.primary_cfg import PrimaryConfig
        from configs.cfg_exts.extended_peg_insert_cfg import ExtendedFactoryTaskPegInsertCfg
        from agents.extended_ppo_cfg import ExtendedPPOConfig
        from configs.cfg_exts.extended_model_cfg import ExtendedModelConfig
        from configs.cfg_exts.extended_wrapper_cfg import ExtendedWrapperConfig

        primary_cfg = PrimaryConfig()
        env_cfg = ExtendedFactoryTaskPegInsertCfg()
        env_cfg.apply_primary_cfg(primary_cfg)
        agent_cfg = ExtendedPPOConfig()
        model_cfg = ExtendedModelConfig()
        wrapper_cfg = ExtendedWrapperConfig()

        config_bundle = ConfigBundle(
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            primary_cfg=primary_cfg,
            model_cfg=model_cfg,
            wrapper_cfg=wrapper_cfg,
            task_name='peg_insert'
        )

        # Should pass validation
        ConfigManagerV2._validate_config_bundle(config_bundle)

    def test_validate_config_bundle_inconsistent_agents(self):
        """Test validation error for inconsistent agent counts."""
        from configs.cfg_exts.primary_cfg import PrimaryConfig
        from configs.cfg_exts.extended_peg_insert_cfg import ExtendedFactoryTaskPegInsertCfg
        from agents.extended_ppo_cfg import ExtendedPPOConfig
        from configs.cfg_exts.extended_model_cfg import ExtendedModelConfig
        from configs.cfg_exts.extended_wrapper_cfg import ExtendedWrapperConfig

        primary_cfg = PrimaryConfig(agents_per_break_force=3)  # total_agents = 3
        env_cfg = ExtendedFactoryTaskPegInsertCfg()
        env_cfg.apply_primary_cfg(primary_cfg)

        # Manually set inconsistent agent count
        env_cfg._primary_cfg = PrimaryConfig(agents_per_break_force=2)  # Different count

        agent_cfg = ExtendedPPOConfig()
        model_cfg = ExtendedModelConfig()
        wrapper_cfg = ExtendedWrapperConfig()

        config_bundle = ConfigBundle(
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            primary_cfg=primary_cfg,
            model_cfg=model_cfg,
            wrapper_cfg=wrapper_cfg,
            task_name='peg_insert'
        )

        with pytest.raises(ValueError, match="Inconsistent agent count"):
            ConfigManagerV2._validate_config_bundle(config_bundle)

    def test_validate_config_bundle_hybrid_mismatch(self):
        """Test validation error for hybrid config mismatch."""
        from configs.cfg_exts.primary_cfg import PrimaryConfig
        from configs.cfg_exts.extended_peg_insert_cfg import ExtendedFactoryTaskPegInsertCfg
        from agents.extended_ppo_cfg import ExtendedPPOConfig
        from configs.cfg_exts.extended_model_cfg import ExtendedModelConfig, ExtendedHybridAgentConfig
        from configs.cfg_exts.extended_wrapper_cfg import ExtendedWrapperConfig

        primary_cfg = PrimaryConfig()
        env_cfg = ExtendedFactoryTaskPegInsertCfg()
        env_cfg.apply_primary_cfg(primary_cfg)
        agent_cfg = ExtendedPPOConfig()
        model_cfg = ExtendedModelConfig(use_hybrid_agent=True)  # Hybrid enabled
        wrapper_cfg = ExtendedWrapperConfig()

        config_bundle = ConfigBundle(
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            primary_cfg=primary_cfg,
            model_cfg=model_cfg,
            wrapper_cfg=wrapper_cfg,
            hybrid_cfg=None,  # But no hybrid config provided
            task_name='peg_insert'
        )

        with pytest.raises(ValueError, match="Model config specifies use_hybrid_agent=True but no hybrid config provided"):
            ConfigManagerV2._validate_config_bundle(config_bundle)

    def test_get_legacy_config_dict(self):
        """Test conversion to legacy configuration format."""
        from configs.cfg_exts.primary_cfg import PrimaryConfig
        from configs.cfg_exts.extended_peg_insert_cfg import ExtendedFactoryTaskPegInsertCfg
        from agents.extended_ppo_cfg import ExtendedPPOConfig
        from configs.cfg_exts.extended_model_cfg import ExtendedModelConfig
        from configs.cfg_exts.extended_wrapper_cfg import ExtendedWrapperConfig

        primary_cfg = PrimaryConfig(agents_per_break_force=3)
        env_cfg = ExtendedFactoryTaskPegInsertCfg()
        env_cfg.apply_primary_cfg(primary_cfg)
        agent_cfg = ExtendedPPOConfig()
        agent_cfg.apply_primary_cfg(primary_cfg)
        model_cfg = ExtendedModelConfig()
        wrapper_cfg = ExtendedWrapperConfig()
        wrapper_cfg.apply_primary_cfg(primary_cfg)

        config_bundle = ConfigBundle(
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            primary_cfg=primary_cfg,
            model_cfg=model_cfg,
            wrapper_cfg=wrapper_cfg,
            task_name='peg_insert'
        )

        legacy_dict = ConfigManagerV2.get_legacy_config_dict(config_bundle)

        # Test legacy structure
        assert 'primary' in legacy_dict
        assert 'derived' in legacy_dict
        assert 'environment' in legacy_dict
        assert 'model' in legacy_dict
        assert 'agent' in legacy_dict

        # Test values
        assert legacy_dict['primary']['agents_per_break_force'] == 3
        assert legacy_dict['derived']['total_agents'] == 3
        assert legacy_dict['environment']['episode_length_s'] == 10.0


class TestConfigManagerV2CompleteLoading:
    """Test complete configuration loading workflow."""

    def test_load_defaults_first_config_simple(self):
        """Test complete configuration loading with simple YAML."""
        # Create simple test YAML
        yaml_content = {
            'task_name': 'peg_insert',
            'primary': {
                'agents_per_break_force': 1,
                'num_envs_per_agent': 64
            },
            'model': {
                'use_hybrid_agent': False
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            config_bundle = ConfigManagerV2.load_defaults_first_config(
                temp_path,
                cli_overrides=['model.actor_latent_size=128'],
                cli_task=None
            )

            # Test that configuration was loaded correctly
            assert config_bundle.task_name == 'peg_insert'
            assert config_bundle.primary_cfg.agents_per_break_force == 1
            assert config_bundle.primary_cfg.num_envs_per_agent == 64
            assert config_bundle.env_cfg.episode_length_s == 10.0  # Isaac Lab default
            assert config_bundle.model_cfg.use_hybrid_agent == False
            assert config_bundle.model_cfg.actor_latent_size == 128  # CLI override
            assert config_bundle.hybrid_cfg is None  # No hybrid agent

        finally:
            os.unlink(temp_path)

    def test_load_defaults_first_config_hybrid(self):
        """Test complete configuration loading with hybrid agent."""
        yaml_content = {
            'task_name': 'Isaac-Factory-PegInsert-Direct-v0',
            'model': {
                'use_hybrid_agent': True,
                'hybrid_agent': {
                    'ctrl_torque': False,
                    'pos_scale': 2.0
                }
            },
            'wrappers': {
                'force_torque_sensor': {
                    'enabled': True
                },
                'hybrid_control': {
                    'enabled': True
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            config_bundle = ConfigManagerV2.load_defaults_first_config(
                temp_path,
                cli_overrides=[],
                cli_task=None
            )

            # Test hybrid configuration
            assert config_bundle.model_cfg.use_hybrid_agent == True
            assert config_bundle.hybrid_cfg is not None
            assert config_bundle.hybrid_cfg.ctrl_torque == False
            assert config_bundle.hybrid_cfg.pos_scale == 2.0
            assert config_bundle.wrapper_cfg.force_torque_sensor_enabled == True
            assert config_bundle.wrapper_cfg.hybrid_control_enabled == True

        finally:
            os.unlink(temp_path)


class TestConfigMergeTagMerging:
    """Test experiment tag merging functionality."""

    def test_merge_configs_experiment_tags_basic(self):
        """Test basic experiment tag merging."""
        base_config = {
            'experiment': {
                'name': 'base_exp',
                'tags': ['baseline', 'factory'],
                'group': 'base_group'
            }
        }

        override_config = {
            'experiment': {
                'name': 'override_exp',
                'tags': ['hybrid_control', 'force_position'],
                'group': 'override_group'
            }
        }

        result = ConfigManagerV2._merge_configs(base_config, override_config)

        # Tags should be merged (base + override, no duplicates)
        assert result['experiment']['tags'] == ['baseline', 'factory', 'hybrid_control', 'force_position']
        # Other fields should be overridden normally
        assert result['experiment']['name'] == 'override_exp'
        assert result['experiment']['group'] == 'override_group'

    def test_merge_configs_experiment_tags_with_duplicates(self):
        """Test tag merging removes duplicates."""
        base_config = {
            'experiment': {
                'tags': ['baseline', 'factory', 'common']
            }
        }

        override_config = {
            'experiment': {
                'tags': ['hybrid_control', 'common', 'factory']
            }
        }

        result = ConfigManagerV2._merge_configs(base_config, override_config)

        # Should keep order from base, add new from override, no duplicates
        expected_tags = ['baseline', 'factory', 'common', 'hybrid_control']
        assert result['experiment']['tags'] == expected_tags

    def test_merge_configs_experiment_tags_base_only(self):
        """Test when only base has tags."""
        base_config = {
            'experiment': {
                'tags': ['baseline', 'factory'],
                'name': 'base'
            }
        }

        override_config = {
            'experiment': {
                'name': 'override'
            }
        }

        result = ConfigManagerV2._merge_configs(base_config, override_config)

        # Base tags should be preserved
        assert result['experiment']['tags'] == ['baseline', 'factory']
        assert result['experiment']['name'] == 'override'

    def test_merge_configs_experiment_tags_override_only(self):
        """Test when only override has tags."""
        base_config = {
            'experiment': {
                'name': 'base'
            }
        }

        override_config = {
            'experiment': {
                'tags': ['hybrid_control', 'force_position'],
                'name': 'override'
            }
        }

        result = ConfigManagerV2._merge_configs(base_config, override_config)

        # Override tags should be used
        assert result['experiment']['tags'] == ['hybrid_control', 'force_position']
        assert result['experiment']['name'] == 'override'

    def test_merge_configs_no_experiment_section(self):
        """Test normal merging when no experiment section."""
        base_config = {
            'primary': {
                'agents_per_break_force': 2
            }
        }

        override_config = {
            'primary': {
                'max_steps': 10000
            }
        }

        result = ConfigManagerV2._merge_configs(base_config, override_config)

        # Should merge normally
        assert result['primary']['agents_per_break_force'] == 2
        assert result['primary']['max_steps'] == 10000


class TestCLIExperimentTagOverrides:
    """Test CLI experiment tag override functionality."""

    def test_cli_experiment_tags_single_tag(self):
        """Test CLI override with single tag."""
        from configs.cfg_exts.primary_cfg import PrimaryConfig
        from configs.cfg_exts.extended_peg_insert_cfg import ExtendedFactoryTaskPegInsertCfg
        from agents.extended_ppo_cfg import ExtendedPPOConfig
        from configs.cfg_exts.extended_model_cfg import ExtendedModelConfig
        from configs.cfg_exts.extended_wrapper_cfg import ExtendedWrapperConfig

        config_bundle = ConfigBundle(
            env_cfg=ExtendedFactoryTaskPegInsertCfg(),
            agent_cfg=ExtendedPPOConfig(),
            primary_cfg=PrimaryConfig(),
            model_cfg=ExtendedModelConfig(),
            wrapper_cfg=ExtendedWrapperConfig(),
            task_name='peg_insert'
        )

        cli_overrides = ['experiment.tags=debug']
        ConfigManagerV2._apply_cli_overrides(config_bundle, cli_overrides)

        assert hasattr(config_bundle, '_cli_experiment_tags')
        assert config_bundle._cli_experiment_tags == ['debug']

    def test_cli_experiment_tags_multiple_comma_separated(self):
        """Test CLI override with comma-separated tags."""
        from configs.cfg_exts.primary_cfg import PrimaryConfig
        from configs.cfg_exts.extended_peg_insert_cfg import ExtendedFactoryTaskPegInsertCfg
        from agents.extended_ppo_cfg import ExtendedPPOConfig
        from configs.cfg_exts.extended_model_cfg import ExtendedModelConfig
        from configs.cfg_exts.extended_wrapper_cfg import ExtendedWrapperConfig

        config_bundle = ConfigBundle(
            env_cfg=ExtendedFactoryTaskPegInsertCfg(),
            agent_cfg=ExtendedPPOConfig(),
            primary_cfg=PrimaryConfig(),
            model_cfg=ExtendedModelConfig(),
            wrapper_cfg=ExtendedWrapperConfig(),
            task_name='peg_insert'
        )

        cli_overrides = ['experiment.tags=debug,v2,test']
        ConfigManagerV2._apply_cli_overrides(config_bundle, cli_overrides)

        assert hasattr(config_bundle, '_cli_experiment_tags')
        assert config_bundle._cli_experiment_tags == ['debug', 'v2', 'test']

    def test_cli_experiment_tags_list_format(self):
        """Test CLI override with Python list format."""
        from configs.cfg_exts.primary_cfg import PrimaryConfig
        from configs.cfg_exts.extended_peg_insert_cfg import ExtendedFactoryTaskPegInsertCfg
        from agents.extended_ppo_cfg import ExtendedPPOConfig
        from configs.cfg_exts.extended_model_cfg import ExtendedModelConfig
        from configs.cfg_exts.extended_wrapper_cfg import ExtendedWrapperConfig

        config_bundle = ConfigBundle(
            env_cfg=ExtendedFactoryTaskPegInsertCfg(),
            agent_cfg=ExtendedPPOConfig(),
            primary_cfg=PrimaryConfig(),
            model_cfg=ExtendedModelConfig(),
            wrapper_cfg=ExtendedWrapperConfig(),
            task_name='peg_insert'
        )

        cli_overrides = ['experiment.tags=["debug", "v2"]']
        ConfigManagerV2._apply_cli_overrides(config_bundle, cli_overrides)

        assert hasattr(config_bundle, '_cli_experiment_tags')
        assert config_bundle._cli_experiment_tags == ['debug', 'v2']


if __name__ == "__main__":
    pytest.main([__file__])