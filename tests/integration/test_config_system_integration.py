"""
Integration tests for the complete configuration system.

Tests end-to-end workflows using real YAML files and ensuring
all components work together correctly.
"""

import pytest
import tempfile
import os
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from configs.config_manager_v2 import ConfigManagerV2
# Integration tests run with the mocked Isaac Lab components built into version_compat.py


class TestConfigSystemIntegration:
    """Integration tests for the complete configuration system."""

    def setup_method(self):
        """Set up test environment."""
        # Create temporary directory for test configs
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')

    def teardown_method(self):
        """Clean up test environment."""
        # Remove temporary files
        if os.path.exists(self.config_path):
            os.unlink(self.config_path)
        os.rmdir(self.temp_dir)

    def _create_test_config(self, config_dict):
        """Helper to create test configuration file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f)

    def test_complete_peg_insert_workflow(self):
        """Test complete workflow for peg insert task."""
        config_dict = {
            'task_name': 'Isaac-Factory-PegInsert-Direct-v0',
            'primary': {
                'agents_per_break_force': 2,
                'num_envs_per_agent': 128,
                'break_forces': [50, 100],
                'decimation': 12,
                'policy_hz': 20,
                'max_steps': 8000000,
                'debug_mode': False
            },
            'environment': {
                'filter_collisions': True,
                'ctrl': {
                    'force_action_bounds': [75.0, 75.0, 75.0],
                    'torque_action_bounds': [1.5, 1.5, 1.5],
                    'force_action_threshold': [15.0, 15.0, 15.0]
                }
            },
            'model': {
                'use_hybrid_agent': True,
                'critic_output_init_mean': 65,
                'actor': {
                    'n': 2,
                    'latent_size': 512
                },
                'critic': {
                    'n': 4,
                    'latent_size': 2048
                },
                'hybrid_agent': {
                    'ctrl_torque': False,
                    'pos_scale': 1.5,
                    'rot_scale': 1.2,
                    'force_scale': 2.0,
                    'selection_adjustment_types': 'batch_norm'
                }
            },
            'wrappers': {
                'force_torque_sensor': {
                    'enabled': True,
                    'use_tanh_scaling': True,
                    'tanh_scale': 0.06
                },
                'hybrid_control': {
                    'enabled': True,
                    'reward_type': 'detailed'
                },
                'observation_noise': {
                    'enabled': True,
                    'global_scale': 0.9,
                    'apply_to_critic': False
                },
                'wandb_logging': {
                    'enabled': True,
                    'wandb_project': 'Integration_Test',
                    'wandb_entity': 'test_entity',
                    'wandb_group': 'peg_insert_tests',
                    'wandb_tags': ['integration', 'peg_insert']
                }
            },
            'agent': {
                'policy_learning_rate': 3.0e-6,
                'critic_learning_rate': 8.0e-6,
                'learning_epochs': 8,
                'discount_factor': 0.995,
                'lambda_': 0.98
            }
        }

        self._create_test_config(config_dict)

        cli_overrides = [
            'primary.decimation=16',
            'model.actor_latent_size=256',
            'agent.learning_epochs=6'
        ]

        # Load configuration
        bundle = ConfigManagerV2.load_defaults_first_config(
            self.config_path,
            cli_overrides=cli_overrides
        )

        # Verify task identification
        assert bundle.task_name == 'Isaac-Factory-PegInsert-Direct-v0'

        # Verify primary configuration with CLI overrides
        assert bundle.primary.decimation == 16  # CLI override
        assert bundle.primary.agents_per_break_force == 2
        assert bundle.primary.num_envs_per_agent == 128
        assert bundle.primary.break_forces == [50, 100]
        assert bundle.primary.policy_hz == 20
        assert bundle.primary.max_steps == 8000000
        assert bundle.primary.debug_mode is False

        # Verify computed properties
        assert bundle.primary.total_agents == 4  # 2 break forces * 2 agents
        assert bundle.primary.total_num_envs == 512  # 4 agents * 128 envs

        # Verify environment configuration
        assert bundle.environment.episode_length_s == 10.0  # Peg insert default
        assert bundle.environment.decimation == 16  # From primary config
        assert bundle.environment.filter_collisions is True

        # Verify ctrl configuration preservation and updates
        from configs.cfg_exts.extended_ctrl_cfg import ExtendedCtrlCfg
        assert isinstance(bundle.environment.ctrl, ExtendedCtrlCfg)
        assert bundle.environment.ctrl.force_action_bounds == [75.0, 75.0, 75.0]
        assert bundle.environment.ctrl.torque_action_bounds == [1.5, 1.5, 1.5]
        assert bundle.environment.ctrl.force_action_threshold == [15.0, 15.0, 15.0]

        # Verify model configuration with CLI overrides
        assert bundle.model.use_hybrid_agent is True
        assert bundle.model.critic_output_init_mean == 65
        assert bundle.model.actor_n == 2
        assert bundle.model.actor_latent_size == 256  # CLI override
        assert bundle.model.critic_n == 4
        assert bundle.model.critic_latent_size == 2048

        # Verify hybrid agent configuration
        assert bundle.hybrid_cfg is not None
        assert bundle.hybrid_cfg.ctrl_torque is False
        assert bundle.hybrid_cfg.pos_scale == 1.5
        assert bundle.hybrid_cfg.rot_scale == 1.2
        assert bundle.hybrid_cfg.force_scale == 2.0
        assert bundle.hybrid_cfg.selection_adjustment_types == 'batch_norm'

        # Verify wrapper configuration
        assert bundle.wrappers.force_torque_sensor_enabled is True
        assert bundle.wrappers.force_torque_use_tanh_scaling is True
        assert bundle.wrappers.force_torque_tanh_scale == 0.06
        assert bundle.wrappers.hybrid_control_enabled is True
        assert bundle.wrappers.hybrid_control_reward_type == 'detailed'
        assert bundle.wrappers.observation_noise_enabled is True
        assert bundle.wrappers.observation_noise_global_scale == 0.9
        assert bundle.wrappers.observation_noise_apply_to_critic is False
        assert bundle.wrappers.wandb_logging_enabled is True
        assert bundle.wrappers.wandb_logging_wandb_project == 'Integration_Test'
        assert bundle.wrappers.wandb_logging_wandb_entity == 'test_entity'
        assert bundle.wrappers.wandb_logging_wandb_group == 'peg_insert_tests'
        assert bundle.wrappers.wandb_logging_wandb_tags == ['integration', 'peg_insert']

        # Verify agent configuration with CLI overrides
        assert bundle.agent.policy_learning_rate == 3.0e-6
        assert bundle.agent.critic_learning_rate == 8.0e-6
        assert bundle.agent.learning_epochs == 6  # CLI override
        assert bundle.agent.discount_factor == 0.995
        assert bundle.agent.lambda_ == 0.98

        # Verify computed agent properties
        rollouts = bundle.agent.get_computed_rollouts(bundle.environment.episode_length_s)
        assert rollouts > 0
        mini_batches = bundle.agent.get_computed_mini_batches(rollouts)
        assert mini_batches > 0

    def test_gear_mesh_task_specifics(self):
        """Test gear mesh task-specific configurations."""
        config_dict = {
            'task_name': 'gear_mesh',
            'primary': {
                'agents_per_break_force': 1,
                'num_envs_per_agent': 256
            },
            'model': {
                'use_hybrid_agent': False
            }
        }

        self._create_test_config(config_dict)

        bundle = ConfigManagerV2.load_defaults_first_config(self.config_path)

        # Verify task resolution
        assert bundle.task_name == 'Isaac-Factory-GearMesh-Direct-v0'

        # Verify gear mesh specific defaults
        assert bundle.environment.episode_length_s == 20.0  # Gear mesh default
        assert bundle.environment.task_name == 'gear_mesh'

        # Verify no hybrid configuration
        assert bundle.hybrid_cfg is None

    def test_nut_thread_task_specifics(self):
        """Test nut thread task-specific configurations."""
        config_dict = {
            'task_name': 'nut_thread',
            'primary': {
                'agents_per_break_force': 3,
                'break_forces': [25, 50, 75, 100]
            }
        }

        self._create_test_config(config_dict)

        bundle = ConfigManagerV2.load_defaults_first_config(self.config_path)

        # Verify task resolution
        assert bundle.task_name == 'Isaac-Factory-NutThread-Direct-v0'

        # Verify nut thread specific defaults
        assert bundle.environment.episode_length_s == 30.0  # Nut thread default
        assert bundle.environment.task_name == 'nut_thread'

        # Verify computed properties with multiple break forces
        assert bundle.primary.total_agents == 12  # 4 break forces * 3 agents
        assert bundle.primary.total_num_envs == 3072  # 12 agents * 256 envs

    def test_legacy_config_conversion(self):
        """Test conversion to legacy configuration format."""
        config_dict = {
            'task_name': 'peg_insert',
            'primary': {
                'agents_per_break_force': 2,
                'num_envs_per_agent': 64,
                'break_forces': [30, 60],
                'decimation': 10
            },
            'model': {
                'use_hybrid_agent': True,
                'hybrid_agent': {
                    'ctrl_torque': True
                }
            },
            'wrappers': {
                'wandb_logging': {
                    'enabled': True,
                    'wandb_project': 'Legacy_Test'
                }
            }
        }

        self._create_test_config(config_dict)

        bundle = ConfigManagerV2.load_defaults_first_config(self.config_path)
        legacy_dict = ConfigManagerV2.get_legacy_config_dict(bundle)

        # Verify legacy structure
        expected_sections = ['primary', 'derived', 'environment', 'model', 'agent', 'wrappers']
        for section in expected_sections:
            assert section in legacy_dict

        # Verify primary values
        assert legacy_dict['primary']['agents_per_break_force'] == 2
        assert legacy_dict['primary']['num_envs_per_agent'] == 64
        assert legacy_dict['primary']['break_forces'] == [30, 60]
        assert legacy_dict['primary']['decimation'] == 10

        # Verify derived values
        assert legacy_dict['derived']['total_agents'] == 4
        assert legacy_dict['derived']['total_num_envs'] == 256

        # Verify environment values
        assert legacy_dict['environment']['episode_length_s'] == 10.0
        assert legacy_dict['environment']['decimation'] == 10
        assert 'ctrl' in legacy_dict['environment']

        # Verify model values
        assert legacy_dict['model']['use_hybrid_agent'] is True
        assert 'hybrid_agent' in legacy_dict['model']
        assert legacy_dict['model']['hybrid_agent']['ctrl_torque'] is True

        # Verify wrapper values
        assert legacy_dict['wrappers']['wandb_logging']['enabled'] is True
        assert legacy_dict['wrappers']['wandb_logging']['wandb_project'] == 'Legacy_Test'

    def test_cli_task_override_workflow(self):
        """Test complete workflow with CLI task override."""
        config_dict = {
            'primary': {
                'agents_per_break_force': 1
            },
            'model': {
                'use_hybrid_agent': False
            }
        }

        self._create_test_config(config_dict)

        # Load with CLI task override
        bundle = ConfigManagerV2.load_defaults_first_config(
            self.config_path,
            cli_task='gear_mesh'
        )

        # Verify CLI task override took precedence
        assert bundle.task_name == 'Isaac-Factory-GearMesh-Direct-v0'
        assert bundle.environment.episode_length_s == 20.0
        assert bundle.environment.task_name == 'gear_mesh'

    def test_configuration_validation_integration(self):
        """Test configuration validation in integration context."""
        config_dict = {
            'task_name': 'peg_insert',
            'primary': {
                'agents_per_break_force': 0,  # Invalid value
                'decimation': -5,  # Invalid value
                'policy_hz': 0  # Invalid value
            }
        }

        self._create_test_config(config_dict)

        with pytest.raises(ValueError, match="agents_per_break_force must be positive"):
            ConfigManagerV2.load_defaults_first_config(self.config_path)

    def test_hybrid_agent_consistency_integration(self):
        """Test hybrid agent configuration consistency in integration."""
        # Test case: hybrid enabled but no hybrid config
        config_dict = {
            'task_name': 'peg_insert',
            'model': {
                'use_hybrid_agent': True
                # Missing hybrid_agent section
            }
        }

        self._create_test_config(config_dict)

        with pytest.raises(ValueError, match="Model config specifies use_hybrid_agent=True but no hybrid config provided"):
            ConfigManagerV2.load_defaults_first_config(self.config_path)

    def test_complex_nested_overrides_integration(self):
        """Test complex nested configuration overrides in integration."""
        config_dict = {
            'task_name': 'peg_insert',
            'primary': {
                'decimation': 8
            },
            'environment': {
                'ctrl': {
                    'force_action_bounds': [25.0, 25.0, 25.0]
                }
            },
            'model': {
                'actor': {
                    'n': 1,
                    'latent_size': 128
                },
                'critic': {
                    'n': 2,
                    'latent_size': 256
                }
            },
            'wrappers': {
                'force_torque_sensor': {
                    'enabled': True,
                    'use_tanh_scaling': False,
                    'tanh_scale': 0.03
                },
                'observation_noise': {
                    'enabled': True,
                    'global_scale': 1.2,
                    'apply_to_critic': True
                }
            }
        }

        self._create_test_config(config_dict)

        cli_overrides = [
            'primary.decimation=12',
            'environment.ctrl.force_action_bounds=[50.0,50.0,50.0]',
            'model.actor_latent_size=512',
            'wrappers.force_torque_sensor_tanh_scale=0.08',
            'wrappers.observation_noise_global_scale=0.8'
        ]

        bundle = ConfigManagerV2.load_defaults_first_config(
            self.config_path,
            cli_overrides=cli_overrides
        )

        # Verify nested overrides were applied correctly
        assert bundle.primary.decimation == 12  # CLI override
        assert bundle.environment.ctrl.force_action_bounds == [50.0, 50.0, 50.0]  # CLI override
        assert bundle.model.actor_n == 1  # YAML value
        assert bundle.model.actor_latent_size == 512  # CLI override
        assert bundle.model.critic_n == 2  # YAML value
        assert bundle.model.critic_latent_size == 256  # YAML value
        assert bundle.wrappers.force_torque_sensor_enabled is True  # YAML value
        assert bundle.wrappers.force_torque_use_tanh_scaling is False  # YAML value
        assert bundle.wrappers.force_torque_tanh_scale == 0.08  # CLI override
        assert bundle.wrappers.observation_noise_enabled is True  # YAML value
        assert bundle.wrappers.observation_noise_global_scale == 0.8  # CLI override
        assert bundle.wrappers.observation_noise_apply_to_critic is True  # YAML value

    def test_real_config_file_loading(self):
        """Test loading actual configuration files from the repository."""
        # Test loading factory base config
        base_config_path = '/home/hunter/Continuous_Force_RL/configs/base/factory_base_v2.yaml'

        if os.path.exists(base_config_path):
            bundle = ConfigManagerV2.load_defaults_first_config(base_config_path)

            # Verify basic structure
            assert bundle.task_name == 'Isaac-Factory-PegInsert-Direct-v0'
            assert isinstance(bundle.primary.agents_per_break_force, int)
            assert isinstance(bundle.primary.num_envs_per_agent, int)
            assert bundle.environment.episode_length_s == 10.0
            assert hasattr(bundle.wrappers, 'wandb_logging_enabled')

        # Test loading hybrid experiment config
        exp_config_path = '/home/hunter/Continuous_Force_RL/configs/experiments/hybrid_control_exp_v2.yaml'

        if os.path.exists(exp_config_path):
            bundle = ConfigManagerV2.load_defaults_first_config(exp_config_path)

            # Verify hybrid-specific configuration
            assert bundle.task_name == 'Isaac-Factory-PegInsert-Direct-v0'
            assert bundle.model.use_hybrid_agent is True
            assert bundle.hybrid_cfg is not None
            assert bundle.wrappers.force_torque_sensor_enabled is True
            assert bundle.wrappers.hybrid_control_enabled is True


class TestConfigSystemErrorHandling:
    """Test error handling in the complete configuration system."""

    def test_missing_config_file(self):
        """Test error handling for missing configuration file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ConfigManagerV2.load_defaults_first_config('/nonexistent/config.yaml')

    def test_invalid_yaml_syntax(self):
        """Test error handling for invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: syntax: [\n")  # Invalid YAML
            temp_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                ConfigManagerV2.load_defaults_first_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_configuration_validation_errors(self):
        """Test various configuration validation errors."""
        invalid_configs = [
            # Invalid decimation
            {
                'task_name': 'peg_insert',
                'primary': {'decimation': 0}
            },
            # Invalid policy_hz
            {
                'task_name': 'peg_insert',
                'primary': {'policy_hz': -1}
            },
            # Invalid agents_per_break_force
            {
                'task_name': 'peg_insert',
                'primary': {'agents_per_break_force': 0}
            }
        ]

        for i, config_dict in enumerate(invalid_configs):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config_dict, f)
                temp_path = f.name

            try:
                with pytest.raises(ValueError):
                    ConfigManagerV2.load_defaults_first_config(temp_path)
            finally:
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])