"""
Unit tests for extended configuration classes.

Tests all the new extended configuration dataclasses including:
- PrimaryConfig
- ExtendedFactoryEnvCfg and task-specific configs
- ExtendedModelConfig and ExtendedHybridAgentConfig
- ExtendedWrapperConfig
- ExtendedPPOConfig
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

# Import configuration classes to test
from configs.cfg_exts.primary_cfg import PrimaryConfig
from configs.cfg_exts.extended_factory_env_cfg import ExtendedFactoryEnvCfg
from configs.cfg_exts.extended_peg_insert_cfg import ExtendedFactoryTaskPegInsertCfg
from configs.cfg_exts.extended_gear_mesh_cfg import ExtendedFactoryTaskGearMeshCfg
from configs.cfg_exts.extended_nut_thread_cfg import ExtendedFactoryTaskNutThreadCfg
from configs.cfg_exts.extended_model_cfg import ExtendedModelConfig, ExtendedHybridAgentConfig
from configs.cfg_exts.extended_wrapper_cfg import ExtendedWrapperConfig
from agents.extended_ppo_cfg import ExtendedPPOConfig


class TestPrimaryConfig:
    """Test PrimaryConfig dataclass."""

    def test_primary_config_creation(self):
        """Test that PrimaryConfig can be created with defaults."""
        primary_cfg = PrimaryConfig()

        assert primary_cfg.agents_per_break_force == 2
        assert primary_cfg.num_envs_per_agent == 256
        assert primary_cfg.break_forces == -1
        assert primary_cfg.decimation == 8
        assert primary_cfg.policy_hz == 15
        assert primary_cfg.max_steps == 10240000
        assert primary_cfg.debug_mode == False
        assert primary_cfg.seed == -1
        assert primary_cfg.ctrl_torque == False

    def test_primary_config_computed_properties(self):
        """Test computed properties work correctly."""
        primary_cfg = PrimaryConfig()

        # Test with default single break force
        assert primary_cfg.total_agents == 2  # 1 * 2
        assert primary_cfg.total_num_envs == 512  # 2 * 256
        assert primary_cfg.sim_dt == pytest.approx(0.008333333333, rel=1e-6)  # (1/15) / 8

    def test_primary_config_computed_properties_multiple_forces(self):
        """Test computed properties with multiple break forces."""
        primary_cfg = PrimaryConfig(break_forces=[10, 20, 30])

        assert primary_cfg.total_agents == 6  # 3 * 2
        assert primary_cfg.total_num_envs == 1536  # 6 * 256

    def test_primary_config_rollout_steps(self):
        """Test rollout steps calculation."""
        primary_cfg = PrimaryConfig()

        rollout_steps = primary_cfg.rollout_steps(10.0)
        expected = int((1.0 / primary_cfg.sim_dt) / primary_cfg.decimation * 10.0)
        assert rollout_steps == expected

    def test_primary_config_validation(self):
        """Test parameter validation."""
        # Test invalid agents_per_break_force
        with pytest.raises(ValueError, match="agents_per_break_force must be a positive integer"):
            PrimaryConfig(agents_per_break_force=0)

        # Test invalid break_forces
        with pytest.raises(ValueError, match="break_forces list cannot be empty"):
            PrimaryConfig(break_forces=[])

        # Test invalid policy_hz
        with pytest.raises(ValueError, match="policy_hz seems unreasonably high"):
            PrimaryConfig(policy_hz=2000)

    def test_primary_config_utility_methods(self):
        """Test utility methods."""
        primary_cfg = PrimaryConfig(break_forces=[10, 20])

        assert primary_cfg.get_num_break_forces() == 2
        assert primary_cfg.is_multi_agent() == True
        assert primary_cfg.is_multi_break_force() == True

        single_force_cfg = PrimaryConfig()
        assert single_force_cfg.get_num_break_forces() == 1
        assert single_force_cfg.is_multi_agent() == True  # Still 2 agents per force
        assert single_force_cfg.is_multi_break_force() == False


class TestExtendedFactoryEnvCfg:
    """Test ExtendedFactoryEnvCfg."""

    def test_extended_factory_env_cfg_creation(self):
        """Test that ExtendedFactoryEnvCfg can be created."""
        env_cfg = ExtendedFactoryEnvCfg()

        assert env_cfg is not None
        assert hasattr(env_cfg, 'decimation')
        assert hasattr(env_cfg, 'episode_length_s')
        assert hasattr(env_cfg, 'ctrl')

    def test_apply_primary_cfg(self):
        """Test applying primary configuration."""
        env_cfg = ExtendedFactoryEnvCfg()
        primary_cfg = PrimaryConfig(decimation=4, num_envs_per_agent=128)

        env_cfg.apply_primary_cfg(primary_cfg)

        assert env_cfg.decimation == 4
        assert hasattr(env_cfg, '_primary_cfg')
        assert env_cfg._primary_cfg == primary_cfg

    def test_computed_properties(self):
        """Test computed properties work with primary config."""
        env_cfg = ExtendedFactoryEnvCfg()
        primary_cfg = PrimaryConfig(agents_per_break_force=3, num_envs_per_agent=128)

        env_cfg.apply_primary_cfg(primary_cfg)

        assert env_cfg.num_agents == 3
        assert env_cfg.total_envs == 384  # 3 * 128
        assert env_cfg.sim_dt == primary_cfg.sim_dt

    def test_get_rollout_steps(self):
        """Test rollout steps calculation."""
        env_cfg = ExtendedFactoryEnvCfg()
        primary_cfg = PrimaryConfig()

        env_cfg.apply_primary_cfg(primary_cfg)

        rollout_steps = env_cfg.get_rollout_steps()
        expected = primary_cfg.rollout_steps(env_cfg.episode_length_s)
        assert rollout_steps == expected

    def test_validation(self):
        """Test configuration validation."""
        env_cfg = ExtendedFactoryEnvCfg()
        primary_cfg = PrimaryConfig()
        env_cfg.apply_primary_cfg(primary_cfg)

        # Should pass validation
        env_cfg.validate_configuration()

        # Test validation failure - invalid decimation
        env_cfg.decimation = 0
        with pytest.raises(ValueError, match="decimation must be positive"):
            env_cfg.validate_configuration()


class TestTaskSpecificConfigs:
    """Test task-specific configuration classes."""

    def test_peg_insert_config(self):
        """Test ExtendedFactoryTaskPegInsertCfg."""
        cfg = ExtendedFactoryTaskPegInsertCfg()
        primary_cfg = PrimaryConfig()

        cfg.apply_primary_cfg(primary_cfg)

        assert cfg.task_name == "peg_insert"
        assert cfg.episode_length_s == 10.0  # Isaac Lab default
        assert hasattr(cfg, 'ctrl')

        # Test validation
        cfg.validate_configuration()

    def test_gear_mesh_config(self):
        """Test ExtendedFactoryTaskGearMeshCfg."""
        cfg = ExtendedFactoryTaskGearMeshCfg()
        primary_cfg = PrimaryConfig()

        cfg.apply_primary_cfg(primary_cfg)

        assert cfg.task_name == "gear_mesh"
        assert cfg.episode_length_s == 20.0  # Isaac Lab default

        # Test validation
        cfg.validate_configuration()

    def test_nut_thread_config(self):
        """Test ExtendedFactoryTaskNutThreadCfg."""
        cfg = ExtendedFactoryTaskNutThreadCfg()
        primary_cfg = PrimaryConfig()

        cfg.apply_primary_cfg(primary_cfg)

        assert cfg.task_name == "nut_thread"
        assert cfg.episode_length_s == 30.0  # Isaac Lab default

        # Test validation
        cfg.validate_configuration()

    def test_task_config_validation_errors(self):
        """Test task-specific validation errors."""
        peg_cfg = ExtendedFactoryTaskPegInsertCfg()
        primary_cfg = PrimaryConfig()
        peg_cfg.apply_primary_cfg(primary_cfg)

        # Test invalid episode length
        peg_cfg.episode_length_s = 100.0
        with pytest.raises(ValueError, match="episode_length_s for peg insert should be 5-60 seconds"):
            peg_cfg.validate_configuration()


class TestExtendedModelConfig:
    """Test ExtendedModelConfig and ExtendedHybridAgentConfig."""

    def test_model_config_creation(self):
        """Test ExtendedModelConfig creation."""
        model_cfg = ExtendedModelConfig()

        assert model_cfg.force_encoding is None
        assert model_cfg.last_layer_scale == 1.0
        assert model_cfg.act_init_std == 1.0
        assert model_cfg.critic_output_init_mean == 50.0
        assert model_cfg.use_hybrid_agent == False
        assert model_cfg.actor_n == 1
        assert model_cfg.actor_latent_size == 256
        assert model_cfg.critic_n == 3
        assert model_cfg.critic_latent_size == 1024

    def test_model_config_validation(self):
        """Test model config validation."""
        # Valid config should pass
        model_cfg = ExtendedModelConfig()
        model_cfg._validate_model_params()  # Should not raise

        # Invalid actor_n
        with pytest.raises(ValueError, match="actor_n must be positive"):
            ExtendedModelConfig(actor_n=0)

        # Invalid latent size
        with pytest.raises(ValueError, match="actor_latent_size must be positive"):
            ExtendedModelConfig(actor_latent_size=-1)

    def test_model_config_methods(self):
        """Test model config utility methods."""
        model_cfg = ExtendedModelConfig(use_hybrid_agent=True)

        assert model_cfg.is_hybrid_agent() == True

        actor_config = model_cfg.get_actor_config()
        assert actor_config == {'n': 1, 'latent_size': 256}

        critic_config = model_cfg.get_critic_config()
        assert critic_config == {'n': 3, 'latent_size': 1024}

    def test_model_config_serialization(self):
        """Test model config to_dict and from_dict."""
        model_cfg = ExtendedModelConfig(
            use_hybrid_agent=True,
            actor_latent_size=128,
            critic_latent_size=512
        )

        config_dict = model_cfg.to_dict()
        assert config_dict['use_hybrid_agent'] == True
        assert config_dict['actor']['latent_size'] == 128
        assert config_dict['critic']['latent_size'] == 512

        # Test from_dict
        new_cfg = ExtendedModelConfig.from_dict(config_dict)
        assert new_cfg.use_hybrid_agent == True
        assert new_cfg.actor_latent_size == 128
        assert new_cfg.critic_latent_size == 512

    def test_hybrid_agent_config(self):
        """Test ExtendedHybridAgentConfig."""
        hybrid_cfg = ExtendedHybridAgentConfig()

        assert hybrid_cfg.ctrl_torque == False
        assert hybrid_cfg.unit_std_init == True
        assert hybrid_cfg.pos_init_std == 1.0
        assert hybrid_cfg.force_init_std == 1.0
        assert hybrid_cfg.pos_scale == 1.0
        assert hybrid_cfg.force_scale == 1.0

    def test_hybrid_agent_config_validation(self):
        """Test hybrid agent config validation."""
        # Valid config should pass
        hybrid_cfg = ExtendedHybridAgentConfig()
        hybrid_cfg._validate_hybrid_params()  # Should not raise

        # Invalid standard deviation
        with pytest.raises(ValueError, match="pos_init_std must be positive"):
            ExtendedHybridAgentConfig(pos_init_std=0)

        # Invalid sampling rate
        with pytest.raises(ValueError, match="uniform_sampling_rate must be in"):
            ExtendedHybridAgentConfig(uniform_sampling_rate=1.5)


class TestExtendedWrapperConfig:
    """Test ExtendedWrapperConfig."""

    def test_wrapper_config_creation(self):
        """Test ExtendedWrapperConfig creation."""
        wrapper_cfg = ExtendedWrapperConfig()

        assert wrapper_cfg.fragile_objects_enabled == True
        assert wrapper_cfg.force_torque_sensor_enabled == False
        assert wrapper_cfg.observation_noise_enabled == False
        assert wrapper_cfg.hybrid_control_enabled == False
        assert wrapper_cfg.wandb_logging_enabled == True

    def test_wrapper_config_apply_primary(self):
        """Test applying primary config to wrapper config."""
        wrapper_cfg = ExtendedWrapperConfig()
        primary_cfg = PrimaryConfig(break_forces=[10, 20], ctrl_torque=True)

        wrapper_cfg.apply_primary_cfg(primary_cfg)

        assert hasattr(wrapper_cfg, '_primary_cfg')
        assert wrapper_cfg._primary_cfg == primary_cfg

    def test_wrapper_config_getters(self):
        """Test wrapper config getter methods."""
        wrapper_cfg = ExtendedWrapperConfig()
        primary_cfg = PrimaryConfig(break_forces=[10, 20], agents_per_break_force=3)
        wrapper_cfg.apply_primary_cfg(primary_cfg)

        # Test fragile objects config
        fragile_config = wrapper_cfg.get_fragile_objects_config()
        assert fragile_config['enabled'] == True
        assert fragile_config['break_force'] == [10, 20]
        assert fragile_config['num_agents'] == 6

        # Test force-torque sensor config
        ft_config = wrapper_cfg.get_force_torque_sensor_config()
        assert ft_config['enabled'] == False
        assert ft_config['tanh_scale'] == 0.03

        # Test wandb config
        wandb_config = wrapper_cfg.get_wandb_config()
        assert wandb_config['enabled'] == True
        assert wandb_config['wandb_project'] == "Continuous_Force_RL"

    def test_wrapper_config_validation(self):
        """Test wrapper config validation."""
        # Valid config should pass
        wrapper_cfg = ExtendedWrapperConfig()
        wrapper_cfg._validate_wrapper_params()  # Should not raise

        # Invalid tanh scale
        with pytest.raises(ValueError, match="force_torque_tanh_scale must be positive"):
            ExtendedWrapperConfig(force_torque_tanh_scale=-1)

        # Invalid merge strategy
        with pytest.raises(ValueError, match="observation_manager_merge_strategy must be one of"):
            ExtendedWrapperConfig(observation_manager_merge_strategy="invalid")


class TestExtendedPPOConfig:
    """Test ExtendedPPOConfig."""

    def test_ppo_config_creation(self):
        """Test ExtendedPPOConfig creation with SKRL defaults."""
        ppo_cfg = ExtendedPPOConfig()

        # Test SKRL defaults
        assert ppo_cfg.rollouts == 16
        assert ppo_cfg.learning_epochs == 8
        assert ppo_cfg.mini_batches == 2
        assert ppo_cfg.discount_factor == 0.99
        assert ppo_cfg.lambda_ == 0.95
        assert ppo_cfg.learning_rate == 1e-3

        # Test custom extensions
        assert ppo_cfg.num_agents == 1
        assert ppo_cfg.policy_learning_rate == 1e-3  # Defaults to learning_rate
        assert ppo_cfg.critic_learning_rate == 1e-3  # Defaults to learning_rate

    def test_ppo_config_apply_primary(self):
        """Test applying primary config to PPO config."""
        ppo_cfg = ExtendedPPOConfig()
        primary_cfg = PrimaryConfig(agents_per_break_force=3, break_forces=[1, 2])

        ppo_cfg.apply_primary_cfg(primary_cfg)

        assert ppo_cfg.num_agents == 6  # 3 * 2
        assert hasattr(ppo_cfg, '_primary_cfg')

    def test_ppo_config_computed_values(self):
        """Test computed rollout and batch values."""
        ppo_cfg = ExtendedPPOConfig()
        primary_cfg = PrimaryConfig()
        ppo_cfg.apply_primary_cfg(primary_cfg)

        # Test rollout steps calculation
        rollout_steps = ppo_cfg.get_rollout_steps(10.0)
        expected = primary_cfg.rollout_steps(10.0)
        assert rollout_steps == expected

        # Test computed rollouts
        computed_rollouts = ppo_cfg.get_computed_rollouts(10.0)
        assert computed_rollouts >= 16  # Minimum rollouts

        # Test computed mini_batches
        computed_mini_batches = ppo_cfg.get_computed_mini_batches(10.0)
        assert computed_mini_batches >= 2  # Minimum mini_batches

    def test_ppo_config_skrl_dict(self):
        """Test conversion to SKRL-compatible dictionary."""
        ppo_cfg = ExtendedPPOConfig(
            learning_rate=5e-4,
            policy_learning_rate=1e-6,
            critic_learning_rate=1e-5
        )
        primary_cfg = PrimaryConfig()
        ppo_cfg.apply_primary_cfg(primary_cfg)

        skrl_dict = ppo_cfg.to_skrl_dict(10.0)

        # Test core parameters
        assert skrl_dict['learning_rate'] == 5e-4
        assert skrl_dict['discount_factor'] == 0.99
        assert skrl_dict['lambda'] == 0.95  # Note: no underscore in SKRL

        # Test custom extensions
        assert skrl_dict['policy_learning_rate'] == 1e-6
        assert skrl_dict['critic_learning_rate'] == 1e-5

        # Test computed values
        assert 'rollouts' in skrl_dict
        assert 'mini_batches' in skrl_dict

        # Test optimizer structure
        assert 'optimizer' in skrl_dict
        assert 'betas' in skrl_dict['optimizer']

        # Test experiment structure
        assert 'experiment' in skrl_dict

    def test_ppo_config_validation(self):
        """Test PPO config validation."""
        # Valid config should pass
        ppo_cfg = ExtendedPPOConfig()
        ppo_cfg._validate_ppo_params()  # Should not raise

        # Invalid rollouts
        with pytest.raises(ValueError, match="rollouts must be positive"):
            ExtendedPPOConfig(rollouts=0)

        # Invalid discount factor
        with pytest.raises(ValueError, match="discount_factor must be in"):
            ExtendedPPOConfig(discount_factor=1.5)

        # Invalid learning rate
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            ExtendedPPOConfig(learning_rate=-1)

        # Invalid optimizer betas
        with pytest.raises(ValueError, match="optimizer_betas must have 2 elements"):
            ExtendedPPOConfig(optimizer_betas=[0.9])


if __name__ == "__main__":
    pytest.main([__file__])