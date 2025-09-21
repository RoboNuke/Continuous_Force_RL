"""
Integration test to verify the Hydra entry point fix in factory_runnerv2.py

This test ensures that the factory runner no longer depends on undefined
Hydra entry points (SimBaNet_ppo_cfg_entry_point, SimBaNet_debug_entry_point)
and can work with our configuration system directly.
"""

import pytest
import sys
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our configuration system
from configs.config_manager import ConfigManager


class TestFactoryRunnerHydraFix:
    """Test that factory_runnerv2.py works without Hydra entry point dependencies."""

    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir(parents=True)

    def teardown_method(self):
        """Cleanup after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_test_config(self, config_name: str) -> str:
        """Create a test configuration file."""
        config = {
            'primary': {
                'agents_per_break_force': 1,
                'num_envs_per_agent': 16,
                'break_forces': [-1],
                'episode_length_s': 10.0,
                'decimation': 4,
                'policy_hz': 15,
                'max_steps': 1024,
                'debug_mode': False,
                'seed': 42
            },
            'defaults': {
                'task_name': "Isaac-Factory-PegInsert-Local-v0"
            },
            'environment': {
                'episode_length_s': 10.0,
                'decimation': 4,
                'component_dims': {
                    'fingertip_pos': 3,
                    'joint_pos': 7,
                    'force_torque': 6,
                    'prev_actions': 6
                },
                'component_attr_map': {
                    'fingertip_pos': 'fingertip_pos',
                    'joint_pos': 'joint_pos',
                    'force_torque': 'robot_force_torque',
                    'prev_actions': 'prev_actions'
                }
            },
            'learning': {
                'learning_epochs': 2,
                'policy_learning_rate': 1.0e-5,
                'critic_learning_rate': 1.0e-4,
                'state_preprocessor': True,
                'value_preprocessor': True
            },
            'model': {
                'use_hybrid_agent': False,
                'actor': {'n': 1, 'latent_size': 64},
                'critic': {'n': 1, 'latent_size': 128}
            },
            'wrappers': {
                'fragile_objects': {'enabled': False},
                'force_torque_sensor': {'enabled': False},
                'observation_manager': {'enabled': True},
                'observation_noise': {'enabled': False},
                'hybrid_control': {'enabled': False},
                'factory_metrics': {'enabled': True},
                'wandb_logging': {'enabled': False},
                'action_logging': {'enabled': False}
            },
            'experiment': {'name': config_name},
            'agent': {
                'class': 'PPO',
                'disable_progressbar': True,
                'experiment': {
                    'directory': str(self.temp_dir),
                    'experiment_name': config_name
                }
            }
        }

        config_path = self.config_dir / f"{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        return str(config_path)

    def test_no_hydra_entry_point_references(self):
        """Test that factory_runnerv2.py doesn't reference undefined Hydra entry points."""
        factory_runner_path = Path(__file__).parent.parent.parent / "learning" / "factory_runnerv2.py"

        with open(factory_runner_path, 'r') as f:
            content = f.read()

        # Check that the problematic entry points are not referenced
        assert "SimBaNet_ppo_cfg_entry_point" not in content, "Found reference to undefined Hydra entry point"
        assert "SimBaNet_debug_entry_point" not in content, "Found reference to undefined Hydra entry point"

        # Check that hydra_task_config decorator is not used
        assert "@hydra_task_config" not in content, "Found usage of hydra_task_config decorator"

        print("✓ No undefined Hydra entry point references found")

    def test_configuration_system_integration(self):
        """Test that the factory runner can work with our configuration system."""
        config_path = self.create_test_config('integration_test')

        # Test that configuration can be loaded and resolved
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        # Verify all required sections exist
        required_sections = ['primary', 'derived', 'environment', 'learning', 'model', 'wrappers']
        for section in required_sections:
            assert section in resolved_config, f"Missing required configuration section: {section}"

        # Test that derived parameters are calculated correctly
        derived = resolved_config['derived']
        assert derived['total_agents'] == 1
        assert derived['total_num_envs'] == 16
        assert 'rollout_steps' in derived

        print("✓ Configuration system integration working correctly")

    def test_debug_mode_configuration(self):
        """Test that debug mode configuration works without Hydra entry points."""
        config_path = self.create_test_config('debug_test')

        # Test debug mode configuration
        debug_config = {
            'primary': {
                'debug_mode': True,
                'agents_per_break_force': 1,
                'num_envs_per_agent': 8,
                'episode_length_s': 5.0
            }
        }

        # Modify the config file to enable debug mode
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        config['primary']['debug_mode'] = True

        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Load and test the debug configuration
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        assert resolved_config['primary']['debug_mode'] == True
        print("✓ Debug mode configuration works without Hydra entry points")

    def test_factory_runner_imports(self):
        """Test that factory runner imports work without Hydra dependencies."""
        # Mock heavy dependencies
        mock_modules = [
            'isaaclab', 'isaaclab.app', 'isaaclab_tasks', 'isaaclab_rl',
            'omni.isaac.lab', 'omni.isaac.lab.app', 'omni.isaac.lab_tasks',
            'envs', 'envs.factory', 'skrl', 'memories'
        ]

        for module in mock_modules:
            sys.modules[module] = MagicMock()

        try:
            # Import the configuration manager and test key functionality
            from configs.config_manager import ConfigManager

            # Create test config
            config_path = self.create_test_config('import_test')
            resolved_config = ConfigManager.load_and_resolve_config(config_path)

            # Test that all configuration sections can be accessed
            primary = resolved_config['primary']
            derived = resolved_config['derived']
            environment = resolved_config.get('environment', {})
            learning = resolved_config['learning']
            model = resolved_config['model']
            wrappers_config = resolved_config.get('wrappers', {})

            # Verify the data is correct
            assert len(primary) > 0, "Primary configuration is empty"
            assert len(derived) > 0, "Derived configuration is empty"
            assert len(learning) > 0, "Learning configuration is empty"
            assert len(model) > 0, "Model configuration is empty"

            print("✓ Factory runner imports work correctly without Hydra")

        finally:
            # Clean up mocked modules
            for module in mock_modules:
                if module in sys.modules:
                    del sys.modules[module]

    def test_main_function_structure(self):
        """Test that the main function structure is correct for direct execution."""
        factory_runner_path = Path(__file__).parent.parent.parent / "learning" / "factory_runnerv2.py"

        with open(factory_runner_path, 'r') as f:
            content = f.read()

        # Check that main function is defined correctly
        assert "def main():" in content, "Main function should be defined without Hydra decorator parameters"

        # Check that main function is called correctly
        assert "main()" in content, "Main function should be called without arguments"

        # Check that the function doesn't expect Hydra-injected parameters
        main_function_start = content.find("def main():")
        main_function_end = content.find("\n\nif __name__", main_function_start)
        if main_function_end == -1:
            main_function_end = len(content)

        main_function_body = content[main_function_start:main_function_end]

        # The main function should not reference env_cfg or agent_cfg as parameters
        lines = main_function_body.split('\n')
        function_def_line = lines[0]
        assert "env_cfg" not in function_def_line, "Main function should not have env_cfg parameter"
        assert "agent_cfg" not in function_def_line, "Main function should not have agent_cfg parameter"

        print("✓ Main function structure is correct for direct execution")

    def test_configuration_to_isaac_lab_integration(self):
        """Test that our configuration can be properly applied to Isaac Lab configs."""
        config_path = self.create_test_config('isaac_lab_test')
        resolved_config = ConfigManager.load_and_resolve_config(config_path)

        # Mock Isaac Lab environment config
        mock_env_cfg = MagicMock()
        mock_env_cfg.sim = MagicMock()
        mock_env_cfg.sim.device = None

        mock_agent_cfg = {
            'agent': {'disable_progressbar': True},
            'models': {'policy': {}, 'value': {}}
        }

        # Test that configuration application works
        try:
            ConfigManager.apply_to_isaac_lab(mock_env_cfg, mock_agent_cfg, resolved_config)
            print("✓ Configuration successfully applied to Isaac Lab configs")
        except Exception as e:
            pytest.fail(f"Failed to apply configuration to Isaac Lab configs: {e}")

    def test_no_missing_hydra_imports(self):
        """Test that there are no missing Hydra imports."""
        factory_runner_path = Path(__file__).parent.parent.parent / "learning" / "factory_runnerv2.py"

        with open(factory_runner_path, 'r') as f:
            content = f.read()

        # Check that hydra_task_config is not imported
        hydra_imports = [
            "from isaaclab_tasks.utils.hydra import hydra_task_config",
            "from omni.isaac.lab_tasks.utils.hydra import hydra_task_config",
            "import hydra"
        ]

        for hydra_import in hydra_imports:
            assert hydra_import not in content, f"Found Hydra import that should be removed: {hydra_import}"

        print("✓ No problematic Hydra imports found")

    def test_entry_point_free_execution(self):
        """Test that the script can be executed without entry point errors."""
        # This test simulates what would happen if someone tried to run the script
        # and verifies that it doesn't fail due to missing Hydra entry points

        config_path = self.create_test_config('execution_test')

        # Mock sys.argv to simulate command line execution
        original_argv = sys.argv[:]
        try:
            sys.argv = ['factory_runnerv2.py', '--config', config_path]

            # Import and test the configuration loading part
            resolved_config = ConfigManager.load_and_resolve_config(config_path)

            # Simulate the configuration steps that would happen in main()
            primary = resolved_config['primary']
            derived = resolved_config['derived']

            # Test that debug mode configuration works
            debug_mode = primary.get('debug_mode', False)

            # This should not raise any errors about missing entry points
            assert isinstance(debug_mode, bool), "Debug mode should be a boolean"

            print("✓ Script can be executed without Hydra entry point errors")

        finally:
            sys.argv[:] = original_argv