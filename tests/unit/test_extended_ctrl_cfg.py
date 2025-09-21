"""
Unit tests for ExtendedCtrlCfg

Tests that our extended control configuration properly inherits from Isaac Lab's
CtrlCfg and adds our custom force control parameters for wandb serialization.
"""

import pytest
import copy
from configs.isaac_lab_extensions.ctrl_cfg import ExtendedCtrlCfg


class TestExtendedCtrlCfg:
    """Test cases for ExtendedCtrlCfg class."""

    def test_extended_ctrl_cfg_creation(self):
        """Test that ExtendedCtrlCfg can be created successfully."""
        ctrl_cfg = ExtendedCtrlCfg()
        assert ctrl_cfg is not None
        assert isinstance(ctrl_cfg, ExtendedCtrlCfg)

    def test_extended_ctrl_cfg_has_inherited_attributes(self):
        """Test that ExtendedCtrlCfg has all inherited Isaac Lab attributes."""
        ctrl_cfg = ExtendedCtrlCfg()

        # Check inherited attributes from Isaac Lab CtrlCfg
        inherited_attrs = [
            'ema_factor', 'pos_action_bounds', 'rot_action_bounds',
            'pos_action_threshold', 'rot_action_threshold', 'default_task_prop_gains',
            'kp_null', 'kd_null', 'reset_joints', 'reset_task_prop_gains',
            'reset_rot_deriv_scale', 'default_dof_pos_tensor'
        ]

        for attr in inherited_attrs:
            assert hasattr(ctrl_cfg, attr), f"Missing inherited attribute: {attr}"
            assert getattr(ctrl_cfg, attr) is not None, f"Inherited attribute {attr} is None"

    def test_extended_ctrl_cfg_has_new_attributes(self):
        """Test that ExtendedCtrlCfg has all our new force control attributes."""
        ctrl_cfg = ExtendedCtrlCfg()

        # Check our new attributes
        new_attrs = [
            'force_action_bounds', 'torque_action_bounds', 'force_action_threshold',
            'torque_action_threshold', 'default_task_force_gains'
        ]

        for attr in new_attrs:
            assert hasattr(ctrl_cfg, attr), f"Missing new attribute: {attr}"
            assert getattr(ctrl_cfg, attr) is not None, f"New attribute {attr} is None"

    def test_extended_ctrl_cfg_default_values(self):
        """Test that ExtendedCtrlCfg has correct default values for new attributes."""
        ctrl_cfg = ExtendedCtrlCfg()

        # Check default values for our new attributes
        assert ctrl_cfg.force_action_bounds == [50.0, 50.0, 50.0]
        assert ctrl_cfg.torque_action_bounds == [0.5, 0.5, 0.5]
        assert ctrl_cfg.force_action_threshold == [10.0, 10.0, 10.0]
        assert ctrl_cfg.torque_action_threshold == [0.1, 0.1, 0.1]
        assert ctrl_cfg.default_task_force_gains == [0.1, 0.1, 0.1, 0.001, 0.001, 0.001]

    def test_extended_ctrl_cfg_attribute_modification(self):
        """Test that attributes can be modified after creation."""
        ctrl_cfg = ExtendedCtrlCfg()

        # Modify new attributes
        ctrl_cfg.force_action_bounds = [75.0, 75.0, 75.0]
        ctrl_cfg.torque_action_bounds = [1.0, 1.0, 1.0]
        ctrl_cfg.force_action_threshold = [15.0, 15.0, 15.0]
        ctrl_cfg.torque_action_threshold = [0.2, 0.2, 0.2]
        ctrl_cfg.default_task_force_gains = [0.2, 0.2, 0.2, 0.002, 0.002, 0.002]

        # Verify modifications
        assert ctrl_cfg.force_action_bounds == [75.0, 75.0, 75.0]
        assert ctrl_cfg.torque_action_bounds == [1.0, 1.0, 1.0]
        assert ctrl_cfg.force_action_threshold == [15.0, 15.0, 15.0]
        assert ctrl_cfg.torque_action_threshold == [0.2, 0.2, 0.2]
        assert ctrl_cfg.default_task_force_gains == [0.2, 0.2, 0.2, 0.002, 0.002, 0.002]

    def test_extended_ctrl_cfg_deepcopy(self):
        """Test that ExtendedCtrlCfg can be deep copied successfully."""
        ctrl_cfg = ExtendedCtrlCfg()

        # Modify some values
        ctrl_cfg.force_action_bounds = [75.0, 75.0, 75.0]
        ctrl_cfg.ema_factor = 0.5

        # Deep copy
        copied_cfg = copy.deepcopy(ctrl_cfg)

        # Verify copy is independent
        assert copied_cfg is not ctrl_cfg
        assert copied_cfg.force_action_bounds == [75.0, 75.0, 75.0]
        assert copied_cfg.ema_factor == 0.5

        # Modify original
        ctrl_cfg.force_action_bounds = [100.0, 100.0, 100.0]

        # Verify copy is unchanged
        assert copied_cfg.force_action_bounds == [75.0, 75.0, 75.0]

    def test_extended_ctrl_cfg_dict_serialization(self):
        """Test that ExtendedCtrlCfg properly serializes to __dict__ for wandb."""
        ctrl_cfg = ExtendedCtrlCfg()

        # Check that __dict__ exists
        assert hasattr(ctrl_cfg, '__dict__'), "ExtendedCtrlCfg should have __dict__"

        # Check that all target attributes are in __dict__
        target_attrs = [
            'force_action_bounds', 'torque_action_bounds', 'force_action_threshold',
            'torque_action_threshold', 'default_task_force_gains'
        ]

        dict_keys = list(ctrl_cfg.__dict__.keys())
        for attr in target_attrs:
            assert attr in dict_keys, f"Attribute {attr} missing from __dict__"

    def test_extended_ctrl_cfg_wandb_serialization_simulation(self):
        """Test that ExtendedCtrlCfg works correctly in wandb-style serialization."""
        ctrl_cfg = ExtendedCtrlCfg()

        # Modify some values to ensure they're preserved
        ctrl_cfg.force_action_bounds = [75.0, 75.0, 75.0]
        ctrl_cfg.torque_action_bounds = [1.0, 1.0, 1.0]

        # Simulate wandb config preparation (what wandb wrapper does)
        combined_config = copy.deepcopy(ctrl_cfg)
        combined_config.__dict__.update({'test_agent_param': 'test_value'})

        # Verify our attributes are still present and correct
        assert hasattr(combined_config, 'force_action_bounds')
        assert hasattr(combined_config, 'torque_action_bounds')
        assert combined_config.force_action_bounds == [75.0, 75.0, 75.0]
        assert combined_config.torque_action_bounds == [1.0, 1.0, 1.0]

        # Verify agent param was added
        assert hasattr(combined_config, 'test_agent_param')
        assert combined_config.test_agent_param == 'test_value'

    def test_extended_ctrl_cfg_with_none_values(self):
        """Test ExtendedCtrlCfg behavior when initialized with None values."""
        # Test that we can create with explicit None values and they get set to defaults
        ctrl_cfg = ExtendedCtrlCfg()
        ctrl_cfg.force_action_bounds = None
        ctrl_cfg.torque_action_bounds = None

        # Call __post_init__ manually to test the default setting logic
        ctrl_cfg.__post_init__()

        # Verify defaults were set
        assert ctrl_cfg.force_action_bounds == [50.0, 50.0, 50.0]
        assert ctrl_cfg.torque_action_bounds == [0.5, 0.5, 0.5]

    def test_extended_ctrl_cfg_inheritance_chain(self):
        """Test that ExtendedCtrlCfg properly inherits from base CtrlCfg."""
        ctrl_cfg = ExtendedCtrlCfg()

        # Check that we inherit from the base class
        from configs.isaac_lab_extensions.version_compat import get_isaac_lab_ctrl_imports
        _, CtrlCfg = get_isaac_lab_ctrl_imports()

        # In testing environment, we use MockCtrlCfg, so check inheritance correctly
        assert isinstance(ctrl_cfg, CtrlCfg), "ExtendedCtrlCfg should inherit from CtrlCfg (or MockCtrlCfg in testing)"

        # Also check that ExtendedCtrlCfg is a subclass of the base
        assert issubclass(ExtendedCtrlCfg, CtrlCfg), "ExtendedCtrlCfg should be a subclass of CtrlCfg"

    def test_extended_ctrl_cfg_all_attributes_accessible(self):
        """Test that all attributes are accessible via getattr for wandb compatibility."""
        ctrl_cfg = ExtendedCtrlCfg()

        # List of all attributes we expect to exist
        all_expected_attrs = [
            # Inherited attributes
            'ema_factor', 'pos_action_bounds', 'rot_action_bounds',
            'pos_action_threshold', 'rot_action_threshold', 'default_task_prop_gains',
            'kp_null', 'kd_null',
            # Our new attributes
            'force_action_bounds', 'torque_action_bounds', 'force_action_threshold',
            'torque_action_threshold', 'default_task_force_gains'
        ]

        # Test that all attributes are accessible via getattr
        for attr in all_expected_attrs:
            try:
                value = getattr(ctrl_cfg, attr)
                assert value is not None, f"Attribute {attr} is None"
            except AttributeError:
                pytest.fail(f"Attribute {attr} not accessible via getattr")


class TestExtendedCtrlCfgIntegration:
    """Integration tests for ExtendedCtrlCfg with other components."""

    def test_extended_ctrl_cfg_with_config_manager(self):
        """Test that ExtendedCtrlCfg works with ConfigManager."""
        from configs.config_manager import ConfigManager

        # Create mock environment with ExtendedCtrlCfg
        class MockEnvCfg:
            def __init__(self):
                self.ctrl = ExtendedCtrlCfg()

        env_cfg = MockEnvCfg()

        # Test configuration that should set our new attributes
        environment_config = {
            'ctrl': {
                'force_action_bounds': [100.0, 100.0, 100.0],
                'torque_action_bounds': [2.0, 2.0, 2.0],
                'force_action_threshold': [20.0, 20.0, 20.0],
                'torque_action_threshold': [0.5, 0.5, 0.5],
                'default_task_force_gains': [0.3, 0.3, 0.3, 0.003, 0.003, 0.003]
            }
        }

        # Apply configuration
        ConfigManager._apply_isaac_lab_basic_config(env_cfg, environment_config)

        # Verify values were set correctly
        assert env_cfg.ctrl.force_action_bounds == [100.0, 100.0, 100.0]
        assert env_cfg.ctrl.torque_action_bounds == [2.0, 2.0, 2.0]
        assert env_cfg.ctrl.force_action_threshold == [20.0, 20.0, 20.0]
        assert env_cfg.ctrl.torque_action_threshold == [0.5, 0.5, 0.5]
        assert env_cfg.ctrl.default_task_force_gains == [0.3, 0.3, 0.3, 0.003, 0.003, 0.003]