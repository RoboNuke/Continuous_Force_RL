"""
Mock Isaac Lab Components for Unit Testing

This module provides mock implementations of Isaac Lab classes and components
used by the factory environment wrappers. These mocks simulate the behavior
of real Isaac Lab components without requiring the full simulation environment.
"""

import torch
import gymnasium as gym
from typing import Optional, Dict, Any, List


class MockRobotView:
    """
    Mock implementation of Isaac Lab's RobotView/ArticulationView for force-torque sensor testing.

    Real class reference:
    - Isaac Lab: omni.isaac.lab.assets.ArticulationView
    - Isaac Sim: omni.isaac.core.articulations.ArticulationView

    Real method: get_measured_joint_forces() -> torch.Tensor
    Documentation: https://isaac-sim.github.io/IsaacLab/source/api/omni.isaac.lab.assets.html#omni.isaac.lab.assets.ArticulationView
    """

    def __init__(self, prim_paths_expr: str = "/World/envs/env_.*/Robot", device: str = "cpu"):
        """
        Mock RobotView initialization.

        Real Isaac Lab RobotView.__init__():
        - Takes prim_paths_expr to find robot assets in scene
        - Initializes USD prim references
        - Sets up physics handles

        Mock behavior:
        - Stores parameters for reference
        - No actual USD/physics setup
        """
        self.prim_paths_expr = prim_paths_expr
        self.device = torch.device(device)
        self._initialized = False
        self._num_envs = 64  # Default for testing
        self._num_joints = 9  # Typical for factory robot

    def initialize(self) -> None:
        """
        Mock initialization of robot view.

        Real Isaac Lab ArticulationView.initialize():
        - Finds USD prims matching prim_paths_expr
        - Creates physics handles
        - Validates robot configurations

        Mock behavior:
        - Simply sets initialized flag
        - No actual physics initialization
        """
        self._initialized = True

    def get_measured_joint_forces(self, joint_indices=None) -> torch.Tensor:
        """
        Mock joint force measurement.

        Real Isaac Lab ArticulationView.get_measured_joint_forces():
        Documentation: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationView.get_measured_joint_forces
        - Returns measured joint reaction forces and torques (link incoming joint forces)
        - Shape: (num_envs, num_joints, 6) where 6 = [fx, fy, fz, tx, ty, tz]
        - First 3 values are forces, last 3 are torques
        - Units: Forces in Newtons, torques in Newton-meters
        - Joint forces are link incoming joint forces (forces exerted by joint connecting child to parent)

        Mock behavior:
        - Returns simple test values
        - Joint 8 (end-effector) has non-zero forces: [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        - Other joints have zero forces for simplicity
        - Ignores joint_indices parameter for simplicity
        """
        if not self._initialized:
            raise RuntimeError("RobotView not initialized. Call initialize() first.")

        # Create force tensor: (num_envs, num_joints, 6)
        forces = torch.zeros((self._num_envs, self._num_joints, 6), device=self.device)

        # Set joint 8 (end-effector) to have test force values
        forces[:, 8, :] = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], device=self.device)

        return forces

    def set_joint_position_target(self, targets):
        """
        Mock set joint position targets.

        Real Isaac Lab ArticulationView.set_joint_position_target():
        - Sets position targets for position-controlled joints
        - targets: tensor shape (num_envs, num_joints)

        Mock behavior:
        - Stores the targets for verification
        """
        self.last_position_targets = targets.clone()

    def set_joint_effort_target(self, efforts):
        """
        Mock set joint effort targets.

        Real Isaac Lab ArticulationView.set_joint_effort_target():
        - Sets torque targets for effort-controlled joints
        - efforts: tensor shape (num_envs, num_joints)

        Mock behavior:
        - Stores the efforts for verification
        """
        self.last_effort_targets = efforts.clone()

    @property
    def _data(self):
        """Mock data attribute with simulation timestamp."""
        class MockData:
            def __init__(self):
                self._sim_timestamp = 1.0
        return MockData()


class MockArticulation:
    """
    Mock articulation for EfficientResetWrapper testing.

    Real class reference:
    - Isaac Lab: omni.isaac.lab.assets.Articulation
    - Documentation: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.Articulation
    """

    def __init__(self, num_envs=64, device="cpu"):
        """Mock articulation initialization."""
        self.num_envs = num_envs
        self.device = torch.device(device)

    def write_root_pose_to_sim(self, pose, env_ids):
        """
        Mock write root pose to simulation.

        Real Isaac Lab Articulation.write_root_pose_to_sim():
        - Sets root pose (position + quaternion) in simulation
        - pose: tensor shape (num_envs, 7) [x, y, z, qw, qx, qy, qz]
        - env_ids: environment indices to update

        Mock behavior:
        - Simply stores the operation for verification
        """
        self.last_root_pose_write = {"pose": pose.clone(), "env_ids": env_ids}

    def write_root_velocity_to_sim(self, velocity, env_ids):
        """
        Mock write root velocity to simulation.

        Real Isaac Lab Articulation.write_root_velocity_to_sim():
        - Sets root velocity (linear + angular) in simulation
        - velocity: tensor shape (num_envs, 6) [vx, vy, vz, wx, wy, wz]
        - env_ids: environment indices to update

        Mock behavior:
        - Simply stores the operation for verification
        """
        self.last_root_velocity_write = {"velocity": velocity.clone(), "env_ids": env_ids}

    def write_joint_state_to_sim(self, position, velocity, env_ids):
        """
        Mock write joint state to simulation.

        Real Isaac Lab Articulation.write_joint_state_to_sim():
        - Sets joint positions and velocities in simulation
        - position: tensor shape (num_envs, num_joints)
        - velocity: tensor shape (num_envs, num_joints)
        - env_ids: environment indices to update

        Mock behavior:
        - Simply stores the operation for verification
        """
        self.last_joint_state_write = {
            "position": position.clone(),
            "velocity": velocity.clone(),
            "env_ids": env_ids
        }

    def reset(self, env_ids):
        """
        Mock articulation reset.

        Real Isaac Lab Articulation.reset():
        - Resets articulation physics state
        - Called after setting new states

        Mock behavior:
        - Simply stores the reset call for verification
        """
        self.last_reset_call = env_ids


class MockScene:
    """
    Mock scene for EfficientResetWrapper testing.

    Real class reference:
    - Isaac Lab: omni.isaac.lab.scene.InteractiveScene
    - Documentation: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html
    """

    def __init__(self, num_envs=64, device="cpu"):
        """Mock scene initialization."""
        self.num_envs = num_envs
        self.device = torch.device(device)

        # Mock articulations dictionary
        self.articulations = {
            "robot": MockArticulation(num_envs, device)
        }

        # Mock environment origins
        self.env_origins = torch.zeros(num_envs, 3, device=self.device)

    def get_state(self):
        """
        Mock get scene state.

        Real Isaac Lab InteractiveScene.get_state():
        - Returns complete scene state including all articulations
        - Format: nested dict with articulation states
        - Each articulation state contains: root_pose, root_velocity, joint_position, joint_velocity

        Mock behavior:
        - Returns mock state data for testing
        """
        return {
            "articulation": {
                "robot": {
                    "root_pose": torch.randn(self.num_envs, 7, device=self.device),
                    "root_velocity": torch.randn(self.num_envs, 6, device=self.device),
                    "joint_position": torch.randn(self.num_envs, 9, device=self.device),
                    "joint_velocity": torch.randn(self.num_envs, 9, device=self.device),
                }
            }
        }


class MockEnvironment(gym.Env):
    """
    Mock base environment for wrapper testing.

    Real class reference:
    - Isaac Lab: omni.isaac.lab.envs.ManagerBasedRLEnv
    - Documentation: https://isaac-sim.github.io/IsaacLab/source/api/omni.isaac.lab.envs.html#omni.isaac.lab.envs.ManagerBasedRLEnv
    """

    def __init__(self, num_envs: int = 64, device: str = "cpu"):
        """
        Mock environment initialization.

        Real Isaac Lab ManagerBasedRLEnv:
        - Sets up simulation scene
        - Initializes observation/action managers
        - Creates robot and object assets

        Mock behavior:
        - Sets basic attributes needed by wrappers
        - No actual simulation setup
        """
        # Initialize gym.Env
        super().__init__()

        # Set up basic gym spaces (required by gymnasium)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(10,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = MockConfig()
        self.cfg_task = MockTaskConfig()

        # Attributes that wrappers check for initialization
        self._robot = MockRobotView(device=str(device))
        self.scene = MockScene(num_envs, str(device))

        # Episode tracking (used by some wrappers)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.common_step_counter = 1
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        # Physics simulation
        self.physics_dt = 1.0 / 120.0  # 120 Hz
        self.last_update_timestamp = 0.0

        # Robot state variables (for HybridForcePositionWrapper)
        self.joint_pos = torch.zeros(num_envs, 9, device=self.device)
        self.joint_vel_fd = torch.zeros(num_envs, 9, device=self.device)
        self.actions = torch.zeros(num_envs, 12, device=self.device)  # Extended for hybrid control
        self.prev_action = torch.zeros(num_envs, 12, device=self.device)

        # End-effector state
        self.fingertip_midpoint_pos = torch.tensor([[0.5, 0.0, 0.5]] * num_envs, device=self.device)
        self.fingertip_midpoint_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, device=self.device)
        self.ee_linvel_fd = torch.zeros(num_envs, 3, device=self.device)
        self.ee_angvel_fd = torch.zeros(num_envs, 3, device=self.device)

        # Control targets
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor([[0.5, 0.0, 0.5]] * num_envs, device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, device=self.device)
        self.ctrl_target_joint_pos = torch.zeros(num_envs, 9, device=self.device)
        self.ctrl_target_gripper_dof_pos = torch.zeros(num_envs, device=self.device)

        # Control gains
        self.task_prop_gains = torch.tensor([[100.0] * 6] * num_envs, device=self.device)
        self.task_deriv_gains = torch.tensor([[10.0] * 6] * num_envs, device=self.device)

        # Joint torques and forces
        self.joint_torque = torch.zeros(num_envs, 9, device=self.device)
        self.robot_force_torque = torch.zeros(num_envs, 6, device=self.device)

        # Jacobian and dynamics
        self.fingertip_midpoint_jacobian = torch.eye(6, 7, device=self.device).unsqueeze(0).repeat(num_envs, 1, 1)
        self.arm_mass_matrix = torch.eye(7, device=self.device).unsqueeze(0).repeat(num_envs, 1, 1)

        # Action bounds and thresholds
        self.pos_threshold = torch.tensor(0.1, device=self.device)
        self.rot_threshold = torch.tensor(0.1, device=self.device)
        self.fixed_pos_action_frame = torch.tensor([[0.5, 0.0, 0.5]] * num_envs, device=self.device)

        # Episode statistics (for force-torque tracking)
        self.ep_max_force = torch.zeros(num_envs, device=self.device)
        self.ep_max_torque = torch.zeros(num_envs, device=self.device)
        self.ep_sum_force = torch.zeros(num_envs, device=self.device)
        self.ep_sum_torque = torch.zeros(num_envs, device=self.device)
        self.ep_ssv = torch.zeros(num_envs, device=self.device)  # smoothness metric

        # Current orientation tracking
        self.curr_yaw = torch.zeros(num_envs, device=self.device)

        # Extras dictionary for logging
        self.extras = {}

    def _init_tensors(self):
        """
        Mock tensor initialization.

        Real Isaac Lab ManagerBasedRLEnv._init_tensors():
        - Initializes observation and action buffers
        - Sets up episode tracking tensors
        - Creates physics state tensors

        Mock behavior:
        - Called by wrappers during initialization
        - No actual tensor setup needed for mocks
        """
        pass

    def _compute_intermediate_values(self, dt: float):
        """
        Mock intermediate value computation.

        Real Isaac Lab ManagerBasedRLEnv._compute_intermediate_values():
        - Updates observations from physics state
        - Computes derived quantities (velocities, etc.)
        - Called every physics step

        Mock behavior:
        - Called by wrappers during physics step
        - No actual computation needed for mocks
        """
        pass

    def _reset_buffers(self, env_ids: torch.Tensor):
        """
        Mock buffer reset.

        Real Isaac Lab ManagerBasedRLEnv._reset_buffers():
        - Resets episode tracking for specified environments
        - Clears observation and action buffers
        - Resets episode counters

        Mock behavior:
        - Called by wrappers during environment reset
        - Resets episode length buffer for testing
        """
        self.episode_length_buf[env_ids] = 0

    def _pre_physics_step(self, action: torch.Tensor):
        """
        Mock pre-physics step processing.

        Real Isaac Lab ManagerBasedRLEnv._pre_physics_step():
        - Processes actions before physics simulation
        - Updates control targets
        - Applies actions to robots

        Mock behavior:
        - Called by wrappers before physics step
        - Increments step counter for testing
        """
        self.common_step_counter += 1

    def step(self, action):
        """
        Mock step method required by gymnasium.Env.

        Returns dummy observation, reward, terminated, truncated, info.
        """
        obs = torch.randn(self.num_envs, 10, device=self.device)
        reward = torch.zeros(self.num_envs, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Mock reset method required by gymnasium.Env.

        Returns dummy observation and info.
        """
        obs = torch.randn(self.num_envs, 10, device=self.device)
        info = {}
        return obs, info

    def render(self):
        """Mock render method required by gymnasium.Env."""
        pass

    def close(self):
        """Mock close method required by gymnasium.Env."""
        pass

    def _get_dones(self):
        """
        Mock _get_dones method used by some wrappers.

        Real Isaac Lab ManagerBasedRLEnv._get_dones():
        - Returns (terminated, time_out) tensors
        - terminated: episodes that ended due to task completion/failure
        - time_out: episodes that ended due to time limit

        Mock behavior:
        - Returns all False for both terminated and time_out
        """
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

    def _get_rewards(self):
        """
        Mock _get_rewards method used by FactoryMetricsWrapper.

        Real Isaac Lab ManagerBasedRLEnv._get_rewards():
        - Computes task-specific rewards
        - Returns reward tensor for all environments

        Mock behavior:
        - Returns simple test rewards
        """
        return torch.ones(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids):
        """
        Mock _reset_idx method used by EfficientResetWrapper.

        Real Isaac Lab ManagerBasedRLEnv._reset_idx():
        - Resets specified environments to initial state
        - Called when episodes end or manually triggered
        - Updates environment state and buffers

        Mock behavior:
        - Simply stores the reset call for verification
        """
        self.last_reset_idx_call = env_ids

    def _pre_physics_step(self, action):
        """
        Mock _pre_physics_step method used by HybridForcePositionWrapper.

        Real Isaac Lab ManagerBasedRLEnv._pre_physics_step():
        - Processes actions before physics simulation
        - Updates control targets
        - Applies actions to robots

        Mock behavior:
        - Stores action for verification
        """
        self.last_pre_physics_step_action = action

    def _apply_action(self):
        """
        Mock _apply_action method used by HybridForcePositionWrapper.

        Real Isaac Lab ManagerBasedRLEnv._apply_action():
        - Applies computed torques to robot joints
        - Sets position targets for position-controlled joints

        Mock behavior:
        - Stores call for verification
        """
        self.last_apply_action_call = True

    def _update_rew_buf(self, curr_successes):
        """
        Mock _update_rew_buf method used by HybridForcePositionWrapper.

        Real Isaac Lab ManagerBasedRLEnv._update_rew_buf():
        - Computes rewards based on task success
        - Updates reward buffer

        Mock behavior:
        - Returns dummy rewards
        """
        return torch.ones_like(curr_successes).float() * 0.5

    def _configure_gym_env_spaces(self):
        """
        Mock _configure_gym_env_spaces method.

        Real Isaac Lab ManagerBasedRLEnv._configure_gym_env_spaces():
        - Sets up gymnasium observation and action spaces
        - Called when environment configuration changes

        Mock behavior:
        - Updates action space size
        """
        if hasattr(self, 'cfg') and hasattr(self.cfg, 'action_space'):
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.cfg.action_space,))


class MockControlConfig:
    """
    Mock control configuration for HybridForcePositionWrapper.

    Real class reference:
    - Isaac Lab: FactoryControlConfig, various control config classes
    """

    def __init__(self):
        """Mock control configuration with hybrid force-position control attributes."""
        # Force task gains
        self.default_task_force_gains = [0.1, 0.1, 0.1, 0.001, 0.001, 0.001]

        # Action thresholds and bounds
        self.force_action_threshold = [10, 10, 10]
        self.force_action_bounds = [50, 50, 50]
        self.torque_action_threshold = [0.1, 0.1, 0.1]
        self.torque_action_bounds = [0.5, 0.5, 0.5]
        self.pos_action_bounds = [0.1, 0.1, 0.1]

        # EMA smoothing
        self.ema_factor = 0.2
        self.no_sel_ema = True

        # Null space control
        self.kp_null = 10.0
        self.kd_null = 1.0
        self.default_dof_pos_tensor = [0.0, -1.57, 0.0, -2.356, 0.0, 1.57, 0.785]


class MockTaskConfig:
    """
    Mock task configuration for HybridForcePositionWrapper rewards.

    Real class reference:
    - Isaac Lab: FactoryTaskConfig and task-specific configurations
    """

    def __init__(self):
        """Mock task configuration with reward parameters."""
        # Force control rewards
        self.force_active_threshold = 1.0
        self.torque_active_threshold = 0.1
        self.good_force_cmd_rew = 0.1
        self.bad_force_cmd_rew = -0.1
        self.wrench_norm_scale = 0.01

        # Rotation constraints
        self.unidirectional_rot = False


class MockSceneConfig:
    """
    Mock scene configuration.

    Real class reference:
    - Isaac Lab: InteractiveSceneConfig
    """

    def __init__(self):
        """Mock scene configuration."""
        self.num_envs = 64


class MockConfig:
    """
    Mock configuration object.

    Real class reference:
    - Isaac Lab: Various config classes in omni.isaac.lab.envs
    - Example: FactoryTaskConfig, RLTaskEnvConfig
    """

    def __init__(self):
        """Mock configuration with attributes used by various wrappers."""
        # Observation noise configuration (used by ObservationManagerWrapper)
        self.obs_noise_mean = {
            "force_torque": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
        self.obs_noise_std = {
            "force_torque": [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
        }

        # Control configuration (used by HybridForcePositionWrapper)
        self.ctrl = MockControlConfig()

        # Scene configuration
        self.scene = MockSceneConfig()

        # Physics configuration (used by FactoryMetricsWrapper)
        self.decimation = 4


class MockCfg:
    """
    Alias for MockConfig for backward compatibility.
    """

    def __init__(self):
        """Initialize with same attributes as MockConfig."""
        # Physics configuration (used by FactoryMetricsWrapper)
        self.decimation = 4


class MockCfgTask:
    """
    Mock task-specific configuration for FactoryMetricsWrapper.

    Real class reference:
    - Isaac Lab: Factory task configurations with success/engagement thresholds
    """

    def __init__(self):
        """Mock task configuration with factory-specific attributes."""
        # Success and engagement thresholds
        self.success_threshold = 0.02
        self.engage_threshold = 0.05

        # Task name for specific task behaviors
        self.name = "test_task"


# Utility function to create a complete mock environment for testing
def create_mock_env(num_envs: int = 64, device: str = "cpu") -> MockEnvironment:
    """
    Create a mock environment with all necessary attributes for wrapper testing.

    Args:
        num_envs: Number of parallel environments
        device: Device for tensors ('cpu' or 'cuda')

    Returns:
        MockEnvironment configured for testing
    """
    return MockEnvironment(num_envs=num_envs, device=device)