"""
Real Robot Interface (STUB)

ROS2-based interface for Franka Panda robot. All methods raise NotImplementedError.
Uses ros2_control (topic_based_ros2_control pattern from MoveIt/Isaac bridge).

Same ros2_control interface works with:
- Isaac Sim (for testing via topic_based_ros2_control)
- Real Franka hardware (via franka_ros2)

Implementation will be filled in when ROS2 environment is available.
"""

import torch


class SafetyViolation(Exception):
    """Raised when a safety check fails on the real robot."""
    pass


class FrankaROS2Interface:
    """ROS2 interface for Franka Panda robot.

    Provides methods for reading robot state (joint positions, EE pose,
    force/torque, dynamics matrices) and sending commands in multiple
    control modes (effort, Cartesian impedance, joint position).

    The active control mode is selected via config['ros2']['control_mode'].

    Args:
        config: Dictionary loaded from real_robot_exps/config.yaml
        device: Torch device for tensor outputs (default: "cpu")
    """

    def __init__(self, config: dict, device: str = "cpu"):
        raise NotImplementedError(
            "FrankaROS2Interface is a stub. Implementation requires ROS2 "
            "environment with franka_ros2 or topic_based_ros2_control."
        )

    # -------------------------------------------------------------------------
    # State reading
    # -------------------------------------------------------------------------

    def get_ee_position(self) -> torch.Tensor:
        """Get end-effector (fingertip midpoint) position in world frame.

        Returns:
            [3] tensor in meters.
        """
        raise NotImplementedError

    def get_ee_orientation(self) -> torch.Tensor:
        """Get end-effector orientation as quaternion.

        Returns:
            [4] tensor (w, x, y, z).
        """
        raise NotImplementedError

    def get_ee_linear_velocity(self) -> torch.Tensor:
        """Get end-effector linear velocity (finite difference or from robot state).

        Returns:
            [3] tensor in m/s.
        """
        raise NotImplementedError

    def get_ee_angular_velocity(self) -> torch.Tensor:
        """Get end-effector angular velocity.

        Returns:
            [3] tensor in rad/s.
        """
        raise NotImplementedError

    def get_force_torque(self) -> torch.Tensor:
        """Get force/torque sensor readings at end-effector.

        Returns:
            [6] tensor (Fx, Fy, Fz, Tx, Ty, Tz) in N and Nm.
        """
        raise NotImplementedError

    def get_joint_positions(self) -> torch.Tensor:
        """Get 7-DOF arm joint positions.

        Returns:
            [7] tensor in radians.
        """
        raise NotImplementedError

    def get_joint_velocities(self) -> torch.Tensor:
        """Get 7-DOF arm joint velocities.

        Returns:
            [7] tensor in rad/s.
        """
        raise NotImplementedError

    def get_joint_torques(self) -> torch.Tensor:
        """Get 7-DOF arm joint torques (measured or commanded).

        Returns:
            [7] tensor in Nm.
        """
        raise NotImplementedError

    def get_jacobian(self) -> torch.Tensor:
        """Get geometric Jacobian at end-effector frame.

        Returns:
            [6, 7] tensor mapping joint velocities to task-space velocities.
        """
        raise NotImplementedError

    def get_mass_matrix(self) -> torch.Tensor:
        """Get 7x7 arm mass (inertia) matrix.

        Returns:
            [7, 7] tensor in kg*m^2.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Action execution (multiple control modes, selected by config)
    # -------------------------------------------------------------------------

    def send_joint_torques(self, torques: torch.Tensor):
        """Send joint torque commands (effort control mode).

        Args:
            torques: [9] tensor (7 arm + 2 gripper) in Nm.
        """
        raise NotImplementedError

    def send_cartesian_impedance(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        target_force: torch.Tensor,
        stiffness: torch.Tensor,
        damping: torch.Tensor,
    ):
        """Send Cartesian impedance command.

        Args:
            target_pos: [3] target position in meters.
            target_quat: [4] target orientation (w, x, y, z).
            target_force: [6] target wrench (Fx, Fy, Fz, Tx, Ty, Tz).
            stiffness: [6] Cartesian stiffness.
            damping: [6] Cartesian damping.
        """
        raise NotImplementedError

    def send_joint_positions(self, positions: torch.Tensor):
        """Send joint position commands (position control mode via IK).

        Args:
            positions: [7] target joint positions in radians.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def reset_to_start_pose(self):
        """Move robot to predefined start pose for episode initialization.

        Blocks until motion is complete or timeout.
        """
        raise NotImplementedError

    def check_safety(self):
        """Run safety checks (joint limits, velocity limits, workspace bounds).

        Raises:
            SafetyViolation: If any safety check fails.
        """
        raise NotImplementedError

    def is_connected(self) -> bool:
        """Check if ROS2 connection to robot is active.

        Returns:
            True if connected and receiving state updates.
        """
        raise NotImplementedError

    def shutdown(self):
        """Cleanly shut down ROS2 node and stop robot."""
        raise NotImplementedError
