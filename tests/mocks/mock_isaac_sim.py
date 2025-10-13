"""
Mock Isaac Sim modules for testing.
Simple stub implementations to replace Isaac Sim functionality.
"""

import torch
import numpy as np
from typing import Any, Dict, Optional


class MockSimulationApp:
    """Mock Isaac Sim application."""

    def __init__(self):
        self.is_running = True

    def update(self):
        pass

    def close(self):
        self.is_running = False


class MockWorld:
    """Mock Isaac Sim world."""

    def __init__(self):
        self.current_time_step_index = 0

    def step(self, render=True):
        self.current_time_step_index += 1

    def reset(self):
        self.current_time_step_index = 0

    def play(self):
        pass

    def pause(self):
        pass

    def stop(self):
        pass


class MockRobot:
    """Mock Isaac Sim robot."""

    def __init__(self, name="robot"):
        self.name = name
        self.num_dof = 6

    def get_joint_positions(self):
        return torch.randn(self.num_dof)

    def get_joint_velocities(self):
        return torch.randn(self.num_dof)

    def set_joint_position_targets(self, positions):
        pass

    def apply_joint_efforts(self, efforts):
        pass


# Mock torch utilities for Isaac Sim
def quat_mul(q1, q2):
    """Multiply two quaternions."""
    # q = [w, x, y, z]
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=1)


def quat_conjugate(q):
    """Compute quaternion conjugate."""
    # q = [w, x, y, z] -> [w, -x, -y, -z]
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=1)


def quat_from_angle_axis(angle, axis):
    """Create quaternion from angle-axis representation."""
    # Ensure angle is a column vector
    if angle.dim() == 1:
        angle = angle.unsqueeze(-1)

    half_angle = angle * 0.5
    sin_half = torch.sin(half_angle)
    cos_half = torch.cos(half_angle)

    # q = [cos(θ/2), sin(θ/2) * axis]
    w = cos_half.squeeze(-1)
    xyz = sin_half * axis

    return torch.cat([w.unsqueeze(-1), xyz], dim=1)


def quat_from_euler_xyz(roll, pitch, yaw):
    """Create quaternion from Euler angles (XYZ convention)."""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=1)


def get_euler_xyz(quat):
    """Extract Euler angles from quaternion (XYZ convention)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1,
                       torch.sign(sinp) * np.pi / 2,
                       torch.asin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class MockTorchUtils:
    """Mock torch utilities module."""

    @staticmethod
    def quat_mul(q1, q2):
        return quat_mul(q1, q2)

    @staticmethod
    def quat_conjugate(q):
        return quat_conjugate(q)

    @staticmethod
    def quat_from_angle_axis(angle, axis):
        return quat_from_angle_axis(angle, axis)

    @staticmethod
    def quat_from_euler_xyz(roll, pitch, yaw):
        return quat_from_euler_xyz(roll, pitch, yaw)

    @staticmethod
    def get_euler_xyz(quat):
        return get_euler_xyz(quat)


# Mock Isaac Sim modules structure
class isaac_sim:
    class core:
        SimulationApp = MockSimulationApp
        World = MockWorld

        class utils:
            torch = MockTorchUtils()

    class robots:
        Robot = MockRobot


# Make modules available
core = isaac_sim.core
robots = isaac_sim.robots