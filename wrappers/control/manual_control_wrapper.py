"""
Manual Control Wrapper

Enables keyboard-based manual control of the robot for testing and debugging.
Supports both pose control and hybrid force-position control modes.
"""

import torch
import gymnasium as gym
import numpy as np
import weakref
from typing import Optional

try:
    import carb.input
    import omni.appwindow
    CARB_AVAILABLE = True
except ImportError:
    CARB_AVAILABLE = False
    print("WARNING: carb.input not available. Manual control will not work.")


class ManualControlWrapper(gym.Wrapper):
    """
    Wrapper for keyboard-based manual control of the robot.

    Key mappings:
    - W/S: +X/-X position
    - A/D: +Y/-Y position
    - Q/E: +Z/-Z position
    - I/K: +Rx/-Rx rotation
    - J/L: +Ry/-Ry rotation
    - U/O: +Rz/-Rz rotation
    - T/G: +Fx/-Fx force (hybrid only)
    - F/H: +Fy/-Fy force (hybrid only)
    - Y/B: +Fz/-Fz force (hybrid only)
    - SPACE: Toggle manual/RL mode
    - R: Reset environment
    - ESC: Exit
    - 0: Zero all actions
    """

    def __init__(
        self,
        env: gym.Env,
        config,
        is_hybrid_control: bool = False
    ):
        """
        Initialize manual control wrapper.

        Args:
            env: Environment to wrap
            config: ManualControlConfig instance
            is_hybrid_control: Whether hybrid control is enabled
        """
        super().__init__(env)

        self.config = config
        self.is_hybrid = is_hybrid_control

        # Validate single environment
        if env.num_envs != 1:
            raise ValueError(
                f"Manual control requires single environment (num_envs=1), got {env.num_envs}"
            )

        # Action space setup - get from unwrapped config, not from action_space which may be incorrect
        self.action_dim = getattr(env.unwrapped.cfg, 'action_space', 6)
        self.manual_action = torch.zeros(1, self.action_dim, device=env.device, dtype=torch.float32)

        # Mode tracking
        self.manual_mode = config.start_in_manual_mode
        self.reset_requested = False
        self.exit_requested = False

        # RL agent (if checkpoint provided)
        self.rl_agent = None
        if config.checkpoint_path:
            self._load_checkpoint(config.checkpoint_path)

        # Keyboard tracking - store key states
        self.key_states = {}

        # Set up keyboard input
        if CARB_AVAILABLE:
            self._setup_keyboard_input()
        else:
            print("WARNING: Manual control wrapper initialized but carb.input not available!")

        print(f"[INFO]: Manual Control Wrapper initialized")
        print(f"[INFO]:   - Action dimension: {self.action_dim}")
        print(f"[INFO]:   - Hybrid control: {self.is_hybrid}")
        print(f"[INFO]:   - Starting mode: {'Manual' if self.manual_mode else 'RL'}")
        print(f"[INFO]:   - RL checkpoint: {config.checkpoint_path if config.checkpoint_path else 'None'}")
        self._print_controls()

    def _setup_keyboard_input(self):
        """Set up keyboard input callbacks using carb.input."""
        if not CARB_AVAILABLE:
            return

        # Get omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        # Subscribe to keyboard events with weakref
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        print("[INFO]: Keyboard input configured")

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._handle_key_press(event.input.name)
            self.key_states[event.input.name] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.key_states[event.input.name] = False

    def _handle_key_press(self, key):
        """Handle key press events for control commands."""
        # Toggle mode
        if key == "SPACE":
            if self.rl_agent is not None:
                self.manual_mode = not self.manual_mode
                mode_str = "Manual" if self.manual_mode else "RL"
                print(f"[INFO]: Switched to {mode_str} mode")
            else:
                print("[INFO]: No RL checkpoint loaded - manual mode only")

        # Reset
        elif key == "R":
            self.reset_requested = True
            print("[INFO]: Reset requested")

        # Exit
        elif key == "ESCAPE":
            self.exit_requested = True
            print("[INFO]: Exit requested")

        # Zero all actions
        elif key == "0" or key == "NUMPAD_0":
            self.manual_action.zero_()
            print("[INFO]: All actions zeroed")

    def _update_manual_action(self):
        """Update manual action based on currently pressed keys."""
        if not self.manual_mode:
            return

        # Store previous action for change detection
        prev_action = self.manual_action.clone()

        # Scale factors from config
        action_scale = self.config.action_scale
        force_scale = self.config.force_scale

        # Pose control actions (indices 0-5 or after selection in hybrid mode)
        if self.is_hybrid:
            # Hybrid control: [sel_x, sel_y, sel_z, px, py, pz, rx, ry, rz, fx, fy, fz] (12D)
            # or [sel_x, sel_y, sel_z, px, py, pz, rx, ry, rz, fx, fy, fz, sel_rx, sel_ry, sel_rz, tx, ty, tz] (18D)

            # Selection values are typically discrete (0 or 1), not continuous increments
            # For now, we'll focus on position, rotation, and force commands

            pos_offset = 3  # Position starts after 3 selection values
            rot_offset = 6  # Rotation starts after position
            force_offset = 9  # Force starts after rotation

            # Position (X, Y, Z)
            if self.key_states.get("W", False):
                self.manual_action[0, pos_offset] += action_scale
            if self.key_states.get("S", False):
                self.manual_action[0, pos_offset] -= action_scale
            if self.key_states.get("A", False):
                self.manual_action[0, pos_offset + 1] += action_scale
            if self.key_states.get("D", False):
                self.manual_action[0, pos_offset + 1] -= action_scale
            if self.key_states.get("Q", False):
                self.manual_action[0, pos_offset + 2] += action_scale
            if self.key_states.get("E", False):
                self.manual_action[0, pos_offset + 2] -= action_scale

            # Rotation (Rx, Ry, Rz)
            if self.key_states.get("I", False):
                self.manual_action[0, rot_offset] += action_scale
            if self.key_states.get("K", False):
                self.manual_action[0, rot_offset] -= action_scale
            if self.key_states.get("J", False):
                self.manual_action[0, rot_offset + 1] += action_scale
            if self.key_states.get("L", False):
                self.manual_action[0, rot_offset + 1] -= action_scale
            if self.key_states.get("U", False):
                self.manual_action[0, rot_offset + 2] += action_scale
            if self.key_states.get("O", False):
                self.manual_action[0, rot_offset + 2] -= action_scale

            # Force (Fx, Fy, Fz)
            if self.key_states.get("T", False):
                self.manual_action[0, force_offset] += force_scale
            if self.key_states.get("G", False):
                self.manual_action[0, force_offset] -= force_scale
            if self.key_states.get("F", False):
                self.manual_action[0, force_offset + 1] += force_scale
            if self.key_states.get("H", False):
                self.manual_action[0, force_offset + 1] -= force_scale
            if self.key_states.get("Y", False):
                self.manual_action[0, force_offset + 2] += force_scale
            if self.key_states.get("B", False):
                self.manual_action[0, force_offset + 2] -= force_scale

            # Set selection to position control by default (selection = 0)
            # User can manually adjust if needed

        else:
            # Pose control: [px, py, pz, rx, ry, rz] (6D)

            # Position (X, Y, Z)
            if self.key_states.get("W", False):
                self.manual_action[0, 0] += action_scale
            if self.key_states.get("S", False):
                self.manual_action[0, 0] -= action_scale
            if self.key_states.get("A", False):
                self.manual_action[0, 1] += action_scale
            if self.key_states.get("D", False):
                self.manual_action[0, 1] -= action_scale
            if self.key_states.get("Q", False):
                self.manual_action[0, 2] += action_scale
            if self.key_states.get("E", False):
                self.manual_action[0, 2] -= action_scale

            # Rotation (Rx, Ry, Rz)
            if self.key_states.get("I", False):
                self.manual_action[0, 3] += action_scale
            if self.key_states.get("K", False):
                self.manual_action[0, 3] -= action_scale
            if self.key_states.get("J", False):
                self.manual_action[0, 4] += action_scale
            if self.key_states.get("L", False):
                self.manual_action[0, 4] -= action_scale
            if self.key_states.get("U", False):
                self.manual_action[0, 5] += action_scale
            if self.key_states.get("O", False):
                self.manual_action[0, 5] -= action_scale

        # Clamp actions to [-1, 1] range
        self.manual_action.clamp_(-1.0, 1.0)

        # Print current action values if they changed
        # if not torch.allclose(prev_action, self.manual_action, atol=1e-6):
        #     self._print_current_action()

    def _print_current_action(self):
        """Print current manual action values in a readable format."""
        action = self.manual_action[0]  # Remove batch dimension

        if self.is_hybrid:
            # Hybrid control: [sel_x, sel_y, sel_z, px, py, pz, rx, ry, rz, fx, fy, fz]
            pos_offset = 3
            rot_offset = 6
            force_offset = 9

            # print("\n[MANUAL CONTROL] Current Actions:")
            # print(f"  Position: X={action[pos_offset]:.3f}, Y={action[pos_offset+1]:.3f}, Z={action[pos_offset+2]:.3f}")
            # print(f"  Rotation: Rx={action[rot_offset]:.3f}, Ry={action[rot_offset+1]:.3f}, Rz={action[rot_offset+2]:.3f}")
            # print(f"  Force:    Fx={action[force_offset]:.3f}, Fy={action[force_offset+1]:.3f}, Fz={action[force_offset+2]:.3f}")
            pass
        else:
            # Pose control: [px, py, pz, rx, ry, rz]
            # print("\n[MANUAL CONTROL] Current Actions:")
            # print(f"  Position: X={action[0]:.3f}, Y={action[1]:.3f}, Z={action[2]:.3f}")
            # print(f"  Rotation: Rx={action[3]:.3f}, Ry={action[4]:.3f}, Rz={action[5]:.3f}")
            pass

    def _print_contact_info(self):
        """Print contact sensor state and values."""
        # Access the unwrapped environment to get contact sensor data
        unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env

        contact_parts = []

        # Get end-effector position for diagnostics
        if hasattr(unwrapped_env, 'ee_pose'):
            ee_pos = unwrapped_env.ee_pose[0, :3]  # First env, position (x, y, z)
            contact_parts.append(f"EE_Z={ee_pos[2].item():.4f}")

        # Check if contact sensor data is available
        if hasattr(unwrapped_env, 'in_contact'):
            in_contact = unwrapped_env.in_contact[0]  # First environment, shape: (3,) for x, y, z
            contact_x = in_contact[0].item() if len(in_contact) > 0 else 0
            contact_y = in_contact[1].item() if len(in_contact) > 1 else 0
            contact_z = in_contact[2].item() if len(in_contact) > 2 else 0
            contact_parts.append(f"in_contact: X={contact_x}, Y={contact_y}, Z={contact_z}")

        # Check if ContactSensor is available in the scene
        if hasattr(unwrapped_env, 'scene') and hasattr(unwrapped_env.scene, 'sensors'):
            sensors = unwrapped_env.scene.sensors

            # Look for contact sensor (commonly named 'held_fixed_contact_sensor')
            if 'held_fixed_contact_sensor' in sensors:
                contact_sensor = sensors['held_fixed_contact_sensor']
                if hasattr(contact_sensor, 'data') and hasattr(contact_sensor.data, 'net_forces_w'):
                    force = contact_sensor.data.net_forces_w[0]  # First environment
                    # Force might be [3] or [N, 3], handle both cases
                    if force.dim() == 1:
                        fx, fy, fz = force[0].item(), force[1].item(), force[2].item()
                    else:
                        fx, fy, fz = force[0, 0].item(), force[0, 1].item(), force[0, 2].item()
                    contact_parts.append(f"Force: X={fx:.3f}, Y={fy:.3f}, Z={fz:.3f}")

                if hasattr(contact_sensor, 'data') and hasattr(contact_sensor.data, 'in_contact'):
                    sensor_contact = contact_sensor.data.in_contact[0].item()
                    contact_parts.append(f"Sensor contact: {sensor_contact}")

                # Add contact time tracking
                if hasattr(contact_sensor, 'data') and hasattr(contact_sensor.data, 'current_contact_time'):
                    contact_time = contact_sensor.data.current_contact_time[0].item()
                    contact_parts.append(f"Contact time: {contact_time:.3f}s")

                if hasattr(contact_sensor, 'data') and hasattr(contact_sensor.data, 'current_air_time'):
                    air_time = contact_sensor.data.current_air_time[0].item()
                    contact_parts.append(f"Air time: {air_time:.3f}s")

        # Print all on one line with tabs
        if contact_parts:
            # print(f"[CONTACT]\t" + "\t".join(contact_parts))
            pass

    def step(self, action):
        """
        Step the environment with manual or RL action.

        Args:
            action: Action from agent (ignored in manual mode)

        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
        """
        # Update manual action based on key states
        self._update_manual_action()

        # Choose action source
        if self.manual_mode:
            action = self.manual_action.clone()
        elif self.rl_agent is not None:
            action = self._get_rl_action()
        # else use provided action

        # Check for reset request
        if self.reset_requested:
            self.reset_requested = False
            print("[INFO]: Performing manual reset...")
            # Reset environment and get new observation
            obs, info = self.reset()
            # Create proper step() return format with zero reward and all terminated
            reward = torch.zeros(self.env.num_envs, device=self.env.device)
            terminated = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
            truncated = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
            return obs, reward, terminated, truncated, info

        # Check for exit request
        if self.exit_requested:
            print("[INFO]: Exit requested - terminating episode")
            # Return terminal state
            obs, reward, terminated, truncated, info = self.env.step(action)
            terminated = torch.ones_like(terminated)
            return obs, reward, terminated, truncated, info

        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Print contact sensor information if in manual mode
        if self.manual_mode:
            self._print_contact_info()

        #self.manual_action *= 0
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and clear manual actions."""
        self.manual_action.zero_()
        self.key_states.clear()
        return self.env.reset(**kwargs)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load RL agent checkpoint for toggle mode."""
        # TODO: Implement checkpoint loading when RL toggle is needed
        print(f"[WARNING]: Checkpoint loading not yet implemented: {checkpoint_path}")
        self.rl_agent = None

    def _get_rl_action(self):
        """Get action from RL agent."""
        if self.rl_agent is None:
            return torch.zeros(1, self.action_dim, device=self.env.device)

        # TODO: Implement RL agent inference
        raise NotImplementedError("RL agent inference not yet implemented")

    def _print_controls(self):
        """Print control scheme to console."""
        print("\n" + "=" * 80)
        print("MANUAL CONTROL SCHEME")
        print("=" * 80)
        print("Position Control:")
        print("  W/S - X axis (forward/backward)")
        print("  A/D - Y axis (left/right)")
        print("  Q/E - Z axis (up/down)")
        print("\nRotation Control:")
        print("  I/K - Rx (roll)")
        print("  J/L - Ry (pitch)")
        print("  U/O - Rz (yaw)")

        if self.is_hybrid:
            print("\nForce Control (Hybrid Mode):")
            print("  T/G - Fx")
            print("  F/H - Fy")
            print("  Y/B - Fz")

        print("\nMode Control:")
        if self.rl_agent is not None:
            print("  SPACE - Toggle between Manual and RL mode")
        else:
            print("  SPACE - (RL mode unavailable - no checkpoint loaded)")
        print("  R     - Reset environment")
        print("  0     - Zero all actions")
        print("  ESC   - Exit")
        print("=" * 80 + "\n")

    def close(self):
        """Clean up keyboard subscription."""
        if CARB_AVAILABLE and hasattr(self, '_sub_keyboard'):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None
        super().close()
