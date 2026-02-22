"""
Real World in Sim Runner — Sim Observation Tool

Sets the sim robot to joint angles from the real robot config and prints
the resulting sim observations. Uses the full wrapper chain so the observation
pipeline can be inspected without needing the real robot connected.

Flow:
  1. Load training config, create sim env with full wrapper stack
  2. Load real robot config, move sim hole to match
  3. Read joint angles from real robot config, set sim to those angles
  4. Print sim observations and raw state

Usage:
    python real_robot_exps/real_world_in_sim_runner.py \
        --config configs/experiments/my_experiment.yaml \
        --real-config real_robot_exps/config.yaml
"""

import argparse
import sys
import os

try:
    from isaaclab.app import AppLauncher
except ImportError:
    from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Real World in Sim Observation Tool")
parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
parser.add_argument("--real-config", type=str, default="real_robot_exps/config.yaml",
                    help="Path to real robot config YAML")
parser.add_argument("--override", action="append", default=[], help="Override config values (repeatable)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Force headless=False so user can see the sim robot
args_cli.video = False
args_cli.enable_cameras = False

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Enable PhysX contact processing
import carb
settings = carb.settings.get_settings()
settings.set("/physics/disableContactProcessing", False)

# ============================================================================
# Post-AppLauncher imports
# ============================================================================

import torch
import yaml

try:
    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
except ImportError:
    from omni.isaac.lab_tasks.direct.factory.factory_env import FactoryEnv

from wrappers.sensors.factory_env_with_sensors import create_sensor_enabled_factory_env
from configs.config_manager_v3 import ConfigManagerV3

# Wrapper imports (same as launch_utils_v3)
from wrappers.mechanics.fragile_object_wrapper import FragileObjectWrapper
from wrappers.mechanics.force_reward_wrapper import ForceRewardWrapper
from wrappers.mechanics.efficient_reset_wrapper import EfficientResetWrapper
from wrappers.mechanics.plate_spawn_offset_wrapper import PlateSpawnOffsetWrapper
from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper
from wrappers.observations.observation_manager_wrapper import ObservationManagerWrapper
from wrappers.observations.ee_pose_noise_wrapper import EEPoseNoiseWrapper
from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper
from wrappers.control.vic_pose_wrapper import VICPoseWrapper
from wrappers.mechanics.two_stage_keypoint_reward_wrapper import TwoStageKeypointRewardWrapper
from wrappers.mechanics.keypoint_offset_wrapper import KeypointOffsetWrapper
from wrappers.mechanics.spawn_height_curriculum_wrapper import SpawnHeightCurriculumWrapper
from wrappers.mechanics.dynamics_randomization_wrapper import DynamicsRandomizationWrapper

from dataclasses import asdict

print("\n\nImports complete\n\n")

# Observation dimension map (matches ObservationBuilder)
OBS_DIM_MAP = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "fingertip_yaw_rel_fixed": 1,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "force_torque": 6,
    "in_contact": 3,
}


# ============================================================================
# Config loading
# ============================================================================

def load_real_robot_config(config_path, overrides=None):
    """Load real robot config from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if overrides:
        for override in overrides:
            if '=' not in override:
                raise ValueError(f"Override must be 'key=value', got: {override}")
            key_path, value_str = override.split('=', 1)
            keys = key_path.split('.')
            parent = config
            for k in keys[:-1]:
                if k not in parent:
                    raise ValueError(f"Config key not found: {key_path}")
                parent = parent[k]
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    if value_str.lower() == 'true':
                        value = True
                    elif value_str.lower() == 'false':
                        value = False
                    else:
                        value = value_str
            parent[keys[-1]] = value
    return config


def reconstruct_obs_order(configs):
    """Reconstruct the obs_order used during training."""
    obs_order = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]

    ft_cfg = configs['wrappers'].force_torque_sensor
    if getattr(ft_cfg, 'add_force_obs', False):
        obs_order.append("force_torque")
    if getattr(ft_cfg, 'add_contact_obs', False):
        obs_order.append("in_contact")

    env_cfg = configs['environment']
    if hasattr(env_cfg, 'obs_rand') and getattr(env_cfg.obs_rand, 'use_yaw_noise', False):
        obs_order.append("fingertip_yaw_rel_fixed")

    return obs_order


# ============================================================================
# Wrapper application (mirrors launch_utils_v3.apply_wrappers)
# ============================================================================

def apply_wrappers(env, configs):
    """Apply the full wrapper stack matching training."""
    wrappers_config = configs['wrappers']

    if wrappers_config.fragile_objects.enabled:
        print(f"  - Applying Fragile Object Wrapper")
        env = FragileObjectWrapper(
            env,
            break_force=configs['primary'].break_forces,
            num_agents=configs['primary'].total_agents,
            config=wrappers_config.fragile_objects
        )

    goal_offset = getattr(configs['environment'].task, 'goal_offset', (0.0, 0.0))
    if goal_offset[0] != 0.0 or goal_offset[1] != 0.0:
        print(f"  - Applying Plate Spawn Offset Wrapper")
        env = PlateSpawnOffsetWrapper(env)

    if wrappers_config.keypoint_offset.enabled:
        print("  - Applying Keypoint Offset Wrapper")
        env = KeypointOffsetWrapper(env, config=asdict(wrappers_config.keypoint_offset))

    if wrappers_config.efficient_reset.enabled:
        print(f"  - Applying Efficient Reset Wrapper")
        env = EfficientResetWrapper(env, config=wrappers_config.efficient_reset)

    if wrappers_config.force_torque_sensor.enabled:
        print("  - Applying Force Torque Wrapper")
        env = ForceTorqueWrapper(
            env,
            use_tanh_scaling=wrappers_config.force_torque_sensor.use_tanh_scaling,
            tanh_scale=wrappers_config.force_torque_sensor.tanh_scale,
            add_force_obs=wrappers_config.force_torque_sensor.add_force_obs,
            add_contact_obs=wrappers_config.force_torque_sensor.add_contact_obs,
            add_contact_state=wrappers_config.force_torque_sensor.add_contact_state,
            contact_force_threshold=wrappers_config.force_torque_sensor.contact_force_threshold,
            contact_torque_threshold=wrappers_config.force_torque_sensor.contact_torque_threshold,
            log_contact_state=wrappers_config.force_torque_sensor.log_contact_state,
            use_contact_sensor=wrappers_config.force_torque_sensor.use_contact_sensor
        )

    if wrappers_config.force_reward.enabled:
        print("  - Applying Force Reward Wrapper")
        env = ForceRewardWrapper(env, config=asdict(wrappers_config.force_reward))

    if wrappers_config.two_stage_keypoint_reward.enabled:
        print("  - Applying Two-Stage Keypoint Reward Wrapper")
        env = TwoStageKeypointRewardWrapper(env, config=asdict(wrappers_config.two_stage_keypoint_reward))

    if wrappers_config.observation_manager.enabled:
        print("  - Applying Observation Manager Wrapper")
        env = ObservationManagerWrapper(
            env, merge_strategy=wrappers_config.observation_manager.merge_strategy
        )

    if wrappers_config.ee_pose_noise.enabled:
        print("  - Applying EE Pose Noise Wrapper")
        env = EEPoseNoiseWrapper(env)

    if wrappers_config.hybrid_control.enabled:
        print("  - Applying Hybrid Force Position Wrapper")
        env = HybridForcePositionWrapper(
            env,
            ctrl_mode=configs['primary'].ctrl_mode,
            reward_type=wrappers_config.hybrid_control.reward_type,
            ctrl_cfg=configs['environment'].ctrl,
            task_cfg=configs['environment'].hybrid_task,
            num_agents=configs['primary'].total_agents,
            use_ground_truth_selection=wrappers_config.hybrid_control.use_ground_truth_selection
        )

    if wrappers_config.vic_pose.enabled:
        print("  - Applying VIC Pose Wrapper")
        env = VICPoseWrapper(
            env,
            ctrl_cfg=configs['environment'].ctrl,
            apply_ema_to_gains=wrappers_config.vic_pose.apply_ema_to_gains
        )

    if wrappers_config.dynamics_randomization.enabled:
        print("  - Applying Dynamics Randomization Wrapper")
        env = DynamicsRandomizationWrapper(env, config=asdict(wrappers_config.dynamics_randomization))

    return env


# ============================================================================
# Sim state override
# ============================================================================

def override_sim_hole_position(env, real_config, device):
    """Move the sim fixed asset (hole) to match the real robot config position.

    Updates the fixed asset root pose and the observation frame variables
    so sim observations use the same reference point as the real robot.

    Args:
        env: Wrapped environment (wrappers modify unwrapped methods).
        real_config: Real robot config dict.
        device: Torch device string.
    """
    task_cfg = real_config['task']
    fixed_pos = torch.tensor(task_cfg['fixed_asset_position'], device=device, dtype=torch.float32)
    fixed_quat = torch.tensor(task_cfg['fixed_asset_orientation_quat'], device=device, dtype=torch.float32)

    # Account for env_origins (sim positions are world-frame = env_origin + local)
    env_origin = env.unwrapped.scene.env_origins[0]  # [3]
    world_pos = env_origin + fixed_pos

    # Write fixed asset root pose to sim
    root_pose = torch.zeros((1, 7), device=device, dtype=torch.float32)
    root_pose[0, 0:3] = world_pos
    root_pose[0, 3:7] = fixed_quat  # [w, x, y, z]
    env.unwrapped._fixed_asset.write_root_pose_to_sim(root_pose, env_ids=torch.tensor([0], device=device))

    # Compute observation frame: fixed_pos + [0, 0, hole_height + base_height]
    hole_height = task_cfg['hole_height']
    base_height = task_cfg['fixed_asset_base_height']
    obs_frame = world_pos.clone()
    obs_frame[2] += hole_height + base_height
    env.unwrapped.fixed_pos_obs_frame[0] = obs_frame

    # Zero out observation noise (real robot has no sim noise)
    env.unwrapped.init_fixed_pos_obs_noise[0] = 0.0

    # Update fixed_pos_action_frame to match (obs_frame + noise = obs_frame)
    if hasattr(env.unwrapped, 'fixed_pos_action_frame'):
        env.unwrapped.fixed_pos_action_frame[0] = obs_frame.clone()

    # Update stored fixed_pos/fixed_quat used by state observations
    env.unwrapped.fixed_pos[0] = world_pos
    env.unwrapped.fixed_quat[0] = fixed_quat

    print(f"  Sim hole moved to: pos={fixed_pos.tolist()}, "
          f"obs_frame={obs_frame.tolist()}")


def set_sim_robot_joints(env, joint_pos_real, action_dim, device, settle_steps=5):
    """Set the sim robot to the given joint angles and run normal steps.

    Writes joint positions, propagates via step_sim_no_action to compute FK,
    sets the controller target to the new EE pose (so zero actions = hold
    position), then runs settle_steps normal env.step() calls to let all
    buffers (FD velocity, EMA, force sensors, etc.) update through the
    standard pipeline.

    Args:
        env: Wrapped environment.
        joint_pos_real: [7] joint positions tensor.
        action_dim: Action dimension for zero-action tensor.
        device: Torch device string.
        settle_steps: Number of env.step() calls to settle buffers.

    Returns:
        The obs dict from the last env.step() call.
    """
    uw = env.unwrapped

    # --- 1. Write desired joint positions to sim ---
    joint_pos = uw._robot.data.joint_pos.clone()
    joint_pos[0, :7] = joint_pos_real.to(device)
    joint_vel = torch.zeros_like(joint_pos)
    uw._robot.write_joint_state_to_sim(joint_pos, joint_vel,
                                       env_ids=torch.tensor([0], device=device))

    # --- 2. Propagate state to compute FK (EE pose from joint angles) ---
    # Uses the env's own method: write_data_to_sim -> sim.step -> scene.update
    # -> _compute_intermediate_values (through wrapper chain)
    uw.step_sim_no_action()

    # --- 3. Set controller target to current EE pose ---
    # This makes zero actions = "hold this position" so the robot doesn't
    # drift toward the old reset target during the settle steps.
    uw.ctrl_target_fingertip_midpoint_pos[0] = uw.fingertip_midpoint_pos[0].clone()
    uw.ctrl_target_fingertip_midpoint_quat[0] = uw.fingertip_midpoint_quat[0].clone()

    # --- 4. Set prev buffers so FD velocities start near zero ---
    uw.prev_fingertip_pos[0] = uw.fingertip_midpoint_pos[0].clone()
    uw.prev_fingertip_quat[0] = uw.fingertip_midpoint_quat[0].clone()
    uw.prev_joint_pos[0] = joint_pos_real.to(device)
    uw.actions[:] = 0.0

    # --- 5. Run normal env.step() to settle all buffers ---
    zero_action = torch.zeros((1, action_dim), device=device, dtype=torch.float32)
    for i in range(settle_steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)

    return obs


# ============================================================================
# Observation printing
# ============================================================================

def print_sim_observations(sim_obs, obs_order, action_dim):
    """Print labeled sim observation components.

    Args:
        sim_obs: [obs_dim] sim observation tensor.
        obs_order: List of observation component names.
        action_dim: Action dimension (for prev_actions label).
    """
    w = 50
    print(f"\n{'=' * w}")
    print(f"  SIM OBSERVATIONS")
    print(f"{'=' * w}")

    idx = 0
    for name in obs_order:
        dim = OBS_DIM_MAP[name]
        print(f"--- {name} ---")
        for i in range(dim):
            val = sim_obs[idx].item()
            print(f"  [{i}] {val:>12.6f}")
            idx += 1

    print(f"--- prev_actions ---")
    for i in range(action_dim):
        val = sim_obs[idx].item()
        print(f"  [{i}] {val:>12.6f}")
        idx += 1

    print(f"{'=' * w}")


def print_raw_state(env):
    """Print raw sim state (joint pos, EE pose, velocities, F/T).

    Args:
        env: Wrapped environment.
    """
    def fmt_vec(t, n=6):
        return "[" + ", ".join(f"{v:.{n}f}" for v in t.tolist()) + "]"

    uw = env.unwrapped
    env_origin = uw.scene.env_origins[0]

    joint_pos = uw._robot.data.joint_pos[0, :7].cpu()
    ee_pos = uw.fingertip_midpoint_pos[0].cpu() - env_origin.cpu()
    ee_quat = uw.fingertip_midpoint_quat[0].cpu()
    ee_linvel = uw.ee_linvel_fd[0].cpu()
    ee_angvel = uw.ee_angvel_fd[0].cpu()
    force_torque = (uw.robot_force_torque[0].cpu()
                    if hasattr(uw, 'robot_force_torque')
                    else torch.zeros(6))

    w = 50
    print(f"\n{'=' * w}")
    print(f"  RAW SIM STATE")
    print(f"{'=' * w}")
    print(f"  Joint Pos:  {fmt_vec(joint_pos, 4)}")
    print(f"  EE Pos:     {fmt_vec(ee_pos)}")
    print(f"  EE Quat:    {fmt_vec(ee_quat)}")
    print(f"  EE LinVel:  {fmt_vec(ee_linvel)}")
    print(f"  EE AngVel:  {fmt_vec(ee_angvel)}")
    print(f"  F/T:        {fmt_vec(force_torque)}")
    print(f"{'=' * w}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("REAL WORLD IN SIM — OBSERVATION TOOL")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load training config
    # ------------------------------------------------------------------
    print("\n[1] Loading training config...")
    config_manager = ConfigManagerV3()
    configs = config_manager.process_config(args_cli.config, args_cli.override)

    # Force num_envs=1 (we have one real robot)
    configs['primary'].num_envs_per_agent = 1
    configs['primary'].total_agents = 1
    config_manager._apply_primary_config_to_all(configs)

    configs['environment'].seed = args_cli.seed

    # Should not matter but removes warning
    configs['environment'].sim.render_interval = configs['primary'].decimation

    device = configs['environment'].sim.device
    print(f"  Task: {configs['experiment'].task_name}")
    print(f"  Device: {device}")

    # Apply asset variant if specified
    task_cfg = configs['environment'].task
    if hasattr(task_cfg, 'apply_asset_variant_if_specified'):
        if task_cfg.apply_asset_variant_if_specified():
            print("  Asset variant applied")

    # ------------------------------------------------------------------
    # 2. Create sim environment with full wrapper stack
    # ------------------------------------------------------------------
    print("\n[2] Creating sim environment...")

    if (hasattr(configs['environment'].task, 'held_fixed_contact_sensor') and
            configs['wrappers'].force_torque_sensor.use_contact_sensor):
        EnvClass = create_sensor_enabled_factory_env(FactoryEnv)
        print("  Using sensor-enabled factory environment")
    else:
        EnvClass = FactoryEnv
        print("  Using standard factory environment")

    env = EnvClass(cfg=configs['environment'], render_mode=None)
    print("  Environment created")

    print("\n[3] Applying wrapper stack...")
    env = apply_wrappers(env, configs)
    print("  Wrappers applied")

    # ------------------------------------------------------------------
    # 3. Reset env (triggers lazy init of all wrappers)
    # ------------------------------------------------------------------
    print("\n[4] Initial environment reset...")
    obs, info = env.reset()
    print(f"  Reset complete (num_envs={env.num_envs})")

    # ------------------------------------------------------------------
    # 4. Load real robot config
    # ------------------------------------------------------------------
    print(f"\n[5] Loading real robot config: {args_cli.real_config}")
    real_config = load_real_robot_config(args_cli.real_config)

    # ------------------------------------------------------------------
    # 5. Move sim hole to match real robot config
    # ------------------------------------------------------------------
    print("\n[6] Moving sim hole to match real robot config...")
    override_sim_hole_position(env, real_config, device)

    # Propagate the hole position change through sim
    env.unwrapped.step_sim_no_action()

    # ------------------------------------------------------------------
    # 6. Determine obs_order and action_dim
    # ------------------------------------------------------------------
    obs_order = reconstruct_obs_order(configs)

    hybrid_enabled = configs['wrappers'].hybrid_control.enabled
    if hybrid_enabled:
        from configs.cfg_exts.ctrl_mode import get_force_size
        ctrl_mode = getattr(configs['primary'], 'ctrl_mode', 'force_only')
        force_size = get_force_size(ctrl_mode)
        action_dim = 2 * force_size + 6
    else:
        action_dim = 6

    print(f"  obs_order: {obs_order}")
    print(f"  action_dim: {action_dim}")

    # ------------------------------------------------------------------
    # 7. Read joint angles from real robot config
    # ------------------------------------------------------------------
    task_cfg_r = real_config['task']
    if 'joint_angles' not in task_cfg_r:
        raise RuntimeError(
            "task.joint_angles not found in real robot config. "
            "Run first_real_robot_test.py phase 1 and copy the joint angles "
            "into config.yaml under task.joint_angles."
        )

    joint_angles = torch.tensor(task_cfg_r['joint_angles'], device=device, dtype=torch.float32)
    if joint_angles.shape[0] != 7:
        raise RuntimeError(
            f"task.joint_angles must have exactly 7 values, got {joint_angles.shape[0]}"
        )

    print(f"\n[7] Joint angles from config: {joint_angles.tolist()}")

    # ------------------------------------------------------------------
    # 8. Set sim robot to config joint angles
    # ------------------------------------------------------------------
    print("\n[8] Setting sim robot to config joint angles...")
    obs = set_sim_robot_joints(env, joint_angles, action_dim, device)

    # Extract sim policy observation from env.step() output.
    # Format depends on wrapper chain — may be dict or tensor.
    if isinstance(obs, dict):
        sim_obs = obs["policy"][0].cpu()
    else:
        sim_obs = obs[0].cpu()

    # ------------------------------------------------------------------
    # 9. Print results
    # ------------------------------------------------------------------
    print_sim_observations(sim_obs, obs_order, action_dim)
    print_raw_state(env)

    print("\nDone.")


if __name__ == "__main__":
    main()
    simulation_app.close()
