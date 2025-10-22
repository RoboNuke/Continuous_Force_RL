#!/usr/bin/env python3
"""
Block SimBa Evaluation Script - Daemon Mode

Monitors checkpoint log, loads checkpoints, runs evaluation rollouts,
calculates comprehensive metrics, and logs to WandB.
"""

import argparse
import sys

# Parse arguments BEFORE importing anything else (AppLauncher needs this)
def parse_arguments():
    """
    Parse command-line arguments for evaluation script.

    Returns:
        Tuple of (args, hydra_args): Parsed arguments and remaining Hydra arguments
    """
    # Import AppLauncher here to avoid circular dependencies
    try:
        from isaaclab.app import AppLauncher
    except:
        from omni.isaac.lab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Block SimBa Checkpoint Evaluation Daemon")

    # Required arguments
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")

    # Checkpoint loading modes (mutually exclusive)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--ckpt_tracker_path", type=str, help="Path to checkpoint tracker file (Mode 1: Tracker)")
    checkpoint_group.add_argument("--wandb_checkpoint", type=str, help="WandB checkpoint: project/run_id/step_number (Mode 2: WandB Download)")
    checkpoint_group.add_argument("--local_checkpoint", type=str, help="Path to local checkpoint file (Mode 3: Local)")

    # Mode 3 required argument
    parser.add_argument("--wandb_run", type=str, help="WandB run: project/run_id (required for --local_checkpoint)")

    # Optional arguments
    parser.add_argument("--eval_seed", type=int, default=42, help="Fixed seed for consistent evaluation")
    parser.add_argument("--enable_video", action="store_true", default=False, help="Toggle video generation")
    parser.add_argument("--num_envs_per_agent", type=int, default=None, help="Number of environments per agent (overrides config)")
    parser.add_argument("--max_rollout_steps", type=int, default=None, help="Maximum rollout steps (overrides config calculation)")
    parser.add_argument("--batch_mode", type=str, default="max_throughput", choices=["wait_full", "max_throughput"],
                        help="Checkpoint batching strategy")
    parser.add_argument("--sensitivity_mask_freq", type=int, default=10, help="Frequency for sensitivity masking (0 to disable)")
    parser.add_argument("--show_progress", action="store_true", default=False, help="Show progress bar during evaluation rollout")
    parser.add_argument("--override", action="append", help="Override config values: key=value")
    parser.add_argument("--upload_ckpt", action="store_true", default=False, help="Upload checkpoint to WandB after evaluation (Modes 1 & 3 only)")

    # Append AppLauncher args
    AppLauncher.add_app_launcher_args(parser)

    # Parse arguments (separate cli args from hydra args)
    args_cli, hydra_args = parser.parse_known_args()

    return args_cli, hydra_args


# Parse arguments and launch AppLauncher BEFORE importing anything else
print("Parsing arguments...")
args_cli, hydra_args = parse_arguments()

# Configure video/camera settings for evaluation
args_cli.video = args_cli.enable_video
args_cli.enable_cameras = args_cli.enable_video
args_cli.headless = True  # Always run headless

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

print("Launching Isaac Sim AppLauncher...")
try:
    from isaaclab.app import AppLauncher
except:
    from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("AppLauncher initialized, importing modules...")

# NOW import everything else after AppLauncher is initialized
import os
import time
import random
import signal
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
import gymnasium as gym
from filelock import FileLock
import tqdm
import wandb

from skrl.utils import set_seed
from configs.config_manager_v3 import ConfigManagerV3
import learning.launch_utils_v3 as lUtils
from memories.multi_random import MultiRandomMemory

# Wrappers
from wrappers.mechanics.fragile_object_wrapper import FragileObjectWrapper
from wrappers.mechanics.force_reward_wrapper import ForceRewardWrapper
from wrappers.mechanics.efficient_reset_wrapper import EfficientResetWrapper
from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper
from wrappers.observations.observation_manager_wrapper import ObservationManagerWrapper
from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper
# from wrappers.smoothness_obs_wrapper import SmoothnessObservationWrapper  # TODO: Module doesn't exist yet

# Environment and SKRL wrappers
try:
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    import isaaclab_tasks
except ImportError:
    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
    import omni.isaac.lab_tasks

# Models
from models.block_simba import BlockSimBaActor, BlockSimBaCritic
from agents.block_ppo import BlockPPO

# Image processing for video
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F

# Import factory environment class for direct instantiation
try:
    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
except ImportError:
    try:
        from omni.isaac.lab_tasks.direct.factory.factory_env import FactoryEnv
    except ImportError:
        print("ERROR: Could not import FactoryEnv.")
        print("Please ensure Isaac Lab tasks are properly installed.")
        sys.exit(1)

from wrappers.sensors.factory_env_with_sensors import create_sensor_enabled_factory_env

print("All modules imported successfully")

# Global state for rollback on shutdown
_current_batch = None
_tracker_path = None


class Img2InfoWrapper(gym.Wrapper):
    """
    Wrapper to capture camera images and add them to info dict for video generation.
    """
    def __init__(self, env, key='tiled_camera'):
        super().__init__(env)
        self.cam_key  = key

    def step(self, action):
        """Steps through environment and captures images."""
        observations, rewards, terminateds, truncateds, infos = self.env.step(action)

        # Capture camera images if available
        if hasattr(self.unwrapped.scene, 'sensors') and self.cam_key in self.unwrapped.scene.sensors:
            infos['img'] = self.unwrapped.scene.sensors[self.cam_key].data.output['rgb']

        return observations, rewards, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Reset environment and capture initial images."""
        observations, info = super().reset(**kwargs)

        # Capture initial camera images if available
        if hasattr(self.unwrapped.scene, 'sensors') and self.cam_key in self.unwrapped.scene.sensors:
            info['img'] = self.unwrapped.scene.sensors[self.cam_key].data.output['rgb']

        return observations, info


def setup_environment_once(config_path: str, args: argparse.Namespace) -> Tuple[Any, Dict[str, Any], int]:
    """
    Create and configure evaluation environment (one-time setup).

    Args:
        config_path: Path to YAML configuration file
        args: Parsed command-line arguments

    Returns:
        Tuple of (env, configs, max_rollout_steps, policy_hz)
    """
    # Import camera config
    try:
        from isaaclab.sensors import TiledCameraCfg, CameraCfg
        import isaaclab.sim as sim_utils
    except:
        from omni.isaac.lab.sensors import TiledCameraCfg, CameraCfg
        import omni.isaac.lab.sim as sim_utils

    print("  Loading configuration from:", config_path)

    # Build CLI overrides list
    cli_overrides = []
    if args.override:
        cli_overrides.extend(args.override)

    # Add num_envs override if specified
    if args.num_envs_per_agent is not None:
        # We need to compute total_agents first to calculate num_envs
        # For now, load config without override, then apply manually
        pass

    # Load configuration using ConfigManagerV3
    config_manager = ConfigManagerV3()
    configs = config_manager.process_config(config_path, cli_overrides=cli_overrides if cli_overrides else None)

    print("  Configuration loaded successfully")

    # Apply num_envs_per_agent override if specified
    if args.num_envs_per_agent is not None:
        total_agents = configs['primary'].total_agents
        total_envs = args.num_envs_per_agent * total_agents
        configs['environment'].scene.num_envs = total_envs
        print(f"  Override num_envs: {total_envs} ({args.num_envs_per_agent} per agent × {total_agents} agents)")

    # Handle seed configuration
    if configs['primary'].seed == -1:
        configs['primary'].seed = random.randint(0, 10000)
        print(f"  Generated random seed: {configs['primary'].seed}")
    else:
        print(f"  Using config seed: {configs['primary'].seed}")

    # Set seed
    set_seed(configs['primary'].seed)
    configs['environment'].seed = configs['primary'].seed

    # Set render interval to decimation for faster execution
    configs['environment'].sim.render_interval = configs['environment'].decimation
    print(f"  Set render_interval to decimation: {configs['environment'].decimation}")

    # Calculate max_rollout_steps
    env_cfg = configs['environment']
    if args.max_rollout_steps is not None:
        max_rollout_steps = args.max_rollout_steps
        print(f"  Override max_rollout_steps: {max_rollout_steps}")
    else:
        max_rollout_steps = int(
            (1 / env_cfg.sim.dt) / env_cfg.decimation * env_cfg.episode_length_s
        )
        print(f"  Calculated max_rollout_steps: {max_rollout_steps}")
        print(f"    sim.dt: {env_cfg.sim.dt}")
        print(f"    decimation: {env_cfg.decimation}")
        print(f"    episode_length_s: {env_cfg.episode_length_s}")

    # Calculate policy_hz
    policy_hz = max_rollout_steps / env_cfg.episode_length_s
    print(f"  Calculated policy_hz: {policy_hz}")

    # Setup camera configuration for video recording if enabled
    if args.enable_video:
        print("  Setting up camera configuration for video recording...")
        env_cfg.scene.tiled_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.35),
                #rot=(0.6123724, 0.3535534, 0.3535534, 0.6123724),
                #convention="opengl"
                rot=(-0.3535534, 0.6123724, 0.6123724, -0.3535534),
                convention="ros"

            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=0.05,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 20.0)
            ),
            width=240,
            height=180,
            debug_vis=False,
        )
        print("    Camera configured: 240x180, RGB")

    # Create environment
    print("  Creating environment...")
    task_name = configs['experiment'].task_name

    # Wrap with sensor support if contact sensor config is present AND use_contact_sensor is enabled
    if (hasattr(env_cfg.task, 'held_fixed_contact_sensor') and
        configs['wrappers'].force_torque_sensor.use_contact_sensor):
        EnvClass = create_sensor_enabled_factory_env(FactoryEnv)
        print("    Using sensor-enabled factory environment")
    else:
        EnvClass = FactoryEnv
        print("    Using standard factory environment")

    # Create environment directly
    env = EnvClass(cfg=env_cfg, render_mode="rgb_array" if args.enable_video else None)
    print(f"    Environment created: {task_name}")

    # Reduce scene lighting intensity for better video visibility (after env creation)
    # print("  Reducing scene lighting intensity...")
    #import isaacsim.core.utils.prims as prim_utils
    #light_prim = prim_utils.get_prim_at_path("/World/Light")
    #if light_prim.IsValid():
    #    light_prim.GetAttribute("inputs:intensity").Set(800.0)
    #    print("    Dome light intensity reduced to 800.0")
    #else:
    #    print("    Warning: Light prim not found at /World/Light")

    # Disable wrappers not needed for evaluation
    print("  Configuring wrappers for evaluation...")
    # Keep efficient_reset enabled to avoid partial resets (ensures consistent episode lengths)
    # configs['wrappers'].efficient_reset_enabled = False  # Keep enabled
    configs['wrappers'].wandb_logging.enabled = False
    configs['wrappers'].action_logging.enabled = False

    # Enable factory_metrics but disable WandB publishing (for smoothness data in info)
    configs['wrappers'].factory_metrics.enabled = True
    configs['wrappers'].factory_metrics.publish_to_wandb = False

    # Apply wrappers using launch_utils_v3
    print("  Applying evaluation wrappers...")
    env = lUtils.apply_wrappers(env, configs)

    # Add image capture wrapper for video
    if args.enable_video:
        print("  - Img2InfoWrapper for video capture")
        env = Img2InfoWrapper(env)

    # Apply AsyncCriticIsaacLabWrapper (flattens policy+critic observations)
    print("  - AsyncCriticIsaacLabWrapper")
    from wrappers.skrl.async_critic_isaaclab_wrapper import AsyncCriticIsaacLabWrapper
    env = AsyncCriticIsaacLabWrapper(env)

    print("  Environment setup complete")

    return env, configs, max_rollout_steps, policy_hz


def create_models_and_agent(env: Any, configs: Dict[str, Any], args: argparse.Namespace) -> Tuple[Dict[str, Any], Any]:
    """
    Create block models and agent for checkpoint loading.

    Uses launch_utils_v3 to create models and agent with proper configuration.

    Args:
        env: Configured environment
        configs: Configuration dictionary
        args: Parsed command-line arguments

    Returns:
        Tuple of (models, agent)
    """
    print("  Creating block models and agent...")

    # Create models using launch_utils_v3
    print("    Creating models...")
    models = lUtils.create_policy_and_value_models(env, configs)
    print("    Models created")
    print(f"    DEBUG: Policy model type: {type(models['policy'])}")
    print(f"    DEBUG: Value model type: {type(models['value'])}")

    # Create memory (required for agent initialization, even for eval)
    print("    Creating memory...", flush=True)
    memory = MultiRandomMemory(memory_size=16, num_envs=env.num_envs, device=env.device)
    print("    Memory created", flush=True)

    # Disable checkpoint tracking for evaluation
    configs['agent'].track_ckpts = False

    # Create agent using launch_utils_v3
    print("    Creating BlockPPO agent...", flush=True)
    agent = lUtils.create_block_ppo_agents(env, configs, models, memory)
    print("    Agent created", flush=True)

    # Debug preprocessor state
    print(f"    DEBUG: Agent has _state_preprocessor: {hasattr(agent, '_state_preprocessor')}")
    if hasattr(agent, '_state_preprocessor'):
        print(f"    DEBUG: _state_preprocessor type: {type(agent._state_preprocessor)}")
        print(f"    DEBUG: _state_preprocessor bool: {bool(agent._state_preprocessor)}")

    print(f"  Created BlockPPO agent with {configs['primary'].total_agents} agents")

    return models, agent


def load_checkpoint_from_wandb(wandb_checkpoint: str) -> List[Dict[str, str]]:
    """
    Download checkpoint from WandB and construct checkpoint dict.

    Args:
        wandb_checkpoint: String in format "project/run_id/step_number"

    Returns:
        List containing single checkpoint dict

    Raises:
        RuntimeError: If download fails or config is missing required fields
    """
    # Parse wandb_checkpoint string
    parts = wandb_checkpoint.split('/')
    if len(parts) != 3:
        raise RuntimeError(
            f"Invalid wandb_checkpoint format: '{wandb_checkpoint}'. "
            f"Expected format: 'project/run_id/step_number'"
        )

    project, run_id, step_str = parts

    try:
        step_number = int(step_str)
    except ValueError:
        raise RuntimeError(f"Invalid step number: '{step_str}'. Must be an integer.")

    print(f"  Downloading checkpoint from WandB: {project}/{run_id} at step {step_number}")

    # Fetch run from WandB
    api = wandb.Api()
    try:
        run = api.run(f"{project}/{run_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch WandB run {project}/{run_id}: {e}")

    # Extract required config fields
    required_fields = ['agent_idx', 'break_force', 'directory', 'experiment_name']
    config = {}
    missing_fields = []

    for field in required_fields:
        value = run.config.get(field)
        # If not found at top level and field is break_force, check agent_specific
        if value is None and field == 'break_force' and 'agent_specific' in run.config:
            value = run.config['agent_specific'].get('break_force')

        if value is None:
            missing_fields.append(field)
        else:
            config[field] = value

    if missing_fields:
        raise RuntimeError(
            f"Missing required config fields in WandB run {project}/{run_id}: {missing_fields}. "
            f"Available config keys: {list(run.config.keys())}"
        )

    agent_idx = config['agent_idx']
    task_name = run.config.get('task_name', 'Isaac-Factory-PegInsert-Direct-v0')

    # Download checkpoint files from WandB
    policy_filename = f"agent_{agent_idx}_{step_number}.pt"
    critic_filename = f"critic_{agent_idx}_{step_number}.pt"

    # Create temporary download directory
    import tempfile
    download_dir = tempfile.mkdtemp(prefix="wandb_ckpt_")

    try:
        # Download policy checkpoint
        policy_file = run.file(f"ckpts/policies/{policy_filename}")
        policy_path = policy_file.download(root=download_dir, replace=True)
        print(f"    Downloaded policy: {policy_path}")

        # Download critic checkpoint
        critic_file = run.file(f"ckpts/critics/{critic_filename}")
        critic_path = critic_file.download(root=download_dir, replace=True)
        print(f"    Downloaded critic: {critic_path}")

    except Exception as e:
        raise RuntimeError(
            f"Failed to download checkpoint files from WandB run {project}/{run_id}: {e}"
        )

    # Construct video path
    vid_path = os.path.join(
        config['directory'],
        config['experiment_name'],
        "eval_videos",
        f"agent_{agent_idx}_{step_number}.gif"
    )

    # Construct checkpoint dict
    checkpoint_dict = {
        'ckpt_path': str(policy_path),
        'task': task_name,
        'vid_path': vid_path,
        'project': project,
        'run_id': run_id
    }

    print(f"  Successfully loaded checkpoint from WandB")
    return [checkpoint_dict]


def load_local_checkpoint(checkpoint_path: str, wandb_run: str) -> List[Dict[str, str]]:
    """
    Load local checkpoint and fetch config from WandB.

    Args:
        checkpoint_path: Path to local checkpoint file
        wandb_run: String in format "project/run_id"

    Returns:
        List containing single checkpoint dict

    Raises:
        RuntimeError: If checkpoint doesn't exist or WandB config is missing required fields
    """
    # Validate checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Local checkpoint not found: {checkpoint_path}")

    # Parse wandb_run string
    parts = wandb_run.split('/')
    if len(parts) != 2:
        raise RuntimeError(
            f"Invalid wandb_run format: '{wandb_run}'. "
            f"Expected format: 'project/run_id'"
        )

    project, run_id = parts

    print(f"  Loading local checkpoint: {checkpoint_path}")
    print(f"  Fetching config from WandB: {project}/{run_id}")

    # Fetch run from WandB
    api = wandb.Api()
    try:
        run = api.run(f"{project}/{run_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch WandB run {project}/{run_id}: {e}")

    # Extract required config fields
    required_fields = ['agent_idx', 'break_force', 'directory', 'experiment_name']
    config = {}
    missing_fields = []

    for field in required_fields:
        value = run.config.get(field)
        # If not found at top level and field is break_force, check agent_specific
        if value is None and field == 'break_force' and 'agent_specific' in run.config:
            value = run.config['agent_specific'].get('break_force')

        if value is None:
            missing_fields.append(field)
        else:
            config[field] = value

    if missing_fields:
        raise RuntimeError(
            f"Missing required config fields in WandB run {project}/{run_id}: {missing_fields}. "
            f"Available config keys: {list(run.config.keys())}"
        )

    agent_idx = config['agent_idx']
    task_name = run.config.get('task_name', 'Isaac-Factory-PegInsert-Direct-v0')

    # Extract step number from checkpoint filename
    # Expected format: agent_{agent_idx}_{step}.pt
    import re
    match = re.search(r'agent_\d+_(\d+)\.pt$', checkpoint_path)
    if not match:
        raise RuntimeError(
            f"Could not extract step number from checkpoint path: {checkpoint_path}. "
            f"Expected format: agent_{{agent_idx}}_{{step}}.pt"
        )

    step_number = int(match.group(1))

    # Construct video path
    vid_path = os.path.join(
        config['directory'],
        config['experiment_name'],
        "eval_videos",
        f"agent_{agent_idx}_{step_number}.gif"
    )

    # Construct checkpoint dict
    checkpoint_dict = {
        'ckpt_path': checkpoint_path,
        'task': task_name,
        'vid_path': vid_path,
        'project': project,
        'run_id': run_id
    }

    print(f"  Successfully loaded local checkpoint")
    return [checkpoint_dict]


def read_and_remove_tracker_batch(tracker_path: str, num_agents: int, batch_mode: str) -> Optional[List[Dict[str, str]]]:
    """
    Atomically read and remove batch of checkpoints from tracker file.

    This function uses FileLock to ensure multiple daemon instances don't
    process the same checkpoints. Lines are removed immediately upon reading.

    Args:
        tracker_path: Path to checkpoint tracker file
        num_agents: Maximum number of agents to load
        batch_mode: 'wait_full' or 'max_throughput'

    Returns:
        List of checkpoint dicts [{ckpt_path, task, vid_path, project, run_id}, ...] or None if no checkpoints
    """
    lock = FileLock(tracker_path + ".lock")

    with lock:
        # Check if tracker file exists
        if not os.path.exists(tracker_path):
            print(f"  DEBUG: Tracker file does not exist: {tracker_path}")
            return None

        # Read all lines from tracker
        with open(tracker_path, 'r') as f:
            lines = f.readlines()

        # Filter out empty lines
        lines = [line.strip() for line in lines if line.strip()]

        print(f"  DEBUG: Read {len(lines)} lines from tracker")

        if len(lines) == 0:
            print(f"  DEBUG: No lines in tracker after filtering")
            return None

        # Determine how many checkpoints to take
        print(f"  DEBUG: batch_mode={batch_mode}, num_agents={num_agents}")
        if batch_mode == 'wait_full':
            # Wait until we have at least num_agents checkpoints
            if len(lines) < num_agents:
                print(f"  DEBUG: wait_full mode - only {len(lines)} lines available, need {num_agents}")
                return None
            num_to_take = num_agents
        else:  # max_throughput
            # Take whatever is available (1 to num_agents)
            num_to_take = min(len(lines), num_agents)

        print(f"  DEBUG: Will take {num_to_take} checkpoints")

        # Take first num_to_take checkpoints
        selected_lines = lines[:num_to_take]
        remaining_lines = lines[num_to_take:]

        # Parse selected lines into checkpoint dicts
        checkpoint_dicts = []
        for line in selected_lines:
            parts = line.split()
            if len(parts) == 5:
                ckpt_path, task, vid_path, project, run_id = parts
                checkpoint_dicts.append({
                    'ckpt_path': ckpt_path,
                    'task': task,
                    'vid_path': vid_path,
                    'project': project,
                    'run_id': run_id
                })
            else:
                print(f"  Warning: Skipping malformed line: {line}")

        if len(checkpoint_dicts) == 0:
            return None

        # Write remaining lines back to tracker (atomic removal)
        with open(tracker_path, 'w') as f:
            for line in remaining_lines:
                f.write(line + '\n')

        return checkpoint_dicts


def write_checkpoints_back_to_tracker(tracker_path: str, checkpoint_dicts: List[Dict[str, str]]) -> None:
    """
    Write checkpoint lines back to tracker file (rollback on failure).

    Uses FileLock to safely append checkpoints back to tracker.

    Args:
        tracker_path: Path to checkpoint tracker file
        checkpoint_dicts: List of checkpoint dicts to write back
    """
    lock = FileLock(tracker_path + ".lock")

    with lock:
        # Append checkpoint lines back to tracker
        with open(tracker_path, 'a') as f:
            for ckpt_dict in checkpoint_dicts:
                line = f"{ckpt_dict['ckpt_path']} {ckpt_dict['task']} {ckpt_dict['vid_path']} {ckpt_dict['project']} {ckpt_dict['run_id']}\n"
                f.write(line)


def upload_checkpoints_to_wandb(checkpoint_dicts: List[Dict[str, str]]) -> None:
    """
    Upload checkpoint files to WandB run.

    Uploads both policy and critic checkpoints to organized directories:
    - ckpts/policies/agent_{agent_idx}_{step}.pt
    - ckpts/critics/critic_{agent_idx}_{step}.pt

    Args:
        checkpoint_dicts: List of checkpoint dicts with project, run_id, ckpt_path

    Raises:
        RuntimeError: If upload fails
    """
    import re

    for i, ckpt_dict in enumerate(checkpoint_dicts):
        project = ckpt_dict['project']
        run_id = ckpt_dict['run_id']
        ckpt_path = ckpt_dict['ckpt_path']

        # Extract step number and agent_idx from checkpoint filename
        match = re.search(r'agent_(\d+)_(\d+)\.pt$', ckpt_path)
        if not match:
            raise RuntimeError(
                f"Could not extract agent_idx and step from checkpoint path: {ckpt_path}. "
                f"Expected format: agent_{{agent_idx}}_{{step}}.pt"
            )

        agent_idx = int(match.group(1))
        step_number = int(match.group(2))

        # Construct critic path
        critic_path = ckpt_path.replace(f"agent_{agent_idx}_", f"critic_{agent_idx}_")

        if not os.path.exists(critic_path):
            raise RuntimeError(f"Critic checkpoint not found: {critic_path}")

        print(f"  Uploading checkpoints for {project}/{run_id} at step {step_number}")

        try:
            # Resume WandB run
            run = wandb.init(project=project, id=run_id, resume="must")

            # Upload policy checkpoint
            run.save(ckpt_path, base_path="ckpts/policies", policy="now")
            print(f"    Uploaded policy: {ckpt_path}")

            # Upload critic checkpoint
            run.save(critic_path, base_path="ckpts/critics", policy="now")
            print(f"    Uploaded critic: {critic_path}")

            wandb.finish()
            print(f"  Successfully uploaded checkpoints to WandB")

        except Exception as e:
            wandb.finish()
            raise RuntimeError(
                f"Failed to upload checkpoints to WandB for {project}/{run_id}: {e}"
            )


def fetch_break_forces_from_wandb(checkpoint_dicts: List[Dict[str, str]]) -> List[float]:
    """
    Query WandB for break_force config for each checkpoint.

    Args:
        checkpoint_dicts: List of checkpoint dicts with project and run_id

    Returns:
        List of break_force values

    Raises:
        RuntimeError: If break_force not found in any WandB config
    """
    api = wandb.Api()
    break_forces = []

    for i, ckpt_dict in enumerate(checkpoint_dicts):
        project = ckpt_dict['project']
        run_id = ckpt_dict['run_id']

        print(f"  Fetching break_force for checkpoint {i+1}/{len(checkpoint_dicts)}: {project}/{run_id}")

        try:
            # Fetch run from WandB
            run = api.run(f"{project}/{run_id}")

            # Extract break_force from config (check both top-level and agent_specific)
            break_force = run.config.get('break_force', None)

            # If not found at top level, check agent_specific
            if break_force is None and 'agent_specific' in run.config:
                break_force = run.config['agent_specific'].get('break_force', None)

            if break_force is None:
                raise RuntimeError(
                    f"break_force not found in WandB config for run {project}/{run_id}. "
                    f"Available config keys: {list(run.config.keys())}"
                )

            print(f"    Found break_force: {break_force}")
            break_forces.append(float(break_force))

        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch break_force from WandB for checkpoint {ckpt_dict['ckpt_path']}: {e}"
            )

    return break_forces


def load_checkpoints_into_agent(agent: Any, checkpoint_dicts: List[Dict[str, str]]) -> None:
    """
    Load checkpoint weights into agent's block models.

    Uses pack_agents_into_block() from block_simba.py to insert single-agent
    checkpoints into the correct agent slices.

    Args:
        agent: BlockPPO agent with block models
        checkpoint_dicts: List of checkpoint dicts with ckpt_path

    Raises:
        RuntimeError: If checkpoint load fails
    """
    from models.block_simba import pack_agents_into_block
    from models.SimBa import SimBaNet

    print(f"  Loading {len(checkpoint_dicts)} checkpoints into agent...")

    # Load all checkpoints and create single-agent models
    policy_agents = {}
    critic_agents = {}

    for agent_idx, ckpt_dict in enumerate(checkpoint_dicts):
        ckpt_path = ckpt_dict['ckpt_path']

        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

        print(f"    Loading checkpoint {agent_idx+1}/{len(checkpoint_dicts)}: {ckpt_path}")

        try:
            # Load policy checkpoint
            policy_checkpoint = torch.load(ckpt_path, map_location=agent.device, weights_only=False)
            if 'net_state_dict' not in policy_checkpoint:
                raise RuntimeError(f"Checkpoint missing 'net_state_dict': {ckpt_path}")

            # Load critic checkpoint
            critic_path = ckpt_path.replace("agent_", "critic_")
            if not os.path.exists(critic_path):
                raise RuntimeError(f"Critic checkpoint not found: {critic_path}")

            critic_checkpoint = torch.load(critic_path, map_location=agent.device, weights_only=False)
            if 'net_state_dict' not in critic_checkpoint:
                raise RuntimeError(f"Critic checkpoint missing 'net_state_dict': {critic_path}")

            # Create single-agent SimBaNet models for policy and critic
            policy_agent = SimBaNet(
                n=len(agent.models['policy'].actor_mean.resblocks),
                in_size=agent.models['policy'].actor_mean.obs_dim,
                out_size=agent.models['policy'].actor_mean.act_dim,
                latent_size=agent.models['policy'].actor_mean.hidden_dim,
                device=agent.device,
                tan_out=agent.models['policy'].actor_mean.use_tanh
            )
            policy_agent.load_state_dict(policy_checkpoint['net_state_dict'])
            policy_agents[agent_idx] = policy_agent

            critic_agent = SimBaNet(
                n=len(agent.models['value'].critic.resblocks),
                in_size=agent.models['value'].critic.obs_dim,
                out_size=agent.models['value'].critic.act_dim,
                latent_size=agent.models['value'].critic.hidden_dim,
                device=agent.device,
                tan_out=agent.models['value'].critic.use_tanh
            )
            critic_agent.load_state_dict(critic_checkpoint['net_state_dict'])
            critic_agents[agent_idx] = critic_agent

            # Load log_std for policy
            if 'log_std' in policy_checkpoint:
                agent.models['policy'].actor_logstd[agent_idx].data.copy_(policy_checkpoint['log_std'].data)
                print(f"      DEBUG: Loaded log_std with shape {policy_checkpoint['log_std'].shape}")
            else:
                print(f"      WARNING: No log_std found in policy checkpoint")

            # Load state preprocessor for this agent
            if 'state_preprocessor' in policy_checkpoint:
                # Initialize preprocessor lists if not already done
                if not hasattr(agent, '_per_agent_state_preprocessors'):
                    agent._per_agent_state_preprocessors = [None] * agent.num_agents

                # Create preprocessor instance if needed
                if agent._per_agent_state_preprocessors[agent_idx] is None:
                    from skrl.resources.preprocessors.torch import RunningStandardScaler
                    obs_size = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else env.observation_space
                    agent._per_agent_state_preprocessors[agent_idx] = RunningStandardScaler(
                        size=obs_size,
                        device=agent.device
                    )

                # Load the state dict
                agent._per_agent_state_preprocessors[agent_idx].load_state_dict(policy_checkpoint['state_preprocessor'])

                # Verify and log statistics
                preprocessor = agent._per_agent_state_preprocessors[agent_idx]
                mean_avg = preprocessor.running_mean.mean().item()
                mean_std = preprocessor.running_mean.std().item()
                var_avg = preprocessor.running_variance.mean().item()
                var_std = preprocessor.running_variance.std().item()
                count = preprocessor.current_count.item()

                print(f"      Loaded state preprocessor: mean_avg={mean_avg:.4f}, mean_std={mean_std:.4f}, var_avg={var_avg:.4f}, var_std={var_std:.4f}, count={count}")
            else:
                print(f"      WARNING: No state_preprocessor in checkpoint for agent {agent_idx}")
                raise RuntimeError(f"Checkpoint missing state_preprocessor for agent {agent_idx}")

            # Load value preprocessor for this agent
            if 'value_preprocessor' in critic_checkpoint:
                # Initialize preprocessor lists if not already done
                if not hasattr(agent, '_per_agent_value_preprocessors'):
                    agent._per_agent_value_preprocessors = [None] * agent.num_agents

                # Create preprocessor instance if needed
                if agent._per_agent_value_preprocessors[agent_idx] is None:
                    from skrl.resources.preprocessors.torch import RunningStandardScaler
                    agent._per_agent_value_preprocessors[agent_idx] = RunningStandardScaler(
                        size=1,
                        device=agent.device
                    )

                # Load the state dict
                agent._per_agent_value_preprocessors[agent_idx].load_state_dict(critic_checkpoint['value_preprocessor'])

                # Verify and log statistics
                preprocessor = agent._per_agent_value_preprocessors[agent_idx]
                mean_val = preprocessor.running_mean.item()
                var_val = preprocessor.running_variance.item()
                count = preprocessor.current_count.item()

                print(f"      Loaded value preprocessor: mean={mean_val:.4f}, var={var_val:.4f}, count={count}")
            else:
                print(f"      WARNING: No value_preprocessor in checkpoint for agent {agent_idx}")
                raise RuntimeError(f"Checkpoint missing value_preprocessor for agent {agent_idx}")

            # Verify
            sample_value = list(policy_checkpoint['net_state_dict'].values())[0].flatten()[0].item()
            print(f"      Loaded successfully (sample weight: {sample_value:.4f})")

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {e}")

    # Pack all single-agent models into block models
    pack_agents_into_block(agent.models['policy'].actor_mean, policy_agents)
    pack_agents_into_block(agent.models['value'].critic, critic_agents)

    # Wrap preprocessors for SKRL compatibility
    if hasattr(agent, '_per_agent_state_preprocessors') and agent._per_agent_state_preprocessors:
        from agents.block_ppo import PerAgentPreprocessorWrapper
        agent._state_preprocessor = PerAgentPreprocessorWrapper(agent, agent._per_agent_state_preprocessors)
        print("  DEBUG: Wrapped state preprocessors for agent compatibility")
    else:
        print("  WARNING: No state preprocessors loaded")

    if hasattr(agent, '_per_agent_value_preprocessors') and agent._per_agent_value_preprocessors:
        from agents.block_ppo import PerAgentPreprocessorWrapper
        agent._value_preprocessor = PerAgentPreprocessorWrapper(agent, agent._per_agent_value_preprocessors)
        print("  DEBUG: Wrapped value preprocessors for agent compatibility")
    else:
        print("  WARNING: No value preprocessors loaded")

    # Set models to eval mode
    agent.models['policy'].eval()
    agent.models['value'].eval()


def update_environment_break_forces(env: Any, break_forces: List[float], num_envs_per_agent: int) -> None:
    """
    Update per-agent break_force settings in environment.

    Searches through wrapper chain to find FragileObjectWrapper and updates
    break forces for each agent's environments.

    Args:
        env: Environment instance
        break_forces: List of break_force values per agent
        num_envs_per_agent: Number of environments per agent
    """
    print(f"  Updating environment break forces for {len(break_forces)} agents...")

    # Find FragileObjectWrapper in wrapper chain
    fragile_wrapper = None
    current_env = env
    search_depth = 0
    max_depth = 10

    while current_env is not None and search_depth < max_depth:
        if isinstance(current_env, FragileObjectWrapper):
            fragile_wrapper = current_env
            break

        # Try different wrapper access patterns
        next_env = None
        for attr in ['env', '_env', 'unwrapped']:
            if hasattr(current_env, attr):
                next_env = getattr(current_env, attr)
                break

        current_env = next_env
        search_depth += 1

    if fragile_wrapper is None:
        print("    Warning: FragileObjectWrapper not found in environment wrapper chain")
        print("    Break forces will not be updated (checkpoints may have been trained without fragile objects)")
        return

    # Update break forces per agent
    # Each agent controls a slice of environments [agent_idx * num_envs_per_agent : (agent_idx + 1) * num_envs_per_agent]
    num_agents = len(break_forces)
    total_envs = num_agents * num_envs_per_agent

    for agent_idx in range(num_agents):
        start_env = agent_idx * num_envs_per_agent
        end_env = (agent_idx + 1) * num_envs_per_agent
        break_force = break_forces[agent_idx]

        # Update break force for this agent's environments
        fragile_wrapper.break_force[start_env:end_env] = break_force

        print(f"    Agent {agent_idx}: envs [{start_env}:{end_env}] -> break_force = {break_force}")

    print(f"  Successfully updated break forces for all {num_agents} agents")


def get_masked_output(model: torch.nn.Module, input_vec: torch.Tensor, mask_indices: torch.Tensor,
                      is_policy: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute model output and gradients with masked input.

    Args:
        model: Neural network module (policy or critic)
        input_vec: Input tensor [num_envs, features]
        mask_indices: Tensor of indices to mask (zero out)
        is_policy: If True, this is a policy model; if False, it's a critic model

    Returns:
        Tuple of (output, gradients, log_std):
            - output: Model output tensor [num_envs, output_size]
            - gradients: Gradient of output w.r.t. input [num_envs, features]
            - log_std: Log std for policy (None for critic)
    """
    # Create masked input (clone and zero out masked indices) - avoid in-place operation
    masked_input = input_vec.clone()
    masked_input[:, mask_indices] = 0.0
    masked_input = masked_input.requires_grad_(True)

    # Forward pass - use compute() method to preserve gradients
    model.train()  # Ensure model is in train mode for gradient computation

    if is_policy:
        # For policy, compute() returns (mean, log_std, {})
        action_mean, log_std, _ = model.compute({"states": masked_input}, role="policy")
        output = action_mean
    else:
        # For critic, compute() returns (value, {})
        value, _ = model.compute({"states": masked_input}, role="value")
        output = value
        log_std = None

    # Compute gradients
    output_sum = output.sum()
    grads = torch.autograd.grad(output_sum, masked_input, create_graph=False)[0]

    model.eval()  # Return model to eval mode

    return output.detach(), grads.detach(), log_std.detach() if log_std is not None else None


def reset_evaluation_seed(seed: int) -> None:
    """
    Reset all random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    print(f"  Resetting all random seeds to {seed}")

    # SKRL set_seed (handles torch, numpy, random)
    set_seed(seed)

    # Also set additional generators for completeness
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_evaluation_rollout(env: Any, agent: Any, max_rollout_steps: int, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute full evaluation episode and collect data.

    Args:
        env: Environment instance
        agent: BlockPPO agent
        max_rollout_steps: Maximum number of steps to run
        args: Command-line arguments

    Returns:
        Dictionary with collected rollout data:
            - images: tensor of images (if video enabled)
            - values: list of value predictions
            - rewards: list of rewards
            - infos: list of info dicts
            - success_history: list of success masks
            - engagement_history: list of engagement masks
            - force_control: tensor of force control selections
            - terminated_episodes: boolean tensor
            - truncated_episodes: boolean tensor
            - total_returns: tensor of total returns per env
            - success_step: tensor of step when each env succeeded
    """
    print(f"  Running rollout for {max_rollout_steps} steps...")

    # Initialize data collection
    rollout_data = {
        "observations": [],
        "actions": [],
        "images": [] if args.enable_video else None,
        "values": [],
        "rewards": [],
        "infos": [],
        "success_history": [],
        "engagement_history": [],
        "force_control": [],
        "terminated_episodes": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        "truncated_episodes": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        "total_returns": torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
        "success_step": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
        "termination_step": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
        "truncation_step": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
    }

    # Reset environment
    states, info = env.reset()

    print(f"  DEBUG: Initial reset - states shape: {states.shape}, dtype: {states.dtype}")
    print(f"  DEBUG: Initial states - min: {states.min().item():.4f}, max: {states.max().item():.4f}, mean: {states.mean().item():.4f}")

    # Evaluation rollout loop
    step_iterator = range(max_rollout_steps)
    if args.show_progress:
        step_iterator = tqdm.tqdm(step_iterator, desc="Evaluation Rollout", file=sys.stdout)

    for step in step_iterator:
        # Debug first few steps in detail
        debug_step = (step == 0 or step == 1 or step == 10 or step == 50)

        if debug_step:
            print(f"\n  DEBUG STEP {step}:")
            print(f"    Input states shape: {states.shape}, dtype: {states.dtype}")
            print(f"    Input states - min: {states.min().item():.4f}, max: {states.max().item():.4f}, mean: {states.mean().item():.4f}")
            print(f"    Input states sample [0, :5]: {states[0, :5]}")

        # Get action from agent (use mean action for deterministic evaluation)
        with torch.no_grad():
            outputs = agent.act(states, timestep=step, timesteps=max_rollout_steps)[-1]
            # Use mean actions for evaluation (deterministic)
            actions = outputs['mean_actions']

        if debug_step:
            print(f"    Agent output keys: {outputs.keys()}")
            print(f"    Actions shape: {actions.shape}, dtype: {actions.dtype}")
            print(f"    Actions - min: {actions.min().item():.4f}, max: {actions.max().item():.4f}, mean: {actions.mean().item():.4f}")
            print(f"    Actions sample [0, :]: {actions[0, :]}")
            if 'preprocessed_states' in outputs:
                prep_states = outputs['preprocessed_states']
                print(f"    Preprocessed states - min: {prep_states.min().item():.4f}, max: {prep_states.max().item():.4f}, mean: {prep_states.mean().item():.4f}")

        # Store observation and action
        rollout_data["observations"].append(states.clone())
        rollout_data["actions"].append(actions.clone())

        # Store value estimate if available
        with torch.no_grad():
            values = agent.value.act({"states": states}, role="value")[0]
            rollout_data["values"].append(values.clone())

        if debug_step:
            print(f"    Values shape: {values.shape}, dtype: {values.dtype}")
            print(f"    Values - min: {values.min().item():.4f}, max: {values.max().item():.4f}, mean: {values.mean().item():.4f}")

        # Step environment
        states, rewards, terminated, truncated, info = env.step(actions)

        # Squeeze rewards, terminated, truncated to remove trailing dimension [N, 1] -> [N]
        rewards = rewards.squeeze(-1)
        terminated = terminated.squeeze(-1)
        truncated = truncated.squeeze(-1)

        if debug_step:
            print(f"    After env.step:")
            print(f"      New states shape: {states.shape}, min: {states.min().item():.4f}, max: {states.max().item():.4f}, mean: {states.mean().item():.4f}")
            print(f"      Rewards - min: {rewards.min().item():.4f}, max: {rewards.max().item():.4f}, mean: {rewards.mean().item():.4f}, sum: {rewards.sum().item():.4f}")
            print(f"      Terminated: {terminated.sum().item()}/{len(terminated)}, Truncated: {truncated.sum().item()}/{len(truncated)}")
            if 'to_log' in info:
                print(f"      info['to_log'] has {len(info['to_log'])} keys: {list(info['to_log'].keys())[:5]}...")
            if 'success' in info:
                print(f"      Success count: {info['success'].sum().item()}/{len(info['success'])}")

        # Store rewards
        rollout_data["rewards"].append(rewards.clone())

        # Deep clone info dict with explicit tensor cloning for nested dicts
        info_copy = {}
        for key, value in info.items():
            if key == 'to_log' and isinstance(value, dict):
                # Clone all tensors in to_log dict
                info_copy['to_log'] = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            elif isinstance(value, torch.Tensor):
                # Clone other tensor values
                info_copy[key] = value.clone()
            elif isinstance(value, dict):
                # Recursively handle other dicts (like 'smoothness')
                info_copy[key] = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                # Copy primitives/other types directly
                info_copy[key] = value

        rollout_data["infos"].append(info_copy)

        # Accumulate returns
        rollout_data["total_returns"] += rewards

        # Track termination/truncation
        rollout_data["terminated_episodes"] |= terminated
        rollout_data["truncated_episodes"] |= truncated

        # Track success (first success timestep)
        if "success" in info:
            success_mask = info["success"] & (rollout_data["success_step"] == -1)
            rollout_data["success_step"][success_mask] = step
            rollout_data["success_history"].append(info["success"].clone())

        # Track termination (first termination timestep)
        termination_mask = terminated & (rollout_data["termination_step"] == -1)
        rollout_data["termination_step"][termination_mask] = step

        # Track truncation (first truncation timestep)
        truncation_mask = truncated & (rollout_data["truncation_step"] == -1)
        rollout_data["truncation_step"][truncation_mask] = step

        # Track engagement if available (get directly from environment)
        if hasattr(env.unwrapped, '_get_curr_successes') and hasattr(env.unwrapped, 'cfg_task'):
            if hasattr(env.unwrapped.cfg_task, 'engage_threshold'):
                engage_threshold = env.unwrapped.cfg_task.engage_threshold
                curr_engaged = env.unwrapped._get_curr_successes(engage_threshold, False)
                rollout_data["engagement_history"].append(curr_engaged.clone())

        # Track force control from action space (hybrid control)
        # Collected after step so that environment's action processing (e.g., EMA) is applied
        # Force control is the first force_size (3) dimensions of the action space
        # Apply threshold > 0.5 to convert to boolean selection
        unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        if hasattr(unwrapped_env, 'actions'):
            force_control_actions = unwrapped_env.actions[:, :3]  # First 3 dimensions
            force_control_mask = force_control_actions > 0.5
            rollout_data["force_control"].append(force_control_mask.clone())

        # Capture image if video enabled
        if args.enable_video and 'img' in info:
            rollout_data["images"].append(info['img'].clone())

        # Check if all environments are done
        if (terminated | truncated).all():
            print(f"  All environments terminated/truncated at step {step+1}/{max_rollout_steps}")
            break

    print(f"  Rollout complete: {step+1} steps collected")

    # Debug: Rollout summary
    print(f"\n  DEBUG: Rollout Summary:")
    print(f"    Total steps: {len(rollout_data['observations'])}")
    print(f"    Total rewards sum: {rollout_data['total_returns'].sum().item():.2f}")
    print(f"    Terminated envs: {rollout_data['terminated_episodes'].sum().item()}/{env.num_envs}")
    print(f"    Truncated envs: {rollout_data['truncated_episodes'].sum().item()}/{env.num_envs}")
    success_count = (rollout_data['success_step'] != -1).sum().item()
    print(f"    Successful envs: {success_count}/{env.num_envs}")
    if len(rollout_data["infos"]) > 0:
        print(f"    Info dicts collected: {len(rollout_data['infos'])}")
        if 'to_log' in rollout_data["infos"][0]:
            print(f"    to_log keys in first info: {len(rollout_data['infos'][0]['to_log'])}")
        if 'to_log' in rollout_data["infos"][-1]:
            print(f"    to_log keys in last info: {len(rollout_data['infos'][-1]['to_log'])}")

    # Convert lists to tensors for easier processing
    if len(rollout_data["observations"]) > 0:
        rollout_data["observations"] = torch.stack(rollout_data["observations"])
        rollout_data["actions"] = torch.stack(rollout_data["actions"])
        rollout_data["rewards"] = torch.stack(rollout_data["rewards"])

        if len(rollout_data["values"]) > 0:
            rollout_data["values"] = torch.stack(rollout_data["values"])

        if len(rollout_data["success_history"]) > 0:
            rollout_data["success_history"] = torch.stack(rollout_data["success_history"])

        if len(rollout_data["engagement_history"]) > 0:
            rollout_data["engagement_history"] = torch.stack(rollout_data["engagement_history"])

        if len(rollout_data["force_control"]) > 0:
            rollout_data["force_control"] = torch.stack(rollout_data["force_control"])

        if args.enable_video and rollout_data["images"] is not None and len(rollout_data["images"]) > 0:
            rollout_data["images"] = torch.stack(rollout_data["images"])

    # Compute sensitivity metrics by masking each observation/state group
    if args.sensitivity_mask_freq > 0:
        print("  Computing sensitivity metrics...")

        # Import dimension configs
        try:
            from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
        except ImportError:
            from omni.isaac.lab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG

        # Add force_torque dimension if not present
        if 'force_torque' not in OBS_DIM_CFG:
            OBS_DIM_CFG['force_torque'] = 6
        if 'force_torque' not in STATE_DIM_CFG:
            STATE_DIM_CFG['force_torque'] = 6

        # Get the last observation/state from rollout
        last_obs = rollout_data["observations"][-1]  # [num_envs, obs_dim]
        raw_obs = env.unwrapped._get_observations()
        last_state = raw_obs["critic"] if isinstance(raw_obs, dict) and "critic" in raw_obs else last_obs

        # Get unmasked outputs for comparison
        agent.policy.eval()
        agent.value.eval()
        with torch.no_grad():
            unmasked_policy_mean, unmasked_policy_logstd, _ = agent.policy.compute({"states": last_obs}, role="policy")
            unmasked_value, _ = agent.value.compute({"states": last_state}, role="value")

        # Initialize sensitivity storage
        rollout_data["sensitivity"] = {
            "policy": {},  # obs_group -> {"output": tensor, "gradients": tensor, "unmasked_output": tensor}
            "critic": {},   # state_group -> {"output": tensor, "gradients": tensor, "unmasked_output": tensor}
            "unmasked_policy_mean": unmasked_policy_mean,
            "unmasked_policy_logstd": unmasked_policy_logstd,
            "unmasked_value": unmasked_value
        }

        # Iterate over observation groups for policy
        current_idx = 0
        for obs_group in env.unwrapped.cfg.obs_order:
            if obs_group in OBS_DIM_CFG:
                group_dim = OBS_DIM_CFG[obs_group]
                mask_indices = torch.arange(current_idx, current_idx + group_dim, device=env.device)

                # Get masked output and gradients
                output, grads, masked_logstd = get_masked_output(agent.policy, last_obs, mask_indices, is_policy=True)
                rollout_data["sensitivity"]["policy"][obs_group] = {
                    "output": output,
                    "gradients": grads,
                    "masked_logstd": masked_logstd,
                    "unmasked_output": unmasked_policy_mean
                }

                current_idx += group_dim

        # Iterate over state groups for critic
        current_idx = 0
        for state_group in env.unwrapped.cfg.state_order:
            if state_group in STATE_DIM_CFG:
                group_dim = STATE_DIM_CFG[state_group]
                mask_indices = torch.arange(current_idx, current_idx + group_dim, device=env.device)

                # Get masked output and gradients
                output, grads, _ = get_masked_output(agent.value, last_state, mask_indices, is_policy=False)
                rollout_data["sensitivity"]["critic"][state_group] = {
                    "output": output,
                    "gradients": grads,
                    "unmasked_output": unmasked_value
                }

                current_idx += group_dim

        print(f"    Computed sensitivity for {len(rollout_data['sensitivity']['policy'])} policy groups, {len(rollout_data['sensitivity']['critic'])} critic groups")

    return rollout_data


def extract_to_log_metrics(infos: List[Dict[str, Any]], num_agents: int, num_envs_per_agent: int) -> List[Dict[str, float]]:
    """
    Extract and aggregate metrics from info['to_log'] across timesteps.

    Args:
        infos: List of info dicts from rollout (one per timestep)
        num_agents: Number of agents
        num_envs_per_agent: Number of environments per agent

    Returns:
        List of metric dicts (one per agent) with keys prefixed by "Eval/"
    """
    if len(infos) == 0:
        return [{} for _ in range(num_agents)]

    # Collect all to_log metrics across timesteps
    metric_accumulator = defaultdict(lambda: defaultdict(list))  # {agent_idx: {metric_name: [values]}}

    for info_dict in infos:
        if 'to_log' not in info_dict:
            continue

        to_log = info_dict['to_log']

        for metric_name, metric_value in to_log.items():
            if isinstance(metric_value, torch.Tensor):
                # Check tensor size to determine aggregation strategy
                if metric_value.shape[0] == num_agents:
                    # Already aggregated per agent (e.g., Active Wrench, Action Effect metrics)
                    for agent_idx in range(num_agents):
                        agent_value = metric_value[agent_idx].item()
                        # Skip NaN values (they indicate no data for this agent at this timestep)
                        if not torch.isnan(torch.tensor(agent_value)):
                            metric_accumulator[agent_idx][metric_name].append(agent_value)
                else:
                    # Per-environment tensor - need to slice and average per agent
                    num_envs = num_agents * num_envs_per_agent
                    if metric_value.shape[0] == num_envs:
                        for agent_idx in range(num_agents):
                            start_env = agent_idx * num_envs_per_agent
                            end_env = (agent_idx + 1) * num_envs_per_agent
                            agent_value = metric_value[start_env:end_env].mean().item()
                            metric_accumulator[agent_idx][metric_name].append(agent_value)
                    else:
                        # Unexpected tensor size - skip with warning
                        print(f"        Warning: Unexpected tensor size for {metric_name}: {metric_value.shape[0]} (expected {num_agents} or {num_envs})")
            else:
                # Scalar value - same for all agents
                for agent_idx in range(num_agents):
                    agent_value = float(metric_value)
                    metric_accumulator[agent_idx][metric_name].append(agent_value)

    # Compute averages across timesteps and format with "Eval/" prefix
    metrics_list = []
    for agent_idx in range(num_agents):
        agent_metrics = {}
        for metric_name, values in metric_accumulator[agent_idx].items():
            avg_value = sum(values) / len(values) if len(values) > 0 else 0.0
            agent_metrics[f"Eval {metric_name}"] = avg_value
        metrics_list.append(agent_metrics)

    return metrics_list


def calculate_metrics(rollout_data: Dict[str, Any], env: Any, agent: Any, args: argparse.Namespace) -> List[Dict[str, float]]:
    """
    Calculate evaluation metrics from rollout data.

    Args:
        rollout_data: Dictionary with collected rollout data
        env: Environment instance
        agent: BlockPPO agent
        args: Command-line arguments

    Returns:
        List of metric dictionaries, one per agent
    """
    print("  Calculating evaluation metrics...")

    # Determine number of agents and environments per agent
    num_envs = rollout_data["terminated_episodes"].shape[0]
    num_agents = agent.num_agents if hasattr(agent, 'num_agents') else 1
    num_envs_per_agent = num_envs // num_agents

    print(f"    Total environments: {num_envs}, Agents: {num_agents}, Envs per agent: {num_envs_per_agent}")

    metrics_list = []

    # Calculate metrics for each agent
    for agent_idx in range(num_agents):
        start_env = agent_idx * num_envs_per_agent
        end_env = (agent_idx + 1) * num_envs_per_agent

        print(f"    Agent {agent_idx}: environments [{start_env}:{end_env}]")

        metrics = {}

        # 1. Success/Failure/Truncation Rates
        agent_success_step = rollout_data["success_step"][start_env:end_env]
        agent_terminated = rollout_data["terminated_episodes"][start_env:end_env]
        agent_truncated = rollout_data["truncated_episodes"][start_env:end_env]

        # Success rate: environments where success_step != -1
        num_succeeded = (agent_success_step != -1).sum().item()
        success_rate = num_succeeded / num_envs_per_agent
        metrics["Eval/success_rate"] = success_rate

        # Failure rate: environments that terminated without success
        failed_mask = agent_terminated & (agent_success_step == -1)
        failure_rate = failed_mask.sum().item() / num_envs_per_agent
        metrics["Eval/failure_rate"] = failure_rate

        # Truncation rate
        truncation_rate = agent_truncated.sum().item() / num_envs_per_agent
        metrics["Eval/truncation_rate"] = truncation_rate

        # Engagement rate (if available)
        if len(rollout_data["engagement_history"]) > 0:
            agent_engagement = rollout_data["engagement_history"][:, start_env:end_env]
            ever_engaged = agent_engagement.any(dim=0)
            engagement_rate = ever_engaged.sum().item() / num_envs_per_agent
            metrics["Eval/engagement_rate"] = engagement_rate

        print(f"      Success: {success_rate:.3f}, Failure: {failure_rate:.3f}, Truncation: {truncation_rate:.3f}")

        # 2. Returns
        agent_returns = rollout_data["total_returns"][start_env:end_env]
        total_returns_mean = agent_returns.mean().item()
        total_returns_std = agent_returns.std().item()
        metrics["Eval/total_returns_mean"] = total_returns_mean
        metrics["Eval/total_returns_std"] = total_returns_std

        print(f"      Returns: {total_returns_mean:.3f} ± {total_returns_std:.3f}")

        # 3. Component Rewards
        # Extract from infos list - look for keys starting with 'logs_rew_'
        if len(rollout_data["infos"]) > 0:
            component_rewards = {}
            component_counts = {}

            # Iterate through all info dicts
            for info_dict in rollout_data["infos"]:
                for key, value in info_dict.items():
                    if key.startswith("logs_rew_"):
                        component_name = key.replace("logs_rew_", "")

                        # Initialize if first time seeing this component
                        if component_name not in component_rewards:
                            component_rewards[component_name] = 0.0
                            component_counts[component_name] = 0

                        # Accumulate (take mean across this agent's environments)
                        if isinstance(value, torch.Tensor):
                            agent_component_value = value[start_env:end_env].mean().item()
                            component_rewards[component_name] += agent_component_value
                        else:
                            component_rewards[component_name] += float(value)
                        component_counts[component_name] += 1

            # Calculate averages and add to metrics
            for component_name, total_reward in component_rewards.items():
                avg_reward = total_reward / component_counts[component_name]
                metrics[f"Eval Component Reward/{component_name}"] = avg_reward

            if len(component_rewards) > 0:
                print(f"      Found {len(component_rewards)} reward components")

        # 4. Smoothness Metrics
        # Extract from final step's info['smoothness']
        if len(rollout_data["infos"]) > 0:
            final_info = rollout_data["infos"][-1]
            if "smoothness" in final_info:
                smoothness = final_info["smoothness"]

                # Average across this agent's environments
                for key, value in smoothness.items():
                    if isinstance(value, torch.Tensor):
                        agent_smoothness = value[start_env:end_env]
                        avg_value = agent_smoothness.mean().item()
                        metrics[f"Eval Smoothness/{key}"] = avg_value

                print(f"      Smoothness: ssv={smoothness['ssv'][start_env:end_env].mean().item():.3f}, ssjv={smoothness['ssjv'][start_env:end_env].mean().item():.3f}")

        # 5. Sensitivity Metrics
        if "sensitivity" in rollout_data:
            print(f"      Processing sensitivity metrics...")

            # Get sigma_idx for hybrid control (number of Bernoulli selection dimensions)
            from models.SimBa_hybrid_control import HybridControlBlockSimBaActor
            if type(agent.policy) == HybridControlBlockSimBaActor:
                sigma_idx = 2*agent.policy.force_size
            else:
                sigma_idx = 0
            #sigma_idx = agent.policy.sigma_idx if hasattr(agent.policy, 'sigma_idx') else 0

            # Process policy sensitivity
            for obs_group, data in rollout_data["sensitivity"]["policy"].items():
                # Slice for this agent's environments
                agent_grads = data["gradients"][start_env:end_env]  # [envs_per_agent, obs_dim]
                masked_output = data["output"][start_env:end_env]  # [envs_per_agent, action_dim]
                unmasked_output = data["unmasked_output"][start_env:end_env]  # [envs_per_agent, action_dim]

                # 1. Policy Saliency: Gradient magnitude (L2 norm per environment, then mean)
                grad_magnitude = torch.norm(agent_grads, p=2, dim=1).mean().item()
                metrics[f"Policy Sensitivity/{obs_group}_saliency"] = grad_magnitude

                # 2. Policy Fisher Information: Squared gradient magnitude
                fisher_info = (agent_grads ** 2).sum(dim=1).mean().item()
                metrics[f"Policy Sensitivity/{obs_group}_fisher"] = fisher_info

                # 3. Policy KL Divergence: KL(unmasked || masked) for Gaussian distributions
                # Get unmasked and masked distributions (mean and stddev)
                unmasked_logstd = rollout_data["sensitivity"]["unmasked_policy_logstd"][start_env:end_env]
                masked_logstd = data["masked_logstd"][start_env:end_env]

                # For hybrid control, .compute() returns raw output with 2*sigma_idx selection logits
                # Need to skip first 2*sigma_idx dimensions to get Gaussian component means
                # Note: log_std doesn't include Bernoulli dimensions, so it's already the correct size
                if sigma_idx > 0:
                    mu1 = unmasked_output[:, sigma_idx:]  # Skip 2*force_size selection logits
                    mu2 = masked_output[:, sigma_idx:]    # Skip 2*force_size selection logits
                    # log_std already has correct size (doesn't include Bernoulli dims)
                    sigma1 = torch.exp(unmasked_logstd)
                    sigma2 = torch.exp(masked_logstd)
                else:
                    mu1 = unmasked_output
                    mu2 = masked_output
                    sigma1 = torch.exp(unmasked_logstd)
                    sigma2 = torch.exp(masked_logstd)

                # KL divergence: KL(p1||p2) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
                var1 = sigma1 ** 2
                var2 = sigma2 ** 2
                print(mu1.size(), mu2.size(), var1.size(), var2.size())
                kl_div = (
                    torch.log(sigma2 / sigma1) +
                    (var1 + (mu1 - mu2) ** 2) / (2 * var2) -
                    0.5
                )
                # Average across action dimensions and environments
                kl_div = kl_div.mean().item()
                metrics[f"Policy Sensitivity/{obs_group}_kl_divergence"] = kl_div

            # Process critic sensitivity
            for state_group, data in rollout_data["sensitivity"]["critic"].items():
                # Slice for this agent's environments
                agent_grads = data["gradients"][start_env:end_env]  # [envs_per_agent, state_dim]
                masked_value = data["output"][start_env:end_env]  # [envs_per_agent, 1]
                unmasked_value = data["unmasked_output"][start_env:end_env]  # [envs_per_agent, 1]

                # 1. Critic Saliency: Gradient magnitude (L2 norm per environment, then mean)
                grad_magnitude = torch.norm(agent_grads, p=2, dim=1).mean().item()
                metrics[f"Critic Sensitivity/{state_group}_saliency"] = grad_magnitude

                # 2. Critic Value Change: Absolute difference between masked and unmasked
                value_change = (masked_value - unmasked_value).abs().mean().item()
                metrics[f"Critic Sensitivity/{state_group}_value_change"] = value_change

            print(f"        Computed sensitivity metrics for {len(rollout_data['sensitivity']['policy'])} policy + {len(rollout_data['sensitivity']['critic'])} critic groups")

        metrics_list.append(metrics)

    # Extract and merge to_log metrics
    print("  Extracting to_log metrics from infos...")
    to_log_metrics_list = extract_to_log_metrics(rollout_data["infos"], num_agents, num_envs_per_agent)

    # Merge to_log metrics into metrics_list
    for agent_idx in range(num_agents):
        metrics_list[agent_idx].update(to_log_metrics_list[agent_idx])

    if len(to_log_metrics_list[0]) > 0:
        print(f"    Found {len(to_log_metrics_list[0])} to_log metrics per agent")

    print(f"  Calculated metrics for {num_agents} agents")

    return metrics_list


def _create_checkpoint_gif(images: torch.Tensor, values: torch.Tensor, success_step: torch.Tensor,
                          termination_step: torch.Tensor, truncation_step: torch.Tensor,
                          engagement_history: torch.Tensor, force_control: torch.Tensor,
                          output_path: str, duration: int, font: Any) -> None:
    """
    Create a 3x4 grid GIF from 12 selected environments.

    Args:
        images: [num_steps, 12, 180, 240, 3] tensor
        values: [num_steps, 12] tensor
        success_step: [12] tensor
        termination_step: [12] tensor
        truncation_step: [12] tensor
        engagement_history: [num_steps, 12] tensor
        force_control: [num_steps, 12, 3] tensor
        output_path: Path to save GIF
        duration: Frame duration in milliseconds
        font: PIL ImageFont to use
    """
    num_steps = images.shape[0]
    pil_frames = []

    # Process each timestep
    for step_idx in range(num_steps):
        # Create empty canvas for 3x4 grid (540 height × 960 width)
        canvas = torch.zeros((540, 960, 3), dtype=torch.float32, device=images.device)

        # Process each of 12 environments
        for env_idx in range(12):
            # Calculate grid position
            row = env_idx // 4
            col = env_idx % 4

            # Get image for this environment at this step [180, 240, 3]
            img = images[step_idx, env_idx].clone()

            # Invert colors
            img = 1.0 - img

            # Check if environment is done at this step
            is_succeeded = step_idx >= success_step[env_idx] and success_step[env_idx] != -1
            is_terminated = step_idx >= termination_step[env_idx] and termination_step[env_idx] != -1
            is_truncated = step_idx >= truncation_step[env_idx] and truncation_step[env_idx] != -1
            is_done = is_succeeded or is_terminated or is_truncated

            # Apply gray overlay if done (0.4 gray + 0.6 original)
            if is_done:
                gray = torch.full_like(img, 0.5)
                img = 0.4 * gray + 0.6 * img

            # Place image in canvas
            y_start = row * 180
            y_end = (row + 1) * 180
            x_start = col * 240
            x_end = (col + 1) * 240
            canvas[y_start:y_end, x_start:x_end] = img

        # Convert canvas to PIL Image
        canvas_np = (canvas.cpu().numpy() * 255).astype(np.uint8)
        #canvas_np = (canvas.cpu().numpy()).astype(np.uint8)
        pil_img = Image.fromarray(canvas_np)
        draw = ImageDraw.Draw(pil_img)

        # Draw annotations for each environment
        for env_idx in range(12):
            row = env_idx // 4
            col = env_idx % 4
            x_offset = col * 240
            y_offset = row * 180

            # Determine border color (precedence: success > termination/truncation > engagement)
            is_succeeded = step_idx >= success_step[env_idx] and success_step[env_idx] != -1
            is_terminated = step_idx >= termination_step[env_idx] and termination_step[env_idx] != -1
            is_truncated = step_idx >= truncation_step[env_idx] and truncation_step[env_idx] != -1
            is_engaged = engagement_history[step_idx, env_idx].item()

            border_color = None
            if is_succeeded:
                border_color = "green"
            elif is_terminated or is_truncated:
                border_color = "red"
            elif is_engaged:
                border_color = "orange"

            # Draw border if needed
            if border_color is not None:
                draw.rectangle(
                    [(x_offset, y_offset), (x_offset + 240, y_offset + 180)],
                    outline=border_color,
                    width=3
                )

            # Draw value estimate text
            val = values[step_idx, env_idx].item()
            draw.text(
                (x_offset, y_offset + 160),
                f"Value Est={val:.2f}",
                fill=(0, 255, 0),
                font=font
            )

            # Draw force control indicators (X/Y/Z)
            force_x = force_control[step_idx, env_idx, 0].item() > 0.5
            force_y = force_control[step_idx, env_idx, 1].item() > 0.5
            force_z = force_control[step_idx, env_idx, 2].item() > 0.5

            draw.text(
                (x_offset + 20, y_offset + 135),
                "X",
                fill=(0, 255, 0) if force_x else (255, 0, 0),
                font=font
            )
            draw.text(
                (x_offset + 50, y_offset + 135),
                "Y",
                fill=(0, 255, 0) if force_y else (255, 0, 0),
                font=font
            )
            draw.text(
                (x_offset + 80, y_offset + 135),
                "Z",
                fill=(0, 255, 0) if force_z else (255, 0, 0),
                font=font
            )

        pil_frames.append(pil_img)

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )


def generate_videos(rollout_data: Dict[str, Any], checkpoint_dicts: List[Dict[str, str]],
                   num_envs_per_agent: int, policy_hz: float, args: argparse.Namespace) -> List[str]:
    """
    Generate 3x4 grid GIFs for each checkpoint.

    Args:
        rollout_data: Dictionary with collected rollout data including images
        checkpoint_dicts: List of checkpoint dicts with vid_path
        num_envs_per_agent: Number of environments per agent
        policy_hz: Policy frequency for GIF frame rate
        args: Command-line arguments

    Returns:
        List of video file paths
    """
    print(f"  Generating videos for {len(checkpoint_dicts)} checkpoint(s)...")

    # Verify images are available
    if rollout_data["images"] is None:
        raise RuntimeError("Cannot generate videos: rollout_data['images'] is None")

    if len(rollout_data["images"]) == 0:
        raise RuntimeError(
            "Cannot generate videos: No images were captured during rollout. "
            "This likely means the camera was not properly configured in the scene. "
            "Check that the environment has a 'tiled_camera' in the scene."
        )

    # Calculate frame duration in milliseconds
    duration = int(1000 / policy_hz)
    print(f"    Frame duration: {duration}ms (policy_hz: {policy_hz})")

    # Try to load font with fallback
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        print("    Using DejaVuSans.ttf font")
    except:
        font = ImageFont.load_default()
        print("    Using default font (DejaVuSans.ttf not found)")

    video_paths = []

    # Generate one video per checkpoint
    for i, ckpt_dict in enumerate(checkpoint_dicts):
        print(f"    Processing checkpoint {i+1}/{len(checkpoint_dicts)}: {ckpt_dict['ckpt_path']}")

        # Calculate environment slice for this checkpoint
        start_env = i * num_envs_per_agent
        end_env = (i + 1) * num_envs_per_agent

        # Slice data for this checkpoint's environments
        agent_images = rollout_data["images"][:, start_env:end_env]  # [steps, num_envs_per_agent, 180, 240, 3]
        agent_values = rollout_data["values"][:, start_env:end_env]  # [steps, num_envs_per_agent, 1] or [steps, num_envs_per_agent]
        agent_returns = rollout_data["total_returns"][start_env:end_env]  # [num_envs_per_agent]
        agent_success_step = rollout_data["success_step"][start_env:end_env]  # [num_envs_per_agent]
        agent_termination_step = rollout_data["termination_step"][start_env:end_env]  # [num_envs_per_agent]
        agent_truncation_step = rollout_data["truncation_step"][start_env:end_env]  # [num_envs_per_agent]

        agent_engagement = rollout_data["engagement_history"][:, start_env:end_env]  # [steps, num_envs_per_agent]

        agent_force_control = rollout_data["force_control"][:, start_env:end_env]  # [steps, num_envs_per_agent, 3]

        # Squeeze values if needed (remove trailing dimension if present)
        if agent_values.dim() == 3 and agent_values.shape[2] == 1:
            agent_values = agent_values.squeeze(2)  # [steps, num_envs_per_agent]

        # Select 12 environments based on returns (worst 4, middle 4, best 4)
        sorted_indices = torch.argsort(agent_returns)
        n = sorted_indices.shape[0]
        selected_indices = torch.cat([
            sorted_indices[-4:],              # best 4
            sorted_indices[n//2-2:n//2+2],   # middle 4
            sorted_indices[:4]              # worst 4
        ])

        # Slice data for selected 12 environments
        selected_images = agent_images[:, selected_indices]
        selected_values = agent_values[:, selected_indices]
        selected_success_step = agent_success_step[selected_indices]
        selected_termination_step = agent_termination_step[selected_indices]
        selected_truncation_step = agent_truncation_step[selected_indices]
        selected_engagement = agent_engagement[:, selected_indices]
        selected_force_control = agent_force_control[:, selected_indices]

        # Generate GIF
        output_path = ckpt_dict['vid_path']
        _create_checkpoint_gif(
            selected_images,
            selected_values,
            selected_success_step,
            selected_termination_step,
            selected_truncation_step,
            selected_engagement,
            selected_force_control,
            output_path,
            duration,
            font
        )

        video_paths.append(output_path)
        print(f"      Saved video to: {output_path}")

    print(f"  Successfully generated {len(video_paths)} videos")
    return video_paths


def log_to_wandb(checkpoint_dicts: List[Dict[str, str]], metrics_list: List[Dict[str, float]],
                 video_paths: List[str], args: argparse.Namespace) -> None:
    """
    Resume training runs and log metrics/videos to WandB.

    Args:
        checkpoint_dicts: List of checkpoint dicts with project, run_id
        metrics_list: List of metric dicts (one per checkpoint)
        video_paths: List of video file paths
        args: Command-line arguments
    """
    import re

    print(f"  Logging metrics for {len(checkpoint_dicts)} checkpoint(s) to WandB...")

    for i, ckpt_dict in enumerate(checkpoint_dicts):
        project = ckpt_dict['project']
        run_id = ckpt_dict['run_id']
        ckpt_path = ckpt_dict['ckpt_path']

        # Extract step number from checkpoint filename
        match = re.search(r'agent_\d+_(\d+)\.pt$', ckpt_path)
        if not match:
            raise RuntimeError(f"Failed to parse step number from checkpoint path (expected format: agent_{{agent_idx}}_{{step}}.pt): {ckpt_path}")

        step_num = int(match.group(1))
        exp_step = step_num // 256

        print(f"    Checkpoint {i+1}/{len(checkpoint_dicts)}: {run_id} at step {step_num}")

        try:
            # Resume WandB run
            wandb.init(project=project, id=run_id, resume="must")

            # Prepare metrics dict
            metrics = metrics_list[i].copy()
            metrics['env_steps'] = step_num
            metrics['total_steps'] = exp_step
            metrics['eval_seed'] = args.eval_seed

            # Add video if available
            video_uploaded = False
            if args.enable_video and i < len(video_paths):
                video_path = video_paths[i]
                if os.path.exists(video_path):
                    metrics['Eval/Checkpoint Videos'] = wandb.Video(video_path)
                    video_uploaded = True

            # Log to WandB
            wandb.log(metrics)

            # Print status
            if video_uploaded:
                print(f"      Logged metrics for {run_id} at checkpoint {step_num} and uploaded associated gif")
            else:
                print(f"      Logged metrics for {run_id} at checkpoint {step_num} and didn't upload associated gif")

        except Exception as e:
            # Ensure wandb.finish() is called before re-raising
            wandb.finish()
            raise RuntimeError(f"Failed to log metrics to WandB for {run_id} at step {step_num}: {e}")
        finally:
            # Always finish the run
            wandb.finish()

    print(f"  Successfully logged all {len(checkpoint_dicts)} checkpoint(s) to WandB")


def _shutdown_handler(signum, frame):
    """
    Signal handler for graceful shutdown with rollback.

    Writes current batch back to tracker if evaluation incomplete.
    """
    global _current_batch, _tracker_path

    print("\n[SHUTDOWN] Received shutdown signal")

    if _current_batch is not None and _tracker_path is not None:
        print(f"[SHUTDOWN] Writing {len(_current_batch)} incomplete checkpoint(s) back to tracker...")
        write_checkpoints_back_to_tracker(_tracker_path, _current_batch)
        print("[SHUTDOWN] Rollback complete")

    print("[SHUTDOWN] Exiting...")
    sys.exit(0)


def main(args):
    """
    Main daemon loop for checkpoint evaluation.

    Args:
        args: Parsed command-line arguments

    Flow:
        1. Setup signal handlers for graceful shutdown
        2. Setup environment once
        3. Create models/agent once
        4. Loop forever:
            a. Atomically read and remove tracker batch
            b. If no checkpoints: wait 2 min, continue
            c. Fetch break forces from WandB
            d. Load checkpoints into agent
            e. Update environment break forces
            f. Reset eval seed
            g. Run evaluation rollout
            h. Calculate metrics
            i. Generate videos (if enabled)
            j. Log to WandB
            k. Clear current batch (evaluation succeeded, no rollback needed)
    """
    global _current_batch, _tracker_path

    print("=" * 80)
    print("Block SimBa Evaluation Script")
    print("=" * 80)

    # Determine checkpoint loading mode
    if args.wandb_checkpoint:
        mode = "wandb"
        print("\n[MODE] WandB Download")
        print(f"  Checkpoint: {args.wandb_checkpoint}")
    elif args.local_checkpoint:
        mode = "local"
        print("\n[MODE] Local Checkpoint")
        print(f"  Checkpoint: {args.local_checkpoint}")
        print(f"  WandB run: {args.wandb_run}")
        # Validate required argument
        if not args.wandb_run:
            raise ValueError("--wandb_run is required when using --local_checkpoint")
    else:
        mode = "tracker"
        print("\n[MODE] Tracker Daemon")
        print(f"  Tracker: {args.ckpt_tracker_path}")
        # Validate required argument
        if not args.ckpt_tracker_path:
            raise ValueError("--ckpt_tracker_path is required when using tracker mode (default)")

    # Display common arguments
    print("\n[INFO] Configuration:")
    print(f"  Config: {args.config}")
    print(f"  Eval seed: {args.eval_seed}")
    print(f"  Video enabled: {args.enable_video}")
    print(f"  Upload checkpoint: {args.upload_ckpt}")

    # Step 1: Setup signal handlers for graceful shutdown (tracker mode only)
    if mode == "tracker":
        print("\n[STEP 1] Setting up signal handlers...")
        _tracker_path = args.ckpt_tracker_path
        signal.signal(signal.SIGINT, _shutdown_handler)
        signal.signal(signal.SIGTERM, _shutdown_handler)
        print("  Signal handlers registered (SIGINT, SIGTERM)")

    # Step 2: Setup environment once
    print("\n[STEP 2] Setting up environment (one-time)...")
    env, configs, max_rollout_steps, policy_hz = setup_environment_once(args.config, args)
    print(f"  Max rollout steps: {max_rollout_steps}")
    print(f"  Policy Hz: {policy_hz}")

    # Step 3: Create models/agent once
    print("\n[STEP 3] Creating models and agent (one-time)...")
    models, agent = create_models_and_agent(env, configs, args)
    print("  Models and agent creation completed")

    # Determine num_agents from config
    print("  Extracting num_agents from config...")
    print(f"  Primary config: break_forces={configs['primary'].break_forces}, agents_per_break_force={configs['primary'].agents_per_break_force}")
    num_agents = configs['primary'].total_agents
    print(f"  Number of agents: {num_agents}")

    # Validate num_envs_per_agent for video generation
    num_envs_per_agent = args.num_envs_per_agent or (configs['environment'].scene.num_envs // num_agents)
    if num_envs_per_agent < 12:
        raise ValueError(
            f"num_envs_per_agent must be >= 12 for video generation (need 12 environments for 3x4 grid). "
            f"Got {num_envs_per_agent}. Please increase --num_envs_per_agent or total num_envs in config."
        )
    print(f"  Environments per agent: {num_envs_per_agent}")

    # Step 4: Main evaluation loop (daemon for tracker mode, single-shot for others)
    if mode == "tracker":
        print("\n[STEP 4] Starting daemon loop...")
        print("=" * 80)
        run_once = False
    else:
        print("\n[STEP 4] Loading checkpoint for single evaluation...")
        print("=" * 80)
        run_once = True

    while True:
        # 4a. Load checkpoints based on mode
        if mode == "tracker":
            print("\n[4a] Reading and removing checkpoint batch from tracker...")
            checkpoint_dicts = read_and_remove_tracker_batch(args.ckpt_tracker_path, num_agents, args.batch_mode)

            # 4b. If no checkpoints: wait 2 min, continue
            if checkpoint_dicts is None or len(checkpoint_dicts) == 0:
                print("  No checkpoints available. Waiting 2 minutes...")
                for _ in tqdm.tqdm(range(120), desc="Waiting", file=sys.stdout):
                    time.sleep(1)
                continue

        elif mode == "wandb":
            print("\n[4a] Downloading checkpoint from WandB...")
            checkpoint_dicts = load_checkpoint_from_wandb(args.wandb_checkpoint)

        elif mode == "local":
            print("\n[4a] Loading local checkpoint...")
            checkpoint_dicts = load_local_checkpoint(args.local_checkpoint, args.wandb_run)

        print(f"  Found {len(checkpoint_dicts)} checkpoint(s) to evaluate")

        # Track current batch for rollback on failure (tracker mode only)
        if mode == "tracker":
            _current_batch = checkpoint_dicts

        try:
            # 4c. Fetch break forces from WandB
            print("\n[4c] Fetching break forces from WandB...")
            break_forces = fetch_break_forces_from_wandb(checkpoint_dicts)
            print(f"  Break forces: {break_forces}")
            
            # 4d. Load checkpoints into agent
            print("\n[4d] Loading checkpoints into agent...")
            load_checkpoints_into_agent(agent, checkpoint_dicts)
            print("  Checkpoints loaded successfully")

            # Debug: Verify model state after loading
            print("  DEBUG: Post-checkpoint loading state:")
            print(f"    Policy model in eval mode: {not agent.models['policy'].training}")
            print(f"    Value model in eval mode: {not agent.models['value'].training}")
            # Sample a weight to verify loading
            sample_weight = agent.models['policy'].actor_mean.resblocks[0].fc1.weight.data.flatten()[0].item()
            print(f"    Sample policy weight: {sample_weight:.6f}")

            # 4e. Update environment break forces
            print("\n[4e] Updating environment break forces...")
            update_environment_break_forces(env, break_forces, num_envs_per_agent)
            print("  Break forces updated")

            # 4f. Reset eval seed
            print("\n[4f] Resetting evaluation seed...")
            reset_evaluation_seed(args.eval_seed)
            print(f"  Seed set to: {args.eval_seed}")

            # 4g. Run evaluation rollout
            print("\n[4g] Running evaluation rollout...")
            rollout_data = run_evaluation_rollout(env, agent, max_rollout_steps, args)
            print("  Rollout complete")

            # 4h. Calculate metrics (per agent)
            print("\n[4h] Calculating metrics...")
            metrics_list = calculate_metrics(rollout_data, env, agent, args)
            print(f"  Calculated metrics for {len(metrics_list)} agents")

            # 4i. Generate videos (if enabled)
            video_paths = []
            if args.enable_video:
                print("\n[4i] Generating videos...")
                video_paths = generate_videos(rollout_data, checkpoint_dicts, num_envs_per_agent, policy_hz, args)
                print(f"  Generated {len(video_paths)} videos")
            else:
                print("\n[4i] Video generation disabled, skipping...")

            # Free GPU memory
            del rollout_data

            # 4j. Log to WandB
            print("\n[4j] Logging to WandB...")
            log_to_wandb(checkpoint_dicts, metrics_list, video_paths, args)
            print("  Logged to WandB successfully")

            # 4k. Upload checkpoints to WandB (if enabled, modes 1 & 3 only)
            if args.upload_ckpt and mode != "wandb":
                print("\n[4k] Uploading checkpoints to WandB...")
                upload_checkpoints_to_wandb(checkpoint_dicts)
                print("  Checkpoints uploaded successfully")
            elif args.upload_ckpt and mode == "wandb":
                print("\n[4k] Skipping checkpoint upload (already in WandB)")
            else:
                print("\n[4k] Checkpoint upload disabled")

            # 4l. Clear current batch (evaluation succeeded, no rollback needed)
            if mode == "tracker":
                print("\n[4l] Evaluation successful, clearing current batch...")
                _current_batch = None
                print("  Batch cleared")

            print("\n" + "=" * 80)
            if run_once:
                print("Evaluation complete. Exiting...")
                print("=" * 80)
                break  # Exit loop for single-shot modes
            else:
                print("Batch evaluation complete. Continuing to next batch...")
                print("=" * 80)

        except Exception as e:
            print(f"\n[ERROR] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

            # Rollback only for tracker mode
            if mode == "tracker":
                print(f"[ERROR] Writing {len(checkpoint_dicts)} checkpoint(s) back to tracker...")
                write_checkpoints_back_to_tracker(args.ckpt_tracker_path, checkpoint_dicts)
                print("[ERROR] Rollback complete")

            print("[ERROR] Stopping...")
            raise


if __name__ == "__main__":
    try:
        main(args_cli)
    except Exception as e:
        print(f"\n[FATAL ERROR] Script failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always close simulation app
        simulation_app.close()
