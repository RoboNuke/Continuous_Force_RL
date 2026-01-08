#!/usr/bin/env python3
"""
WandB Tag-Based Evaluation Script

Discovers and evaluates checkpoints using WandB tags.
Queries runs by tag, downloads checkpoints, runs evaluation, and logs results.
"""

import argparse
import sys

# Parse arguments BEFORE importing anything else (AppLauncher needs this)
def parse_arguments():
    """
    Parse command-line arguments for WandB evaluation script.

    Returns:
        Tuple of (args, hydra_args): Parsed arguments and remaining Hydra arguments
    """
    # Import AppLauncher here to avoid circular dependencies
    try:
        from isaaclab.app import AppLauncher
    except:
        from omni.isaac.lab.app import AppLauncher

    parser = argparse.ArgumentParser(description="WandB Tag-Based Checkpoint Evaluation")

    # Required arguments
    parser.add_argument("--tag", type=str, required=True,
                        help="Experiment tag to query (format: group_name:YYYY-MM-DD_HH:MM)")

    # Optional arguments
    parser.add_argument("--entity", type=str, default="hur",
                        help="WandB entity (username or team name)")
    parser.add_argument("--project", type=str, default="Peg_in_Hole",
                        help="WandB project name")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Specific run ID to evaluate (if not provided, evaluates all runs with tag)")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Specific checkpoint step to evaluate (e.g., 10000)")
    parser.add_argument("--checkpoint_range", type=str, default=None,
                        help="Checkpoint range to evaluate (format: start:end:step, e.g., 5000:50000:5000)")
    parser.add_argument("--eval_seed", type=int, default=42,
                        help="Fixed seed for consistent evaluation")
    parser.add_argument("--enable_video", action="store_true", default=False,
                        help="Enable video generation")
    parser.add_argument("--show_progress", action="store_true", default=True,
                        help="Show progress bar during evaluation rollout")
    parser.add_argument("--eval_mode", type=str, required=True,
                        choices=["performance", "noise", "rotation", "gain", "dynamics", "trajectory"],
                        help="Evaluation mode: 'performance' (100 envs, videos), 'noise' (500 envs, noise analysis), 'rotation' (500 envs, rotation analysis), 'gain' (500 envs, gain robustness), 'dynamics' (1200 envs, friction/mass robustness), or 'trajectory' (100 envs, detailed trajectory data)")
    parser.add_argument("--no_wandb", action="store_true", default=False,
                        help="Disable WandB logging and save results locally for debugging")
    parser.add_argument("--debug_project", action="store_true", default=False,
                        help="Log results to 'debug' project instead of original run (for testing)")
    parser.add_argument("--report_to_base_run", action="store_true", default=False,
                        help="Log results to the original training run instead of creating a new eval run")
    parser.add_argument("--traj_output_dir", type=str, default="./traj_eval_results",
                        help="Output directory for trajectory evaluation .pkl files (only used with --eval_mode trajectory)")
    parser.add_argument("--override", action="append",
                        help="Override config values (e.g., --override environment.task.asset_variant=hex_short_small)")

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
import re
import tempfile
import shutil
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# Enable camera view lighting mode (camera headlight)
import carb
carb_settings = carb.settings.get_settings()
carb_settings.set_bool("/rtx/useViewLightingMode", True)
print("Enabled camera view lighting mode (/rtx/useViewLightingMode)")



import torch
import numpy as np
import gymnasium as gym
import tqdm
import wandb

from skrl.utils import set_seed
from configs.config_manager_v3 import ConfigManagerV3
import learning.launch_utils_v3 as lUtils

# Wrappers
from wrappers.mechanics.fragile_object_wrapper import FragileObjectWrapper
from wrappers.mechanics.force_reward_wrapper import ForceRewardWrapper
from wrappers.mechanics.efficient_reset_wrapper import EfficientResetWrapper
from wrappers.sensors.force_torque_wrapper import ForceTorqueWrapper
from wrappers.observations.observation_manager_wrapper import ObservationManagerWrapper
from wrappers.control.hybrid_force_position_wrapper import HybridForcePositionWrapper

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

# Import factory environment class
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

# Import camera and sensor utils
try:
    from isaaclab.sensors.camera import TiledCameraCfg
    import isaaclab.sim as sim_utils
except ImportError:
    from omni.isaac.lab.sensors.camera import TiledCameraCfg
    import omni.isaac.lab.sim as sim_utils

print("All modules imported successfully")


# ===== NOISE EVALUATION CONSTANTS =====

# Define noise ranges for noise robustness evaluation
# Format: (min_val_meters, max_val_meters, display_name)
NOISE_RANGES = [
    (0.0, 0.001, "0mm-1mm"),        # 0-1mm
    (0.001, 0.0025, "1mm-2.5mm"),   # 1-2.5mm
    (0.0025, 0.005, "2.5mm-5mm"),   # 2.5-5mm
    (0.005, 0.0075, "5mm-7.5mm"),   # 5-7.5mm
    #(0.0075, 0.01, "7.5mm-10mm")    # 7.5-10mm
]

# Number of environments per noise range
ENVS_PER_NOISE_RANGE = 100

# Total environments for noise mode
NOISE_MODE_TOTAL_ENVS = len(NOISE_RANGES) * ENVS_PER_NOISE_RANGE


# ===== ROTATION EVALUATION CONSTANTS =====

# Define rotation angles for rotation robustness evaluation (in degrees)
ROTATION_ANGLES_DEG = [1, 2, 3, 4, 5]

# Number of environments per rotation angle
ENVS_PER_ROTATION_ANGLE = 100

# Total environments for rotation mode
ROTATION_MODE_TOTAL_ENVS = len(ROTATION_ANGLES_DEG) * ENVS_PER_ROTATION_ANGLE


# ===== GAIN EVALUATION CONSTANTS =====

# Define gain multipliers for gain robustness evaluation
# These multiply the default proportional gains [100, 100, 100, 30, 30, 30]
GAIN_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]

# Number of environments per gain multiplier
ENVS_PER_GAIN_MULTIPLIER = 100

# Total environments for gain mode
GAIN_MODE_TOTAL_ENVS = len(GAIN_MULTIPLIERS) * ENVS_PER_GAIN_MULTIPLIER


# ===== DYNAMICS EVALUATION CONSTANTS =====

# Define friction multipliers for dynamics robustness evaluation
DYNAMICS_FRICTION_MULTIPLIERS = [0.5, 0.75, 1.0]

# Define mass multipliers for dynamics robustness evaluation
DYNAMICS_MASS_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]

# Number of environments per friction/mass combination
ENVS_PER_DYNAMICS_COMBO = 100

# Total environments for dynamics mode (friction-major ordering)
# 3 friction × 4 mass × 100 envs = 1200 total
DYNAMICS_MODE_TOTAL_ENVS = len(DYNAMICS_FRICTION_MULTIPLIERS) * len(DYNAMICS_MASS_MULTIPLIERS) * ENVS_PER_DYNAMICS_COMBO


# ===== TRAJECTORY EVALUATION CONSTANTS =====

# Import trajectory evaluation module
from eval.traj_eval import (
    TrajectoryEvalWrapper,
    run_trajectory_evaluation,
    save_trajectory_data,
    TRAJECTORY_MODE_TOTAL_ENVS,
)


# ===== PARALLEL MODE DETECTION =====

def is_parallel_mode(args: argparse.Namespace) -> bool:
    """
    Determine if parallel evaluation mode should be used.

    Returns True for ALL cases EXCEPT performance mode with video enabled,
    which has memory constraints due to video frame capture for multiple agents.

    Args:
        args: Command-line arguments namespace

    Returns:
        True if parallel mode should be used, False for sequential mode
    """
    if args.eval_mode == "performance" and args.enable_video:
        return False
    return True


# ===== METRIC CATEGORIES =====

# Core metrics (10 total)
CORE_METRICS = [
    'total_episodes',
    'num_successful_completions',
    'num_breaks',
    'num_failed_timeouts',
    'episode_length',
    'ssv',
    'ssjv',
    'max_force',
    'avg_force_in_contact',
    'energy',
]

# Contact-control metrics (20 total)
CONTACT_METRICS = [
    # Per-axis raw counts (12)
    'force_control_contact_x',
    'force_control_no_contact_x',
    'pos_control_contact_x',
    'pos_control_no_contact_x',
    'force_control_contact_y',
    'force_control_no_contact_y',
    'pos_control_contact_y',
    'pos_control_no_contact_y',
    'force_control_contact_z',
    'force_control_no_contact_z',
    'pos_control_contact_z',
    'pos_control_no_contact_z',
    # Per-axis accuracy (6)
    'force_control_accuracy_x',
    'pos_control_accuracy_x',
    'force_control_accuracy_y',
    'pos_control_accuracy_y',
    'force_control_accuracy_z',
    'pos_control_accuracy_z',
    # Total accuracy (2)
    'force_control_accuracy_total',
    'pos_control_accuracy_total',
]


def prefix_metrics_by_category(
    metrics: Dict[str, float],
    core_prefix: str,
    contact_prefix: str
) -> Dict[str, float]:
    """
    Prefix metrics based on their category (core vs contact).

    Args:
        metrics: Dictionary of metric name -> value
        core_prefix: Prefix for core metrics (e.g., "Eval_Core")
        contact_prefix: Prefix for contact metrics (e.g., "Eval_Contact")

    Returns:
        Dictionary with prefixed metric names

    Raises:
        RuntimeError: If a metric is not in either category
    """
    prefixed = {}
    for key, value in metrics.items():
        if key in CORE_METRICS:
            prefixed[f"{core_prefix}/{key}"] = value
        elif key in CONTACT_METRICS:
            prefixed[f"{contact_prefix}/{key}"] = value
        else:
            raise RuntimeError(
                f"Metric '{key}' not found in CORE_METRICS or CONTACT_METRICS. "
                f"Please add it to the appropriate category list."
            )
    return prefixed


# ===== WRAPPERS =====

class Img2InfoWrapper(gym.Wrapper):
    """
    Wrapper to capture camera images and add them to info dict for video generation.
    """
    def __init__(self, env, key='tiled_camera'):
        super().__init__(env)
        self.cam_key = key

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


class FixedNoiseWrapper(gym.Wrapper):
    """
    Wrapper to override fixed asset position noise with predetermined values.
    Used for noise robustness evaluation.
    """
    def __init__(self, env, noise_assignments):
        """
        Initialize the fixed noise wrapper.

        Args:
            env: Base environment to wrap
            noise_assignments: [num_envs, 3] tensor with (x, y, z) noise values in meters for each environment
        """
        super().__init__(env)
        self.noise_assignments = noise_assignments
        self._init_wrapper()

    def _init_wrapper(self):
        if hasattr(self.unwrapped, '_reset_idx'):
            self._original_reset_idx = self.unwrapped._reset_idx
            self.unwrapped._reset_idx = self._wrapped_reset_idx

    def _wrapped_reset_idx(self, env_ids):
        """Reset environment and override noise with predetermined values."""
        # Let the base environment complete its reset
        self._original_reset_idx(env_ids)

        # Override the noise with our custom 3D values
        self.unwrapped.init_fixed_pos_obs_noise[env_ids, 0] = self.noise_assignments[env_ids, 0]
        self.unwrapped.init_fixed_pos_obs_noise[env_ids, 1] = self.noise_assignments[env_ids, 1]
        self.unwrapped.init_fixed_pos_obs_noise[env_ids, 2] = self.noise_assignments[env_ids, 2]

        # CRITICAL: Update fixed_pos_action_frame to maintain consistency
        # between what the policy sees and how its actions are interpreted
        self.unwrapped.fixed_pos_action_frame[env_ids] = (
            self.unwrapped.fixed_pos_obs_frame[env_ids] +
            self.unwrapped.init_fixed_pos_obs_noise[env_ids]
        )

        # also must update pos actions for initial ema
        pos_actions = self.unwrapped.fingertip_midpoint_pos - self.unwrapped.fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.unwrapped.cfg.ctrl.pos_action_bounds, device=self.unwrapped.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.unwrapped.actions[env_ids, 0:3] = self.unwrapped.prev_actions[env_ids, 0:3] = pos_actions[env_ids, 0:3]



class FixedAssetRotationWrapper(gym.Wrapper):
    """
    Wrapper to rotate the fixed asset by predetermined angles around random axes.
    Used for rotation robustness evaluation.
    """
    def __init__(self, env, rotation_assignments):
        """
        Initialize the fixed asset rotation wrapper.

        Args:
            env: Base environment to wrap
            rotation_assignments: [num_envs, 5] tensor with (angle_rad, axis_x, axis_y, axis_z, direction) for each environment
                - angle_rad: rotation angle in radians
                - axis_x, axis_y, axis_z: rotation axis (unit vector in xy-plane, z=0)
                - direction: +1 or -1 for rotation direction
        """
        super().__init__(env)
        self.rotation_assignments = rotation_assignments

        # Import torch utils for quaternion operations
        import isaacsim.core.utils.torch as torch_utils
        self.torch_utils = torch_utils

    def reset(self, **kwargs):
        """Reset environment and apply rotation to fixed asset."""
        # Let the base environment complete its reset
        observations, info = super().reset(**kwargs)

        # Get current fixed asset state
        fixed_asset = self.unwrapped._fixed_asset
        current_state = fixed_asset.data.root_state_w.clone()

        # Extract rotation parameters for each environment
        angles = self.rotation_assignments[:, 0] * self.rotation_assignments[:, 4]  # angle * direction
        axes = self.rotation_assignments[:, 1:4]  # (x, y, z) - z should be 0

        # Create rotation quaternion from angle-axis
        rotation_quat = self.torch_utils.quat_from_angle_axis(angles, axes)

        # Compose rotation with current orientation
        current_quat = current_state[:, 3:7]
        new_quat = self.torch_utils.quat_mul(rotation_quat, current_quat)

        # Update the state with new orientation
        current_state[:, 3:7] = new_quat

        # Write the rotated pose to simulation
        env_ids = torch.arange(self.unwrapped.num_envs, device=self.unwrapped.device)
        fixed_asset.write_root_pose_to_sim(current_state[:, 0:7], env_ids=env_ids)
        fixed_asset.write_root_velocity_to_sim(current_state[:, 7:13], env_ids=env_ids)

        # Reset the asset to apply changes
        fixed_asset.reset()

        # The observations don't need to be recomputed - the agent is unaware of rotation
        return observations, info


class FixedGainWrapper(gym.Wrapper):
    """
    Wrapper to apply fixed gain multipliers to controller gains.
    Used for gain robustness evaluation.

    Multiplies the default proportional gains by a predetermined multiplier,
    then computes derivative gains via critical damping: kd = 2 * sqrt(kp).
    All 6 axes (linear xyz + rotational xyz) receive the same multiplier.
    """
    def __init__(self, env, gain_multipliers: torch.Tensor):
        """
        Initialize the fixed gain wrapper.

        Args:
            env: Base environment to wrap
            gain_multipliers: [num_envs] tensor with gain multiplier for each environment
        """
        super().__init__(env)
        self.gain_multipliers = gain_multipliers  # [num_envs]
        # CRITICAL: Store a CLONE of default_gains because the base env's _set_gains
        # creates an alias between task_prop_gains and default_gains. Without cloning,
        # our modifications to task_prop_gains would corrupt default_gains.
        self.original_default_gains = self.unwrapped.default_gains.clone()
        self._init_wrapper()

    def _init_wrapper(self):
        """Monkey-patch _reset_idx to apply custom gains after reset."""
        if hasattr(self.unwrapped, '_reset_idx'):
            self._original_reset_idx = self.unwrapped._reset_idx
            self.unwrapped._reset_idx = self._wrapped_reset_idx
        else:
            raise RuntimeError(
                "FixedGainWrapper requires unwrapped environment to have _reset_idx method. "
                "Ensure the base environment is a FactoryEnv or compatible class."
            )

    def _wrapped_reset_idx(self, env_ids):
        """Reset environment and override gains with predetermined multipliers."""
        # Let the base environment complete its reset (which sets default gains)
        self._original_reset_idx(env_ids)

        # Compute scaled proportional gains: original_default_gains * multiplier
        # gain_multipliers is [num_envs], original_default_gains is [num_envs, 6]
        # Expand multipliers to [len(env_ids), 1] for broadcasting
        multipliers = self.gain_multipliers[env_ids].unsqueeze(1)  # [len(env_ids), 1]

        # Get the ORIGINAL default gains for these environments (not the corrupted ones)
        default_gains = self.original_default_gains[env_ids]  # [len(env_ids), 6]

        # Apply multiplier to all 6 axes equally
        scaled_prop_gains = default_gains * multipliers  # [len(env_ids), 6]

        # Set proportional gains
        self.unwrapped.task_prop_gains[env_ids] = scaled_prop_gains.to(
            dtype=self.unwrapped.task_prop_gains.dtype
        )

        # Compute derivative gains via critical damping: kd = 2 * sqrt(kp)
        deriv_gains = 2.0 * torch.sqrt(scaled_prop_gains)

        # Set derivative gains
        self.unwrapped.task_deriv_gains[env_ids] = deriv_gains.to(
            dtype=self.unwrapped.task_deriv_gains.dtype
        )


class FixedDynamicsWrapper(gym.Wrapper):
    """
    Wrapper to apply fixed friction and mass multipliers to the held object.
    Used for dynamics robustness evaluation.

    Multiplies the held object's friction and mass by predetermined multipliers.
    Inertia is scaled linearly with mass (assuming uniform density).
    Applied BEFORE reset to ensure PhysX has correct values during simulation setup.
    """
    def __init__(self, env, friction_multipliers: torch.Tensor, mass_multipliers: torch.Tensor):
        """
        Initialize the fixed dynamics wrapper.

        Args:
            env: Base environment to wrap
            friction_multipliers: [num_envs] tensor with friction multiplier for each environment
            mass_multipliers: [num_envs] tensor with mass multiplier for each environment
        """
        super().__init__(env)
        self.friction_multipliers = friction_multipliers  # [num_envs]
        self.mass_multipliers = mass_multipliers          # [num_envs]

        # Store original values (clone to prevent aliasing issues)
        held_asset = self.unwrapped._held_asset

        # Friction: from material properties [num_envs, num_shapes, 3] where 3 = [static, dynamic, restitution]
        materials = held_asset.root_physx_view.get_material_properties()
        self.original_static_friction = materials[..., 0].clone()  # [num_envs, num_shapes]
        self.materials_device = materials.device  # PhysX uses CPU

        # Mass/inertia: from default data
        self.original_mass = held_asset.data.default_mass.clone()  # [num_envs, num_bodies]
        self.original_inertia = held_asset.data.default_inertia.clone()  # [num_envs, num_bodies, 9]

        self._init_wrapper()

    def _init_wrapper(self):
        """Monkey-patch _reset_idx to apply custom dynamics before reset."""
        if hasattr(self.unwrapped, '_reset_idx'):
            self._original_reset_idx = self.unwrapped._reset_idx
            self.unwrapped._reset_idx = self._wrapped_reset_idx
        else:
            raise RuntimeError(
                "FixedDynamicsWrapper requires unwrapped environment to have _reset_idx method. "
                "Ensure the base environment is a FactoryEnv or compatible class."
            )

    def _wrapped_reset_idx(self, env_ids):
        """Apply dynamics changes BEFORE reset, then perform reset."""
        # Apply friction/mass before reset
        self._apply_dynamics(env_ids)

        # Let the base environment complete its reset
        self._original_reset_idx(env_ids)

    def _apply_dynamics(self, env_ids):
        """Apply friction and mass multipliers to the held asset."""
        held_asset = self.unwrapped._held_asset

        # Move env_ids to materials device (PhysX is on CPU)
        env_ids_cpu = env_ids.to(self.materials_device)

        # 1. Apply friction
        materials = held_asset.root_physx_view.get_material_properties()
        fric_mults = self.friction_multipliers[env_ids].to(self.materials_device).unsqueeze(-1)  # [len(env_ids), 1]
        orig_fric = self.original_static_friction[env_ids_cpu]  # [len(env_ids), num_shapes]
        new_fric = orig_fric * fric_mults
        materials[env_ids_cpu, ..., 0] = new_fric  # Static friction
        materials[env_ids_cpu, ..., 1] = new_fric  # Dynamic friction
        held_asset.root_physx_view.set_material_properties(materials, env_ids_cpu)

        # 2. Apply mass with inertia recomputation
        masses = held_asset.root_physx_view.get_masses()
        inertias = held_asset.root_physx_view.get_inertias()
        mass_mults = self.mass_multipliers[env_ids].to(masses.device)

        new_masses = self.original_mass[env_ids_cpu] * mass_mults.unsqueeze(-1)
        masses[env_ids_cpu] = new_masses

        new_inertias = self.original_inertia[env_ids_cpu] * mass_mults.unsqueeze(-1).unsqueeze(-1)
        inertias[env_ids_cpu] = new_inertias

        held_asset.root_physx_view.set_masses(masses, env_ids_cpu)
        held_asset.root_physx_view.set_inertias(inertias, env_ids_cpu)


# ===== WANDB QUERY FUNCTIONS =====

def query_runs_by_tag(tag: str, entity: str, project: str, run_id: Optional[str] = None) -> List[wandb.Run]:
    """
    Query WandB for runs with specified tag.

    Args:
        tag: Experiment tag (format: group_name:YYYY-MM-DD_HH:MM)
        entity: WandB entity (username or team name)
        project: WandB project name
        run_id: Optional specific run ID to filter by

    Returns:
        List of WandB runs

    Raises:
        RuntimeError: If no runs found or API query fails
    """
    print(f"Querying WandB for runs with tag: {tag}")
    print(f"  Entity: {entity}")
    print(f"  Project: {project}")

    api = wandb.Api(timeout=60)
    max_retries = 5
    retry_delay = 2.0

    for attempt in range(max_retries):
        try:
            # Query all runs with the tag in the specified project
            project_path = f"{entity}/{project}"
            runs = api.runs(project_path, filters={"tags": {"$in": [tag]}})

            # Convert to list to actually execute query
            runs_list = list(runs)

            if len(runs_list) == 0:
                raise RuntimeError(f"No runs found with tag: {tag}")

            # Filter out evaluation runs (those with "Eval_" in the name)
            original_count = len(runs_list)
            runs_list = [r for r in runs_list if "Eval_" not in r.name]
            if original_count != len(runs_list):
                print(f"  Filtered out {original_count - len(runs_list)} evaluation run(s)")

            if len(runs_list) == 0:
                raise RuntimeError(f"No training runs found with tag: {tag} (all runs were evaluation runs)")

            # Filter by run_id if specified
            if run_id is not None:
                runs_list = [r for r in runs_list if r.id == run_id]
                if len(runs_list) == 0:
                    raise RuntimeError(f"No runs found with tag '{tag}' and run_id '{run_id}'")

            print(f"  Found {len(runs_list)} run(s) with tag '{tag}'")
            for r in runs_list:
                print(f"    - {r.project}/{r.id} ({r.name})")

            return runs_list

        except wandb.errors.CommError as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < max_retries - 1:
                    print(f"    Rate limit hit, waiting {retry_delay:.1f}s before retry {attempt+1}/{max_retries-1}...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise RuntimeError(
                        f"Failed to query runs after {max_retries} attempts due to rate limiting. "
                        f"Please wait a few minutes before running again."
                    )
            else:
                raise RuntimeError(f"Failed to query WandB runs: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to query WandB runs: {e}")

    raise RuntimeError(f"Failed to query WandB runs after all retries")


# ===== CONFIG RECONSTRUCTION =====

def reconstruct_config_from_wandb(run: wandb.Run) -> Dict[str, Any]:
    """Reconstruct configuration from WandB run using config files.

    Downloads config YAML files and uses ConfigManagerV3 to load them.

    Args:
        run: WandB run object

    Returns:
        Dictionary with config instances compatible with setup_environment_once

    Raises:
        RuntimeError: If config files missing or loading fails
    """
    print(f"Reconstructing config from WandB run: {run.project}/{run.id}")

    # Use ConfigManagerV3's new config_from_wandb method
    config_manager = ConfigManagerV3()
    configs = config_manager.config_from_wandb(run)

    print(f"  Successfully reconstructed {len(configs)} config sections")
    return configs


# ===== CHECKPOINT DISCOVERY =====

def get_checkpoint_steps(run: wandb.Run, args: argparse.Namespace) -> List[int]:
    """
    Determine which checkpoint steps to evaluate.

    Args:
        run: WandB run object
        args: Command-line arguments

    Returns:
        List of checkpoint step numbers

    Raises:
        RuntimeError: If no checkpoints found or invalid range specified
    """
    print(f"Determining checkpoint steps for run {run.id}...")

    # If specific checkpoint specified
    if args.checkpoint is not None:
        print(f"  Using specific checkpoint: {args.checkpoint}")
        return [args.checkpoint]

    # If checkpoint range specified
    if args.checkpoint_range is not None:
        try:
            parts = args.checkpoint_range.split(':')
            if len(parts) != 3:
                raise ValueError(f"Invalid format: {args.checkpoint_range}")

            start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
            checkpoint_steps = list(range(start, end + 1, step))
            print(f"  Using checkpoint range: {start}:{end}:{step} ({len(checkpoint_steps)} checkpoints)")
            return checkpoint_steps
        except Exception as e:
            raise RuntimeError(f"Failed to parse checkpoint range '{args.checkpoint_range}': {e}")

    # Otherwise, query WandB for all available checkpoints
    print("  Querying WandB for all available checkpoints...")

    try:
        # List all files in the run
        files = run.files()

        # Filter for policy checkpoint files in ckpts/policies/
        policy_files = [f for f in files if f.name.startswith('ckpts/policies/') and f.name.endswith('.pt')]

        if len(policy_files) == 0:
            raise RuntimeError(f"No checkpoint files found in run {run.id}")

        # Extract step numbers from filenames
        checkpoint_steps = []
        for f in policy_files:
            # Extract step number from ckpts/policies/{step}.pt
            match = re.search(r'ckpts/policies/(\d+)\.pt$', f.name)
            if match:
                step_num = int(match.group(1))
                checkpoint_steps.append(step_num)

        checkpoint_steps.sort()

        if len(checkpoint_steps) == 0:
            raise RuntimeError(f"Could not extract step numbers from checkpoint files in run {run.id}")

        print(f"  Found {len(checkpoint_steps)} checkpoint(s): {checkpoint_steps}")
        return checkpoint_steps

    except Exception as e:
        raise RuntimeError(f"Failed to query checkpoints from WandB run {run.id}: {e}")


def download_checkpoint_pair(run: wandb.Run, step: int) -> Tuple[str, str]:
    """
    Download policy and critic checkpoints from WandB.

    Args:
        run: WandB run object
        step: Checkpoint step number

    Returns:
        Tuple of (policy_path, critic_path)

    Raises:
        RuntimeError: If download fails
    """
    print(f"  Downloading checkpoint pair for step {step}...")

    # Construct file paths on WandB
    policy_filename = f"ckpts/policies/{step}.pt"
    critic_filename = f"ckpts/critics/{step}.pt"

    # Create temporary download directory
    download_dir = tempfile.mkdtemp(prefix="wandb_ckpt_")

    try:
        # Download policy checkpoint
        policy_file = run.file(policy_filename)
        policy_path = policy_file.download(root=download_dir, replace=True).name
        print(f"    Downloaded policy: {policy_path}")

        # Download critic checkpoint
        critic_file = run.file(critic_filename)
        critic_path = critic_file.download(root=download_dir, replace=True).name
        print(f"    Downloaded critic: {critic_path}")

        return policy_path, critic_path

    except Exception as e:
        # Clean up temp directory on failure
        shutil.rmtree(download_dir, ignore_errors=True)
        raise RuntimeError(
            f"Failed to download checkpoint files from WandB run {run.project}/{run.id} at step {step}: {e}"
        )


def load_checkpoints_parallel(
    runs: List[wandb.Run],
    step: int,
    env: Any,
    agent: Any,
) -> List[str]:
    """
    Download and load checkpoints from all runs into their respective agent slots.

    Args:
        runs: List of WandB run objects
        step: Checkpoint step number
        env: Environment instance
        agent: Agent instance with num_agents = len(runs)

    Returns:
        List of download directories (for cleanup after evaluation)

    Raises:
        RuntimeError: If any checkpoint fails to load (fail fast)
    """
    from models.SimBa import SimBaNet
    from models.block_simba import pack_agents_into_block
    from agents.block_ppo import PerAgentPreprocessorWrapper
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    print(f"  Loading {len(runs)} checkpoints in parallel for step {step}...")

    # Download all checkpoints first
    checkpoint_pairs = []
    download_dirs = []
    for run_idx, run in enumerate(runs):
        print(f"    Downloading checkpoint for run {run_idx}: {run.id}")
        policy_path, critic_path = download_checkpoint_pair(run, step)
        checkpoint_pairs.append((policy_path, critic_path))
        # Track download directory for cleanup
        download_dirs.append(os.path.dirname(os.path.dirname(policy_path)))

    # Initialize per-agent preprocessor lists if not already present
    if not hasattr(agent, '_per_agent_state_preprocessors'):
        agent._per_agent_state_preprocessors = [None] * agent.num_agents
    if not hasattr(agent, '_per_agent_value_preprocessors'):
        agent._per_agent_value_preprocessors = [None] * agent.num_agents

    # Load each checkpoint into its agent slot
    for agent_idx, (run, (policy_path, critic_path)) in enumerate(zip(runs, checkpoint_pairs)):
        print(f"    Loading run {run.id} into agent slot {agent_idx}...")

        # Load checkpoint files
        policy_checkpoint = torch.load(policy_path, map_location=env.unwrapped.device, weights_only=False)
        critic_checkpoint = torch.load(critic_path, map_location=env.unwrapped.device, weights_only=False)

        # Validate checkpoint contents
        if 'net_state_dict' not in policy_checkpoint:
            raise RuntimeError(f"Policy checkpoint for run {run.id} missing 'net_state_dict'")
        if 'net_state_dict' not in critic_checkpoint:
            raise RuntimeError(f"Critic checkpoint for run {run.id} missing 'net_state_dict'")
        if 'state_preprocessor' not in policy_checkpoint:
            raise RuntimeError(f"Policy checkpoint for run {run.id} missing 'state_preprocessor'")
        if 'value_preprocessor' not in critic_checkpoint:
            raise RuntimeError(f"Critic checkpoint for run {run.id} missing 'value_preprocessor'")
        if 'log_std' not in policy_checkpoint:
            raise RuntimeError(f"Policy checkpoint for run {run.id} missing 'log_std'")

        # Create single-agent SimBaNet for policy
        policy_agent = SimBaNet(
            n=len(agent.policy.actor_mean.resblocks),
            in_size=agent.policy.actor_mean.obs_dim,
            out_size=agent.policy.actor_mean.act_dim,
            latent_size=agent.policy.actor_mean.hidden_dim,
            device=agent.device,
            tan_out=agent.policy.actor_mean.use_tanh
        )
        policy_agent.load_state_dict(policy_checkpoint['net_state_dict'])

        # Create single-agent SimBaNet for critic
        critic_agent = SimBaNet(
            n=len(agent.value.critic.resblocks),
            in_size=agent.value.critic.obs_dim,
            out_size=agent.value.critic.act_dim,
            latent_size=agent.value.critic.hidden_dim,
            device=agent.device,
            tan_out=agent.value.critic.use_tanh
        )
        critic_agent.load_state_dict(critic_checkpoint['net_state_dict'])

        # Pack into block models at this agent's index
        pack_agents_into_block(agent.policy.actor_mean, {agent_idx: policy_agent})
        pack_agents_into_block(agent.value.critic, {agent_idx: critic_agent})

        # Load log_std
        agent.policy.actor_logstd[agent_idx].data.copy_(policy_checkpoint['log_std'].data)

        # Load state preprocessor for this agent
        obs_size = policy_checkpoint['state_preprocessor']['running_mean'].shape[0]
        if agent._per_agent_state_preprocessors[agent_idx] is None:
            agent._per_agent_state_preprocessors[agent_idx] = RunningStandardScaler(
                size=obs_size, device=agent.device
            )
        agent._per_agent_state_preprocessors[agent_idx].load_state_dict(
            policy_checkpoint['state_preprocessor']
        )

        # Load value preprocessor for this agent
        if agent._per_agent_value_preprocessors[agent_idx] is None:
            agent._per_agent_value_preprocessors[agent_idx] = RunningStandardScaler(
                size=1, device=agent.device
            )
        agent._per_agent_value_preprocessors[agent_idx].load_state_dict(
            critic_checkpoint['value_preprocessor']
        )

        print(f"      Loaded run {run.id} -> agent {agent_idx}")

    # Wrap preprocessors for SKRL compatibility
    agent._state_preprocessor = PerAgentPreprocessorWrapper(agent, agent._per_agent_state_preprocessors)
    agent._value_preprocessor = PerAgentPreprocessorWrapper(agent, agent._per_agent_value_preprocessors)

    # Set agent to eval mode
    agent.set_running_mode("eval")
    print(f"  All {len(runs)} checkpoints loaded successfully!")

    return download_dirs


# ===== ENVIRONMENT SETUP =====

def setup_environment_once(
    configs: Dict[str, Any],
    args: argparse.Namespace,
    num_runs: int = 1,
    parallel_mode: bool = False
) -> Tuple[Any, Any, int, float, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """
    Create environment and agent once for all evaluations.

    Args:
        configs: Configuration dictionary from WandB
        args: Command-line arguments
        num_runs: Number of WandB runs to evaluate (used in parallel mode)
        parallel_mode: If True, create num_runs agents to evaluate all runs simultaneously

    Returns:
        Tuple of (env, agent, max_rollout_steps, policy_hz, noise_assignments,
                  rotation_assignments, gain_assignments, friction_assignments,
                  mass_assignments, envs_per_agent)
        - noise_assignments is None except for noise mode ([num_envs, 3] tensor)
        - rotation_assignments is None except for rotation mode ([num_envs, 5] tensor)
        - envs_per_agent: Number of environments per agent (for metrics splitting)

    Raises:
        RuntimeError: If environment or agent creation fails
    """
    print("Setting up environment and agent...")

    # Import camera config
    try:
        from isaaclab.sensors import TiledCameraCfg, CameraCfg
        import isaaclab.sim as sim_utils
    except:
        from omni.isaac.lab.sensors import TiledCameraCfg, CameraCfg
        import omni.isaac.lab.sim as sim_utils

    from memories.multi_random import MultiRandomMemory

    # TODO: The config structure needs to be properly reconstructed from WandB
    # For now, we assume configs has the same structure as from ConfigManagerV3
    # This may need adjustment based on actual WandB run.config structure

    # Configure agent count based on parallel mode
    original_break_forces = configs['primary'].break_forces
    original_agents_per_break_force = configs['primary'].agents_per_break_force

    if parallel_mode:
        # Parallel mode: one agent per run
        configs['primary'].agents_per_break_force = num_runs
        print(f"  PARALLEL MODE: {num_runs} agents (one per run)")
    else:
        # Sequential mode: force 1 agent
        configs['primary'].agents_per_break_force = 1
        print(f"  SEQUENTIAL MODE: 1 agent")

    # Use a single break force (all agents use same training conditions)
    if isinstance(original_break_forces, list) and len(original_break_forces) > 0:
        configs['primary'].break_forces = [original_break_forces[0]]
    else:
        configs['primary'].break_forces = [original_break_forces]

    print(f"    break_forces={configs['primary'].break_forces}, agents_per_break_force={configs['primary'].agents_per_break_force}")
    print(f"    (original: break_forces={original_break_forces}, agents_per_break_force={original_agents_per_break_force})")
    print(f"    total_agents property now returns: {configs['primary'].total_agents}")

    # Set num_envs based on eval mode
    if args.eval_mode == "performance":
        num_envs_per_agent = 100
    elif args.eval_mode == "noise":
        num_envs_per_agent = NOISE_MODE_TOTAL_ENVS
    elif args.eval_mode == "rotation":
        num_envs_per_agent = ROTATION_MODE_TOTAL_ENVS
    elif args.eval_mode == "gain":
        num_envs_per_agent = GAIN_MODE_TOTAL_ENVS
    elif args.eval_mode == "dynamics":
        num_envs_per_agent = DYNAMICS_MODE_TOTAL_ENVS
    elif args.eval_mode == "trajectory":
        num_envs_per_agent = TRAJECTORY_MODE_TOTAL_ENVS
    else:
        raise RuntimeError(f"Unknown eval_mode: {args.eval_mode}")

    total_agents = configs['primary'].total_agents
    total_envs = num_envs_per_agent * total_agents
    configs['environment'].scene.num_envs = total_envs
    print(f"  Set num_envs: {total_envs} ({num_envs_per_agent} per agent × {total_agents} agents) [mode: {args.eval_mode}]")

    # Handle seed configuration - use eval_seed from command line for deterministic evaluation
    eval_seed = args.eval_seed
    print(f"  Using eval_seed for deterministic evaluation: {eval_seed}")

    # Set environment variables for CUDA deterministic operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Set seed with deterministic mode for PyTorch
    set_seed(eval_seed, deterministic=True)
    configs['environment'].seed = eval_seed

    # Set render interval to decimation for faster execution
    configs['environment'].sim.render_interval = configs['environment'].decimation
    print(f"  Set render_interval to decimation: {configs['environment'].decimation}")

    # Calculate max_rollout_steps
    env_cfg = configs['environment']

    # Apply asset variant if specified in config (e.g., custom peg/hole from training)
    task_cfg = env_cfg.task
    if hasattr(task_cfg, 'apply_asset_variant_if_specified'):
        result = task_cfg.apply_asset_variant_if_specified()
        if result:
            print(f"  Applied asset variant: {task_cfg.asset_variant}")

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

    # Disable wrappers not needed for evaluation
    print("  Configuring wrappers for evaluation...")
    configs['wrappers'].wandb_logging.enabled = False
    configs['wrappers'].action_logging.enabled = False

    # Enable factory_metrics but disable WandB publishing (for smoothness data in info)
    configs['wrappers'].factory_metrics.enabled = True
    configs['wrappers'].factory_metrics.publish_to_wandb = False

    # Apply wrappers using launch_utils_v3
    print("  Applying evaluation wrappers...")
    env = lUtils.apply_wrappers(env, configs)#, eval_mode=args.eval_mode)

    # Generate noise assignments for noise mode
    noise_assignments = None
    if args.eval_mode == "noise":
        print("  Generating noise assignments for noise mode...")

        # Generate base noise assignments for one agent
        base_noise_assignments_list = []
        for min_val, max_val, range_name in NOISE_RANGES:
            # Generate 2D noise using polar coordinates in x-y plane
            # Sample uniformly in circular annulus [min_val, max_val]

            # Random angle in x-y plane [0, 2π]
            theta = torch.rand(ENVS_PER_NOISE_RANGE, device=env.device) * 2 * 3.14159265359

            # Random radii within the circular annulus [min_val, max_val]
            # We square the radii to ensure uniform distribution by area, since area scales with r²
            # This prevents bias toward the inner edge of the annulus
            radii_squared = torch.rand(ENVS_PER_NOISE_RANGE, device=env.device) * (max_val**2 - min_val**2) + min_val**2
            radii = torch.sqrt(radii_squared)

            # Convert polar to Cartesian coordinates (x-y plane only, z = 0)
            x_noise = radii * torch.cos(theta)
            y_noise = radii * torch.sin(theta)
            z_noise = torch.zeros(ENVS_PER_NOISE_RANGE, device=env.device)

            base_noise_assignments_list.append(torch.stack([x_noise, y_noise, z_noise], dim=1))

        base_noise_assignments = torch.cat(base_noise_assignments_list, dim=0)  # [NOISE_MODE_TOTAL_ENVS, 3]

        # Replicate for all agents in parallel mode
        if parallel_mode:
            noise_assignments = base_noise_assignments.repeat(num_runs, 1)
            print(f"    Generated {noise_assignments.shape[0]} noise assignments "
                  f"({len(NOISE_RANGES)} ranges × {ENVS_PER_NOISE_RANGE} envs × {num_runs} agents)")
        else:
            noise_assignments = base_noise_assignments
            print(f"    Generated {noise_assignments.shape[0]} noise assignments "
                  f"({len(NOISE_RANGES)} ranges × {ENVS_PER_NOISE_RANGE} envs)")

        # Apply FixedNoiseWrapper
        print("  - FixedNoiseWrapper for noise mode")
        env = FixedNoiseWrapper(env, noise_assignments)

    # Generate rotation assignments for rotation mode
    rotation_assignments = None
    if args.eval_mode == "rotation":
        print("  Generating rotation assignments for rotation mode...")

        # Generate base rotation assignments for one agent
        base_rotation_assignments_list = []
        for angle_deg in ROTATION_ANGLES_DEG:
            # Convert angle to radians
            angle_rad = torch.tensor(angle_deg * 3.14159265359 / 180.0, device=env.device)

            # Generate 100 environments for this angle
            for i in range(ENVS_PER_ROTATION_ANGLE):
                # Random angle in xy-plane [0, 2π]
                theta = torch.rand(1, device=env.device) * 2 * 3.14159265359

                # Unit vector in xy-plane
                axis_x = torch.cos(theta)
                axis_y = torch.sin(theta)
                axis_z = torch.zeros(1, device=env.device)

                # Random direction: +1 or -1
                direction = torch.tensor(1.0 if torch.rand(1).item() > 0.5 else -1.0, device=env.device)

                # Store: [angle_rad, axis_x, axis_y, axis_z, direction]
                # Note: axis_z is always 0 for xy-plane rotation
                base_rotation_assignments_list.append(
                    torch.tensor([angle_rad, axis_x.item(), axis_y.item(), 0.0, direction], device=env.device)
                )

        base_rotation_assignments = torch.stack(base_rotation_assignments_list, dim=0)  # [ROTATION_MODE_TOTAL_ENVS, 5]

        # Replicate for all agents in parallel mode
        if parallel_mode:
            rotation_assignments = base_rotation_assignments.repeat(num_runs, 1)
            print(f"    Generated {rotation_assignments.shape[0]} rotation assignments "
                  f"({len(ROTATION_ANGLES_DEG)} angles × {ENVS_PER_ROTATION_ANGLE} envs × {num_runs} agents)")
        else:
            rotation_assignments = base_rotation_assignments
            print(f"    Generated {rotation_assignments.shape[0]} rotation assignments "
                  f"({len(ROTATION_ANGLES_DEG)} angles × {ENVS_PER_ROTATION_ANGLE} envs)")

        # Apply FixedAssetRotationWrapper
        print("  - FixedAssetRotationWrapper for rotation mode")
        env = FixedAssetRotationWrapper(env, rotation_assignments)

    # Generate gain multiplier assignments for gain mode
    gain_assignments = None
    if args.eval_mode == "gain":
        print("  Generating gain multiplier assignments for gain mode...")

        # Generate base gain assignments for one agent
        base_gain_assignments_list = []
        for multiplier in GAIN_MULTIPLIERS:
            # Create tensor of same multiplier for ENVS_PER_GAIN_MULTIPLIER environments
            multiplier_tensor = torch.full(
                (ENVS_PER_GAIN_MULTIPLIER,),
                multiplier,
                device=env.device,
                dtype=torch.float32
            )
            base_gain_assignments_list.append(multiplier_tensor)

        base_gain_assignments = torch.cat(base_gain_assignments_list, dim=0)  # [GAIN_MODE_TOTAL_ENVS]

        # Replicate for all agents in parallel mode
        if parallel_mode:
            gain_assignments = base_gain_assignments.repeat(num_runs)
            print(f"    Generated {gain_assignments.shape[0]} gain assignments "
                  f"({len(GAIN_MULTIPLIERS)} multipliers × {ENVS_PER_GAIN_MULTIPLIER} envs × {num_runs} agents)")
        else:
            gain_assignments = base_gain_assignments
            print(f"    Generated {gain_assignments.shape[0]} gain assignments "
                  f"({len(GAIN_MULTIPLIERS)} multipliers × {ENVS_PER_GAIN_MULTIPLIER} envs)")

        # Apply FixedGainWrapper
        print("  - FixedGainWrapper for gain mode")
        env = FixedGainWrapper(env, gain_assignments)

    # Generate friction/mass assignments for dynamics mode
    friction_assignments = None
    mass_assignments = None
    if args.eval_mode == "dynamics":
        print("  Generating friction/mass assignments for dynamics mode...")

        # Generate base dynamics assignments for one agent
        # Friction-major ordering: for each friction, iterate all masses
        base_friction_list = []
        base_mass_list = []
        for fric_mult in DYNAMICS_FRICTION_MULTIPLIERS:
            for mass_mult in DYNAMICS_MASS_MULTIPLIERS:
                base_friction_list.extend([fric_mult] * ENVS_PER_DYNAMICS_COMBO)
                base_mass_list.extend([mass_mult] * ENVS_PER_DYNAMICS_COMBO)

        base_friction_assignments = torch.tensor(base_friction_list, device=env.device, dtype=torch.float32)
        base_mass_assignments = torch.tensor(base_mass_list, device=env.device, dtype=torch.float32)

        # Replicate for all agents in parallel mode
        if parallel_mode:
            friction_assignments = base_friction_assignments.repeat(num_runs)
            mass_assignments = base_mass_assignments.repeat(num_runs)
            print(f"    Generated {friction_assignments.shape[0]} dynamics assignments "
                  f"({len(DYNAMICS_FRICTION_MULTIPLIERS)} friction × {len(DYNAMICS_MASS_MULTIPLIERS)} mass × "
                  f"{ENVS_PER_DYNAMICS_COMBO} envs × {num_runs} agents)")
        else:
            friction_assignments = base_friction_assignments
            mass_assignments = base_mass_assignments
            print(f"    Generated {friction_assignments.shape[0]} dynamics assignments "
                  f"({len(DYNAMICS_FRICTION_MULTIPLIERS)} friction × {len(DYNAMICS_MASS_MULTIPLIERS)} mass × "
                  f"{ENVS_PER_DYNAMICS_COMBO} envs)")

        # Apply FixedDynamicsWrapper
        print("  - FixedDynamicsWrapper for dynamics mode")
        env = FixedDynamicsWrapper(env, friction_assignments, mass_assignments)

    # Apply TrajectoryEvalWrapper for trajectory mode
    if args.eval_mode == "trajectory":
        print("  - TrajectoryEvalWrapper for trajectory mode")
        env = TrajectoryEvalWrapper(env)

    # Add image capture wrapper for video (only in performance mode)
    if args.enable_video and args.eval_mode == "performance":
        print("  - Img2InfoWrapper for video capture")
        env = Img2InfoWrapper(env)

    # Apply AsyncCriticIsaacLabWrapper (flattens policy+critic observations)
    print("  - AsyncCriticIsaacLabWrapper")
    from wrappers.skrl.async_critic_isaaclab_wrapper import AsyncCriticIsaacLabWrapper
    env = AsyncCriticIsaacLabWrapper(env)
    print("  Environment setup complete")

    # Create models and agent
    print("  Creating block models and agent...")

    # Create models using launch_utils_v3
    print("    Creating models...")
    models = lUtils.create_policy_and_value_models(env, configs)
    print("    Models created")

    # Create memory (required for agent initialization, even for eval)
    print("    Creating memory...")
    memory = MultiRandomMemory(memory_size=16, num_envs=env.num_envs, device=env.device)
    print("    Memory created")

    # Disable checkpoint tracking for evaluation
    configs['agent'].track_ckpts = False

    # Create agent using launch_utils_v3
    print("    Creating BlockPPO agent...")
    agent = lUtils.create_block_ppo_agents(env, configs, models, memory)
    print("    Agent created")

    print(f"  Created BlockPPO agent with {configs['primary'].total_agents} agents")
    print("  Setup complete!")

    return env, agent, max_rollout_steps, policy_hz, noise_assignments, rotation_assignments, gain_assignments, friction_assignments, mass_assignments, num_envs_per_agent


# ===== WANDB LOGGING =====

def log_results_to_wandb(run: wandb.Run, step: int, metrics: Dict[str, float],
                        media_path: Optional[str], args: argparse.Namespace) -> None:
    """
    Log evaluation results back to the WandB run, or save locally if --no_wandb is set.

    Args:
        run: WandB run object
        step: Checkpoint step number
        metrics: Dictionary of evaluation metrics
        media_path: Path to generated media (video for performance mode, image for noise mode)
        args: Command-line arguments

    Raises:
        RuntimeError: If logging fails
    """
    # Add eval_seed and step metrics
    metrics_to_log = metrics.copy()
    metrics_to_log['eval_seed'] = args.eval_seed
    metrics_to_log['total_steps'] = step
    metrics_to_log['env_steps'] = step / 256

    if args.no_wandb:
        # Debug mode: Print metrics and save media locally
        print(f"\n{'=' * 80}")
        print(f"EVALUATION METRICS FOR STEP {step} (--no_wandb mode)")
        print(f"{'=' * 80}")
        for key, value in sorted(metrics_to_log.items()):
            print(f"  {key}: {value}")
        print(f"{'=' * 80}\n")

        # Save media locally with descriptive filename
        if media_path is not None and os.path.exists(media_path):
            import shutil
            # Create eval_results directory if it doesn't exist
            os.makedirs("eval_results", exist_ok=True)

            # Determine file extension from media_path
            _, ext = os.path.splitext(media_path)

            # Create descriptive filename
            if args.eval_mode == "performance":
                local_media_path = f"./eval_results/performance_step_{step}{ext}"
            elif args.eval_mode == "noise":
                local_media_path = f"./eval_results/noise_step_{step}{ext}"
            elif args.eval_mode == "rotation":
                local_media_path = f"./eval_results/rotation_step_{step}{ext}"
            elif args.eval_mode == "gain":
                local_media_path = f"./eval_results/gain_step_{step}{ext}"
            elif args.eval_mode == "dynamics":
                local_media_path = f"./eval_results/dynamics_step_{step}{ext}"
            elif args.eval_mode == "trajectory":
                os.makedirs(f"./eval_results/{run.id}", exist_ok=True)
                local_media_path = f"./eval_results/{run.id}/traj_step_{step}{ext}"
            else:
                local_media_path = f"./eval_results/eval_step_{step}{ext}"

            shutil.copy(media_path, local_media_path)
            print(f"    Saved media locally to: {local_media_path}")

        print(f"    (WandB logging disabled with --no_wandb flag)")
    else:
        # WandB mode: Log to appropriate run
        if args.report_to_base_run:
            # Legacy mode: Resume original run per checkpoint
            print(f"  [REPORT_TO_BASE_RUN] Logging to original run {run.id} at step {step}...")
            try:
                wandb.init(project=run.project, id=run.id, resume="must")
            except Exception as e:
                raise RuntimeError(f"Failed to resume WandB run {run.id}: {e}")
        elif args.debug_project:
            # Debug mode: Use existing debug run (already initialized in main loop)
            print(f"  [DEBUG MODE] Logging to debug run at step {step}...")
        else:
            # Default mode: Use existing eval run (already initialized in main loop)
            print(f"  Logging to eval run at step {step}...")

        # Add media based on eval mode
        media_uploaded = False
        if args.eval_mode == "performance":
            # Add video if available
            if args.enable_video and media_path is not None and os.path.exists(media_path):
                caption = f"Evaluation at step {step}"
                metrics_to_log['Eval/Checkpoint Videos'] = wandb.Video(media_path, caption=caption)
                media_uploaded = True
        elif args.eval_mode == "noise":
            # Add noise visualization image
            if media_path is not None and os.path.exists(media_path):
                caption = f"Noise Evaluation at step {step}"
                metrics_to_log['eval_media/noise_images'] = wandb.Image(media_path, caption=caption)
                media_uploaded = True
        elif args.eval_mode == "rotation":
            # Add rotation visualization image
            if media_path is not None and os.path.exists(media_path):
                caption = f"Rotation Evaluation at step {step}"
                metrics_to_log['eval_media/rotation_images'] = wandb.Image(media_path, caption=caption)
                media_uploaded = True
        elif args.eval_mode == "trajectory":
            # Trajectory data is saved locally to --traj_output_dir, not uploaded to wandb
            # Keeping wandb upload code commented out for potential future use:
            # if media_path is not None and os.path.exists(media_path):
            #     # Save file to wandb files
            #     wandb.save(media_path, base_path=os.path.dirname(media_path), policy="now")
            #     media_uploaded = True
            pass  # Trajectory .pkl files are saved locally, not to wandb

        try:
            # Log metrics with step number
            wandb.log(metrics_to_log, step=step)

            # Print status
            if media_uploaded:
                if args.eval_mode == "performance":
                    print(f"    Logged metrics and video for step {step}")
                elif args.eval_mode == "noise":
                    print(f"    Logged metrics and noise visualization for step {step}")
                elif args.eval_mode == "rotation":
                    print(f"    Logged metrics and rotation visualization for step {step}")
                # elif args.eval_mode == "trajectory":
                #     print(f"    Logged metrics and trajectory data file for step {step}")
            elif args.eval_mode == "trajectory":
                print(f"    Logged metrics for step {step} (trajectory data saved locally)")
            else:
                print(f"    Logged metrics for step {step}")

        except Exception as e:
            raise RuntimeError(f"Failed to log metrics to WandB for {run.id} at step {step}: {e}")
        finally:
            # Only finish run in report_to_base_run mode (other modes finish after all checkpoints)
            if args.report_to_base_run:
                wandb.finish()


# ===== EVALUATION FUNCTIONS =====

def run_basic_evaluation(
    env: Any,
    agent: Any,
    configs: Dict[str, Any],
    max_rollout_steps: int,
    enable_video: bool = False,
    show_progress: bool = False,
    eval_seed: int = None
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Run basic evaluation rollout for 100 episodes (one per environment).

    Args:
        env: Environment instance (wrapped with metrics wrappers)
        agent: Agent instance with loaded checkpoint
        configs: Configuration dictionary
        max_rollout_steps: Maximum steps per episode
        enable_video: Whether to capture video frames
        show_progress: Whether to show progress bar
        eval_seed: Seed for deterministic evaluation

    Returns:
        Tuple of (metrics_dict, rollout_data_dict)
            - metrics_dict: Aggregated metrics
            - rollout_data_dict: Raw rollout data for video generation (or None if video disabled)

    Raises:
        RuntimeError: If evaluation fails
    """
    print("  Running basic evaluation rollout...")

    # Set seed for deterministic evaluation
    if eval_seed is not None:
        from skrl.utils import set_seed
        set_seed(eval_seed, deterministic=True)
        print(f"    Set eval seed: {eval_seed}")

    # Get device and num_envs
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    # Pre-allocate all tracking arrays to maximum possible size
    # Episode data storage [num_envs]
    episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_rewards = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # Success/engagement tracking [num_envs]
    episode_succeeded = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_success_times = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_engaged = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_engagement_times = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_engagement_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)

    # Smoothness tracking [num_envs]
    episode_ssv = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_ssjv = torch.zeros(num_envs, dtype=torch.float32, device=device)

    # Force tracking [num_envs]
    episode_max_force = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_sum_force_in_contact = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_contact_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    # Energy tracking [num_envs]
    episode_energy = torch.zeros(num_envs, dtype=torch.float32, device=device)

    # Contact metrics [num_envs, 3] for X/Y/Z axes
    contact_force_counts = torch.zeros((num_envs, 3), dtype=torch.long, device=device)
    contact_pos_counts = torch.zeros((num_envs, 3), dtype=torch.long, device=device)
    no_contact_force_counts = torch.zeros((num_envs, 3), dtype=torch.long, device=device)
    no_contact_pos_counts = torch.zeros((num_envs, 3), dtype=torch.long, device=device)

    # Episode completion tracking
    completed_episodes = torch.zeros(num_envs, dtype=torch.bool, device=device)
    termination_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    truncation_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    success_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    # Reset environment
    obs_dict, info = env.reset()

    # Preallocate tensors for video generation (only if enabled)
    # We need max_rollout_steps + 1 frames (initial + each step)
    if enable_video:
        # Get image dimensions from first frame
        if 'img' in info:
            img_shape = info['img'].shape  # [num_envs, H, W, C]
            max_video_steps = max_rollout_steps + 1

            # Preallocate on CPU to save GPU memory
            video_frames = torch.zeros((max_video_steps, num_envs, img_shape[1], img_shape[2], img_shape[3]),
                                      dtype=torch.float32, device='cpu')
            step_values = torch.zeros((max_video_steps, num_envs), dtype=torch.float32, device=device)
            step_rewards = torch.zeros((max_video_steps, num_envs), dtype=torch.float32, device=device)
            step_engagement = torch.zeros((max_video_steps, num_envs), dtype=torch.bool, device=device)
            step_success = torch.zeros((max_video_steps, num_envs), dtype=torch.bool, device=device)
            step_force_control = torch.zeros((max_video_steps, num_envs, 3), dtype=torch.bool, device=device)

            # Capture initial frame
            video_frames[0] = info['img'].cpu()

            # Capture initial value estimate
            with torch.no_grad():
                if isinstance(obs_dict, dict) and 'critic' in obs_dict:
                    critic_obs = obs_dict['critic']
                else:
                    critic_obs = obs_dict
                values = agent.value.act({"states": critic_obs}, role="value")[0]
                if values.dim() > 1:
                    values = values.squeeze(-1)
                step_values[0] = values

            # Initial reward is 0 (already initialized)

            # Initial engagement and success state - find factory metrics wrapper
            factory_wrapper = env
            while hasattr(factory_wrapper, 'env'):
                if hasattr(factory_wrapper, 'curr_engaged') and hasattr(factory_wrapper, 'successes'):
                    break
                factory_wrapper = factory_wrapper.env

            if hasattr(factory_wrapper, 'curr_engaged'):
                step_engagement[0] = factory_wrapper.curr_engaged
            # else: already initialized to False

            if hasattr(factory_wrapper, 'successes'):
                step_success[0] = factory_wrapper.successes
            # else: already initialized to False

            # Initial force control is all zeros (already initialized)
        else:
            # No video available
            video_frames = None
            step_values = None
            step_rewards = None
            step_engagement = None
            step_force_control = None
    else:
        video_frames = None
        step_values = None
        step_rewards = None
        step_engagement = None
        step_force_control = None

    # Rollout loop
    step_count = 0
    progress_bar = tqdm.tqdm(total=max_rollout_steps + 1, desc="Steps completed", disable=not show_progress)

    while not completed_episodes.all():
        # Get actions from agent (deterministic evaluation)
        with torch.no_grad():
            #actions = agent.act(obs_dict, deterministic=True)[0]
            outputs = agent.act(obs_dict, timestep=step_count, timesteps=max_rollout_steps)[-1]
            # Use mean actions for evaluation (deterministic)
            actions = outputs['mean_actions']

        # Get value estimates for video if enabled
        if enable_video and step_values is not None:
            with torch.no_grad():
                # Get critic observations - handle both flat and dict observations
                if isinstance(obs_dict, dict) and 'critic' in obs_dict:
                    critic_obs = obs_dict['critic']
                else:
                    critic_obs = obs_dict
                values = agent.value.act({"states": critic_obs}, role="value")[0]
                if values.dim() > 1:
                    values = values.squeeze(-1)
                step_values[step_count + 1] = values

        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions)

        # Ensure all tensors are 1D [num_envs]
        if rewards.dim() > 1:
            rewards = rewards.squeeze(-1)
        if terminated.dim() > 1:
            terminated = terminated.squeeze(-1)
        if truncated.dim() > 1:
            truncated = truncated.squeeze(-1)

        # Store per-step data for video (write into preallocated tensors)
        if enable_video and video_frames is not None:
            # Capture video frame
            if 'img' in info:
                # For environments that have already completed, reuse their last frame
                # For active environments, capture the new frame
                new_frame = info['img'].cpu()

                # Start with previous frame for all environments
                video_frames[step_count + 1] = video_frames[step_count].clone()

                # Update only the active (not yet completed) environments with new frames
                active_mask = ~completed_episodes
                if active_mask.any():
                    # Get indices of active environments
                    active_indices = torch.where(active_mask)[0].cpu()
                    for idx in active_indices:
                        video_frames[step_count + 1, idx] = new_frame[idx]

            # Store rewards
            step_rewards[step_count + 1] = rewards

            # Track engagement and success - find factory metrics wrapper
            factory_wrapper = env
            while hasattr(factory_wrapper, 'env'):
                if hasattr(factory_wrapper, 'curr_engaged') and hasattr(factory_wrapper, 'successes'):
                    break
                factory_wrapper = factory_wrapper.env

            if hasattr(factory_wrapper, 'curr_engaged'):
                step_engagement[step_count + 1] = factory_wrapper.curr_engaged

            if hasattr(factory_wrapper, 'successes'):
                step_success[step_count + 1] = factory_wrapper.successes

            # Track force control (first 3 dimensions of action space)
            unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
            if hasattr(unwrapped_env, 'actions'):
                force_control_actions = unwrapped_env.actions[:, :3]  # First 3 dimensions
                force_control_mask = force_control_actions > 0.5
                step_force_control[step_count + 1] = force_control_mask

        # Active environments mask (not yet completed)
        active_mask = ~completed_episodes

        # Update episode rewards for active environments
        episode_rewards[active_mask] += rewards[active_mask]

        # Update episode lengths for active environments
        episode_lengths[active_mask] += 1

        # Track success and engagement states from info dict
        if 'smoothness' in info:
            # Factory metrics wrapper adds smoothness info
            smoothness_info = info['smoothness']

            # Find the factory metrics wrapper in the wrapper stack
            # It stores successes and curr_engaged attributes
            factory_wrapper = env
            while hasattr(factory_wrapper, 'env'):
                if hasattr(factory_wrapper, 'successes') and hasattr(factory_wrapper, 'curr_engaged'):
                    break
                factory_wrapper = factory_wrapper.env

            # Check if we have success/engagement data
            if hasattr(factory_wrapper, 'successes'):
                curr_successes = factory_wrapper.successes  # Boolean tensor [num_envs]

                # Track first success for active environments
                first_success = curr_successes & ~episode_succeeded & active_mask
                episode_succeeded[curr_successes & active_mask] = True
                if first_success.any():
                    episode_success_times[first_success] = episode_lengths[first_success].clone()
                    success_step[first_success] = step_count

            # Check for engagement state
            if hasattr(factory_wrapper, 'curr_engaged'):
                curr_engaged = factory_wrapper.curr_engaged  # Boolean tensor [num_envs]

                # Track first engagement for active environments
                first_engaged = curr_engaged & ~episode_engaged & active_mask
                episode_engaged[curr_engaged & active_mask] = True
                if first_engaged.any():
                    episode_engagement_times[first_engaged] = episode_lengths[first_engaged].clone()

                # Increment engagement length for currently engaged active environments
                episode_engagement_lengths[curr_engaged & active_mask] += 1

        # Calculate SSV directly from environment velocity data
        if hasattr(env.unwrapped, 'fingertip_midpoint_linvel'):
            # Get end-effector linear velocity (real velocity from simulation) [num_envs, 3]
            ee_vel = env.unwrapped.fingertip_midpoint_linvel
            # Calculate velocity magnitude (L2 norm) for each environment [num_envs]
            velocity_norm = torch.linalg.norm(ee_vel, dim=1)
            # Accumulate for active environments
            episode_ssv[active_mask] += velocity_norm[active_mask]

        else:
            pass

        # Calculate SSJV directly from environment joint velocity data
        if hasattr(env.unwrapped, 'joint_vel'):
            # Get joint velocities [num_envs, num_joints]
            joint_vel = env.unwrapped.joint_vel
            # Calculate norm of squared velocities for each environment
            ssjv_step = torch.linalg.norm(joint_vel * joint_vel, dim=1)
            # Accumulate for active environments
            episode_ssjv[active_mask] += ssjv_step[active_mask]

        else:
            pass

        # Force metrics - only accumulate when in contact
        if hasattr(env.unwrapped, 'robot_force_torque') and hasattr(env.unwrapped, 'in_contact'):
            # Get current force magnitude [num_envs]
            force_magnitude = torch.linalg.norm(env.unwrapped.robot_force_torque[:, :3], dim=1)

            # Check if any axis is in contact [num_envs]
            any_contact = env.unwrapped.in_contact.any(dim=1)

            # Only accumulate for environments that are: in contact AND active
            in_contact_and_active = any_contact & active_mask

            if in_contact_and_active.any():
                episode_sum_force_in_contact[in_contact_and_active] += force_magnitude[in_contact_and_active]
                episode_contact_steps[in_contact_and_active] += 1

            # Max force tracked across ALL steps (not just contact)
            episode_max_force[active_mask] = torch.max(
                episode_max_force[active_mask],
                force_magnitude[active_mask]
            )

        # Energy metric: |joint_vel × joint_torque| for arm joints only (indices 0:7)
        if hasattr(env.unwrapped, 'joint_vel') and hasattr(env.unwrapped, 'joint_torque'):
            # Get arm joint data only (exclude gripper joints 7-8)
            arm_joint_vel = env.unwrapped.joint_vel[:, :7]  # [num_envs, 7]
            arm_joint_torque = env.unwrapped.joint_torque[:, :7]  # [num_envs, 7]

            # Sum of absolute values across all arm joints
            energy_step = torch.sum(torch.abs(arm_joint_vel * arm_joint_torque), dim=1)  # [num_envs]

            # Accumulate for active environments
            episode_energy[active_mask] += energy_step[active_mask]

        # Accumulate contact metrics from to_log for active environments
        if hasattr(env.unwrapped, 'extras') and 'to_log' in env.unwrapped.extras:
            to_log = env.unwrapped.extras['to_log']

            # Check if hybrid control wrapper is present
            has_hybrid_control = 'Control Mode / Force Control X' in to_log

            if has_hybrid_control:
                for axis_idx, axis in enumerate(['X', 'Y', 'Z']):
                    # Get control mode for this axis from to_log
                    force_control_key = f'Control Mode / Force Control {axis}'
                    if force_control_key in to_log:
                        force_control = to_log[force_control_key] > 0.5  # [num_envs]
                        pos_control = ~force_control

                        # Get contact state for this axis
                        # Check if in_contact is available
                        if hasattr(env.unwrapped, 'in_contact'):
                            in_contact = env.unwrapped.in_contact[:, axis_idx]  # [num_envs]

                            # Update counts for active environments only
                            contact_force_counts[active_mask, axis_idx] += (in_contact[active_mask] & force_control[active_mask]).long()
                            contact_pos_counts[active_mask, axis_idx] += (in_contact[active_mask] & pos_control[active_mask]).long()
                            no_contact_force_counts[active_mask, axis_idx] += (~in_contact[active_mask] & force_control[active_mask]).long()
                            no_contact_pos_counts[active_mask, axis_idx] += (~in_contact[active_mask] & pos_control[active_mask]).long()

        # Check for newly completed episodes (terminated OR truncated ONLY - success is not terminal)
        newly_completed = ((terminated | truncated) & ~completed_episodes)

        # Track termination and truncation steps
        if newly_completed.any():
            termination_mask = newly_completed & terminated
            truncation_mask = newly_completed & truncated
            termination_step[termination_mask] = step_count
            truncation_step[truncation_mask] = step_count

        # Mark completed
        completed_episodes = completed_episodes | newly_completed

        # Track which episodes terminated vs truncated
        episode_terminated[newly_completed & terminated] = True

        # Update progress bar (increment by 1 step)
        if show_progress:
            progress_bar.update(1)

        step_count += 1

        # Safety check to prevent infinite loops
        if step_count > max_rollout_steps * 2:
            raise RuntimeError(f"Evaluation exceeded maximum steps ({max_rollout_steps * 2}). Some environments may not be terminating.")

    progress_bar.close()
    print(f"    Completed {num_envs} episodes in {step_count} steps")

    # Compute aggregated metrics from raw data
    print("  Computing aggregated metrics...")
    metrics = _compute_aggregated_metrics(
        episode_lengths=episode_lengths,
        episode_terminated=episode_terminated,
        episode_succeeded=episode_succeeded,
        episode_ssv=episode_ssv,
        episode_ssjv=episode_ssjv,
        episode_max_force=episode_max_force,
        episode_sum_force_in_contact=episode_sum_force_in_contact,
        episode_contact_steps=episode_contact_steps,
        episode_energy=episode_energy,
        contact_force_counts=contact_force_counts,
        contact_pos_counts=contact_pos_counts,
        no_contact_force_counts=no_contact_force_counts,
        no_contact_pos_counts=no_contact_pos_counts,
    )

    # Prepare rollout data for video generation and/or noise evaluation
    rollout_data = None
    if enable_video and video_frames is not None:
        # Slice tensors to actual number of steps taken (step_count + 1 for initial frame)
        actual_steps = step_count + 1
        rollout_data = {
            'images': video_frames[:actual_steps],  # [actual_steps, num_envs, H, W, C]
            'values': step_values[:actual_steps],    # [actual_steps, num_envs]
            'rewards': step_rewards[:actual_steps],  # [actual_steps, num_envs]
            'engagement_history': step_engagement[:actual_steps],  # [actual_steps, num_envs]
            'success_history': step_success[:actual_steps],  # [actual_steps, num_envs]
            'force_control': step_force_control[:actual_steps],  # [actual_steps, num_envs, 3]
            'total_returns': episode_rewards,  # [num_envs]
            'success_step': success_step,      # [num_envs]
            'termination_step': termination_step,  # [num_envs]
            'truncation_step': truncation_step,    # [num_envs]
        }
    else:
        # Even without video, include per-environment data for noise evaluation
        rollout_data = {
            'episode_lengths': episode_lengths,
            'episode_terminated': episode_terminated,
            'episode_succeeded': episode_succeeded,
            'episode_ssv': episode_ssv,
            'episode_ssjv': episode_ssjv,
            'episode_max_force': episode_max_force,
            'episode_sum_force_in_contact': episode_sum_force_in_contact,
            'episode_contact_steps': episode_contact_steps,
            'episode_energy': episode_energy,
            'contact_force_counts': contact_force_counts,
            'contact_pos_counts': contact_pos_counts,
            'no_contact_force_counts': no_contact_force_counts,
            'no_contact_pos_counts': no_contact_pos_counts,
            'termination_step': termination_step,
            'truncation_step': truncation_step,
        }

    return metrics, rollout_data


def _compute_aggregated_metrics(
    episode_lengths: torch.Tensor,
    episode_terminated: torch.Tensor,
    episode_succeeded: torch.Tensor,
    episode_ssv: torch.Tensor,
    episode_ssjv: torch.Tensor,
    episode_max_force: torch.Tensor,
    episode_sum_force_in_contact: torch.Tensor,
    episode_contact_steps: torch.Tensor,
    episode_energy: torch.Tensor,
    contact_force_counts: torch.Tensor,
    contact_pos_counts: torch.Tensor,
    no_contact_force_counts: torch.Tensor,
    no_contact_pos_counts: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute aggregated metrics from raw episode data.

    Args:
        episode_lengths: Episode lengths [num_envs]
        episode_terminated: Whether episode terminated (vs truncated) [num_envs]
        episode_succeeded: Whether episode succeeded [num_envs]
        episode_ssv: Sum squared velocity [num_envs]
        episode_ssjv: Sum squared joint velocity [num_envs]
        episode_max_force: Maximum force magnitude [num_envs]
        episode_sum_force_in_contact: Sum of force magnitudes when in contact [num_envs]
        episode_contact_steps: Number of steps with contact [num_envs]
        episode_energy: Energy used (arm joints only) [num_envs]
        contact_force_counts: Contact+Force counts [num_envs, 3]
        contact_pos_counts: Contact+Pos counts [num_envs, 3]
        no_contact_force_counts: NoContact+Force counts [num_envs, 3]
        no_contact_pos_counts: NoContact+Pos counts [num_envs, 3]

    Returns:
        Dictionary of aggregated metrics (38 total)
    """
    metrics = {}
    num_envs = episode_lengths.shape[0]

    # ===== Core Episode Metrics (10 total) =====
    metrics['total_episodes'] = num_envs

    # Mutually exclusive categories by final outcome:
    # 1. Successful completions: Succeeded AND did NOT break (may have truncated due to staying in success)
    # 2. Breaks: Terminated (broke object), regardless of whether succeeded first
    # 3. Failed timeouts: Truncated without success and without breaking

    successful_completions = episode_succeeded & ~episode_terminated
    breaks = episode_terminated
    failed_timeouts = ~episode_terminated & ~episode_succeeded

    metrics['num_successful_completions'] = successful_completions.sum().item()
    metrics['num_breaks'] = breaks.sum().item()
    metrics['num_failed_timeouts'] = failed_timeouts.sum().item()

    # Verify they sum to total (sanity check)
    total_check = metrics['num_successful_completions'] + metrics['num_breaks'] + metrics['num_failed_timeouts']
    if total_check != num_envs:
        print(f"WARNING: Episode counts don't sum correctly! {total_check} != {num_envs}")

    metrics['episode_length'] = episode_lengths.float().mean().item()

    # Smoothness metrics
    metrics['ssv'] = episode_ssv.mean().item()
    metrics['ssjv'] = episode_ssjv.mean().item()

    # Force metrics
    metrics['max_force'] = episode_max_force.max().item()
    if episode_contact_steps.sum() > 0:
        metrics['avg_force_in_contact'] = (
            episode_sum_force_in_contact.sum().item() /
            episode_contact_steps.sum().item()
        )
    else:
        metrics['avg_force_in_contact'] = 0.0

    # Energy metric
    metrics['energy'] = episode_energy.mean().item()

    # ===== Contact-Control Metrics (28 total) =====
    # Raw counts per axis (12) + accuracy metrics (16)
    axis_names = ['x', 'y', 'z']
    for axis_idx, axis in enumerate(axis_names):
        # Sum across all environments for total counts
        fc_contact = contact_force_counts[:, axis_idx].sum().item()
        fc_no_contact = no_contact_force_counts[:, axis_idx].sum().item()
        pc_contact = contact_pos_counts[:, axis_idx].sum().item()
        pc_no_contact = no_contact_pos_counts[:, axis_idx].sum().item()

        # Raw counts (12 metrics: 4 per axis × 3 axes)
        metrics[f'force_control_contact_{axis}'] = fc_contact
        metrics[f'force_control_no_contact_{axis}'] = fc_no_contact
        metrics[f'pos_control_contact_{axis}'] = pc_contact
        metrics[f'pos_control_no_contact_{axis}'] = pc_no_contact

        # Force control accuracy: force_control_contact / (force_control_contact + force_control_no_contact)
        fc_total = fc_contact + fc_no_contact
        if fc_total > 0:
            metrics[f'force_control_accuracy_{axis}'] = fc_contact / fc_total
        else:
            metrics[f'force_control_accuracy_{axis}'] = 0.0

        # Position control accuracy: pos_control_no_contact / (pos_control_no_contact + pos_control_contact)
        pc_total = pc_contact + pc_no_contact
        if pc_total > 0:
            metrics[f'pos_control_accuracy_{axis}'] = pc_no_contact / pc_total
        else:
            metrics[f'pos_control_accuracy_{axis}'] = 0.0

    # Total accuracy metrics (averaged across axes)
    force_accuracies = [
        metrics[f'force_control_accuracy_{axis}']
        for axis in axis_names
    ]
    pos_accuracies = [
        metrics[f'pos_control_accuracy_{axis}']
        for axis in axis_names
    ]

    metrics['force_control_accuracy_total'] = sum(force_accuracies) / 3.0
    metrics['pos_control_accuracy_total'] = sum(pos_accuracies) / 3.0

    return metrics


def run_noise_evaluation(
    env: Any,
    agent: Any,
    configs: Dict[str, Any],
    max_rollout_steps: int,
    noise_assignments: torch.Tensor,
    show_progress: bool = False,
    eval_seed: int = None
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Run noise robustness evaluation with 500 environments across 5 noise ranges.

    Args:
        env: Environment instance (wrapped with metrics wrappers and FixedNoiseWrapper)
        agent: Agent instance with loaded checkpoint
        configs: Configuration dictionary
        max_rollout_steps: Maximum steps per episode
        noise_assignments: [500, 2] tensor with (x, y) noise for each environment
        show_progress: Whether to show progress bar
        eval_seed: Seed for deterministic evaluation

    Returns:
        Tuple of (metrics_dict, rollout_data_dict)
            - metrics_dict: Aggregated metrics split by noise range
            - rollout_data_dict: Contains success/failure data per environment for visualization
    """
    print("  Running noise robustness evaluation...")

    # Run basic evaluation (no video)
    metrics, rollout_data = run_basic_evaluation(
        env=env,
        agent=agent,
        configs=configs,
        max_rollout_steps=max_rollout_steps,
        enable_video=False,  # No video in noise mode
        show_progress=show_progress,
        eval_seed=eval_seed
    )

    # Split metrics by noise range
    split_metrics = {}
    for range_idx, (min_val, max_val, range_name) in enumerate(NOISE_RANGES):
        start_idx = range_idx * ENVS_PER_NOISE_RANGE
        end_idx = start_idx + ENVS_PER_NOISE_RANGE

        # Extract data for this noise range
        range_metrics = _compute_range_metrics(
            rollout_data=rollout_data,
            start_idx=start_idx,
            end_idx=end_idx,
            range_name=range_name
        )

        # Add to split metrics with category-based prefixes
        prefixed_range_metrics = prefix_metrics_by_category(
            range_metrics,
            core_prefix=f"Noise_Eval({range_name})_Core",
            contact_prefix=f"Noise_Eval({range_name})_Contact"
        )
        split_metrics.update(prefixed_range_metrics)

    # Prepare rollout data for visualization
    rollout_data_for_viz = {
        'noise_assignments': noise_assignments,  # [500, 2]
        'episode_succeeded': rollout_data['episode_succeeded'],  # [500]
        'termination_step': rollout_data['termination_step'],  # [500]
        'truncation_step': rollout_data['truncation_step'],    # [500]
    }

    return split_metrics, rollout_data_for_viz


def run_rotation_evaluation(
    env: Any,
    agent: Any,
    configs: Dict[str, Any],
    max_rollout_steps: int,
    rotation_assignments: torch.Tensor,
    show_progress: bool = False,
    eval_seed: int = None
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Run rotation robustness evaluation with 500 environments across 5 rotation angles.

    Args:
        env: Environment instance (wrapped with metrics wrappers and FixedAssetRotationWrapper)
        agent: Agent instance with loaded checkpoint
        configs: Configuration dictionary
        max_rollout_steps: Maximum steps per episode
        rotation_assignments: [500, 5] tensor with (angle_rad, axis_x, axis_y, axis_z, direction) for each environment
        show_progress: Whether to show progress bar
        eval_seed: Seed for deterministic evaluation

    Returns:
        Tuple of (metrics_dict, rollout_data_dict)
            - metrics_dict: Aggregated metrics split by rotation angle
            - rollout_data_dict: Contains success/failure data per environment for visualization
    """
    print("  Running rotation robustness evaluation...")

    # Run basic evaluation (no video)
    metrics, rollout_data = run_basic_evaluation(
        env=env,
        agent=agent,
        configs=configs,
        max_rollout_steps=max_rollout_steps,
        enable_video=False,  # No video in rotation mode
        show_progress=show_progress,
        eval_seed=eval_seed
    )

    # Split metrics by rotation angle
    split_metrics = {}
    for angle_idx, angle_deg in enumerate(ROTATION_ANGLES_DEG):
        start_idx = angle_idx * ENVS_PER_ROTATION_ANGLE
        end_idx = start_idx + ENVS_PER_ROTATION_ANGLE

        # Extract data for this rotation angle
        angle_metrics = _compute_range_metrics(
            rollout_data=rollout_data,
            start_idx=start_idx,
            end_idx=end_idx,
            range_name=f"{angle_deg}deg"
        )

        # Add to split metrics with category-based prefixes
        prefixed_angle_metrics = prefix_metrics_by_category(
            angle_metrics,
            core_prefix=f"Rot_Eval({angle_deg}deg)_Core",
            contact_prefix=f"Rot_Eval({angle_deg}deg)_Contact"
        )
        split_metrics.update(prefixed_angle_metrics)

    # Prepare rollout data for visualization
    rollout_data_for_viz = {
        'rotation_assignments': rotation_assignments,  # [500, 5]
        'episode_succeeded': rollout_data['episode_succeeded'],  # [500]
        'termination_step': rollout_data['termination_step'],  # [500]
        'truncation_step': rollout_data['truncation_step'],    # [500]
    }

    return split_metrics, rollout_data_for_viz


def run_gain_evaluation(
    env: Any,
    agent: Any,
    configs: Dict[str, Any],
    max_rollout_steps: int,
    gain_assignments: torch.Tensor,
    show_progress: bool = False,
    eval_seed: int = None
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Run gain robustness evaluation with 500 environments across 5 gain multipliers.

    Args:
        env: Environment instance (wrapped with metrics wrappers and FixedGainWrapper)
        agent: Agent instance with loaded checkpoint
        configs: Configuration dictionary
        max_rollout_steps: Maximum steps per episode
        gain_assignments: [500] tensor with gain multiplier for each environment
        show_progress: Whether to show progress bar
        eval_seed: Seed for deterministic evaluation

    Returns:
        Tuple of (metrics_dict, rollout_data_dict)
            - metrics_dict: Aggregated metrics split by gain multiplier
            - rollout_data_dict: Contains success/failure data per environment
    """
    print("  Running gain robustness evaluation...")

    # Run basic evaluation (no video)
    metrics, rollout_data = run_basic_evaluation(
        env=env,
        agent=agent,
        configs=configs,
        max_rollout_steps=max_rollout_steps,
        enable_video=False,  # No video in gain mode
        show_progress=show_progress,
        eval_seed=eval_seed
    )

    # Split metrics by gain multiplier
    split_metrics = {}
    for mult_idx, multiplier in enumerate(GAIN_MULTIPLIERS):
        start_idx = mult_idx * ENVS_PER_GAIN_MULTIPLIER
        end_idx = start_idx + ENVS_PER_GAIN_MULTIPLIER

        # Extract data for this gain multiplier
        mult_metrics = _compute_range_metrics(
            rollout_data=rollout_data,
            start_idx=start_idx,
            end_idx=end_idx,
            range_name=f"{multiplier}x"
        )

        # Add to split metrics with category-based prefixes
        prefixed_mult_metrics = prefix_metrics_by_category(
            mult_metrics,
            core_prefix=f"Gain_Eval({multiplier}x)_Core",
            contact_prefix=f"Gain_Eval({multiplier}x)_Contact"
        )
        split_metrics.update(prefixed_mult_metrics)

    # Prepare rollout data (no visualization for gain mode)
    rollout_data_for_return = {
        'gain_assignments': gain_assignments,  # [500]
        'episode_succeeded': rollout_data['episode_succeeded'],  # [500]
        'termination_step': rollout_data['termination_step'],  # [500]
        'truncation_step': rollout_data['truncation_step'],    # [500]
    }

    return split_metrics, rollout_data_for_return


def run_dynamics_evaluation(
    env: Any,
    agent: Any,
    configs: Dict[str, Any],
    max_rollout_steps: int,
    friction_assignments: torch.Tensor,
    mass_assignments: torch.Tensor,
    show_progress: bool = False,
    eval_seed: int = None
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Run dynamics robustness evaluation with 1200 environments across 12 friction/mass combinations.

    Args:
        env: Environment instance (wrapped with metrics wrappers and FixedDynamicsWrapper)
        agent: Agent instance with loaded checkpoint
        configs: Configuration dictionary
        max_rollout_steps: Maximum steps per episode
        friction_assignments: [1200] tensor with friction multiplier for each environment
        mass_assignments: [1200] tensor with mass multiplier for each environment
        show_progress: Whether to show progress bar
        eval_seed: Seed for deterministic evaluation

    Returns:
        Tuple of (metrics_dict, rollout_data_dict)
            - metrics_dict: Aggregated metrics split by friction/mass combination
            - rollout_data_dict: Contains success/failure data per environment
    """
    print("  Running dynamics robustness evaluation...")

    # Run basic evaluation (no video)
    metrics, rollout_data = run_basic_evaluation(
        env=env,
        agent=agent,
        configs=configs,
        max_rollout_steps=max_rollout_steps,
        enable_video=False,  # No video in dynamics mode
        show_progress=show_progress,
        eval_seed=eval_seed
    )

    # Split metrics by friction/mass combination (friction-major ordering)
    split_metrics = {}
    combo_idx = 0
    for fric_mult in DYNAMICS_FRICTION_MULTIPLIERS:
        for mass_mult in DYNAMICS_MASS_MULTIPLIERS:
            start_idx = combo_idx * ENVS_PER_DYNAMICS_COMBO
            end_idx = start_idx + ENVS_PER_DYNAMICS_COMBO

            # Extract data for this friction/mass combination
            combo_metrics = _compute_range_metrics(
                rollout_data=rollout_data,
                start_idx=start_idx,
                end_idx=end_idx,
                range_name=f"fric={fric_mult}x,mass={mass_mult}x"
            )

            # Add to split metrics with category-based prefixes
            prefixed_combo_metrics = prefix_metrics_by_category(
                combo_metrics,
                core_prefix=f"Dyn_Eval(fric={fric_mult}x,mass={mass_mult}x)_Core",
                contact_prefix=f"Dyn_Eval(fric={fric_mult}x,mass={mass_mult}x)_Contact"
            )
            split_metrics.update(prefixed_combo_metrics)
            combo_idx += 1

    # Prepare rollout data (no visualization for dynamics mode)
    rollout_data_for_return = {
        'friction_assignments': friction_assignments,  # [1200]
        'mass_assignments': mass_assignments,          # [1200]
        'episode_succeeded': rollout_data['episode_succeeded'],  # [1200]
        'termination_step': rollout_data['termination_step'],    # [1200]
        'truncation_step': rollout_data['truncation_step'],      # [1200]
    }

    return split_metrics, rollout_data_for_return


def _compute_range_metrics(
    rollout_data: Dict[str, torch.Tensor],
    start_idx: int,
    end_idx: int,
    range_name: str
) -> Dict[str, float]:
    """
    Compute metrics for a specific noise/rotation/gain range from per-environment data.

    Args:
        rollout_data: Per-environment data from run_basic_evaluation
        start_idx: Start environment index for this range
        end_idx: End environment index for this range
        range_name: Name of the noise/rotation range (for logging)

    Returns:
        Dictionary of metrics for this range (same 38 metrics as main aggregation)
    """
    print(f"    Computing metrics for range {range_name} (envs {start_idx}-{end_idx})")

    # Extract per-environment data for this range
    episode_lengths = rollout_data['episode_lengths'][start_idx:end_idx]
    episode_terminated = rollout_data['episode_terminated'][start_idx:end_idx]
    episode_succeeded = rollout_data['episode_succeeded'][start_idx:end_idx]
    episode_ssv = rollout_data['episode_ssv'][start_idx:end_idx]
    episode_ssjv = rollout_data['episode_ssjv'][start_idx:end_idx]
    episode_max_force = rollout_data['episode_max_force'][start_idx:end_idx]
    episode_sum_force_in_contact = rollout_data['episode_sum_force_in_contact'][start_idx:end_idx]
    episode_contact_steps = rollout_data['episode_contact_steps'][start_idx:end_idx]
    episode_energy = rollout_data['episode_energy'][start_idx:end_idx]
    contact_force_counts = rollout_data['contact_force_counts'][start_idx:end_idx]
    contact_pos_counts = rollout_data['contact_pos_counts'][start_idx:end_idx]
    no_contact_force_counts = rollout_data['no_contact_force_counts'][start_idx:end_idx]
    no_contact_pos_counts = rollout_data['no_contact_pos_counts'][start_idx:end_idx]

    # Reuse the main aggregation function for consistency
    return _compute_aggregated_metrics(
        episode_lengths=episode_lengths,
        episode_terminated=episode_terminated,
        episode_succeeded=episode_succeeded,
        episode_ssv=episode_ssv,
        episode_ssjv=episode_ssjv,
        episode_max_force=episode_max_force,
        episode_sum_force_in_contact=episode_sum_force_in_contact,
        episode_contact_steps=episode_contact_steps,
        episode_energy=episode_energy,
        contact_force_counts=contact_force_counts,
        contact_pos_counts=contact_pos_counts,
        no_contact_force_counts=no_contact_force_counts,
        no_contact_pos_counts=no_contact_pos_counts,
    )


def split_rollout_data_by_agent(
    rollout_data: Dict[str, torch.Tensor],
    num_agents: int,
    envs_per_agent: int
) -> List[Dict[str, torch.Tensor]]:
    """
    Split rollout data by agent for parallel evaluation.

    The environment partitions environments as:
        agent_0: envs [0, envs_per_agent)
        agent_1: envs [envs_per_agent, 2*envs_per_agent)
        ...

    Args:
        rollout_data: Dictionary with tensors of shape [num_envs, ...] or [num_envs]
        num_agents: Number of agents (runs)
        envs_per_agent: Environments per agent

    Returns:
        List of rollout_data dicts, one per agent
    """
    agent_rollout_data = []

    for agent_idx in range(num_agents):
        start_idx = agent_idx * envs_per_agent
        end_idx = (agent_idx + 1) * envs_per_agent

        agent_data = {}
        for key, tensor in rollout_data.items():
            if tensor is None:
                agent_data[key] = None
            elif isinstance(tensor, torch.Tensor):
                # Slice along first dimension (env dimension)
                agent_data[key] = tensor[start_idx:end_idx].clone()
            else:
                # Non-tensor data (pass through)
                agent_data[key] = tensor

        agent_rollout_data.append(agent_data)

    return agent_rollout_data


def compute_run_metrics(
    rollout_data: Dict[str, torch.Tensor],
    eval_mode: str
) -> Dict[str, float]:
    """
    Compute metrics for one run's slice of rollout data.

    Args:
        rollout_data: Per-environment data from run_basic_evaluation for one agent
        eval_mode: Evaluation mode (performance, noise, rotation, gain, dynamics, trajectory)

    Returns:
        Dictionary of prefixed metrics ready for WandB logging

    Raises:
        RuntimeError: If eval_mode is unknown
    """
    if eval_mode == "performance":
        # Compute basic aggregated metrics
        base_metrics = _compute_aggregated_metrics(
            episode_lengths=rollout_data['episode_lengths'],
            episode_terminated=rollout_data['episode_terminated'],
            episode_succeeded=rollout_data['episode_succeeded'],
            episode_ssv=rollout_data['episode_ssv'],
            episode_ssjv=rollout_data['episode_ssjv'],
            episode_max_force=rollout_data['episode_max_force'],
            episode_sum_force_in_contact=rollout_data['episode_sum_force_in_contact'],
            episode_contact_steps=rollout_data['episode_contact_steps'],
            episode_energy=rollout_data['episode_energy'],
            contact_force_counts=rollout_data['contact_force_counts'],
            contact_pos_counts=rollout_data['contact_pos_counts'],
            no_contact_force_counts=rollout_data['no_contact_force_counts'],
            no_contact_pos_counts=rollout_data['no_contact_pos_counts'],
        )
        return prefix_metrics_by_category(base_metrics, "Eval_Core", "Eval_Contact")

    elif eval_mode == "noise":
        # Split by noise range
        split_metrics = {}
        for range_idx, (min_val, max_val, range_name) in enumerate(NOISE_RANGES):
            start_idx = range_idx * ENVS_PER_NOISE_RANGE
            end_idx = start_idx + ENVS_PER_NOISE_RANGE
            range_metrics = _compute_range_metrics(rollout_data, start_idx, end_idx, range_name)
            prefixed = prefix_metrics_by_category(
                range_metrics,
                f"Noise_Eval({range_name})_Core",
                f"Noise_Eval({range_name})_Contact"
            )
            split_metrics.update(prefixed)
        return split_metrics

    elif eval_mode == "rotation":
        # Split by rotation angle
        split_metrics = {}
        for angle_idx, angle_deg in enumerate(ROTATION_ANGLES_DEG):
            start_idx = angle_idx * ENVS_PER_ROTATION_ANGLE
            end_idx = start_idx + ENVS_PER_ROTATION_ANGLE
            angle_metrics = _compute_range_metrics(rollout_data, start_idx, end_idx, f"{angle_deg}deg")
            prefixed = prefix_metrics_by_category(
                angle_metrics,
                f"Rot_Eval({angle_deg}deg)_Core",
                f"Rot_Eval({angle_deg}deg)_Contact"
            )
            split_metrics.update(prefixed)
        return split_metrics

    elif eval_mode == "gain":
        # Split by gain multiplier
        split_metrics = {}
        for mult_idx, multiplier in enumerate(GAIN_MULTIPLIERS):
            start_idx = mult_idx * ENVS_PER_GAIN_MULTIPLIER
            end_idx = start_idx + ENVS_PER_GAIN_MULTIPLIER
            mult_metrics = _compute_range_metrics(rollout_data, start_idx, end_idx, f"{multiplier}x")
            prefixed = prefix_metrics_by_category(
                mult_metrics,
                f"Gain_Eval({multiplier}x)_Core",
                f"Gain_Eval({multiplier}x)_Contact"
            )
            split_metrics.update(prefixed)
        return split_metrics

    elif eval_mode == "dynamics":
        # Split by friction/mass combination
        split_metrics = {}
        combo_idx = 0
        for fric_mult in DYNAMICS_FRICTION_MULTIPLIERS:
            for mass_mult in DYNAMICS_MASS_MULTIPLIERS:
                start_idx = combo_idx * ENVS_PER_DYNAMICS_COMBO
                end_idx = start_idx + ENVS_PER_DYNAMICS_COMBO
                combo_name = f"fric={fric_mult}x,mass={mass_mult}x"
                combo_metrics = _compute_range_metrics(rollout_data, start_idx, end_idx, combo_name)
                prefixed = prefix_metrics_by_category(
                    combo_metrics,
                    f"Dyn_Eval({combo_name})_Core",
                    f"Dyn_Eval({combo_name})_Contact"
                )
                split_metrics.update(prefixed)
                combo_idx += 1
        return split_metrics

    elif eval_mode == "trajectory":
        # Trajectory mode returns minimal metrics (data is in .pkl file)
        raise RuntimeError("compute_run_metrics should not be called for trajectory mode")

    else:
        raise RuntimeError(f"Unknown eval_mode: {eval_mode}")


def _create_checkpoint_gif(images: torch.Tensor, values: torch.Tensor, cumulative_rewards: torch.Tensor,
                          success_step: torch.Tensor, termination_step: torch.Tensor,
                          truncation_step: torch.Tensor, engagement_history: torch.Tensor,
                          success_history: torch.Tensor, force_control: torch.Tensor,
                          output_path: str, duration: int, font: Any) -> None:
    """
    Create a 3x4 grid GIF from 12 selected environments.

    Args:
        images: [num_steps, 12, 180, 240, 3] tensor
        values: [num_steps, 12] tensor
        cumulative_rewards: [num_steps, 12] tensor of accumulated rewards
        success_step: [12] tensor (kept for backward compatibility, not used)
        termination_step: [12] tensor
        truncation_step: [12] tensor
        engagement_history: [num_steps, 12] tensor
        success_history: [num_steps, 12] tensor
        force_control: [num_steps, 12, 3] tensor
        output_path: Path to save GIF
        duration: Frame duration in milliseconds
        font: PIL ImageFont to use
    """
    from PIL import Image, ImageDraw, ImageFont

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

            # Place image in canvas
            y_start = row * 180
            y_end = (row + 1) * 180
            x_start = col * 240
            x_end = (col + 1) * 240
            canvas[y_start:y_end, x_start:x_end] = img

        # Convert canvas to PIL Image
        canvas_np = (canvas.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(canvas_np)
        draw = ImageDraw.Draw(pil_img)

        # Draw annotations for each environment
        for env_idx in range(12):
            row = env_idx // 4
            col = env_idx % 4
            x_offset = col * 240
            y_offset = row * 180

            # Determine border color (precedence: success > termination/truncation > engagement)
            is_succeeded = success_history[step_idx, env_idx].item()
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

            # Draw accumulated reward text (top left of image)
            cum_rew = cumulative_rewards[step_idx, env_idx].item()
            draw.text(
                (x_offset + 2, y_offset + 2),
                f"TotRew: {cum_rew:.2f}",
                fill=(0, 255, 0),
                font=font
            )

            # Draw value estimate (bottom of image)
            val = values[step_idx, env_idx].item()
            draw.text(
                (x_offset + 2, y_offset + 160),
                f"V-Est: {val:.2f}",
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


def generate_evaluation_video(rollout_data: Dict[str, Any], agent: Any,
                              policy_hz: float, output_path: str) -> str:
    """
    Generate 3x4 grid GIF for evaluation checkpoint.

    Args:
        rollout_data: Dictionary with collected rollout data including images, values, rewards, etc.
        agent: Agent instance (for value preprocessing)
        policy_hz: Policy frequency for GIF frame rate
        output_path: Path to save the video

    Returns:
        Path to generated video file

    Raises:
        RuntimeError: If video generation fails
    """
    print(f"  Generating evaluation video...")

    # Verify images are available
    if rollout_data["images"] is None:
        raise RuntimeError("Cannot generate video: rollout_data['images'] is None")

    if len(rollout_data["images"]) == 0:
        raise RuntimeError(
            "Cannot generate video: No images were captured during rollout. "
            "This likely means the camera was not properly configured in the scene. "
            "Check that the environment has a 'tiled_camera' in the scene."
        )

    # Calculate frame duration in milliseconds
    duration = int(1000 / policy_hz)
    print(f"    Frame duration: {duration}ms (policy_hz: {policy_hz})")

    # Try to load font with fallback
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        print("    Using DejaVuSans.ttf font")
    except:
        from PIL import ImageFont
        font = ImageFont.load_default()
        print("    Using default font (DejaVuSans.ttf not found)")

    # Apply inverse preprocessing to ALL values at once
    all_values = rollout_data["values"]  # [steps, num_envs]

    # Ensure values have shape [steps, num_envs, 1] for preprocessor
    if all_values.dim() == 2:
        all_values = all_values.unsqueeze(-1)  # [steps, num_envs, 1]

    # Apply inverse preprocessing (handles 3D batched input)
    all_values_unscaled = agent._value_preprocessor(all_values, inverse=True)  # [steps, num_envs, 1]

    # Squeeze to [steps, num_envs] for easier slicing
    all_values_unscaled = all_values_unscaled.squeeze(-1)  # [steps, num_envs]

    # Get data dimensions
    num_envs = rollout_data["total_returns"].shape[0]

    # Compute cumulative rewards
    # Mask rewards after termination/truncation
    step_rewards = rollout_data["rewards"]  # [steps, num_envs]
    num_steps = step_rewards.shape[0]

    # Create mask for active environments at each step
    termination_step = rollout_data["termination_step"]  # [num_envs]
    truncation_step = rollout_data["truncation_step"]    # [num_envs]

    # For each step, determine which envs are still active
    # An env is inactive after its termination or truncation step
    reward_mask = torch.ones_like(step_rewards, dtype=torch.bool)
    for step_idx in range(num_steps):
        # Mark envs as inactive if they terminated/truncated at or before this step
        inactive = ((termination_step != -1) & (termination_step <= step_idx)) | \
                   ((truncation_step != -1) & (truncation_step <= step_idx))
        reward_mask[step_idx, inactive] = False

    # Apply mask and compute cumulative sum
    masked_rewards = step_rewards.clone()
    masked_rewards[~reward_mask] = 0.0
    cumulative_rewards = torch.cumsum(masked_rewards, dim=0)  # [steps, num_envs]

    # Select 12 environments based on returns (worst 4, middle 4, best 4)
    sorted_indices = torch.argsort(rollout_data["total_returns"])
    n = sorted_indices.shape[0]

    # Ensure we have at least 12 environments
    if n < 12:
        raise RuntimeError(f"Need at least 12 environments for video generation, but only have {n}")

    selected_indices = torch.cat([
        sorted_indices[-4:],              # best 4
        sorted_indices[n//2-2:n//2+2],   # middle 4
        sorted_indices[:4]                # worst 4
    ])

    # Slice data for selected 12 environments (move to CPU if needed to save GPU memory)
    images_tensor = rollout_data["images"]
    if images_tensor.device.type == 'cuda':
        images_tensor = images_tensor.cpu()

    # Move selected_indices to same device as the tensors we're indexing
    selected_indices_cpu = selected_indices.cpu()
    selected_images = images_tensor[:, selected_indices_cpu]

    selected_values = all_values_unscaled[:, selected_indices].cpu()
    selected_success_step = rollout_data["success_step"][selected_indices].cpu()
    selected_termination_step = rollout_data["termination_step"][selected_indices].cpu()
    selected_truncation_step = rollout_data["truncation_step"][selected_indices].cpu()
    selected_engagement = rollout_data["engagement_history"][:, selected_indices].cpu()
    selected_success = rollout_data["success_history"][:, selected_indices].cpu()
    selected_force_control = rollout_data["force_control"][:, selected_indices].cpu()
    selected_cumulative_rewards = cumulative_rewards[:, selected_indices].cpu()

    # Generate GIF
    _create_checkpoint_gif(
        selected_images,
        selected_values,
        selected_cumulative_rewards,
        selected_success_step,
        selected_termination_step,
        selected_truncation_step,
        selected_engagement,
        selected_success,
        selected_force_control,
        output_path,
        duration,
        font
    )

    print(f"    Saved video to: {output_path}")
    return output_path


def generate_noise_visualization(rollout_data: Dict[str, Any], step: int, output_path: str) -> str:
    """
    Generate noise robustness visualization image.

    Creates concentric circles representing noise ranges with dots showing
    success (green) or failure (red) for each environment, and ring shading
    based on success rate.

    Args:
        rollout_data: Dictionary with noise_assignments, episode_succeeded, termination_step, truncation_step
        step: Checkpoint step number (for caption)
        output_path: Path to save the image

    Returns:
        Path to generated image file
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib import cm
    import numpy as np

    print(f"  Generating noise visualization...")

    # Extract data
    noise_assignments = rollout_data['noise_assignments'].cpu().numpy()  # [500, 2] in meters
    episode_succeeded = rollout_data['episode_succeeded'].cpu().numpy()  # [500]
    termination_step = rollout_data['termination_step'].cpu().numpy()    # [500]
    truncation_step = rollout_data['truncation_step'].cpu().numpy()      # [500]

    # Compute success rate per range
    success_rates = []
    for range_idx in range(len(NOISE_RANGES)):
        start_idx = range_idx * ENVS_PER_NOISE_RANGE
        end_idx = start_idx + ENVS_PER_NOISE_RANGE
        success_rate = episode_succeeded[start_idx:end_idx].mean()
        success_rates.append(success_rate)

    # Create figure (1024x1024)
    fig, ax = plt.subplots(1, 1, figsize=(10.24, 10.24), dpi=100)
    ax.set_xlim(-0.012, 0.012)
    ax.set_ylim(-0.012, 0.012)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw concentric circles with shading based on success rate
    # Draw from largest to smallest so smaller rings appear on top
    colormap = cm.get_cmap('cividis')  # Cividis colormap (perceptually uniform)
    # Extract outer radii from NOISE_RANGES
    ring_radii = [max_val for min_val, max_val, range_name in NOISE_RANGES]

    # Draw in reverse order (largest to smallest)
    for ring_idx in reversed(range(len(NOISE_RANGES))):
        min_val, max_val, range_name = NOISE_RANGES[ring_idx]
        outer_radius = ring_radii[ring_idx] #* 0.1
        inner_radius = 0.0 if ring_idx == 0 else ring_radii[ring_idx - 1] #* 0.1
        success_rate = success_rates[ring_idx]

        # Draw filled circle with color based on success rate
        # Use success_rate directly as it's already in [0, 1] range
        # This ensures 0.0 -> red, 0.5 -> yellow, 1.0 -> green consistently
        circle = patches.Circle(
            (0, 0), outer_radius,
            facecolor=colormap(success_rate), edgecolor='black', linewidth=1.5, alpha=1.0
        )
        ax.add_patch(circle)

    # Plot dots for each environment
    for env_idx in range(NOISE_MODE_TOTAL_ENVS):
        x_noise = noise_assignments[env_idx, 0]#*0.1
        y_noise = noise_assignments[env_idx, 1]#*0.1
        succeeded = episode_succeeded[env_idx]
        terminated = termination_step[env_idx] != -1
        truncated = truncation_step[env_idx] != -1

        # Determine dot color: green for success, red for terminated/truncated
        if succeeded:
            color = 'green'
        elif terminated or truncated:
            color = 'red'
        else:
            color = 'gray'  # Should not happen in practice

        ax.scatter(x_noise, y_noise, c=color, s=20, alpha=0.8, edgecolors='black', linewidths=0.5)

    # Add success rate text labels in a single box in the bottom right corner
    text_lines = []
    for min_val, max_val, range_name in NOISE_RANGES:
        ring_idx = NOISE_RANGES.index((min_val, max_val, range_name))
        success_rate = success_rates[ring_idx]
        text_lines.append(f"{range_name}: {success_rate*100:.1f}%")

    text_content = '\n'.join(text_lines)
    ax.text(
        0.9975, 0.0075, text_content,
        fontsize=18, verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9)
    )

    # Add title with minimal padding
    ax.set_title(f"Noise Robustness at Step {step}", fontsize=16, fontweight='bold', pad=0)

    # Save figure
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    print(f"    Saved noise visualization to: {output_path}")
    return output_path


def generate_rotation_visualization(rollout_data: Dict[str, Any], step: int, output_path: str) -> str:
    """
    Generate bar chart visualization for rotation robustness evaluation.

    Args:
        rollout_data: Dictionary containing:
            - rotation_assignments: [500, 5] tensor with rotation parameters
            - episode_succeeded: [500] boolean tensor
            - termination_step: [500] int tensor
            - truncation_step: [500] int tensor
        step: Training step number (for title)
        output_path: Path to save the visualization

    Returns:
        Path to the saved visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    print(f"  Generating rotation visualization...")

    # Extract data
    rotation_assignments = rollout_data['rotation_assignments'].cpu().numpy()  # [500, 5]
    episode_succeeded = rollout_data['episode_succeeded'].cpu().numpy()  # [500]

    # Compute success rate for each rotation angle
    success_rates = []
    for angle_idx, angle_deg in enumerate(ROTATION_ANGLES_DEG):
        start_idx = angle_idx * ENVS_PER_ROTATION_ANGLE
        end_idx = start_idx + ENVS_PER_ROTATION_ANGLE
        success_rate = episode_succeeded[start_idx:end_idx].mean()
        success_rates.append(success_rate)

    # Create bar chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # X positions and labels
    x_positions = np.arange(len(ROTATION_ANGLES_DEG))
    x_labels = [f"{angle}°" for angle in ROTATION_ANGLES_DEG]

    # Color bars based on success rate using Red-Yellow-Green colormap
    colormap = cm.get_cmap('RdYlGn')
    colors = [colormap(rate) for rate in success_rates]

    # Create bars
    bars = ax.bar(x_positions, success_rates, color=colors, edgecolor='black', linewidth=1.5, alpha=0.9)

    # Customize plot
    ax.set_xlabel('Rotation Angle', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=14, fontweight='bold')
    ax.set_title(f'Rotation Robustness at Step {step}', fontsize=16, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of each bar
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate*100:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Save figure
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    print(f"    Saved rotation visualization to: {output_path}")
    return output_path


def evaluate_checkpoint(run: wandb.Run, step: int, env: Any, agent: Any,
                       configs: Dict[str, Any], args: argparse.Namespace,
                       max_rollout_steps: int, policy_hz: float,
                       noise_assignments: Optional[torch.Tensor] = None,
                       rotation_assignments: Optional[torch.Tensor] = None,
                       gain_assignments: Optional[torch.Tensor] = None,
                       friction_assignments: Optional[torch.Tensor] = None,
                       mass_assignments: Optional[torch.Tensor] = None) -> Tuple[Dict[str, float], Optional[str]]:
    """
    Evaluate a single checkpoint.

    Args:
        run: WandB run object
        step: Checkpoint step number
        env: Environment instance
        agent: Agent instance
        configs: Configuration dictionary
        args: Command-line arguments
        max_rollout_steps: Maximum rollout steps
        policy_hz: Policy frequency

    Returns:
        Tuple of (metrics_dict, video_path)

    Raises:
        RuntimeError: If checkpoint loading or evaluation fails
    """
    print(f"  Evaluating checkpoint at step {step}...")

    # Download checkpoint pair from WandB
    policy_path, critic_path = download_checkpoint_pair(run, step)

    try:
        # Import required modules for checkpoint loading
        from models.SimBa import SimBaNet
        from models.block_simba import pack_agents_into_block
        from agents.block_ppo import PerAgentPreprocessorWrapper

        # Load checkpoints into agent models
        print(f"    Loading policy checkpoint: {policy_path}")
        policy_checkpoint = torch.load(policy_path, map_location=env.unwrapped.device, weights_only=False)

        # Verify policy checkpoint has required fields
        if 'net_state_dict' not in policy_checkpoint:
            raise RuntimeError("Policy checkpoint missing 'net_state_dict'")

        print(f"    Loading critic checkpoint: {critic_path}")
        critic_checkpoint = torch.load(critic_path, map_location=env.unwrapped.device, weights_only=False)

        # Verify critic checkpoint has required fields
        if 'net_state_dict' not in critic_checkpoint:
            raise RuntimeError("Critic checkpoint missing 'net_state_dict'")

        # Create single-agent SimBaNet models for policy and critic
        agent_idx = 0  # We're loading into agent slot 0

        print(f"      Creating SimBaNet for policy...")
        policy_agent = SimBaNet(
            n=len(agent.policy.actor_mean.resblocks),
            in_size=agent.policy.actor_mean.obs_dim,
            out_size=agent.policy.actor_mean.act_dim,
            latent_size=agent.policy.actor_mean.hidden_dim,
            device=agent.device,
            tan_out=agent.policy.actor_mean.use_tanh
        )
        policy_agent.load_state_dict(policy_checkpoint['net_state_dict'])
        print(f"      Loaded policy SimBaNet state dict")

        print(f"      Creating SimBaNet for critic...")
        critic_agent = SimBaNet(
            n=len(agent.value.critic.resblocks),
            in_size=agent.value.critic.obs_dim,
            out_size=agent.value.critic.act_dim,
            latent_size=agent.value.critic.hidden_dim,
            device=agent.device,
            tan_out=agent.value.critic.use_tanh
        )
        critic_agent.load_state_dict(critic_checkpoint['net_state_dict'])
        print(f"      Loaded critic SimBaNet state dict")

        # Pack single-agent models into block models at agent_idx=0
        print(f"      Packing policy into BlockSimBa at agent index {agent_idx}...")
        pack_agents_into_block(agent.policy.actor_mean, {agent_idx: policy_agent})

        print(f"      Packing critic into BlockSimBa at agent index {agent_idx}...")
        pack_agents_into_block(agent.value.critic, {agent_idx: critic_agent})

        # Load log_std for policy
        if 'log_std' in policy_checkpoint:
            agent.policy.actor_logstd[agent_idx].data.copy_(policy_checkpoint['log_std'].data)
            print(f"      Loaded log_std for agent {agent_idx}")
        else:
            print(f"      WARNING: No log_std found in policy checkpoint")

        # Load state preprocessor
        if 'state_preprocessor' in policy_checkpoint:
            if not hasattr(agent, '_per_agent_state_preprocessors'):
                agent._per_agent_state_preprocessors = [None] * agent.num_agents

            if agent._per_agent_state_preprocessors[agent_idx] is None:
                from skrl.resources.preprocessors.torch import RunningStandardScaler
                obs_size = policy_checkpoint['state_preprocessor']['running_mean'].shape[0]
                agent._per_agent_state_preprocessors[agent_idx] = RunningStandardScaler(
                    size=obs_size,
                    device=agent.device
                )

            agent._per_agent_state_preprocessors[agent_idx].load_state_dict(policy_checkpoint['state_preprocessor'])

            preprocessor = agent._per_agent_state_preprocessors[agent_idx]
            mean_avg = preprocessor.running_mean.mean().item()
            mean_std = preprocessor.running_mean.std().item()
            var_avg = preprocessor.running_variance.mean().item()
            count = preprocessor.current_count.item()
            print(f"      Loaded state preprocessor: mean_avg={mean_avg:.4f}, mean_std={mean_std:.4f}, var_avg={var_avg:.4f}, count={count}")
        else:
            raise RuntimeError("Policy checkpoint missing 'state_preprocessor'")

        # Load value preprocessor
        if 'value_preprocessor' in critic_checkpoint:
            if not hasattr(agent, '_per_agent_value_preprocessors'):
                agent._per_agent_value_preprocessors = [None] * agent.num_agents

            if agent._per_agent_value_preprocessors[agent_idx] is None:
                from skrl.resources.preprocessors.torch import RunningStandardScaler
                agent._per_agent_value_preprocessors[agent_idx] = RunningStandardScaler(
                    size=1,
                    device=agent.device
                )

            agent._per_agent_value_preprocessors[agent_idx].load_state_dict(critic_checkpoint['value_preprocessor'])

            preprocessor = agent._per_agent_value_preprocessors[agent_idx]
            mean_val = preprocessor.running_mean.item()
            var_val = preprocessor.running_variance.item()
            count = preprocessor.current_count.item()
            print(f"      Loaded value preprocessor: mean={mean_val:.4f}, var={var_val:.4f}, count={count}")
        else:
            raise RuntimeError("Critic checkpoint missing 'value_preprocessor'")

        # Wrap preprocessors for SKRL compatibility
        print(f"      Wrapping preprocessors for SKRL...")
        if hasattr(agent, '_per_agent_state_preprocessors') and agent._per_agent_state_preprocessors:
            agent._state_preprocessor = PerAgentPreprocessorWrapper(agent, agent._per_agent_state_preprocessors)
        else:
            print("      WARNING: No state preprocessors to wrap")

        if hasattr(agent, '_per_agent_value_preprocessors') and agent._per_agent_value_preprocessors:
            agent._value_preprocessor = PerAgentPreprocessorWrapper(agent, agent._per_agent_value_preprocessors)
        else:
            print("      WARNING: No value preprocessors to wrap")

        # Verify loading
        sample_value = list(policy_checkpoint['net_state_dict'].values())[0].flatten()[0].item()
        print(f"      Checkpoint loading verified (sample weight: {sample_value:.4f})")

        # Set agent to eval mode
        agent.set_running_mode("eval")
        print(f"    Checkpoint loading complete!")

        # Branch based on eval mode
        media_path = None
        if args.eval_mode == "performance":
            # Run basic evaluation with video
            metrics, rollout_data = run_basic_evaluation(
                env=env,
                agent=agent,
                configs=configs,
                max_rollout_steps=max_rollout_steps,
                enable_video=args.enable_video,
                show_progress=args.show_progress,
                eval_seed=args.eval_seed
            )

            # Generate video if enabled and rollout data is available
            if args.enable_video and rollout_data is not None:
                # Create temp video path
                fd, media_path = tempfile.mkstemp(suffix=".gif", prefix=f"eval_step_{step}_")
                os.close(fd)

                media_path = generate_evaluation_video(
                    rollout_data=rollout_data,
                    agent=agent,
                    policy_hz=policy_hz,
                    output_path=media_path
                )
            else:
                media_path = None

            # Add category-based prefixes for WandB namespacing
            eval_metrics = prefix_metrics_by_category(
                metrics,
                core_prefix="Eval_Core",
                contact_prefix="Eval_Contact"
            )

        elif args.eval_mode == "noise":
            # Run noise evaluation (no video)
            if noise_assignments is None:
                raise RuntimeError("noise_assignments is None in noise mode evaluation")

            eval_metrics, rollout_data = run_noise_evaluation(
                env=env,
                agent=agent,
                configs=configs,
                max_rollout_steps=max_rollout_steps,
                noise_assignments=noise_assignments,
                show_progress=args.show_progress,
                eval_seed=args.eval_seed
            )

            # Visualization disabled - keep function for potential future use
            # fd, media_path = tempfile.mkstemp(suffix=".png", prefix=f"noise_eval_step_{step}_")
            # os.close(fd)
            #
            # media_path = generate_noise_visualization(
            #     rollout_data=rollout_data,
            #     step=step,
            #     output_path=media_path
            # )
            media_path = None

        elif args.eval_mode == "rotation":
            # Run rotation evaluation (no video)
            if rotation_assignments is None:
                raise RuntimeError("rotation_assignments is None in rotation mode evaluation")

            eval_metrics, rollout_data = run_rotation_evaluation(
                env=env,
                agent=agent,
                configs=configs,
                max_rollout_steps=max_rollout_steps,
                rotation_assignments=rotation_assignments,
                show_progress=args.show_progress,
                eval_seed=args.eval_seed
            )

            # Visualization disabled - keep function for potential future use
            # fd, media_path = tempfile.mkstemp(suffix=".png", prefix=f"rotation_eval_step_{step}_")
            # os.close(fd)
            #
            # media_path = generate_rotation_visualization(
            #     rollout_data=rollout_data,
            #     step=step,
            #     output_path=media_path
            # )
            media_path = None

        elif args.eval_mode == "gain":
            # Run gain evaluation (no video)
            if gain_assignments is None:
                raise RuntimeError("gain_assignments is None in gain mode evaluation")

            eval_metrics, rollout_data = run_gain_evaluation(
                env=env,
                agent=agent,
                configs=configs,
                max_rollout_steps=max_rollout_steps,
                gain_assignments=gain_assignments,
                show_progress=args.show_progress,
                eval_seed=args.eval_seed
            )

            # No visualization for gain mode
            media_path = None

        elif args.eval_mode == "dynamics":
            # Run dynamics evaluation (no video)
            if friction_assignments is None or mass_assignments is None:
                raise RuntimeError("friction_assignments or mass_assignments is None in dynamics mode evaluation")

            eval_metrics, rollout_data = run_dynamics_evaluation(
                env=env,
                agent=agent,
                configs=configs,
                max_rollout_steps=max_rollout_steps,
                friction_assignments=friction_assignments,
                mass_assignments=mass_assignments,
                show_progress=args.show_progress,
                eval_seed=args.eval_seed
            )

            # No visualization for dynamics mode
            media_path = None

        elif args.eval_mode == "trajectory":
            # Run trajectory evaluation
            trajectory_data = run_trajectory_evaluation(
                env=env,
                agent=agent,
                configs=configs,
                max_rollout_steps=max_rollout_steps,
                show_progress=args.show_progress,
                eval_seed=args.eval_seed
            )

            # Save trajectory data to .pkl file in run-specific subdirectory
            run_output_dir = os.path.join(args.traj_output_dir, run.id)
            os.makedirs(run_output_dir, exist_ok=True)
            media_path = os.path.join(run_output_dir, f"traj_{step}.pkl")
            save_trajectory_data(trajectory_data, media_path)
            print(f"    Saved trajectory data to: {media_path}")

            # Minimal eval_metrics for trajectory mode (data is in the .pkl file)
            eval_metrics = {
                "Traj_Eval/total_rollouts": trajectory_data["metadata"]["total_rollouts"],
                "Traj_Eval/total_policy_steps": trajectory_data["metadata"]["total_policy_steps"],
                "Traj_Eval/total_breaks": trajectory_data["metadata"]["total_breaks"],
            }

        else:
            raise RuntimeError(f"Unknown eval_mode: {args.eval_mode}")

        print(f"  Checkpoint {step} evaluation complete")

        # Print summary (different for each mode)
        if args.eval_mode == "performance":
            # Compute success rate from actual metrics
            total = eval_metrics.get('Eval_Core/total_episodes', 0)
            successes = eval_metrics.get('Eval_Core/num_successful_completions', 0)
            success_rate = successes / total if total > 0 else 0.0
            print(f"    Success rate: {success_rate:.2%} ({successes}/{total})")
        elif args.eval_mode == "noise":
            # Print summary for each noise range
            for min_val, max_val, range_name in NOISE_RANGES:
                total = eval_metrics.get(f'Noise_Eval({range_name})_Core/total_episodes', 0)
                successes = eval_metrics.get(f'Noise_Eval({range_name})_Core/num_successful_completions', 0)
                success_rate = successes / total if total > 0 else 0.0
                print(f"    {range_name}: {success_rate:.2%} ({successes}/{total})")
        elif args.eval_mode == "rotation":
            # Print summary for each rotation angle
            for angle_deg in ROTATION_ANGLES_DEG:
                total = eval_metrics.get(f'Rot_Eval({angle_deg}deg)_Core/total_episodes', 0)
                successes = eval_metrics.get(f'Rot_Eval({angle_deg}deg)_Core/num_successful_completions', 0)
                success_rate = successes / total if total > 0 else 0.0
                print(f"    {angle_deg}deg: {success_rate:.2%} ({successes}/{total})")
        elif args.eval_mode == "gain":
            # Print summary for each gain multiplier
            for multiplier in GAIN_MULTIPLIERS:
                total = eval_metrics.get(f'Gain_Eval({multiplier}x)_Core/total_episodes', 0)
                successes = eval_metrics.get(f'Gain_Eval({multiplier}x)_Core/num_successful_completions', 0)
                success_rate = successes / total if total > 0 else 0.0
                print(f"    {multiplier}x: {success_rate:.2%} ({successes}/{total})")
        elif args.eval_mode == "dynamics":
            # Print summary for each friction/mass combination
            for fric_mult in DYNAMICS_FRICTION_MULTIPLIERS:
                for mass_mult in DYNAMICS_MASS_MULTIPLIERS:
                    combo_name = f"fric={fric_mult}x,mass={mass_mult}x"
                    total = eval_metrics.get(f'Dyn_Eval({combo_name})_Core/total_episodes', 0)
                    successes = eval_metrics.get(f'Dyn_Eval({combo_name})_Core/num_successful_completions', 0)
                    success_rate = successes / total if total > 0 else 0.0
                    print(f"    ({combo_name}): {success_rate:.2%} ({successes}/{total})")
        elif args.eval_mode == "trajectory":
            # Print summary for trajectory mode
            total_rollouts = eval_metrics.get('Traj_Eval/total_rollouts', 0)
            total_steps = eval_metrics.get('Traj_Eval/total_policy_steps', 0)
            total_breaks = eval_metrics.get('Traj_Eval/total_breaks', 0)
            print(f"    Rollouts: {total_rollouts}, Policy steps: {total_steps}, Breaks: {total_breaks}")

        return eval_metrics, media_path

    finally:
        # Clean up downloaded checkpoint files
        if os.path.exists(policy_path):
            download_dir = os.path.dirname(os.path.dirname(policy_path))
            shutil.rmtree(download_dir, ignore_errors=True)


# ===== MAIN EXECUTION =====

def _run_parallel_evaluation(runs: List[wandb.Run], configs: Dict[str, Any]):
    """
    Run evaluation in parallel mode: all runs evaluated simultaneously.

    Checkpoint loop is outermost, run loop is innermost (for logging only).
    Environment has num_agents = num_runs, each agent slot gets one run's checkpoint.
    """
    num_runs = len(runs)
    print(f"\n=== PARALLEL MODE: Evaluating {num_runs} runs simultaneously ===")

    # Setup environment with multiple agents
    print("\nSetting up environment with multiple agents...")
    result = setup_environment_once(
        configs=configs,
        args=args_cli,
        num_runs=num_runs,
        parallel_mode=True
    )
    env, agent, max_rollout_steps, policy_hz, noise_assignments, rotation_assignments, \
        gain_assignments, friction_assignments, mass_assignments, envs_per_agent = result

    print(f"\nEnvironment configured (parallel mode):")
    print(f"  - Number of runs/agents: {num_runs}")
    print(f"  - Environments per agent: {envs_per_agent}")
    print(f"  - Total environments: {num_runs * envs_per_agent}")
    print(f"  - Max rollout steps: {max_rollout_steps}")
    print(f"  - Policy Hz: {policy_hz}")

    # Get checkpoint steps from representative run (all runs have same checkpoints)
    representative_run = runs[0]
    checkpoint_steps = get_checkpoint_steps(representative_run, args_cli)
    print(f"\nCheckpoints to evaluate: {checkpoint_steps}")

    # Initialize all WandB eval runs upfront (using reinit="create_new" for concurrent runs)
    eval_wandb_runs = []  # List of (run, wandb_run_object) tuples
    if not args_cli.no_wandb and not args_cli.report_to_base_run and args_cli.eval_mode != "trajectory":
        if args_cli.debug_project:
            print(f"\n[DEBUG MODE] Initializing {num_runs} WandB eval runs in 'debug' project...")
        else:
            print(f"\nInitializing {num_runs} WandB eval runs...")
        for run_idx, run in enumerate(runs):
            try:
                eval_tags = [f"eval_{args_cli.eval_mode}", f"source_run:{run.id}"] + list(run.tags)

                if args_cli.debug_project:
                    target_project = "debug"
                else:
                    target_project = run.project

                eval_run_name = f"Eval_{args_cli.eval_mode}_{run.name}"
                eval_group_name = f"Eval_{args_cli.eval_mode}_{run.group}_{args_cli.tag}" if run.group else None

                wandb_run = wandb.init(
                    project=target_project,
                    entity=args_cli.entity,
                    name=eval_run_name,
                    group=eval_group_name,
                    tags=eval_tags,
                    reinit="create_new",  # Key: allows multiple concurrent runs
                    config={
                        "source_run_id": run.id,
                        "source_run_name": run.name,
                        "source_run_group": run.group,
                        "source_project": run.project,
                        "eval_mode": args_cli.eval_mode,
                        "eval_seed": args_cli.eval_seed,
                        "num_checkpoints": len(checkpoint_steps),
                        "parallel_mode": True,
                    }
                )
                eval_wandb_runs.append((run, wandb_run))
                print(f"  Initialized eval run {run_idx + 1}/{num_runs}: {eval_run_name}")
            except Exception as e:
                print(f"  WARNING: Failed to initialize eval run for {run.id}: {e}")
                # Continue without this run's wandb logging
                eval_wandb_runs.append((run, None))
    else:
        # No wandb logging - just track runs without wandb objects
        eval_wandb_runs = [(run, None) for run in runs]

    # Checkpoint loop (OUTERMOST)
    for step_idx, step in enumerate(checkpoint_steps):
        print(f"\n{'=' * 80}")
        print(f"Checkpoint {step_idx + 1}/{len(checkpoint_steps)}: step {step}")
        print(f"{'=' * 80}")

        # Load all checkpoints in parallel
        download_dirs = load_checkpoints_parallel(runs, step, env, agent)

        try:
            # Run ONE evaluation (covers all agents at once)
            # For parallel mode, always use run_basic_evaluation to get full rollout_data
            # The mode-specific metric computation happens in compute_run_metrics
            if args_cli.eval_mode == "trajectory":
                # Trajectory mode needs special handling - run and save per agent
                trajectory_data = run_trajectory_evaluation(
                    env=env,
                    agent=agent,
                    configs=configs,
                    max_rollout_steps=max_rollout_steps,
                    show_progress=args_cli.show_progress,
                    eval_seed=args_cli.eval_seed
                )
                # For trajectory mode, we need to handle saving per-agent
                # This is more complex, so for now just save combined data
                for run_idx, run in enumerate(runs):
                    run_output_dir = os.path.join(args_cli.traj_output_dir, run.id)
                    os.makedirs(run_output_dir, exist_ok=True)
                    media_path = os.path.join(run_output_dir, f"traj_{step}.pkl")
                    save_trajectory_data(trajectory_data, media_path)
                    print(f"    Saved trajectory data for {run.id} to: {media_path}")
                continue  # Skip the per-run metrics computation for trajectory mode
            elif args_cli.eval_mode in ("performance", "noise", "rotation", "gain", "dynamics"):
                # All standard modes use run_basic_evaluation in parallel mode
                # This returns full rollout_data that can be split by agent
                eval_metrics, rollout_data = run_basic_evaluation(
                    env=env,
                    agent=agent,
                    configs=configs,
                    max_rollout_steps=max_rollout_steps,
                    enable_video=False,  # No video in parallel mode
                    show_progress=args_cli.show_progress,
                    eval_seed=args_cli.eval_seed
                )
            else:
                raise RuntimeError(f"Unknown eval_mode: {args_cli.eval_mode}")

            print(f"  Evaluation complete, splitting results by agent...")

            # Split rollout data by agent
            per_agent_data = split_rollout_data_by_agent(rollout_data, num_runs, envs_per_agent)

            # Compute and log metrics for each run (INNERMOST loop)
            for run_idx, (run, wandb_run) in enumerate(eval_wandb_runs):
                print(f"\n  Processing results for run {run_idx + 1}/{num_runs}: {run.id}")

                # Compute metrics for this run's slice
                run_metrics = compute_run_metrics(
                    rollout_data=per_agent_data[run_idx],
                    eval_mode=args_cli.eval_mode
                )

                # Add standard metrics
                run_metrics['eval_seed'] = args_cli.eval_seed
                run_metrics['total_steps'] = step
                run_metrics['env_steps'] = step / 256

                # Log to WandB if we have a run object
                if wandb_run is not None:
                    wandb_run.log(run_metrics, step=step, commit=True)
                    print(f"    Logged {len(run_metrics)} metrics to WandB at step {step}")
                elif args_cli.report_to_base_run:
                    # Resume original run, log, and finish (same pattern as sequential)
                    print(f"    [REPORT_TO_BASE_RUN] Logging to original run {run.id}...")
                    try:
                        wandb.init(project=run.project, id=run.id, resume="must")
                        wandb.log(run_metrics, step=step)
                        wandb.finish()
                        print(f"    Logged {len(run_metrics)} metrics to base run at step {step}")
                    except Exception as e:
                        print(f"    WARNING: Failed to log to base run {run.id}: {e}")
                # Print summary
                _print_run_summary(run_metrics, args_cli.eval_mode, run.id)

        finally:
            # Clean up downloaded checkpoint files
            for download_dir in download_dirs:
                if os.path.exists(download_dir):
                    shutil.rmtree(download_dir, ignore_errors=True)

    # Finish all WandB runs
    print(f"\nFinishing {len(eval_wandb_runs)} WandB eval runs...")
    for run, wandb_run in eval_wandb_runs:
        if wandb_run is not None:
            wandb_run.finish()
            print(f"  Finished eval run for {run.id}")


def _print_run_summary(metrics: Dict[str, float], eval_mode: str, run_id: str):
    """Print evaluation summary for a single run."""
    if eval_mode == "performance":
        total = metrics.get('Eval_Core/total_episodes', 0)
        successes = metrics.get('Eval_Core/num_successful_completions', 0)
        success_rate = successes / total if total > 0 else 0.0
        print(f"    [{run_id}] Success rate: {success_rate:.2%} ({successes}/{total})")
    elif eval_mode == "noise":
        for min_val, max_val, range_name in NOISE_RANGES:
            total = metrics.get(f'Noise_Eval({range_name})_Core/total_episodes', 0)
            successes = metrics.get(f'Noise_Eval({range_name})_Core/num_successful_completions', 0)
            success_rate = successes / total if total > 0 else 0.0
            print(f"    [{run_id}] {range_name}: {success_rate:.2%}")
    elif eval_mode == "rotation":
        for angle_deg in ROTATION_ANGLES_DEG:
            total = metrics.get(f'Rot_Eval({angle_deg}deg)_Core/total_episodes', 0)
            successes = metrics.get(f'Rot_Eval({angle_deg}deg)_Core/num_successful_completions', 0)
            success_rate = successes / total if total > 0 else 0.0
            print(f"    [{run_id}] {angle_deg}deg: {success_rate:.2%}")
    elif eval_mode == "gain":
        for multiplier in GAIN_MULTIPLIERS:
            total = metrics.get(f'Gain_Eval({multiplier}x)_Core/total_episodes', 0)
            successes = metrics.get(f'Gain_Eval({multiplier}x)_Core/num_successful_completions', 0)
            success_rate = successes / total if total > 0 else 0.0
            print(f"    [{run_id}] {multiplier}x: {success_rate:.2%}")
    elif eval_mode == "dynamics":
        for fric_mult in DYNAMICS_FRICTION_MULTIPLIERS:
            for mass_mult in DYNAMICS_MASS_MULTIPLIERS:
                combo_name = f"fric={fric_mult}x,mass={mass_mult}x"
                total = metrics.get(f'Dyn_Eval({combo_name})_Core/total_episodes', 0)
                successes = metrics.get(f'Dyn_Eval({combo_name})_Core/num_successful_completions', 0)
                success_rate = successes / total if total > 0 else 0.0
                print(f"    [{run_id}] ({combo_name}): {success_rate:.2%}")


def _run_sequential_evaluation(runs: List[wandb.Run], configs: Dict[str, Any]):
    """
    Run evaluation in sequential mode: one run at a time (existing behavior).

    Used for performance+video mode where memory is constrained.
    """
    print(f"\n=== SEQUENTIAL MODE: Evaluating {len(runs)} runs one at a time ===")

    # Setup environment with 1 agent
    print("\nSetting up environment with single agent...")
    result = setup_environment_once(
        configs=configs,
        args=args_cli,
        num_runs=1,
        parallel_mode=False
    )
    env, agent, max_rollout_steps, policy_hz, noise_assignments, rotation_assignments, \
        gain_assignments, friction_assignments, mass_assignments, envs_per_agent = result

    print(f"\nEnvironment configured (sequential mode):")
    if args_cli.eval_mode == "performance":
        print(f"  - Environments per agent: 100")
    elif args_cli.eval_mode == "noise":
        print(f"  - Environments per agent: {NOISE_MODE_TOTAL_ENVS}")
    elif args_cli.eval_mode == "rotation":
        print(f"  - Environments per agent: {ROTATION_MODE_TOTAL_ENVS}")
    elif args_cli.eval_mode == "gain":
        print(f"  - Environments per agent: {GAIN_MODE_TOTAL_ENVS}")
    elif args_cli.eval_mode == "dynamics":
        print(f"  - Environments per agent: {DYNAMICS_MODE_TOTAL_ENVS}")
    elif args_cli.eval_mode == "trajectory":
        print(f"  - Environments per agent: {TRAJECTORY_MODE_TOTAL_ENVS}")
    print(f"  - Max rollout steps: {max_rollout_steps}")
    print(f"  - Policy Hz: {policy_hz}")

    # Evaluate each run sequentially (existing logic)
    for run_idx, run in enumerate(runs):
        print(f"\n{'=' * 80}")
        print(f"Evaluating run {run_idx + 1}/{len(runs)}: {run.project}/{run.id}")
        print(f"{'=' * 80}")

        # Get checkpoint steps for this run
        checkpoint_steps = get_checkpoint_steps(run, args_cli)

        # Initialize WandB eval run once for all checkpoints
        eval_run_initialized = False
        if not args_cli.no_wandb and not args_cli.report_to_base_run and args_cli.eval_mode != "trajectory":
            try:
                eval_tags = [f"eval_{args_cli.eval_mode}", f"source_run:{run.id}"] + list(run.tags)

                if args_cli.debug_project:
                    target_project = "debug"
                    eval_run_name = f"Eval_{args_cli.eval_mode}_{run.name}"
                    eval_group_name = f"Eval_{args_cli.eval_mode}_{run.group}_{args_cli.tag}" if run.group else None
                    print(f"\n[DEBUG MODE] Initializing eval run in 'debug' project: {eval_run_name}")
                else:
                    target_project = run.project
                    eval_run_name = f"Eval_{args_cli.eval_mode}_{run.name}"
                    eval_group_name = f"Eval_{args_cli.eval_mode}_{run.group}_{args_cli.tag}" if run.group else None
                    print(f"\nInitializing eval run: {eval_run_name}")
                    print(f"  Project: {target_project}")
                    if eval_group_name:
                        print(f"  Group: {eval_group_name}")

                wandb.init(
                    project=target_project,
                    entity=args_cli.entity,
                    name=eval_run_name,
                    group=eval_group_name,
                    tags=eval_tags,
                    config={
                        "source_run_id": run.id,
                        "source_run_name": run.name,
                        "source_run_group": run.group,
                        "source_project": run.project,
                        "eval_mode": args_cli.eval_mode,
                        "eval_seed": args_cli.eval_seed,
                        "num_checkpoints": len(checkpoint_steps),
                    }
                )
                eval_run_initialized = True
                print(f"  Eval run initialized successfully")
            except Exception as e:
                print(f"  WARNING: Failed to initialize eval run: {e}")
                print(f"  Continuing without WandB logging...")
                args_cli.no_wandb = True

        # Evaluate each checkpoint
        for step_idx, step in enumerate(checkpoint_steps):
            print(f"\nCheckpoint {step_idx + 1}/{len(checkpoint_steps)}: step {step}")

            try:
                # Evaluate this checkpoint
                metrics, media_path = evaluate_checkpoint(
                    run, step, env, agent, configs, args_cli,
                    max_rollout_steps, policy_hz, noise_assignments, rotation_assignments, gain_assignments,
                    friction_assignments, mass_assignments
                )

                # Log results to WandB (skip for trajectory mode)
                if args_cli.eval_mode != "trajectory":
                    log_results_to_wandb(run, step, metrics, media_path, args_cli)

            except Exception as e:
                print(f"  ERROR: Failed to evaluate checkpoint at step {step}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Finish eval run after all checkpoints
        if eval_run_initialized:
            print(f"\nFinishing eval run")
            wandb.finish()


def main():
    """Main execution function."""
    print("=" * 80)
    print("WandB Tag-Based Evaluation Script")
    print("=" * 80)

    try:
        # Query runs by tag
        runs = query_runs_by_tag(args_cli.tag, args_cli.entity, args_cli.project, args_cli.run_id)

        # Get representative run for config
        representative_run = runs[0]
        print(f"\nUsing run {representative_run.id} as configuration source")

        # Reconstruct config from WandB
        configs = reconstruct_config_from_wandb(representative_run)

        # Apply CLI overrides if provided
        if args_cli.override:
            print(f"\nApplying {len(args_cli.override)} CLI override(s):")
            for ovr in args_cli.override:
                print(f"  --override {ovr}")

            # Parse and apply overrides directly (ConfigManagerV3.apply_cli_overrides
            # doesn't work here because 'environment' isn't in its section_mapping)
            config_manager = ConfigManagerV3()
            parsed_overrides = config_manager.parse_cli_overrides(args_cli.override)

            # Apply overrides directly to config objects
            for section, section_overrides in parsed_overrides.items():
                if section in configs:
                    config_manager._apply_yaml_overrides(configs[section], section_overrides, indent_level=2)
                    print(f"  Applied overrides to '{section}'")
                else:
                    print(f"  WARNING: Section '{section}' not found in configs")

            print("  Overrides applied successfully")

        # Determine execution mode
        parallel_mode = is_parallel_mode(args_cli)

        if parallel_mode:
            _run_parallel_evaluation(runs, configs)
        else:
            _run_sequential_evaluation(runs, configs)

        print(f"\n{'=' * 80}")
        print("Evaluation complete!")
        print(f"{'=' * 80}")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up simulation
        simulation_app.close()


if __name__ == "__main__":
    main()
