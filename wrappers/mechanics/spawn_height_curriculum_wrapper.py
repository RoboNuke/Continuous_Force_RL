"""
Spawn Height Curriculum Wrapper

This wrapper implements bidirectional curriculum learning for spawn height difficulty:
- Increases difficulty (raises min spawn height) when agent succeeds
- Decreases difficulty (lowers min spawn height) when agent struggles
- Per-agent curriculum progression based on rollout success rates
- Uses full factory environment randomization logic (robot IK + held object positioning)
"""

import torch
import gymnasium as gym
import numpy as np
from typing import Dict, Any

# Import Isaac Lab utilities
try:
    import omni.isaac.lab.utils.torch as torch_utils
    import omni.isaac.lab.sim as sim_utils
    import isaaclab_tasks.direct.factory.factory_control as factory_utils
except ImportError:
    try:
        import isaacsim.core.utils.torch as torch_utils
        import isaaclab.sim as sim_utils
        import isaaclab_tasks.direct.factory.factory_control as factory_utils
    except ImportError:
        raise ImportError("Could not import Isaac Lab utilities")

# Carb for gravity
import carb


class SpawnHeightCurriculumWrapper(gym.Wrapper):
    """
    Wrapper that implements bidirectional curriculum learning for spawn height.

    Features:
    - Per-agent minimum height tracking
    - Bidirectional adjustment: progress (raise min) or regress (lower min)
    - Rollout-based success rate evaluation
    - Full factory env randomization (robot + IK + held object)
    """

    def __init__(self, env, config: Dict[str, Any], num_agents: int):
        """
        Initialize the spawn height curriculum wrapper.

        Args:
            env: Base environment to wrap
            config: Configuration dictionary with curriculum parameters
            num_agents: Number of agents for per-agent curriculum tracking
        """
        super().__init__(env)

        self.config = config
        self.enabled = config.get('enabled', False)

        if not self.enabled:
            return

        # Environment info
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # Agent configuration
        self.num_agents = num_agents
        self.envs_per_agent = self.num_envs // self.num_agents

        # Validate num_agents
        if self.num_envs % self.num_agents != 0:
            raise ValueError(
                f"Number of environments ({self.num_envs}) must be divisible by num_agents ({self.num_agents})"
            )

        # Curriculum parameters
        self.progress_threshold = config.get('progress_threshold', 0.80)
        self.regress_threshold = config.get('regress_threshold', 0.50)
        self.progress_height_delta = config.get('progress_height_delta', 0.01)
        self.regression_height_delta = config.get('regression_height_delta', 0.02)
        self.config_min_height = config.get('min_height', 0.0)
        self.min_episodes_for_evaluation = config.get('min_episodes_for_evaluation', 10)

        # Validate configuration
        if self.progress_threshold <= self.regress_threshold:
            raise ValueError(
                f"progress_threshold ({self.progress_threshold}) must be > regress_threshold ({self.regress_threshold})"
            )

        # Per-environment minimum heights (vectorized for efficiency)
        # Initialize all environments to config min_height
        self.min_heights = torch.full((self.num_envs,), self.config_min_height, device=self.device, dtype=torch.float32)

        # XY threshold for centered spawning (when held object below fixed tip)
        self.xy_threshold = 0.0025  # Same as is_centered threshold in two-stage wrapper

        # Store original randomize_initial_state method
        self._original_randomize_initial_state = None
        self._wrapper_initialized = False

        # Initialize wrapper if environment is ready
        if hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        print(f"[SpawnHeightCurriculumWrapper] Initialized with {self.num_agents} agents")
        print(f"  Progress threshold: {self.progress_threshold}, Regress threshold: {self.regress_threshold}")
        print(f"  Height deltas: +{self.progress_height_delta}m (progress), -{self.regression_height_delta}m (regress)")
        print(f"  Min height: {self.config_min_height}m")

    def _find_factory_metrics_wrapper(self):
        """Find FactoryMetricsWrapper in the wrapper chain."""
        current = self.env
        while hasattr(current, 'env'):
            if hasattr(current, 'agent_episode_data') and hasattr(current, 'num_agents'):
                return current
            current = current.env
        return None

    def _initialize_wrapper(self):
        """Initialize the wrapper after the base environment is set up."""
        if self._wrapper_initialized or not self.enabled:
            return

        # Store and override randomize_initial_state method
        if hasattr(self.unwrapped, 'randomize_initial_state'):
            self._original_randomize_initial_state = self.unwrapped.randomize_initial_state
            self.unwrapped.randomize_initial_state = self._curriculum_randomize_initial_state
        else:
            raise ValueError("Environment missing required randomize_initial_state method")

        self._wrapper_initialized = True
        print("[SpawnHeightCurriculumWrapper] Wrapper initialized")

    def _curriculum_randomize_initial_state(self, env_ids):
        """
        Curriculum-aware version of factory_env's randomize_initial_state.

        Copied from factory_env.py with modification to height sampling.
        """
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1.) Randomize fixed asset pose.
        fixed_state = self.unwrapped._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.unwrapped.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.unwrapped.scene.env_origins[env_ids]
        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.unwrapped.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.unwrapped.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        self.unwrapped._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self.unwrapped._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self.unwrapped._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(
            self.unwrapped.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device
        )
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.unwrapped.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.unwrapped.step_sim_no_action()

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.unwrapped.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.unwrapped.cfg_task.fixed_asset_cfg.base_height
        if self.unwrapped.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self.unwrapped.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.unwrapped.fixed_quat,
            self.unwrapped.fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )
        self.unwrapped.fixed_pos_obs_frame[:] = fixed_tip_pos

        # (2) Move gripper to randomized location above fixed asset. Keep trying until IK succeeds.
        bad_envs = env_ids.clone()
        ik_attempt = 0

        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        while True:
            n_bad = bad_envs.shape[0]

            above_fixed_pos = fixed_tip_pos.clone()

            # === CURRICULUM MODIFICATION: Sample height from per-env range (vectorized) ===
            max_height = self.unwrapped.cfg_task.hand_init_pos[2] + above_fixed_pos[bad_envs,2]

            # Get min heights for bad_envs
            min_heights_bad = self.min_heights[bad_envs]
            #print(min_heights_bad)
            # Sample heights uniformly from [min_height, max_height] for each env
            height_ranges = max_height - min_heights_bad
            rand_factors = torch.rand(len(bad_envs), device=self.device)
            sampled_heights = min_heights_bad + rand_factors * height_ranges
            
            above_fixed_pos[bad_envs, 2] = sampled_heights
            # Edge case: When held object would be below fixed tip, force centered XY spawn
            # This happens when sampled_height is small (gripper near/below fixed tip)
            # Check which envs need centered spawning (held object below fixed tip)
            held_below_fixed = sampled_heights < 0.026  # Gripper below fixed tip means held object definitely below
            
            if torch.any(held_below_fixed):
                # Get the bad_envs that need centered spawning
                centered_env_mask = held_below_fixed
                centered_bad_indices = torch.where(centered_env_mask)[0]
                centered_bad_envs = bad_envs[centered_bad_indices]

                # Force XY position to be within xy_threshold of fixed tip
                # Generate random position within circular region of radius xy_threshold
                n_centered = len(centered_bad_envs)
                random_angles = torch.rand(n_centered, device=self.device) * 2 * 3.14159
                random_radii = torch.rand(n_centered, device=self.device) * self.xy_threshold * 0.9  # 90% of threshold

                offset_x = random_radii * torch.cos(random_angles)
                offset_y = random_radii * torch.sin(random_angles)

                # Apply centered offsets (override XY, keep Z from curriculum)
                above_fixed_pos[centered_bad_envs, 0] = fixed_tip_pos[centered_bad_envs, 0] + offset_x
                above_fixed_pos[centered_bad_envs, 1] = fixed_tip_pos[centered_bad_envs, 1] + offset_y
            # === END CURRICULUM MODIFICATION ===

            # Apply position noise (for non-centered envs, and XY noise for centered envs)
            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_pos_rand = torch.tensor(self.unwrapped.cfg_task.hand_init_pos_noise, device=self.device)
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)

            # Only apply XY noise to non-centered envs (centered envs already have controlled XY)
            if torch.any(held_below_fixed):
                above_fixed_pos_rand[centered_bad_indices, 0:2] = 0.0  # No XY noise for centered spawns

            above_fixed_pos[bad_envs] += above_fixed_pos_rand
            
            # (b) get random orientation facing down
            hand_down_euler = (
                torch.tensor(self.unwrapped.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            )

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_orn_rand = torch.tensor(self.unwrapped.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )
            # (c) iterative IK Method
            self.unwrapped.ctrl_target_fingertip_midpoint_pos[bad_envs, ...] = above_fixed_pos[bad_envs, ...]
            self.unwrapped.ctrl_target_fingertip_midpoint_quat[bad_envs, ...] = hand_down_quat[bad_envs, :]

            pos_error, aa_error = self.unwrapped.set_pos_inverse_kinematics(env_ids=bad_envs)
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            angle_error = torch.norm(aa_error, dim=1) > 1e-3
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            # Check IK succeeded for all envs, otherwise try again for those envs
            if bad_envs.shape[0] == 0:
                break

            self.unwrapped._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], env_ids=bad_envs
            )

            ik_attempt += 1
            print(f"\t[DEBUG]: IK Attempt: {ik_attempt} \tRemaining: {bad_envs.shape[0]}")

        self.unwrapped.step_sim_no_action()

        # Add flanking gears after servo (so arm doesn't move them).
        if self.unwrapped.cfg_task.name == "gear_mesh" and self.unwrapped.cfg_task.add_flanking_gears:
            small_gear_state = self.unwrapped._small_gear_asset.data.default_root_state.clone()[env_ids]
            small_gear_state[:, 0:7] = fixed_state[:, 0:7]
            small_gear_state[:, 7:] = 0.0  # vel
            self.unwrapped._small_gear_asset.write_root_pose_to_sim(small_gear_state[:, 0:7], env_ids=env_ids)
            self.unwrapped._small_gear_asset.write_root_velocity_to_sim(small_gear_state[:, 7:], env_ids=env_ids)
            self.unwrapped._small_gear_asset.reset()

            large_gear_state = self.unwrapped._large_gear_asset.data.default_root_state.clone()[env_ids]
            large_gear_state[:, 0:7] = fixed_state[:, 0:7]
            large_gear_state[:, 7:] = 0.0  # vel
            self.unwrapped._large_gear_asset.write_root_pose_to_sim(large_gear_state[:, 0:7], env_ids=env_ids)
            self.unwrapped._large_gear_asset.write_root_velocity_to_sim(large_gear_state[:, 7:], env_ids=env_ids)
            self.unwrapped._large_gear_asset.reset()

        # (3) Randomize asset-in-gripper location.
        # flip gripper z orientation
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.unwrapped.device).unsqueeze(0).repeat(self.unwrapped.num_envs, 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.unwrapped.fingertip_midpoint_quat,
            t1=self.unwrapped.fingertip_midpoint_pos,
            q2=flip_z_quat,
            t2=torch.zeros_like(self.unwrapped.fingertip_midpoint_pos),
        )

        # get default gripper in asset transform
        held_asset_relative_pos, held_asset_relative_quat = self.unwrapped.get_handheld_asset_relative_pose()
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        # Add asset in hand randomization
        rand_sample = torch.rand((self.unwrapped.num_envs, 3), dtype=torch.float32, device=self.unwrapped.device)
        self.unwrapped.held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
        if self.unwrapped.cfg_task.name == "gear_mesh":
            self.unwrapped.held_asset_pos_noise[:, 2] = -rand_sample[:, 2]  # [-1, 0]

        held_asset_pos_noise = torch.tensor(self.unwrapped.cfg_task.held_asset_pos_noise, device=self.unwrapped.device)
        self.unwrapped.held_asset_pos_noise = self.unwrapped.held_asset_pos_noise @ torch.diag(held_asset_pos_noise)
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=self.unwrapped.identity_quat,
            t2=self.unwrapped.held_asset_pos_noise,
        )

        held_state = self.unwrapped._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.unwrapped.scene.env_origins
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0
        self.unwrapped._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self.unwrapped._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self.unwrapped._held_asset.reset()

        #  Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.unwrapped.cfg.ctrl.reset_task_prop_gains, device=self.unwrapped.device).repeat(
            (self.unwrapped.num_envs, 1)
        )
        reset_rot_deriv_scale = self.unwrapped.cfg.ctrl.reset_rot_deriv_scale
        self.unwrapped._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

        self.unwrapped.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.unwrapped.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.unwrapped.ctrl_target_gripper_dof_pos = 0.0
            self.unwrapped.close_gripper_in_place()
            self.unwrapped.step_sim_no_action()
            grasp_time += self.unwrapped.sim.get_physics_dt()

        self.unwrapped.prev_joint_pos = self.unwrapped.joint_pos[:, 0:7].clone()
        self.unwrapped.prev_fingertip_pos = self.unwrapped.fingertip_midpoint_pos.clone()
        self.unwrapped.prev_fingertip_quat = self.unwrapped.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.unwrapped.actions = torch.zeros_like(self.unwrapped.actions)
        self.unwrapped.prev_actions = torch.zeros_like(self.unwrapped.actions)
        # Back out what actions should be for initial state.
        # Relative position to bolt tip.
        self.unwrapped.fixed_pos_action_frame[:] = self.unwrapped.fixed_pos_obs_frame + self.unwrapped.init_fixed_pos_obs_noise

        pos_actions = self.unwrapped.fingertip_midpoint_pos - self.unwrapped.fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.unwrapped.cfg.ctrl.pos_action_bounds, device=self.unwrapped.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.unwrapped.actions[:, 0:3] = self.unwrapped.prev_actions[:, 0:3] = pos_actions

        # Relative yaw to bolt.
        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.unwrapped.device).repeat(self.unwrapped.num_envs, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )

        fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.unwrapped.fingertip_midpoint_quat)
        fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )

        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.unwrapped.actions[:, 5] = self.unwrapped.prev_actions[:, 5] = yaw_action

        # Zero initial velocity.
        self.unwrapped.ee_angvel_fd[:, :] = 0.0
        self.unwrapped.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self.unwrapped._set_gains(self.unwrapped.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.unwrapped.cfg.sim.gravity))

    def _update_curriculum(self):
        """
        Update curriculum min heights for all agents based on rollout success rates.

        Called after each rollout when episode data is available.
        """
        metrics_wrapper = self._find_factory_metrics_wrapper()
        if metrics_wrapper is None or metrics_wrapper.last_pubbed_agent_metrics is None:
            print("[Curriculum] Skipping update")
            return

        max_height = self.unwrapped.cfg_task.hand_init_pos[2]

        for agent_id in range(self.num_agents):
            # Get success rates from last rollout
            #agent_data = metrics_wrapper.agent_episode_data[agent_id]
            success_rate = metrics_wrapper.last_pubbed_agent_metrics['Episode/success_rate'][agent_id].item()

            #if not agent_data['success_rates']:
            #    continue

            # Calculate success rate over completed episodes in this rollout
            #num_episodes = len(agent_data['success_rates'])
            #if num_episodes < self.min_episodes_for_evaluation:
            #    continue

            #success_rate = sum(agent_data['success_rates']) / num_episodes

            # Get environment indices for this agent
            start_idx = agent_id * self.envs_per_agent
            end_idx = (agent_id + 1) * self.envs_per_agent
            agent_env_indices = torch.arange(start_idx, end_idx, device=self.device)

            # Get current min height for this agent (all envs in agent should have same value)
            old_min = self.min_heights[start_idx].item()

            # Simple delta adjustment
            if success_rate >= self.progress_threshold:
                # Make harder: increase min_height
                new_min = min(old_min + self.progress_height_delta, max_height)
                # Update all environments for this agent
                self.min_heights[agent_env_indices] = new_min
            elif success_rate < self.regress_threshold:
                # Make easier: decrease min_height
                new_min = max(old_min - self.regression_height_delta, self.config_min_height)
                # Update all environments for this agent
                self.min_heights[agent_env_indices] = new_min
            else:
                # No change
                continue

            print(f"[Curriculum] Agent {agent_id}: min_height {old_min:.3f}m -> {new_min:.3f}m "
                  f"(SR={success_rate:.3f}, range=[{new_min:.3f}, {max_height:.3f}])")

    def step(self, action):
        """Step environment and ensure wrapper is initialized."""
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and update curriculum."""
        # Update curriculum based on previous rollout performance
        print("[DEBUG]: About to call parent reset in curriculum")
        obs, info = super().reset(**kwargs)

        # Initialize wrapper if not done yet
        if not self._wrapper_initialized and hasattr(self.unwrapped, '_robot'):
            self._initialize_wrapper()

        if self.enabled and self._wrapper_initialized:
            self._update_curriculum()
            max_height = self.unwrapped.cfg_task.hand_init_pos[2]
            # Get per-agent min heights (sample first env of each agent)
            agent_min_heights = [self.min_heights[i * self.envs_per_agent].item() for i in range(self.num_agents)]
            info['curriculum'] = {
                'agent_min_heights': agent_min_heights,
                'height_ranges': [(min_h, max_height) for min_h in agent_min_heights]
            }

        return obs, info

    def get_curriculum_stats(self):
        """Get current curriculum statistics."""
        if not self.enabled:
            return {}

        max_height = self.unwrapped.cfg_task.hand_init_pos[2]
        # Get per-agent min heights (sample first env of each agent)
        agent_min_heights = [self.min_heights[i * self.envs_per_agent].item() for i in range(self.num_agents)]
        return {
            'agent_min_heights': agent_min_heights,
            'height_ranges': [(min_h, max_height) for min_h in agent_min_heights],
            'progress_threshold': self.progress_threshold,
            'regress_threshold': self.regress_threshold
        }
