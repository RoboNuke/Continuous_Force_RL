from gymnasium import Env, spaces
import gymnasium as gym
from gymnasium.spaces import Box
import torch
import envs.factory.factory_control as fc
try:
    import isaacsim.core.utils.torch as torch_utils
except ImportError:
    import omni.isaac.core.utils.torch as torch_utils
import numpy as np


class ParallelForcePosActionWrapper(gym.ActionWrapper):
    """
    Use this wrapper changes the task to using a parallel control env
        in order to do this within the IsaacLab + Factory framework
        we override the _apply_action function which is called by IsaacLab
        env at each physics step and we overwrite (if we want to) generate_ctrl_signals
        which is called in factory env 

        This is simple parallel control, we take the error in force and multiply it by a gain
        that theoretically converts it to the equivalent of pose error. This is then added 
        to the change in pose and fed to the initial env.  The logical inverse of the selection
        "matrix" (really a vector) is used to select between force control and pose control in 
        a given dimension 
    """

    def __init__(
            self, 
            env,
            history=1):
        super().__init__(env)
        # get these for factory env
        
        self.old_randomize_initial_state = self.env.unwrapped.randomize_initial_state
        self.env.unwrapped.randomize_initial_state = self.randomize_initial_state
        self.old_update_rew = self.env.unwrapped._update_rew_buf
        self.env.unwrapped._update_rew_buf = self._update_reward_buf

        #self.old_pre_physics_step = self.env.unwrapped._pre_physics_step
        self.env.unwrapped._pre_physics_step = self._pre_physics_step
        self.env.unwrapped._apply_action = self._apply_action
        
        self.act_size = 12 * history

        # previous action is now larger
        #self.unwrapped.cfg.state_space += self.act_size - self.unwrapped.cfg.action_space
        #self.unwrapped.cfg.observation_space += self.act_size - self.unwrapped.cfg.action_space
        self.unwrapped.cfg.action_space = self.act_size

        # reconfigure spaces to match
        self.unwrapped._configure_gym_env_spaces()
        self.cfg_task = self.unwrapped.cfg_task
        self.sel_matrix = torch.zeros((self.unwrapped.num_envs, 3), dtype=bool, device=self.unwrapped.device)
        self.force_action = torch.zeros((self.unwrapped.num_envs, 3), device=self.unwrapped.device)
        self.pos_action = torch.zeros_like(self.force_action)

        # this makes it so that the maximum possible position error is equivalent to the maximum force error
        force_task_gains = [self.unwrapped.cfg.ctrl.pos_action_bounds[i] / self.unwrapped.cfg.ctrl.force_action_bounds[i] for i in range(len(self.unwrapped.cfg.ctrl.pos_action_bounds))]
        self.kp = torch.tensor(force_task_gains, device=self.unwrapped.device).repeat(
            (self.unwrapped.num_envs, 1)
        )
        self.force_threshold = torch.tensor(self.unwrapped.cfg.ctrl.force_action_threshold, device=self.unwrapped.device).repeat(
            (self.unwrapped.num_envs, 1)
        )

        self.force_action_bounds = torch.tensor(self.unwrapped.cfg.ctrl.force_action_bounds, device=self.unwrapped.device)

    def randomize_initial_state(self, env_ids):
        self.old_randomize_initial_state(env_ids)
        # set a random force gain
        # this is setup so if we want to vary the gain we can add it here

    def _calc_ctrl_force(self, min_idx=9, max_idx=12):
        # we interpret the action as a change in current force reading
        # threshold defines the maximum change allowed
        self.force_action = self.unwrapped.actions[:,min_idx:max_idx] * self.force_threshold
        # adding force reading gives us a absolute goal
        self.force_action += self.unwrapped.robot_force_torque[:,:3]
        # clipping keeps in in bounds
        self.force_action = torch.clip(
            self.force_action, -self.force_action_bounds, self.force_action_bounds
        )

    def _update_reward_buf(self, curr_successes):
        rew_buf = self.old_update_rew(curr_successes)
        #print("Prio Rew:", rew_buf.T)
        # reward when selection and force/pos match
        #print(self.unwrapped.robot_force_torque[:2,:3])
        active_force = torch.abs(self.unwrapped.robot_force_torque[:,:3]) > self.cfg_task.force_active_threshold
        #print(active_force[:2,:])
        force_ctrl = self.sel_matrix

        bad_force_ctrl = torch.logical_and(force_ctrl, ~active_force)
        good_force_ctrl = torch.logical_and(force_ctrl, active_force)
        good_pos_ctrl = torch.logical_and(~force_ctrl, ~active_force)
        bad_pos_ctrl = torch.logical_and(~force_ctrl, active_force)

        # get n,3 bools for good and bad
        good_dims = torch.logical_or(good_force_ctrl, good_pos_ctrl)
        good_idxs = torch.any(good_dims, dim=1)
        # get n,1 bools for adding reward
        bad_dims = torch.logical_or(bad_force_ctrl, bad_pos_ctrl)
        bad_idxs  = torch.any(bad_dims, dim=1)

        #print(good_dims.size(), good_idxs.size())
        rew_buf += self.cfg_task.good_force_cmd_rew * torch.sum(good_dims,dim=1)
        rew_buf  += self.cfg_task.bad_force_cmd_rew  * torch.sum(bad_dims, dim=1)
        #print("Final:", rew_buf.T)
        return rew_buf
    
    def _pre_physics_step(self, action):
        self.sel_matrix = action[:,:3] > 0.5
        
        self.unwrapped.extras[f'Hybrid Controller / Force Control X'] = action[:,0]#.mean()
        self.unwrapped.extras[f'Hybrid Controller / Force Control Y'] = action[:,1]#.mean()
        self.unwrapped.extras[f'Hybrid Controller / Force Control Z'] = action[:,2]#.mean()

        env_ids = self.unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.env.unwrapped._reset_buffers(env_ids)
        self.unwrapped.actions = (
            self.unwrapped.cfg.ctrl.ema_factor * action.clone().to(self.unwrapped.device) + (1 - self.unwrapped.cfg.ctrl.ema_factor) * self.unwrapped.actions
        )
        #self.unwrapped.actions[:,:3] = action[:,:3] # set the selector variables to fixed values
        
        # set goal 1 time
        self.env.unwrapped._calc_ctrl_pos(min_idx=3, max_idx=6)
        self.pos_action = self.unwrapped.ctrl_target_fingertip_midpoint_pos.clone()
        self.env.unwrapped._calc_ctrl_quat(min_idx=6, max_idx=9)
        self._calc_ctrl_force()

        # update current values
        self.env.unwrapped._compute_intermediate_values(dt=self.env.unwrapped.physics_dt) 

        # smoothness metrics
        self.unwrapped.ep_ssv += (torch.linalg.norm(self.unwrapped.ee_linvel_fd, axis=1))
        if self.unwrapped.use_ft or self.unwrapped.fragile:
            self.unwrapped.ep_sum_force += torch.linalg.norm(self.unwrapped.robot_force_torque[:,:3], axis=1)
            self.unwrapped.ep_sum_torque += torch.linalg.norm(self.unwrapped.robot_force_torque[:,3:], axis=1)
            self.unwrapped.ep_max_force = torch.max(self.unwrapped.ep_max_force, torch.linalg.norm(self.unwrapped.robot_force_torque[:,:3], axis=1 ))
            self.unwrapped.ep_max_torque = torch.max(self.unwrapped.ep_max_torque, torch.linalg.norm(self.unwrapped.robot_force_torque[:,3:]))

    def action(self, action):
        return action

    def _apply_action(self):
        # Get current yaw for success checking.
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.unwrapped.fingertip_midpoint_quat)
        self.unwrapped.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.unwrapped.last_update_timestamp < self.unwrapped._robot._data._sim_timestamp:
            self.env.unwrapped._compute_intermediate_values(dt=self.unwrapped.physics_dt)

        self.unwrapped.ctrl_target_gripper_dof_pos = 0.0
        
        force_error =  self.sel_matrix * ( self.force_action - self.unwrapped.robot_force_torque[:,:3])
        delta_pos = self.pos_action + self.kp * force_error - self.unwrapped.fixed_pos_action_frame

        pos_error_clipped = torch.clip(
            delta_pos,
            -self.unwrapped.cfg.ctrl.pos_action_bounds[0],
            self.unwrapped.cfg.ctrl.pos_action_bounds[0]
        )
        # set control target
        self.unwrapped.ctrl_target_fingertip_midpoint_pos = pos_error_clipped + self.unwrapped.fixed_pos_action_frame
        self.env.unwrapped.generate_ctrl_signals()
