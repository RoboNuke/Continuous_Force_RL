import torch
import gym
from wrappers.factory_wrapper import FactoryWrapper
import envs.factory.factory_control as fc
try:
    import isaacsim.core.utils.torch as torch_utils
except:
    import omni.isaac.core.utils.torch as torch_utils
import numpy as np

class HybridForcePosActionWrapper(gym.Wrapper):#FactoryWrapper):
    """
    Use this wrapper changes the task to using a hybrid control env
        in order to do this within the IsaacLab + Factory framework
        we override the _apply_action function which is called by IsaacLab
        env at each physics step and we overwrite (if we want to) generate_ctrl_signals
        which is called in factory env 

    control law = tau = J^T(S*k_f*(f_d - f) + (1-S) * k_p * (x_d - x))

    """

    def __init__(
            self,
            env,
            reward_type="simp",
            ctrl_torque=False
    ):
        super().__init__(env)
        # have to fix function references
        self.old_pre_physics_step = self.env.unwrapped._pre_physics_step
        self.env.unwrapped._pre_physics_step = self._pre_physics_step

        self.old_apply_action = self.env.unwrapped._apply_action
        self.env.unwrapped._apply_action = self._apply_action

        self.force_size = 6 if ctrl_torque else 3
        
        # set up action space
        self.unwrapped.cfg.action_space = 6 + 2 * self.force_size
        self.unwrapped._configure_gym_env_spaces()

        # get ref to cfgs
        self.cfg_task = self.unwrapped.cfg_task
        self.cfg = self.unwrapped.cfg
        
        # allocate memory
        self.sel_matrix = torch.zeros((self.unwrapped.num_envs, 6), device=self.unwrapped.device)
        self.force_action = torch.zeros((self.unwrapped.num_envs, 6), device=self.unwrapped.device)
        
        # this makes it so that the maximum possible position error is equivalent to the maximum force error
        force_task_gains = self.unwrapped.cfg.ctrl.default_task_force_gains #[
        #    self.unwrapped.cfg.ctrl.pos_action_bounds[0] / self.unwrapped.cfg.ctrl.force_action_bounds[0] \
        #    for i in range(6)
        #]
        print("force task gains:", force_task_gains)
        self.kp = torch.tensor(force_task_gains, device=self.unwrapped.device).repeat(
            (self.unwrapped.num_envs, 1)
        )
        
        if not ctrl_torque:
            self.kp[:,3:] = 0.0
        
        if reward_type == "simp": # any force direction means at least one force control
            self._update_rew_buf = self._simp_update_rew_buf
        elif reward_type == "dirs": # each direction is a seperate selection reward
            self._update_rew_buf = self._dir_update_rew_buf
        elif reward_type == "delta": # punish changing selection matrix
            self._old_sel_matrix = torch.zeros_like(self.sel_matrix)
            self._update_rew_buf = self._delta_update_rew_buf
        elif reward_type == "base": # no rew for hybrid control
            self._update_rew_buf = self.env.unwrapped._update_rew_buf
        elif reward_type == "pos_simp":
            self._update_rew_buf = self._pos_simp_update_rew_buf
        elif reward_type == "wrench_norm":
            self._update_rew_buf = self._low_wrench_update_rew_buf
        else:
            raise NoImplementedError(f"Reward type {reward_type} is not implemented")

        self.old_update_rew_buf = self.env.unwrapped._update_rew_buf
        self.env.unwrapped._update_rew_buf = self._update_rew_buf
        

    def _calc_ctrl_force(self):
        min_idx = self.force_size + 6
        max_idx = min_idx + self.force_size
        # locks delta force goal to force_threshold (2N)
        self.force_action[:,:self.force_size] = self.unwrapped.actions[:,min_idx:max_idx] * self.cfg.ctrl.force_action_threshold[0]

        # keeps force goal in acceptable range
        self.force_action[:,:3] = torch.clip(
            self.force_action[:,:3] + self.unwrapped.robot_force_torque[:,:3], # goal force is change + current value
            -self.cfg.ctrl.force_action_bounds[0],
            self.cfg.ctrl.force_action_bounds[0]
        )

        if self.force_size > 3:
            self.force_action[:,3:] *= self.cfg.ctrl.torque_action_threshold[0] / self.cfg.ctrl.force_action_threshold[0]
        
            self.force_action[:,3:] = torch.clip(
                self.force_action[:,3:] + self.unwrapped.robot_force_torque[:,3:], # goal force is change + current value
                -self.cfg.ctrl.torque_action_bounds[0],
                self.cfg.ctrl.torque_action_bounds[0]
            )
        
    def _pre_physics_step(self, action):
        self.sel_matrix[:,:self.force_size] = torch.where(action[:,:self.force_size] > 0.5, 1.0, 0.0)
        
        self.unwrapped.extras[f'Hybrid Controller / Force Control X'] = action[:,0]#.mean()
        self.unwrapped.extras[f'Hybrid Controller / Force Control Y'] = action[:,1]#.mean()
        self.unwrapped.extras[f'Hybrid Controller / Force Control Z'] = action[:,2]#.mean()

        if self.force_size > 3:
            self.unwrapped.extras[f'Hybrid Controller / Force Control RX'] = action[:,3]#.mean()
            self.unwrapped.extras[f'Hybrid Controller / Force Control RY'] = action[:,4]#.mean()
            self.unwrapped.extras[f'Hybrid Controller / Force Control RZ'] = action[:,5]#.mean()
        

        env_ids = self.unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.unwrapped._reset_buffers(env_ids)
        self.unwrapped.prev_action = self.unwrapped.actions.clone()
        self.unwrapped.actions = (
            self.unwrapped.cfg.ctrl.ema_factor * action.clone().to(self.unwrapped.device) + (1 - self.unwrapped.cfg.ctrl.ema_factor) * self.unwrapped.actions
        )
        
        
        #self.unwrapped.prev_action = self.unwrapped.actions
        
        # set goal 1 time
        self.unwrapped._calc_ctrl_pos(min_idx=self.force_size, max_idx=self.force_size+3)
        self.unwrapped._calc_ctrl_quat(min_idx=self.force_size+3, max_idx=self.force_size+6)
        self._calc_ctrl_force()

        # update current values
        self.unwrapped._compute_intermediate_values(dt=self.unwrapped.physics_dt) 

        # smoothness metrics
        self.unwrapped.ep_ssv += (torch.linalg.norm(self.unwrapped.ee_linvel_fd, axis=1))
        if self.unwrapped.use_ft or self.unwrapped.fragile:
            self.unwrapped.ep_sum_force += torch.linalg.norm(self.unwrapped.robot_force_torque[:,:3], axis=1)
            self.unwrapped.ep_sum_torque += torch.linalg.norm(self.unwrapped.robot_force_torque[:,3:], axis=1)
            self.unwrapped.ep_max_force = torch.max(self.unwrapped.ep_max_force, torch.linalg.norm(self.unwrapped.robot_force_torque[:,:3], axis=1 ))
            self.unwrapped.ep_max_torque = torch.max(self.unwrapped.ep_max_torque, torch.linalg.norm(self.unwrapped.robot_force_torque[:,3:]))

    def get_target_out_of_bounds(self):
        #out_of_bounds = torch.zeros_like(self.sel_matrix)

        # test if fingertip is in range
        delta = self.unwrapped.fingertip_midpoint_pos - self.unwrapped.fixed_pos_action_frame

        out_of_bounds = torch.logical_or(
            delta <= -self.unwrapped.cfg.ctrl.pos_action_bounds[0],
            delta >= self.unwrapped.cfg.ctrl.pos_action_bounds[1]
        )             
        return out_of_bounds
    
    def _apply_action(self):
        # Get current yaw for success checking.
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.unwrapped.fingertip_midpoint_quat)
        self.unwrapped.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)
        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.unwrapped.last_update_timestamp < self.unwrapped._robot._data._sim_timestamp:
            self.unwrapped._compute_intermediate_values(dt=self.unwrapped.physics_dt)

        self.unwrapped.ctrl_target_gripper_dof_pos = 0.0

        out_of_bounds = self.get_target_out_of_bounds()
        #print("out:", out_of_bounds.size())
        
        pose_wrench =  fc.compute_pose_task_wrench(
            cfg=self.unwrapped.cfg,
            dof_pos=self.unwrapped.joint_pos,
            fingertip_midpoint_pos=self.unwrapped.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.unwrapped.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.unwrapped.ee_linvel_fd,
            fingertip_midpoint_angvel=self.unwrapped.ee_angvel_fd,
            ctrl_target_fingertip_midpoint_pos=self.unwrapped.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.unwrapped.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.unwrapped.task_prop_gains,
            task_deriv_gains=self.unwrapped.task_deriv_gains,
            device=self.unwrapped.device
        ) # this calculates task wrench generated by pose error
        
        force_wrench = fc.compute_force_task_wrench(
            cfg=self.unwrapped.cfg,
            dof_pos=self.unwrapped.joint_pos,
            eef_force=self.unwrapped.robot_force_torque,
            ctrl_target_force=self.force_action,
            task_gains=self.kp,
            device=self.unwrapped.device
        )

        # if pos is out of bounds we want there to be no error on force target
        force_wrench[:,:3][out_of_bounds] = 0.0 #self.unwrapped.robot_force_torque[:,:3][out_of_bounds]
        
        task_wrench = (1-self.sel_matrix) * pose_wrench + self.sel_matrix * force_wrench
        if self.force_size > 3:
            task_wrench[:,3:] = pos_wrench[:,3:]

        #task_wrench[:,3:5] = 0.0 # prevent rotating anywhere but parallel to tabletop

        self.unwrapped.joint_torque, task_wrench = fc.compute_dof_torque_from_wrench(
            cfg=self.unwrapped.cfg,
            dof_pos=self.unwrapped.joint_pos,
            dof_vel=self.unwrapped.joint_vel_fd,
            task_wrench=task_wrench,
            jacobian=self.unwrapped.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.unwrapped.arm_mass_matrix,
            device=self.unwrapped.device
        )

        # set target for gripper joints to use physx's PD controller
        self.unwrapped.ctrl_target_joint_pos[:, 7:9] = self.unwrapped.ctrl_target_gripper_dof_pos
        self.unwrapped.joint_torque[:, 7:9] = 0.0

        self.unwrapped._robot.set_joint_position_target(self.unwrapped.ctrl_target_joint_pos) # this is not used since actuator gains = 0
        self.unwrapped._robot.set_joint_effort_target(self.unwrapped.joint_torque)
    
    def _dir_update_rew_buf(self, curr_successes):
        #rew_buf = self.unwrapped._update_reward_buf(curr_successes)
        rew_buf = self.old_update_rew_buf(curr_successes)
        
        # reward when selection and force/pos match
        active_force = torch.abs(self.unwrapped.robot_force_torque[:,:self.force_size]) > self.cfg_task.force_active_threshold
        if self.force_size > 3:
            active_force[:,3:] = torch.abs(self.unwrapped.robot_force_torque[:,3:]) > self.cfg_task.torque_active_threshold
        force_ctrl = self.sel_matrix[:,:self.force_size].bool()

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

        good_rew = self.cfg_task.good_force_cmd_rew * torch.sum(good_dims,dim=1)
        bad_rew = self.cfg_task.bad_force_cmd_rew  * torch.sum(bad_dims, dim=1)
        sel_rew = good_rew + bad_rew
        self.unwrapped.extras['Reward / Selection Matrix'] = good_rew + bad_rew
        
        return rew_buf + good_rew + bad_rew  #+ self.force_size*torch.any(active_force, dim=1).float() * self.cfg_task.good_force_cmd_rew
    
    def _simp_update_rew_buf(self, curr_successes):
        #rew_buf = self.unwrapped._update_reward_buf(curr_successes)
        rew_buf = self.old_update_rew_buf(curr_successes)
        
        # reward when selection and force/pos match
        active_force = torch.abs(self.unwrapped.robot_force_torque[:,:self.force_size]) > self.cfg_task.force_active_threshold
        #active_force[:,3:] = torch.abs(self.unwrapped.robot_force_torque[:,3:]) > self.cfg_task.torque_active_threshold
        force_ctrl = self.sel_matrix[:,:self.force_size].bool()


        good_force_cmd = torch.logical_and(torch.any(active_force,dim=1), torch.any(force_ctrl, dim=1))
        bad_force_cmd = torch.logical_and(torch.all(~active_force, dim=1), torch.any(force_ctrl, dim=1))

        sel_rew = self.cfg_task.good_force_cmd_rew * good_force_cmd
        sel_rew += self.cfg_task.bad_force_cmd_rew * bad_force_cmd

        self.unwrapped.extras['Reward / Selection Matrix'] = sel_rew
        
        return rew_buf + sel_rew #+ torch.any(active_force, dim=1).float() * self.cfg_task.good_force_cmd_rew

    def _delta_update_rew_buf(self, curr_successes):
        rew_buf = self.old_update_rew_buf(curr_successes)

        sel_rew = torch.sum( torch.abs(self.sel_matrix - self._old_sel_matrix), dim=1) * self.cfg_task.bad_force_cmd_rew
        
        self._old_sel_matrix = self.sel_matrix.clone()

        self.unwrapped.extras['Reward / Selection Matrix'] = sel_rew
        
        return rew_buf + sel_rew

    def _pos_simp_update_rew_buf(self, curr_successes):
        #rew_buf = self.unwrapped._update_reward_buf(curr_successes)
        rew_buf = self.old_update_rew_buf(curr_successes)
        
        # reward when selection and force/pos match
        active_force = torch.abs(self.unwrapped.robot_force_torque[:,:self.force_size]) > self.cfg_task.force_active_threshold
        #active_force[:,3:] = torch.abs(self.unwrapped.robot_force_torque[:,3:]) > self.cfg_task.torque_active_threshold
        force_ctrl = self.sel_matrix[:,:self.force_size].bool()


        good_force_cmd = torch.logical_and(torch.any(active_force,dim=1), torch.any(force_ctrl, dim=1))
        bad_force_cmd = torch.logical_and(torch.all(~active_force, dim=1), torch.any(force_ctrl, dim=1))

        sel_rew = self.cfg_task.good_force_cmd_rew * good_force_cmd

        self.unwrapped.extras['Reward / Selection Matrix'] = sel_rew
        
        return rew_buf + sel_rew #+ torch.any(active_force, dim=1).float() * self.cfg_task.good_force_cmd_rew
        
    def _low_wrench_update_rew_buf(self, curr_successes):
        rew_buf = self.old_update_rew_buf(curr_successes)
        wrench_norm = self.unwrapped.actions[:,self.force_size:].norm(dim=-1)
        return rew_buf - wrench_mags * self.cfg_task.wrench_norm_scale
