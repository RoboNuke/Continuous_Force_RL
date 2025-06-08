from .factory_env import FactoryEnv
from .factory_env_cfg import FactoryEnvCfg, OBS_DIM_CFG, STATE_DIM_CFG

import torch


HISTORY_STATE_CFG = [
    "fingertip_pos",
    "fingertip_pos_rel_fixed",
    "fingertip_quat",
    "ee_linvel",
    "ee_angvel",
    "held_pos",
    "held_pos_rel_fixed",
    "held_quat",
    "ee_linacc",
    "ee_angacc",
    "force_torque"
]
class HistoryObsFactoryEnv(FactoryEnv):
    def __init__(
            self, 
            cfg: FactoryEnvCfg, 
            render_mode: str | None = None, 
            calc_accel = False,
            **kwargs
        ):

        self.h_len = cfg.decimation #define before __init__ because super calls _init_tensors()
        self.calc_accel = calc_accel

        if self.calc_accel:
            print("\nCalculate Acceleration\n")

        super().__init__(cfg, render_mode, **kwargs)
        if self.calc_accel:
            OBS_DIM_CFG["ee_linacc"] = 3
            OBS_DIM_CFG["ee_angacc"] = 3
            STATE_DIM_CFG["ee_linacc"] = 3
            STATE_DIM_CFG["ee_angacc"] = 3
            cfg.obs_order.append("ee_linacc")
            cfg.obs_order.append("ee_angacc")

        if self.calc_accel and cfg.use_force_sensor:
            OBS_DIM_CFG['force_jerk'] = 6
            OBS_DIM_CFG['force_snap'] = 6
            cfg.obs_order.append('force_jerk')
            cfg.obs_order.append('force_snap')

        self.num_samples = self.cfg.history_samples
        if self.num_samples == -1:
            self.num_samples = self.h_len
        
        self.keep_idxs = torch.linspace(0.0, self.h_len - 1.0, self.num_samples).type(torch.int32)
        if self.num_samples == 1: # special case where linspace gives first value but we always want to include last value
            self.keep_idxs[0] = self.h_len - 1
        print("Keep idxs:", self.keep_idxs)

        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order]) * self.num_samples #self.h_len
        #if self.calc_accel:
        #    cfg.observation_space += 6 * self.h_len
        cfg.state_space = sum(
            [STATE_DIM_CFG[state] * self.num_samples if state in HISTORY_STATE_CFG else STATE_DIM_CFG[state] for state in cfg.state_order]
        )
        #if self.calc_accel:
        #    cfg.state_space += 6 * self.h_len

        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        self.cfg_task = cfg.task

        # need to reset env observation space size for vectorization
        self._configure_gym_env_spaces()

    def _init_tensors(self):
        super()._init_tensors()
        # initialize containers
        self.fingertip_midpoint_history = torch.zeros((self.num_envs, self.h_len, 3), device=self.device)
        self.fingertip_quat_history = torch.zeros((self.num_envs, self.h_len, 4), device=self.device)

        self.ee_linvel_history = torch.zeros_like(self.fingertip_midpoint_history)
        self.ee_angvel_history = torch.zeros_like(self.fingertip_midpoint_history)

        if self.calc_accel:
            self.ee_linacc_history = torch.zeros_like(self.fingertip_midpoint_history)
            self.ee_angacc_history = torch.zeros_like(self.fingertip_midpoint_history)

        if self.cfg.use_force_sensor:
            self.force_history = torch.zeros((self.num_envs, self.h_len, 6), device=self.device)

        if self.calc_accel and self.cfg.use_force_sensor:
            self.force_jerk_history = torch.zeros((self.num_envs, self.h_len, 6), device=self.device)
            self.force_snap_history = torch.zeros_like(self.force_jerk_history)

        # required for critic
        self.held_pos_history = torch.zeros_like(self.fingertip_midpoint_history)
        self.held_quat_history = torch.zeros_like(self.fingertip_quat_history)
        
    def _update_history(self, reset=False):
        #print("Update with reset: " + (" yes" if reset else ' no'))
        if reset:
            self.fingertip_midpoint_history[:,:,:] = self.fingertip_midpoint_pos[:,None,:]
            self.fingertip_quat_history[:,:,:] = self.fingertip_midpoint_quat[:,None,:]
            self.ee_linvel_history[:,:,:] = self.ee_linvel_fd[:,None,:]
            self.ee_angvel_history[:,:,:] = self.ee_angvel_fd[:,None,:]
            self.held_pos_history[:,:,:] = self.held_pos[:,None,:]
            self.held_quat_history[:,:,:] = self.held_quat[:,None,:]
            if self.calc_accel:
                self.ee_linacc_history[:,:,:] = 0
                self.ee_angacc_history[:,:,:] = 0
            if self.cfg.use_force_sensor:
                self.force_history[:,:,:] = self.robot_force_torque[:,None,:]
            
            if self.cfg.use_force_sensor and self.calc_accel:
                self.force_jerk_history[:,:,:] = 0
                self.force_snap_history[:,:,:] = 0
        else:
            self.fingertip_midpoint_history = torch.roll(self.fingertip_midpoint_history, -1, 1)
            self.fingertip_midpoint_history[:,-1,:] = self.fingertip_midpoint_pos
            self.fingertip_quat_history = torch.roll(self.fingertip_quat_history, -1, 1)
            self.fingertip_quat_history[:,-1,:] = self.fingertip_midpoint_quat
            self.ee_linvel_history = torch.roll(self.ee_linvel_history, -1, 1)
            self.ee_linvel_history[:,-1,:] = self.ee_angvel_fd
            self.ee_angvel_history = torch.roll(self.ee_angvel_history, -1, 1)
            self.ee_angvel_history[:,-1,:] = self.ee_angvel_fd
            self.held_pos_history = torch.roll(self.held_pos_history, -1, 1)
            self.held_pos_history[:,-1,:] = self.held_pos
            self.held_quat_history = torch.roll(self.held_quat_history, -1, 1)
            self.held_quat_history[:,-1,:] = self.held_quat

            if self.calc_accel:
                # use finite difference to get accelerations
                self.ee_linacc_history = torch.roll(self.ee_linacc_history, -1, 1)
                self.ee_angacc_history = torch.roll(self.ee_angacc_history, -1, 1)
                
                self.ee_linacc_history[:,-1,:] = self.fd_last_vals(self.ee_linacc_history)
                self.ee_angacc_history[:,-1,:] = self.fd_last_vals(self.ee_angacc_history)
            
            if self.cfg.use_force_sensor:
                self.force_history = torch.roll(self.force_history, -1, 1)
                self.force_history[:,-1,:] = self.robot_force_torque
            
            if self.cfg.use_force_sensor and self.calc_accel:
                self.force_jerk_history = torch.roll(self.force_jerk_history, -1, 1)
                self.force_snap_history = torch.roll(self.force_snap_history, -1, 1)

                self.force_jerk_history[:,-1,:] = self.fd_last_vals(self.force_jerk_history)
                self.force_snap_history[:,-1,:] = self.fd_last_vals(self.force_snap_history)

    def fd_last_vals(self, history):
        return (history[:,-2,:] - history[:,-1,:]) / self.cfg.sim.dt

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # set initial conditions for histories
        self._update_history(reset=True)


    def _pre_physics_step(self, action):
        super()._pre_physics_step(action)
        
        self._update_history(reset=False)
    
    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise # size (num_envs, 3)

        prev_actions = self.actions.clone()
        
        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_history[:, self.keep_idxs, :].view( (self.num_envs, -1)),
            "fingertip_pos_rel_fixed": (self.fingertip_midpoint_history - noisy_fixed_pos[:,None,:])[:, self.keep_idxs, :].view( (self.num_envs, -1)),
            "fingertip_quat": self.fingertip_quat_history[:, self.keep_idxs, :].view( (self.num_envs, -1)),
            "ee_linvel": self.ee_linvel_history[:, self.keep_idxs, :].view( (self.num_envs, -1)),
            "ee_angvel": self.ee_angvel_history[:, self.keep_idxs, :].view( (self.num_envs, -1)),
            "prev_actions": prev_actions,
        }
        
        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_history[:, self.keep_idxs, :].view( (self.num_envs, -1)),
            "fingertip_pos_rel_fixed": (self.fingertip_midpoint_history - self.fixed_pos_obs_frame[:,None,:])[:, self.keep_idxs, :].view( (self.num_envs, -1)),
            "fingertip_quat": self.fingertip_quat_history[:, self.keep_idxs, :].view( (self.num_envs, -1)),
            "ee_linvel": self.ee_linvel_history[:, self.keep_idxs, :].view( (self.num_envs, -1)),
            "ee_angvel": self.ee_angvel_history[:, self.keep_idxs, :].view( (self.num_envs, -1)),
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos_history[:, self.keep_idxs, :].view ( (self.num_envs, -1) ),
            "held_pos_rel_fixed": (self.held_pos_history - self.fixed_pos_obs_frame[:,None,:])[:, self.keep_idxs, :].view( (self.num_envs, -1)),
            "held_quat": self.held_quat_history[:, self.keep_idxs, :].view( (self.num_envs, -1 )),
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
        }

        if self.calc_accel:
            obs_dict['ee_linacc'] = self.ee_linacc_history[:, self.keep_idxs, :].view( (self.num_envs, -1))
            obs_dict['ee_angacc'] = self.ee_angacc_history[:, self.keep_idxs, :].view( (self.num_envs, -1))

            state_dict['ee_linacc'] = self.ee_linacc_history[:, self.keep_idxs, :].view( (self.num_envs, -1))
            state_dict['ee_angacc'] = self.ee_angacc_history[:, self.keep_idxs, :].view( (self.num_envs, -1))

        if self.cfg.use_force_sensor:
            obs_dict["force_torque"] = self.force_history[:, self.keep_idxs, :].view( (self.num_envs, -1))
            state_dict["force_torque"] = self.force_history[:, self.keep_idxs, :].view( (self.num_envs, -1))

        if self.cfg.use_force_sensor and self.calc_accel:
            obs_dict["force_jerk"] = self.force_jerk_history[:, self.keep_idxs, :].view( (self.num_envs, -1))
            obs_dict["force_snap"] = self.force_snap_history[:, self.keep_idxs, :].view( (self.num_envs, -1))
            state_dict["force_jerk"] = self.force_jerk_history[:, self.keep_idxs, :].view( (self.num_envs, -1))
            state_dict["force_snap"] = self.force_snap_history[:, self.keep_idxs, :].view( (self.num_envs, -1))


        if self.calc_accel:
            return {"policy": obs_dict, "critic": state_dict}
        
        # scale force readings
        if self.cfg.use_force_sensor:
            obs_dict['force_torque'] = torch.tanh( 0.0011 * obs_dict['force_torque'])
            state_dict['force_torque'] = torch.tanh( 0.0011 * state_dict['force_torque'])

        obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order + ["prev_actions"]]
        
        obs_tensors = torch.cat(obs_tensors, dim=-1)
        
        state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order + ["prev_actions"]]
        
        state_tensors = torch.cat(state_tensors, dim=-1)
        
        return {"policy": obs_tensors, "critic": state_tensors}
