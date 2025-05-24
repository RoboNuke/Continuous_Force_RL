from .factory_env import FactoryEnv
from .factory_env_cfg import FactoryEnvCfg

import torch

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}

HISTORY_STATE_CFG = [
    "fingertip_pos",
    "fingertip_pos_rel_fixed",
    "fingertip_quat",
    "ee_linvel",
    "ee_angvel",
    "held_pos",
    "held_pos_rel_fixed",
    "held_quat",

]
class HistoryObsFactoryEnv(FactoryEnv):
    def __init__(
            self, 
            cfg: FactoryEnvCfg, 
            render_mode: str | None = None, 
            history_length=10,
            **kwargs
        ):
        self.h_len = history_length
        super().__init__(cfg, render_mode, **kwargs)
        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order]) * self.h_len
        cfg.state_space = sum(
            [STATE_DIM_CFG[state] * self.h_len if state in HISTORY_STATE_CFG else STATE_DIM_CFG[state] for state in cfg.state_order]
        )
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        self.cfg_task = cfg.task
        
        self._configure_gym_env_spaces()

        self._set_body_inertias()
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._compute_intermediate_values(dt=self.physics_dt)

    def _init_tensors(self):
        super()._init_tensors()
        # initialize containers
        self.fingertip_midpoint_history = torch.zeros((self.num_envs, self.h_len, 3), device=self.device)
        self.fingertip_quat_history = torch.zeros((self.num_envs, self.h_len, 4), device=self.device)

        self.ee_linvel_history = torch.zeros_like(self.fingertip_midpoint_history)
        self.ee_angvel_history = torch.zeros_like(self.fingertip_midpoint_history)

        #self.ee_linacc_history = torch.zeros((self.h_len, self.num_envs, 3), device=self.device)
        #self.ee_angacc_history = torch.zeros((self.h_len, self.num_envs, 3), device=self.device)

        # required for critic
        self.held_pos_history = torch.zeros_like(self.fingertip_midpoint_history)
        self.held_quat_history = torch.zeros_like(self.fingertip_quat_history)
        
    def _update_history(self, reset=False):
        if reset:
            self.fingertip_midpoint_history[:,:,:] = self.fingertip_midpoint_pos[:,None,:]
            self.fingertip_quat_history[:,:,:] = self.fingertip_midpoint_quat[:,None,:]
            self.ee_linvel_history[:,:,:] = self.ee_linvel_fd[:,None,:]
            self.ee_angvel_history[:,:,:] = self.ee_angvel_fd[:,None,:]
            self.held_pos_history[:,:,:] = self.held_pos[:,None,:]
            self.held_quat_history[:,:,:] = self.held_quat[:,None,:]
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

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # set initial conditions for histories
        self._update_history(reset=True)


    def _pre_physics_step(self, action):
        super()._pre_physics_step(action)
        # update current values
        self._compute_intermediate_values(dt=self.physics_dt) 
        
        self._update_history(reset=False)
    
    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise # size (num_envs, 3)

        prev_actions = self.actions.clone()

        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_history.view( (self.num_envs, -1)),
            "fingertip_pos_rel_fixed": (self.fingertip_midpoint_history - noisy_fixed_pos[:,None,:]).view( (self.num_envs, -1)),
            "fingertip_quat": self.fingertip_quat_history.view( (self.num_envs, -1)),
            "ee_linvel": self.ee_linvel_history.view( (self.num_envs, -1)),
            "ee_angvel": self.ee_angvel_history.view( (self.num_envs, -1)),
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_history.view( (self.num_envs, -1)),
            "fingertip_pos_rel_fixed": (self.fingertip_midpoint_history - self.fixed_pos_obs_frame[:,None,:]).view( (self.num_envs, -1)),
            "fingertip_quat": self.fingertip_quat_history.view( (self.num_envs, -1)),
            "ee_linvel": self.ee_linvel_history.view( (self.num_envs, -1)),
            "ee_angvel": self.ee_angvel_history.view( (self.num_envs, -1)),
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos_history.view ( (self.num_envs, -1) ),
            "held_pos_rel_fixed": (self.held_pos_history - self.fixed_pos_obs_frame[:,None,:]).view( (self.num_envs, -1)),
            "held_quat": self.held_quat_history.view( (self.num_envs, -1 )),
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
        }
        obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order + ["prev_actions"]]
        obs_tensors = torch.cat(obs_tensors, dim=-1)
        state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order + ["prev_actions"]]
        state_tensors = torch.cat(state_tensors, dim=-1)
        return {"policy": obs_tensors, "critic": state_tensors}


    
