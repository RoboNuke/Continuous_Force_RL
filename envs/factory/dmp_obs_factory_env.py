import copy
from typing import Dict

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import gymnasium.spaces.utils
import torch
from wrappers.DMP.discrete_dmp import DiscreteDMP
from wrappers.DMP.cs import CS
from wrappers.DMP.quaternion_dmp import QuaternionDMP

import matplotlib.pyplot as plt
from envs.factory.obs_factory_env import OBS_DIM_CFG, STATE_DIM_CFG, HISTORY_STATE_CFG, HistoryObsFactoryEnv
from envs.factory.factory_env_cfg import FactoryEnvCfg

class DMPObsFactoryEnv(HistoryObsFactoryEnv):
    """
        Takes the time series information in observation and replaces it with a set of 
        weights from a DMP that represent the trajectory 
    """

    def __init__(
            self, 
            cfg: FactoryEnvCfg, 
            render_mode: str | None = None, 
            calc_accel = True,
            num_weights=5,
            fit_force_data=False,
            **kwargs
        ):
        
        super().__init__(cfg, render_mode, calc_accel, **kwargs)
        self.num_weights = num_weights
        self.save_fit = False
        
        self.display_fit = False
        # calculating t values is tricky, I assume
        # that x = 0 is equal to update_dt
        # while using a tau may be the correct way, simpling scaling
        # the time variable keeps the topology of the trajectory without
        # so long as the canaical system has the same number of steps
        
        self.sim_dt = cfg.sim.dt #1/120 #self.physics_dt #sim_dt
        self.update_dt = cfg.sim.dt * cfg.decimation #update_dt
        self.dt = self.sim_dt / self.update_dt
        self.dec = int(self.update_dt / self.sim_dt)
        
        # define data containers
        # note: you add one to include the starting point. This is the last location the policy was at in the last timestep
        #self.num_envs = self.num_envs
        self.t = torch.linspace(start=0, end=self.update_dt, steps= self.dec, device=self.device ) 
        self.y = torch.zeros((self.num_envs, self.dec, 3), device=self.device)
        self.dy = torch.zeros_like(self.y)
        self.ddy = torch.zeros_like(self.y)
        self.ay = torch.zeros((self.num_envs, self.dec, 4), device=self.device)
        self.ay[:,:,0] = 1.0
        self.day = torch.zeros_like(self.ay)
        self.dday = torch.zeros_like(self.ay)

        # set a ref list to make step func look pretty
        self.unpack_list = [
            ("ee_linvel", self.dy),
            ("ee_angvel", self.day),
            ("ee_linacc", self.ddy),
            ("ee_angacc", self.dday)
        ]        

        self.dmp_keys = [
            "fingertip_pos",
            "fingertip_pos_rel_fixed",
            "fingertip_quat",
            "held_pos",
            "held_pos_rel_fixed",
            "held_quat"
        ]

        new_obs_shape = 0
        new_state_shape = 0
        self.new_obs = {
            "policy":{},
            "critic":{}
        }
        self.dmp_obs_order = []
        for key in self.cfg.obs_order:

            if key in [self.unpack_list[i][0] for i in range(len(self.unpack_list))]:
                continue
            elif key in self.dmp_keys:
                self.new_obs['policy'][key] = torch.zeros((self.num_envs, 3 * self.num_weights), device=self.device)
                new_obs_shape += 3 * self.num_weights
                self.dmp_obs_order.append(key)
            else:
                self.new_obs['policy'][key] = torch.zeros((self.num_envs, OBS_DIM_CFG[key]), device=self.device)
                new_obs_shape += OBS_DIM_CFG[key]
                self.dmp_obs_order.append(key)
                

        self.dmp_state_order = []
        for key in self.cfg.state_order:
            if key in [self.unpack_list[i][0] for i in range(len(self.unpack_list))]:
                continue
            elif key in self.dmp_keys:
                self.new_obs['critic'][key] = torch.zeros((self.num_envs, 3 * self.num_weights), device=self.device)
                new_state_shape += 3 * self.num_weights
                self.dmp_state_order.append(key)
            else:
                self.new_obs['critic'][key] = torch.zeros((self.num_envs, STATE_DIM_CFG[key]), device=self.device)
                new_state_shape += STATE_DIM_CFG[key]
                self.dmp_state_order.append(key)
                
        # setup gym spaces
        self.cfg.observation_space = new_obs_shape
        self.cfg.state_space = new_state_shape    
        self._configure_gym_env_spaces()   


        # define the DMPs
        self.cs = CS(
            ax=2.5, 
            dt=self.dt, 
            device=self.device
        )

        self.pos_dmp = DiscreteDMP(
            nRBF=self.num_weights, 
            betaY=12.5/4.0, 
            dt=self.dt, 
            cs=self.cs, 
            num_envs=self.num_envs, 
            num_dims=3,
            device=self.device
        )


        self.ang_dmp = QuaternionDMP(
            nRBF=self.num_weights,
            betaY=12/4,
            dt = self.dt,
            cs = CS(
                    ax=1, 
                    dt=self.dt, 
                    device=self.device
            ),
            num_envs= self.num_envs,
            device = self.device
        )


        if self.save_fit or self.display_fit:
            self.fig, self.axs = plt.subplots(self.num_envs, 4)
            self.fig.set_figwidth(3 / 3 * 1600/96)
            self.fig.set_figheight(self.num_envs / 4 * 1000/96)
            self.fig.tight_layout(pad=5.0)
            self.start_time = 0.0



    def _get_observations(self):
        old_obs = super()._get_observations()
        
        # unpack derivatives
        for i in range(len(self.unpack_list)):
            var_name, var_ref = self.unpack_list[i]
            if 'ang' in var_name:
                var_ref[:,:,1:] = old_obs['critic'][var_name].detach().clone().view(self.num_envs, self.dec, 3)
            else:
                var_ref[:,:,:] = old_obs['critic'][var_name].detach().clone().view(self.num_envs, self.dec, 3)
        
        for key in self.cfg.obs_order:
            if key in self.dmp_keys:
                if 'pos' in key:
                    self.y = old_obs['policy'][key].detach().clone().view(self.num_envs, self.dec, 3)
                    self.pos_dmp.learnWeightsCSDT(self.y, self.dy, self.ddy, self.t)
                    self.new_obs['policy'][key] = self.pos_dmp.ws.view(self.num_envs, 3 * self.num_weights)
                elif 'quat' in key:
                    self.ay = old_obs['policy'][key].detach().clone().view(self.num_envs, self.dec, 4)
                    self.ang_dmp.learnWeightsCSDT(self.ay, self.day, self.dday, self.t)
                    self.new_obs['policy'][key] = self.ang_dmp.ws[:,:,1:].reshape(self.num_envs, 3 * self.num_weights)
        
        
        obs_tensors = [self.new_obs['policy'][obs_name].view( (self.num_envs, -1 )) for obs_name in self.dmp_obs_order ]
        obs_tensors = torch.cat(obs_tensors, dim=-1)
        obs_tensors = torch.nan_to_num(obs_tensors)
        for key in self.cfg.state_order:
            if key in self.dmp_keys:
                if 'pos' in key:
                    self.y = old_obs['critic'][key].detach().clone().view(self.num_envs, self.dec, 3)
                    self.pos_dmp.learnWeightsCSDT(self.y, self.dy, self.ddy, self.t)
                    self.new_obs['critic'][key] = self.pos_dmp.ws.view(self.num_envs, 3 * self.num_weights)
                elif 'quat' in key:
                    self.ay = old_obs['critic'][key].detach().clone().view(self.num_envs, self.dec, 4)
                    self.ang_dmp.learnWeightsCSDT(self.ay, self.day, self.dday, self.t)
                    self.new_obs['critic'][key] = self.ang_dmp.ws[:,:,1:].reshape(self.num_envs, 3 * self.num_weights)
        
        # order and concat observations
        state_tensors = [self.new_obs['critic'][state_name].view( (self.num_envs, -1 )) for state_name in self.dmp_state_order ]
        state_tensors = torch.cat(state_tensors, dim=-1)
        
        #print(type(obs_tensors))
        #print(obs_tensors.size(), self.cfg.observation_space)
        #print(type(state_tensors))
        #print(state_tensors.size(), self.cfg.state_space)
        #print("Obs:", obs_tensors)
        return {"policy": obs_tensors, "critic": state_tensors}
    
        

    """def reset(self, **kwargs):
    
        print("DMP reset called")
        if self.save_fit:
            plt.savefig("/home/hunter/Fit.png")
        if self.display_fit:
            plt.show()
        if self.save_fit or self.display_fit:
            plt.clf()
            self.fig, self.axs = plt.subplots(self.num_envs, 4)
            self.fig.set_figwidth(3 / 3 * 1600/96)
            self.fig.set_figheight(self.num_envs / 4 * 1000/96)
            self.fig.tight_layout(pad=5.0)
            self.start_time = 0.0
        
        print("\n\n\n\nstart reset\n\n\n\n\n")
        old_obs, info = super().reset(**kwargs)
        print("\nreturn\n")
        return old_obs, info"""