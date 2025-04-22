import copy
from typing import Dict

import gymnasium as gym
import gymnasium.spaces.utils
import numpy as np
import torch

class SmoothnessObservationWrapper(gym.Wrapper):
    """
        Adds to the observation space data required to describe
        the smoothness of a run 


    """
    def __init__(self, env)->None:
        #self.unwrapped = env
        super().__init__(env)
        self.obs = {}
        self.obs['Smoothness / Squared Joint Velocity'] = torch.zeros((self.num_envs, 1), device=self.device)
        self.obs['Smoothness / Jerk'] = torch.zeros_like(self.obs['Smoothness / Squared Joint Velocity'])
        self.obs['Smoothness / Damage Force'] = torch.zeros_like(self.obs['Smoothness / Squared Joint Velocity'])
        self.obs['Smoothness / Damage Torque'] = torch.zeros_like(self.obs['Smoothness / Squared Joint Velocity'])
        
        self.old_acc = None #torch.zeros_like(self.agent.robot.qacc)
    
    def reset(self, **kwargs):
        #print("\n\nSmoothness reset called\n\n")
        obs, info = self.env.reset(**kwargs)
        info['smoothness'] = self.obs
        for k in self.obs.keys():
            self.obs[k] *= 0
        return obs, info

    def step(self, action):
        #observation, r, term, trun, info = self.unwrapped.step(action)
        observation, r, term, trun, info = self.env.step(action)
        
        if self.old_acc is None:
            self.old_acc = torch.zeros_like(observation['info']['joint_acc'])

        reset_set = torch.logical_or(term, trun)
        self.old_acc[reset_set] *= 0
        #TODO: Add Force
        #self.sdf[reset_set] *= 0


        qvel = observation['info']['joint_vel']
        qacc = observation['info']['joint_acc']

        obs = {}

        #note must index self.obs because linalg.norm returns tensor of size [num_envs], but
        # downstream everything expectes size [num_envs, 1], I know the difference is obvious
        # sum squared velocity
        self.obs['Smoothness / Squared Joint Velocity'][:,0] = torch.linalg.norm(qvel * qvel, axis=1)

        # sum squared accel
        #obs['sqr_qa'] = torch.linalg.norm(qacc * qacc, axis=1)

        # jerk
        jerk = (qacc - self.old_acc) / 0.1
        self.obs['Smoothness / Jerk'][:,0] =  torch.linalg.norm(jerk, axis=1)
        self.old_acc = qacc

        # force 
        #print("Dmg force size:", observation['info']['dmg_force'].size())
        #print(torch.max( torch.linalg.norm(observation['info']['dmg_force'][:,:,:3], axis=2), 1)[0])
        if len(observation['info']['dmg_force'].size()) == 3:
            self.obs['Smoothness / Damage Force'][:,0] = torch.max( torch.linalg.norm(observation['info']['dmg_force'][:,:,:3], axis=2), 1)[0] # look at only force
            self.obs['Smoothness / Damage Torque'][:,0] =torch.max( torch.linalg.norm(observation['info']['dmg_force'][:,:,3:], axis=2), 1)[0]
        else:
            self.obs['Smoothness / Damage Force'][:,0] = torch.max( torch.linalg.norm(observation['info']['dmg_force'][:,None, :3], axis=2), 1)[0] # look at only force
            self.obs['Smoothness / Damage Torque'][:,0] =torch.max( torch.linalg.norm(observation['info']['dmg_force'][:,None, 3:], axis=2), 1)[0]

        info['smoothness'] = self.obs
        
        return observation, r, term, trun, info