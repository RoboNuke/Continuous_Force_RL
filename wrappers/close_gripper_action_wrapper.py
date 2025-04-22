from gym import Env, spaces
import gym
from gym.spaces import Box
import torch

class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        #ub = self.env.action_space
        #self.action_space = Box(ub.low[:6], ub.high[:6])
        #self.new_action = torch.zeros((self.unwrapped.num_envs, 7), device=self.unwrapped.device)

    def action(self, action: torch.tensor) -> torch.tensor:
        #print("new_action:", self.new_action[:,:6].size(), "\taction:", action.size())
        #self.new_action[:,:6] = action.clone()
        #self.new_action[:,6] = -1.0
        #return self.new_action
        action[:,-1] = -1.0
        return action

    def step(self, action):
        #print(action.size())
        obs, rew, done, truncated, info = self.env.step(self.action(action))
        return obs, rew, done, truncated, info