from typing import List, Optional, Union, Any

import atexit
import sys
import tqdm

import torch

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper



class AgentList():
    def __init__(self, agent_list: Union[Agent, List[Agent]], agents_scope=None, cfg=None):
        self.agents = agent_list
        self.is_list = cfg.get("agent_is_list", False)
        self.agents_scope = agents_scope
        

    def init(self, trainer_cfg):
        if self.is_list:
            for agent in self.agents:
                agent.init(trainer_cfg=trainer_cfg)
        else:
            self.agents.init(trainer_cfg=trainer_cfg)


    def reset_memory(self):
        if self.is_list:
            for agent in self.agents:
                agent.memory.reset()
        else:
            try:
                self.agents.memory.reset()
            except AttributeError:
                self.agents.reset_memory()

    def track_video_path(self, tag: str, value: str, timestep)-> None:
        if self.is_list:
            for agent in self.agents:
                agent.track_video_path(tag, value, timestep)
        else:
            self.agents.track_video_path(tag, value, timestep)

    def track_data(self, tag: str, value: float) -> None:
        if self.is_list:
            for agent in self.agents:
                agent.track_data(tag, value)
        else:
            self.agents.track_data(tag, value)

    def write_tracking_data(self, timestep: int, timesteps: int, eval=False) -> None:
        """Write tracking data to TensorBoard

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if self.is_list:
            for agent in self.agents:
                agent.write_tracking_data(timestep, timesteps, eval)
        else:
            self.agents.write_tracking_data(timestep, timesteps, eval)

        self.reset_tracking()
            
    def act(self,
        states: torch.Tensor,
        timestep: int,
        timesteps: int) -> torch.Tensor:
        if self.is_list:
            return torch.vstack(
                [
                    agent.act(states[scope[0]:scope[1]], 
                        timestep=timestep, timesteps=timesteps)[0]    \
                            for agent, scope in zip(self.agents, self.agents_scope)
                ]
            )
        else:
            return self.agents.act(states, timestep=timestep, timesteps=timesteps)#[0]

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
                          infos: Any,
                          timestep: int,
                          timesteps: int,
                          reward_dist: dict,
                          term_dist: dict,
                          alive_mask: torch.Tensor = None) -> None:
        if self.is_list:
            for agent, scope in zip(self.agents, self.agents_scope):
                    agent_rew_dist = {}
                    for rew_type in reward_dist.keys():
                        agent_rew_dist[rew_type] = reward_dist[rew_type][scope[0]:scope[1]]

                    agent_term_dist = {}
                    for con in term_dist.keys():
                        agent_term_dist[con] = term_dist[con][scope[0]:scope[1]]

                    if alive_mask is None:
                        agent.record_transition(
                            states=states[scope[0]:scope[1]],
                            actions=actions[scope[0]:scope[1]],
                            rewards=rewards[scope[0]:scope[1]],
                            next_states=next_states[scope[0]:scope[1]],
                            terminated=terminated[scope[0]:scope[1]],
                            truncated=truncated[scope[0]:scope[1]],
                            infos=infos,
                            timestep=timestep,
                            timesteps=timesteps,
                            reward_dist = agent_rew_dist,
                            term_dist = agent_term_dist
                        )
                    else:
                        alive_mask[scope[0]:scope[1]] = agent.record_transition(
                            states=states[scope[0]:scope[1]],
                            actions=actions[scope[0]:scope[1]],
                            rewards=rewards[scope[0]:scope[1]],
                            next_states=next_states[scope[0]:scope[1]],
                            terminated=terminated[scope[0]:scope[1]],
                            truncated=truncated[scope[0]:scope[1]],
                            infos=infos,
                            timestep=timestep,
                            timesteps=timesteps,
                            reward_dist = agent_rew_dist,
                            term_dist = agent_term_dist,
                            alive_mask=alive_mask[scope[0]:scope[1]]
                        )
            return alive_mask
        else:
            return self.agents.record_transition(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=timesteps,
                reward_dist = reward_dist,
                term_dist = term_dist,
                alive_mask=alive_mask
            )

    def set_running_mode(self, mode: str) -> None:
        if self.is_list:
            for agent in self.agents:
                agent.set_running_mode(mode)
        else:
            self.agents.set_running_mode(mode)

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        if self.is_list:
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=timesteps)
        else:
            self.agents.pre_interaction(timestep=timestep, timesteps=timesteps)
    
    def post_interaction(self, timestep: int, timesteps: int, eval: bool = False) -> None:
        if self.is_list:
            for agent in self.agents:
                if eval:
                    super(type(agent), agent).post_interaction(timestep=timestep, timesteps=timesteps)
                else: 
                    agent.post_interaction(timestep=timestep, timesteps=timesteps)
        else:
            if eval:
                super(type(self.agents),self.agents).post_interaction(timestep=timestep, timesteps=timesteps)
            else:
                self.agents.post_interaction(timestep=timestep, timesteps=timesteps)

    def reset_tracking(self):
        if self.is_list:
            for agent in self.agents:
                agent.reset_tracking()
        else:
            self.agents.reset_tracking()