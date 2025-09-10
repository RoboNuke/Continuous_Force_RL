from agents.MultiWandbLoggerPPO import MultiWandbLoggerPPO, EpisodeTracker
from models.SimBa import SimBaNet
from models.block_simba import export_policies, make_agent_optimizer
from typing import Any, Mapping, Optional, Tuple, Union
from typing import Any, Mapping, Optional, Tuple, Union
import copy
import os
from skrl.models.torch import Model
from skrl import config, logger
from skrl.memories.torch import Memory

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from agents.wandb_logger_ppo_agent import WandbLoggerPPO
from typing import Any, Mapping, Optional, Tuple, Union

#import gym
import gymnasium as gym
import skrl
import torch
import torch.nn as nn
import numpy as np
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl import config, logger

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from data_processing.data_manager import DataManager

import wandb

import collections
import subprocess
from filelock import FileLock

import itertools
from packaging import version

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.resources.schedulers.torch import KLAdaptiveLR
import statistics


class BlockWandbLoggerPPO(MultiWandbLoggerPPO):
    def __init__(
            self,
            models: Mapping[str, Model],
            memory: Optional[Union[Memory, Tuple[Memory]]] = None,
            observation_space: Optional[Union[int, Tuple[int], gym.Space, gym.Space]] = None,
            action_space: Optional[Union[int, Tuple[int], gym.Space, gym.Space]] = None,
            num_agents: int = 1,
            num_envs: int = 256,
            device: Optional[Union[str, torch.device]] = None,
            state_size=-1,
            cfg: Optional[dict] = None,
            track_ckpt_paths = False,
            task_cfg=None,
            task="Isaac-Factory-PegInsert-Local-v0",
            optimizer = None
    ) -> None:
        super().__init__(
            models=models, 
            memory=memory, 
            observation_space=observation_space, 
            action_space=action_space, 
            num_agents=num_agents, 
            num_envs=num_envs, 
            device=device, 
            state_size=state_size, 
            cfg=cfg, 
            track_ckpt_paths=track_ckpt_paths, 
            task=task, 
            optimizers=optimizer, 
            task_cfg=task_cfg
        )


    def write_checkpoint(self, timestep: int, timesteps: int):
        ckpt_paths = []
        critic_paths = []
        vid_paths = []
        for i in range(self.num_agents):
            exp_args = self.agent_exp_cfgs[i]['experiment'] #self.cfg[f'agent_{i}']['experiment']
            ckpt_paths.append(os.path.join(exp_args['directory'], exp_args['experiment_name'], "checkpoints", f"agent_{i}_{timestep*self.num_envs // self.num_agents}.pt"))
            vid_paths.append( os.path.join(exp_args['directory'], exp_args['experiment_name'], "eval_videos", f"agent_{i}_{timestep* self.num_envs // self.num_agents}.gif"))
            critic_paths.append(os.path.join(exp_args['directory'], exp_args['experiment_name'], "checkpoints", f"critic_{i}_{timestep*self.num_envs // self.num_agents}.pt"))
                             
        export_policies(self.models['policy'].actor_mean, self.models['policy'].actor_logstd, ckpt_paths)
        export_policies(self.models['value'].critic, None, critic_paths)
        
        #self.track_data("ckpt_video", (timestep, vid_path) )
        if self.track_ckpt_paths:
            lock = FileLock(self.tracker_path + ".lock")
            with lock:
                with open(self.tracker_path, "a") as f:
                    for i in range(self.num_agents):
                        f.write(f'{ckpt_paths[i]} {self.task_name} {vid_paths[i]} {self.loggers[i].wandb_cfg["project"]} {self.loggers[i].wandb_cfg["run_id"]}\n')
    """
    def _log_minibatch_update(
            self,
            returns=None, #num_samples x num_agents x num_envs_per_agent x dim
            values=None,
            advantages=None,
            old_log_probs=None,
            new_log_probs=None,
            entropies=None,
            policy_losses=None,
            value_losses=None,
            policy_state = None,
            critic_state = None
    ):
        for i, logger in enumerate(self.loggers):
            state = self._get_block_network_state(i)
            
            logger.log_minibatch_update(
                returns[:,i,:],
                values[:,i,:],
                advantages[:,i,:],
                old_log_probs[:,i,:],
                new_log_probs[:,i,:],
                entropies[:,i,:],
                policy_losses[:,i,:],
                value_losses[:,i,:],
                state['policy'],
                state['critic'],
                self.optimizer
            )
    """
    """
    def _get_block_network_state(self, agent_idx):
        #Extract optimizer state for one agent, returning separate dicts for
        #policy and critic. Assumes optimizer param_groups were created with
        #make_agent_optimizer (policy first, critic second per agent).
        
        state = {
            "policy": {"gradients":[], "weight_norms":{}, "optimizer_state":{}},
            "critic":{"gradients":[], "weight_norms":{}, "optimizer_state":{}}
        }

        # Index mapping: policy group = 2*agent_idx, critic group = 2*agent_idx+1
        group_indices = {"policy": 2 * agent_idx, "critic": 2 * agent_idx + 1}
        
        for kind, gidx in group_indices.items():
            param_group = self.optimizer.param_groups[gidx]
            
            for param in param_group["params"]:
                name = getattr(param, "_name", f"param_{id(param)}")

                # Gradients
                if param.grad is not None:
                    state[kind]['gradients'].append(param.grad.detach().norm(2))

                    if param in self.optimizer.state:
                        op_state = self.optimizer.state[param]
                        state[kind]['optimizer_state'][name] = {
                            "exp_avg": op_state["exp_avg"].norm().item(),
                            "exp_avg_sq": op_state["exp_avg_sq"].norm().item(),
                            "step": op_state["step"],
                        }

                    # Optional: clear grad after collection
                    param.grad = None

                # Weight norms
                state[kind]['weight_norms'][name] = param.norm().item()

        return state
    """
    def _get_block_network_state(self, agent_idx):
        """
        Extract optimizer state for one agent, returning separate dicts for
        policy and critic. Assumes optimizer param_groups were created with
        make_agent_optimizer (policy first, critic second per agent).
        """
        state = {
            "policy": {"gradients":[], "weight_norms":{}, "optimizer_state":{}},
            "critic":{"gradients":[], "weight_norms":{}, "optimizer_state":{}}
        }
        for role in ['critic','policy']:
            net = self.value if role=='critic' else self.policy
            for pname, p in net.named_parameters():
                #role = getattr(p, "role")
                if p.grad is not None:
                    try:
                        state[role]['gradients'].append(p.grad.detach()[agent_idx,:,:].norm(2))
                    except IndexError:
                        #print(pname, " re-indexed")
                        state[role]['gradients'].append(p.grad.detach()[agent_idx,:].norm(2))

                    if pname in self.optimizer.state:
                        # this is the same for every agent since they share the optimizer
                        op_state = self.optimizer.state[pname]
                        state[role]['opimizer_state'][pname] = {
                            "exp_avg": op_state["exp_avg"].norm().item(),
                            "exp_avg_sq": op_state["exp_avg_sq"].norm().item(),
                            "step": op_state["step"],
                        }
                    
                    # Optional: clear grad after collection
                    p.grad = None

                # Weight norms
                state[role]['weight_norms'][pname] = p.norm().item()

        return state
                        

