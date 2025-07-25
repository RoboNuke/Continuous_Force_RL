import torch
import torch.nn as nn
import numpy as np
from typing import Any, Mapping, Tuple, Union
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import math
from models.feature_net import NatureCNN, layer_init, he_layer_init
from torch.distributions import MixtureSameFamily, Normal, Bernoulli, Independent, Categorical

class SimBaLayer(nn.Module):
    def __init__(self, size, device):
        super().__init__()

        self.path = nn.Sequential(
            nn.LayerNorm(size),
            he_layer_init(nn.Linear(size, 4 * size)),
            nn.ReLU(),
            he_layer_init(nn.Linear(4 * size, size))
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.path(res)
        return out + res

class SimBaNet(nn.Module):
    def __init__(self, n, in_size, out_size, latent_size, device, tan_out=False):
        super().__init__()
        self.layers = []
        self.n = n
        self.input = nn.Sequential(
            he_layer_init(nn.Linear(in_size, latent_size))
        ).to(device)

        self.layers = nn.ModuleList([SimBaLayer(latent_size, device) for i in range(self.n)])

        if tan_out:
            self.output = nn.Sequential(
                nn.LayerNorm(latent_size),
                he_layer_init(nn.Linear(latent_size, out_size)),
                nn.Tanh()
            ).to(device)
        else:
            self.output = nn.Sequential(
                nn.LayerNorm(latent_size),
                he_layer_init(nn.Linear(latent_size, out_size))
            ).to(device)

    def forward(self, x):
        out =  self.input(x)
        for i in range(self.n):
            out = self.layers[i](out)
        out = self.output(out)
        return out


class SimBaAgent(GaussianMixin, DeterministicMixin, Model):
    def __init__(
            self, 
            observation_space,
            action_space,
            device, 
            act_init_std = 0.60653066, # -0.5 value used in maniskill
            critic_output_init_mean = 0.0,
            force_type=None, 
            critic_n = 1, 
            actor_n = 2,
            critic_latent=128,
            actor_latent=512,

            clip_actions=False,
            clip_log_std=True, 
            min_log_std=-20, 
            max_log_std=2, 
            reduction="sum",
            sigma_idx=0
        ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)
        self.feature_net = NatureCNN(
            obs_size=self.num_observations, 
            with_state=True,
            with_rgb=False,
            force_type=force_type
        )

        in_size = self.feature_net.out_features
        
        self.critic = SimBaNet(
            n=critic_n, 
            in_size=in_size, 
            out_size=1, 
            latent_size=critic_latent, 
            tan_out=False,
            device=device
        )
        self.sigma_idx = sigma_idx
        #self.actors = [BroNet(actor_n, in_size, self.act_size, actor_latent, device, tanh_out=True) for i in range(tot_actors)]
        self.actor_mean = SimBaNet(
            n=actor_n, 
            in_size=in_size, 
            out_size=self.num_actions, 
            latent_size=actor_latent, 
            device=device,
            tan_out=True
        )

        he_layer_init(self.critic.output[-1], bias_const=critic_output_init_mean) # 2.5 is about average return for random policy w/curriculum
        with torch.no_grad():
            self.actor_mean.output[-2].weight *= 0.01

        self.actor_logstd = nn.Parameter(
            torch.ones(1, self.num_actions) * math.log(act_init_std)
        )

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.feature_net(inputs)
            action_mean = self.actor_mean(self._shared_output)
            if self.sigma_idx > 0:
                action_mean[:,:self.sigma_idx] = (action_mean[:,:self.sigma_idx] + 1)/2.0
            return action_mean, self.actor_logstd.expand_as(action_mean), {}
        elif role == "value":
            shared_output = self.feature_net(inputs) if self._shared_output is None else self._shared_output
            self._shared_output = None
            val = self.critic(shared_output)
            return self.critic(shared_output), {}
        

class SimBaActor(GaussianMixin, Model):
    def __init__(
            self,
            observation_space,
            action_space,
            device, 
            act_init_std = 0.60653066, # -0.5 value used in maniskill 
            actor_n = 2,
            actor_latent=512,
            action_gain=1.0,

            clip_actions=False,
            clip_log_std=True, 
            min_log_std=-20, 
            max_log_std=2, 
            reduction="sum",
            sigma_idx=0
        ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        self.action_gain = action_gain
        self.actor_mean = SimBaNet(
            n=actor_n, 
            in_size=self.num_observations, 
            out_size=self.num_actions, 
            latent_size=actor_latent, 
            device=device,
            tan_out=True
        )
        with torch.no_grad():
            self.actor_mean.output[-2].weight *= 1.0 #TODO FIX THIS TO 0.01
            if sigma_idx > 0:
                self.actor_mean.output[-2].bias[:sigma_idx] = -1.1
        
        self.actor_logstd = nn.Parameter(
            torch.ones(1, self.num_actions) * math.log(act_init_std)
        )
        self.sigma_idx = sigma_idx

    def act(self, inputs, role):
        #print("policy act:", inputs, role)
        return GaussianMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        #print("Policy compute:", role, inputs['states'].size(), inputs['states'][:,:self.num_observations].size())
        action_mean = self.action_gain * self.actor_mean(inputs['states'][:,:self.num_observations])
        if self.sigma_idx > 0:
            action_mean[:,:self.sigma_idx] = (action_mean[:,:self.sigma_idx] + 1)/2.0
        #print(action_mean[0,:self.sigma_idx])
        return action_mean, self.actor_logstd.expand_as(action_mean), {}
           

class SimBaCritic(DeterministicMixin, Model):
    def __init__(self, 
            state_space_size,
            device, 
            critic_output_init_mean = 0.0,
            critic_n = 1, 
            critic_latent=128,
            clip_actions=False,
        ):
        Model.__init__(self, state_space_size, 1, device)
        DeterministicMixin.__init__(self, clip_actions)

        in_size = self.num_observations
        #print("state space size:", state_space_size)
        self.critic = SimBaNet(
            n=critic_n, 
            in_size=in_size, 
            out_size=1, 
            latent_size=critic_latent, 
            tan_out=False,
            device=device
        )

        he_layer_init(self.critic.output[-1], bias_const=critic_output_init_mean) 
        

    def act(self, inputs, role):
        #print("critic act:", inputs, role)
        return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        #print("critic compute:", role, inputs['states'].size(), inputs['states'][:,-self.num_observations:].size())
        return self.critic(inputs['states'][:,-self.num_observations:]), {}
