import torch
import torch.nn as nn
import numpy as np
from typing import Any, Mapping, Tuple, Union, Optional
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import math
from models.feature_net import NatureCNN, layer_init, he_layer_init
from torch.distributions import MixtureSameFamily, Normal, Bernoulli, Independent, Categorical
from models.SimBa import SimBaNet
from torch.distributions.distribution import Distribution

class ActionGMM(MixtureSameFamily):
    def __init__(
        self,
        mixture_distribution: Categorical,
        component_distribution: Distribution,
        validate_args: Optional[bool] = None,
        force_weight: int = 1.0,
    ) -> None:
        super().__init__(mixture_distribution, component_distribution, validate_args)
        self.f_w = force_weight
              

    def sample(self, sample_shape=torch.Size()):
        # samples except moves to format consistant for action selection
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # mixture samples [n, B]
            #print("Sample Shape:", sample_shape, self.f_w)
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.sample(sample_shape)
            
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1))
            )
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es
            )
            #samples = torch.gather(comp_samples, gather_dim, mix_sample_r).squeeze(gather_dim)
            out_samples = torch.cat( 
                (   
                    mix_sample_r[:,:3,0], 
                    comp_samples[:,:,0], 
                    (comp_samples[:,:,1] - self.component_distribution.mean[:,:,0])[:,:3] / self.f_w
                ), dim=1 
            )
            return out_samples

    def log_prob(self, action):
        #  convert to GMM format
        sel_matrix = torch.where(action[:,:3] > 0, True, False)
        pose_action = action[:,3:9]
        force_action = action[:,9:] * self.f_w

        x = pose_action 
        x[:,0:3] += ~sel_matrix * force_action
        return super().log_prob(x)


class HybridGMMMixin(GaussianMixin):
    def __init__(
        self,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
        role: str = "",
        force_scale=1.0
    ) -> None:
        super().__init__(clip_actions, clip_log_std, min_log_std, max_log_std, reduction, role)
        self.force_scale = force_scale

    def act(
        self, 
        inputs: Mapping[str, Union[torch.Tensor, Any]], 
        role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act stochastically in response to the state of the environment"""
        
        # map from states/observations to mean actions and log standard deviations
        mean_actions, log_std, outputs = self.compute(inputs, role)
        
        num_envs = mean_actions.size()[0]


        new_log_std = torch.zeros((num_envs,6,2), device=log_std.device)
        new_log_std[:,:,0] = log_std[:,:6].view((num_envs,6))
        new_log_std[:,:3,1] = log_std[:,6:].view((num_envs,3))
        new_log_std[:,3:,1] = log_std[:,3:6].view(num_envs,3)
        log_std = new_log_std + 1e-6
        
        # clamp log standard deviations
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)

        self._g_log_std = log_std
        self._g_num_samples = mean_actions.shape[0]
        
        # distribution
        holder=torch.zeros((num_envs,6,2), device=log_std.device) # batch size, action dim, distribution
        holder[:,:,0] = mean_actions[:,:3].repeat(1,2)
        holder[:,:,1] = (1-mean_actions[:,:3]).repeat(1,2)
        #print("Holder:", holder)
        b_dist = Categorical(holder)
        
        dist_means = torch.zeros((num_envs,6,2), device=log_std.device) # batch size, action dim, distributions
        dist_means[:,:,0] = mean_actions[:,3:9]
        dist_means[:,:3,1] = self.force_scale * mean_actions[:,3:6] + mean_actions[:,9:]
        dist_means[:,3:,1] = mean_actions[:, 6:9]
        
        error_dist = Normal(dist_means, log_std)
        
        self._g_distribution = ActionGMM(b_dist, error_dist, force_weight=self.force_scale)

        actions = self._g_distribution.sample()

        # clip actions
        if self._g_clip_actions:
            actions = torch.clamp(actions, min=self._g_clip_actions_min, max=self._g_clip_actions_max)

        # log of the probability density function
        log_prob = self._g_distribution.log_prob(inputs.get("taken_actions", actions))
        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["mean_actions"] = mean_actions
        return actions, log_prob, outputs
    
class HybridControlSimBaActor(HybridGMMMixin, Model):
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
        force_scale=1.0
        ):
        Model.__init__(self, observation_space, action_space, device)
        HybridGMMMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, force_scale=force_scale)
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
            
        self.actor_logstd = nn.Parameter(
            torch.ones(1, 9) * math.log(act_init_std)
        )
        

    def act(self, inputs, role):
        #print("policy act:", inputs, role)
        return HybridGMMMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        #print("Policy compute:", role, inputs['states'].size(), inputs['states'][:,:self.num_observations].size())
        #print(inputs['states'][:,:self.num_observations])
        action_mean = self.action_gain * self.actor_mean(inputs['states'][:,:self.num_observations])
        #print("raw action mean:", action_mean)
        action_mean[:,:3] = (action_mean[:,:3] + 1.0) / 2.0
        #print("final:", action_mean)
        return action_mean, self.actor_logstd.expand(action_mean.size()[0],-1), {}
    

  