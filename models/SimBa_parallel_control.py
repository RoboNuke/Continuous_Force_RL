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
import torch.nn.functional as F

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
              

    def sample_gumbel_softmax(self,logits, tau=1.0, eps=1e-10):
        """
        logits: (batch, num_components)
        Returns: differentiable soft one-hot (batch, num_components)
        """
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        y_gumble = (logits + gumbel_noise) / tau
        #print("Gumm:", F.softmax(y_gumble[0,:,0],dim=-1)>0.5)
        # sample from uniform distribution
        U = torch.rand_like(logits[:,:,0]) # one per batch
        y_uniform = torch.rand_like(y_gumble[:,:,0])
        y = torch.zeros_like(y_gumble)
        gumble_idxs = U > 0.01
        y[gumble_idxs, ...] = y_gumble[gumble_idxs, ...]
        y[~gumble_idxs][:,0] = y_uniform[~gumble_idxs]
        y[~gumble_idxs][:,1] = 1.0 - y_uniform[~gumble_idxs]
        y = F.softmax(y, dim=-1)
        #print("Uni:", F.softmax(y_uniform[0,:],dim=-1)> 0.5)
        return torch.where(y>0.5, 1.0, 0.0)


    def sample(self, sample_shape=torch.Size()):
        # samples except moves to format consistant for action selection
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # mixture samples [n, B]
            probs = self.mixture_distribution.probs[:,:3,:]
            weights = self.sample_gumbel_softmax(probs)

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.rsample(sample_shape)
            #print(weights.size(), comp_samples.size())
            #pos_sample = torch.sum(weights * comp_samples, dim=-1)
            #samples = torch.gather(comp_samples, gather_dim, mix_sample_r).squeeze(gather_dim)

            out_samples = torch.cat( 
                (   
                    weights[:,:,0], 
                    comp_samples[:,:,0], 
                    (comp_samples[:,:,1] - self.component_distribution.mean[:,:,0])[:,:3] / self.f_w
                ), dim=1 
            )
            #print(out_samples[0,:])
            return out_samples

    def log_prob(self, action):
        """
        #  convert to GMM format
        #print(action[0,:])
        sel_matrix = action[:,:3] #torch.where(action[:,:3]>0.9, True, False)
        pose_action = action[:,3:9]
        force_action = action[:,9:] * self.f_w

        x = pose_action 
        x[:,0:3] += (1.0-sel_matrix) * force_action
        #print("Log Prob log_prob:", super().log_prob(x)[0,:].size())
        return super().log_prob(x)
        """

        #  convert to GMM format
        #print(action[0,:])
        batch_size = action.shape[0]
        logits = torch.ones((batch_size,6), device=action.device)
        logits[:,:3] = action[:,:3] 
        #logits[:,:3,1] = 1.0 - action[:,:3] 
        mean1 = action[:, 3:9]             # (batch, 6)
        offset = action[:, 9:12] * self.f_w   # (batch, 3)
        mean2 = mean1.clone()
        mean2[:,:3] += offset              # (batch, 6)

        # Combine means and expand to component axis
        means = torch.stack([mean1, mean2], dim=2)  # (batch, 3 dims, 2 components)

        log_probs = self.component_distribution.log_prob(means)
        
        log_weights = self.mixture_distribution.log_prob(logits)#F.log_softmax(logits, dim=-1)

        log_weights = torch.stack([log_weights, (1-log_weights.exp()).log()], dim=2)
        #print(log_weights[0,:].exp())
        log_mix = torch.logsumexp(log_weights + log_probs, dim=-1)
        return log_mix


class ParallelGMMMixin(GaussianMixin):
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
        
        batch_size = mean_actions.shape[0]

        logits = torch.ones((batch_size,6), device=mean_actions.device)
        logits[:,:3] = mean_actions[:,:3]            # (batch, 6) – one logit per dim for 2 components
        mean1 = mean_actions[:, 3:9]             # (batch, 6)
        offset = mean_actions[:, 9:12] * self.force_scale    # (batch, 3)
        mean2 = mean1.clone()
        mean2[:,:3] += offset              # (batch, 6)

        # Combine means and expand to component axis
        means = torch.stack([mean1, mean2], dim=2)  # (batch, 3 dims, 2 components)

        # std
        std = torch.ones_like(means)
        std[:,:,0] = log_std[:,:6]
        std[:,:3,1] = log_std[:,6:9]
        std[:,3:,1] = (log_std[:,3:6]**2) * (self.force_scale**4) + (log_std[:,:3]**2)

        # Create component Gaussians: Normal(loc, scale)
        components = Normal(loc=means, scale=std.exp())  # (batch, 6, 2)

        # Categorical logits need to be expanded to match components
        mix_logits = torch.zeros((batch_size, 6, 2), device=mean_actions.device)
        mix_logits[:,:,0] = logits[:,:]
        mix_logits[:,:,1] = 1-logits[:,:]
        
        mix_dist = Categorical(probs=mix_logits)
        
        self._g_distribution = ActionGMM(mix_dist, components, force_weight=self.force_scale)

        actions = self._g_distribution.sample()
        #print("Act:", actions[0,:])
        # clip actions
        if self._g_clip_actions:
            actions = torch.clamp(actions, min=self._g_clip_actions_min, max=self._g_clip_actions_max)

        # log of the probability density function
        log_prob = self._g_distribution.log_prob(inputs.get("taken_actions", actions))
        #print("Act log prob:", log_prob.size())
        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)
        #print("Act log prob redux:", log_prob.size())
        
        outputs["mean_actions"] = mean_actions
        return actions, log_prob, outputs
    
class ParallelControlSimBaActor(ParallelGMMMixin, Model):
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
        ParallelGMMMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, force_scale=force_scale)
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
            self.actor_mean.output[-2].bias[:3] = -1.1
            
        self.actor_logstd = nn.Parameter(
            torch.ones(1, 9) * math.log(act_init_std)
        )
        

    def act(self, inputs, role):
        #print("policy act:", inputs, role)
        return ParallelGMMMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        #print("Policy compute:", role, inputs['states'].size(), inputs['states'][:,:self.num_observations].size())
        #print(inputs['states'][:,:self.num_observations])
        action_mean = self.action_gain * self.actor_mean(inputs['states'][:,:self.num_observations])
        #print("raw action mean:", action_mean)
        action_mean[:,:3] = (action_mean[:,:3] + 1.0) / 2.0
        #print("final:", action_mean[:,:3])
        return action_mean, self.actor_logstd.expand(action_mean.size()[0],-1), {}
    

  
