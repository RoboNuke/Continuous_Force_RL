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

class HybridActionGMM(MixtureSameFamily):
    def __init__(
            self,
            mixture_distribution: Categorical,
            component_distribution: Distribution,
            validate_args: Optional[bool] = None,
            pos_weight: float = 1.0,
            rot_weight: float = 1.0,
            force_weight: float = 1.0,
            torque_weight: float = 1.0,
            uniform_rate: float = 0.01,
            ctrl_torque: bool = False
    ) -> None:
        super().__init__(mixture_distribution, component_distribution, validate_args)
        self.f_w = force_weight
        self.p_w = pos_weight
        self.r_w = rot_weight
        self.t_w = torque_weight
        self.force_size = 6 if ctrl_torque else 3
        self.uniform_rate = uniform_rate
              

    def sample_gumbel_softmax(self,logits, tau=1.0, eps=1e-10):
        """
        logits: (batch, num_components)
        Returns: differentiable soft one-hot (batch, num_components)
        """
        # create sample with gumbel noise
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        y_gumble = F.softmax((logits + gumbel_noise) / tau,dim=-1)
        
        # sample from uniform distribution
        U = torch.rand_like(logits[:,:,0]) # one per batch
        y_uniform = torch.rand_like(y_gumble[:,:,0])
        gumble_idxs = U > self.uniform_rate

        # zip together final product
        y = torch.zeros_like(y_gumble)
        y[:,:,0][gumble_idxs] = y_gumble[:,:,0][gumble_idxs]
        y[:,:,1][gumble_idxs] = y_gumble[:,:,1][gumble_idxs]
        y[:,:,0][~gumble_idxs] = y_uniform[~gumble_idxs]
        y[:,:,1][~gumble_idxs] = 1.0 - y_uniform[~gumble_idxs]
        return y


    def sample(self, sample_shape=torch.Size()):
        # samples except moves to format consistant for action selection
        with torch.no_grad():
            # mixture samples [n, B]
            probs = self.mixture_distribution.probs.log()
            weights = self.sample_gumbel_softmax(probs)

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.rsample(sample_shape)

            if self.force_size > 3:
                out_samples = torch.cat( 
                    (   
                        weights[:,:,0], 
                        comp_samples[:,0:3,0] / self.p_w,
                        comp_samples[:,3:6,0] / self.r_w,
                        comp_samples[:,0:3,1] / self.f_w,
                        comp_samples[:,3:6,1] / self.t_w
                    ), dim=1 
                )
            else:
                #print("weights:", weights.size())
                out_samples = torch.cat( 
                    (   
                        weights[:,:3,0], 
                        comp_samples[:,0:3,0] / self.p_w,
                        comp_samples[:,3:6,0] / self.r_w,
                        comp_samples[:,0:3,1] / self.f_w
                    ), dim=1 
                )
                
            
            return out_samples
        

    def log_prob(self, action):
        """ Action: batch size x 18
            - 0:6(3) - selection
            - 6(3):12(9) - position sample
            - 12(9):18(12) - force sample
        """

        #  convert to GMM format
        batch_size = action.shape[0]
        logits = torch.ones((batch_size,6), device=action.device)
        logits[:,:self.force_size] = action[:,:self.force_size] 
        #logits[:,:3,1] = 1.0 - action[:,:3] 
        mean1 = action[:, self.force_size:self.force_size+6] * self.p_w         # (batch, 6)
        mean1[:,3:] *= self.r_w / self.p_w
        mean2 = torch.zeros_like(mean1)
        mean2[:,:3] = action[:, self.force_size+6:6+2*self.force_size] * self.f_w
        if self.force_size > 3:
            mean2[:,3:] *= self.t_w / self.f_w
        else:
            mean2[:,3:] = mean1[:,3:]

        # Combine means and expand to component axis
        means = torch.stack([mean1, mean2], dim=2)  # (batch, 6 dims, 2 components)

        log_probs = self.component_distribution.log_prob(means)
        log_weights = self.mixture_distribution.log_prob(logits)#F.log_softmax(logits, dim=-1)

        log_weights = torch.stack([log_weights, (1-log_weights.exp()).log()], dim=2)
        log_mix = torch.logsumexp(log_weights + log_probs, dim=-1)
        return log_mix


class HybridGMMMixin(GaussianMixin):
    # note the scales = (control_prop_gain * threshold)
    def __init__(
            self,
            clip_actions: bool = False,
            clip_log_std: bool = True,
            min_log_std: float = -20,
            max_log_std: float = 2,
            reduction: str = "sum",
            role: str = "",
            pos_scale=1.0,
            rot_scale=1.0,
            force_scale=1.0,
            torque_scale=1.0,
            ctrl_torque=False
    ) -> None:
        super().__init__(clip_actions, clip_log_std, min_log_std, max_log_std, reduction, role)
        self.force_size = 6 if ctrl_torque else 3
        self.force_scale = force_scale
        self.torque_scale= torque_scale
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        
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
        logits[:,:self.force_size] = mean_actions[:,:self.force_size]            # (batch, 6) – one logit per dim for 2 components
        
        mean1 = mean_actions[:, self.force_size:self.force_size+6] * self.pos_scale            # (batch, 6)
        mean1[:,3:6] *= self.rot_scale / self.pos_scale

        mean2 = torch.zeros_like(mean1)
        mean2[:,:self.force_size] = mean_actions[:, 6+self.force_size:6+2*self.force_size] * self.force_scale    # (batch, 6)
        if self.force_size > 6:
            mean2[:,3:6] *= self.torque_scale / self.force_scale
        else:
            mean2[:,3:6] = mean1[:,3:6] #set to position values to make mixture work
        
        # Combine means and expand to component axis
        means = torch.stack([mean1, mean2], dim=2)  # (batch, 3 dims, 2 components)

        # std
        std = torch.ones_like(means)
        std[:,:3,0] = log_std[:,:3].exp() * self.pos_scale ** 2 #10000
        std[:,3:,0] = log_std[:,3:6].exp() * self.rot_scale ** 2 #900
        std[:,:3,1] = log_std[:,6:9].exp() * self.force_scale ** 2 #0.25
        if self.force_size > 3:
            std[:,3:,1] = log_std[:,9:].exp() * self.torque_scale ** 2 #0.25
        else:
            std[:,3:,1] = std[:,3:,0]
            
        # Create component Gaussians: Normal(loc, scale)
        components = Normal(loc=means, scale=std)  # (batch, 6, 2)

        # Categorical logits need to be expanded to match components
        mix_logits = torch.zeros((batch_size, 6, 2), device=mean_actions.device)
        mix_logits[:,:,0] = logits[:,:]
        mix_logits[:,:,1] = 1-logits[:,:]
        
        mix_dist = Categorical(probs=mix_logits)
        
        self._g_distribution = HybridActionGMM(
            mix_dist, components,
            force_weight=self.force_scale,
            torque_weight = self.torque_scale,
            pos_weight=self.pos_scale,
            rot_weight = self.rot_scale,
            ctrl_torque = self.force_size > 3
        )

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
            force_scale=1.0,
            torque_scale=1.0,
            pos_scale=1.0,
            rot_scale=1.0,
            ctrl_torque=False
    ):
        Model.__init__(self, observation_space, action_space, device)
        HybridGMMMixin.__init__(
            self,
            clip_actions,
            clip_log_std,
            min_log_std,
            max_log_std,
            reduction,
            pos_scale=pos_scale,
            rot_scale=rot_scale,
            force_scale=force_scale,
            torque_scale=torque_scale,
            ctrl_torque=ctrl_torque
        )
        self.action_gain = action_gain
        self.actor_mean = SimBaNet(
            n=actor_n, 
            in_size=self.num_observations, 
            out_size=self.num_actions, 
            latent_size=actor_latent, 
            device=device,
            tan_out=True
        )
        self.force_size = 6 if ctrl_torque else 3
        with torch.no_grad():
            #self.actor_mean.output[-2].weight *= 0.1 #1.0 #TODO FIX THIS TO 0.01
            self.actor_mean.output[-2].bias[:self.force_size] = -1.1
            
        self.actor_logstd = nn.Parameter(
            torch.ones(1, 6 + self.force_size) * math.log(act_init_std)
        )
        

    def act(self, inputs, role):
        return HybridGMMMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        action_mean = self.action_gain * self.actor_mean(inputs['states'][:,:self.num_observations])
        action_mean[:,:self.force_size] = (action_mean[:,:self.force_size] + 1.0) / 2.0
        return action_mean, self.actor_logstd.expand(action_mean.size()[0],-1), {}
    

  
