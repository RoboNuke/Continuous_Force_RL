
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
              

    def sample_gumbel_softmax(self,logits, sample_shape=torch.Size(), tau=1.0, eps=1e-10):
        """
        logits: (batch, num_components)
        Returns: differentiable soft one-hot (batch, num_components)
        """
        # create sample with gumbel noise
        #print("Sample Size:", sample_shape)
        #print("comb:", (sample_shape + logits.size()))
        U = torch.rand((sample_shape + logits.size()), device=logits.device)
        #print("U size:", U.size())
                       
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        y_gumble = F.softmax((logits + gumbel_noise) / tau,dim=-1)
        #print("gumble:", gumbel_noise.size(), y_gumble.size())
        # sample from uniform distribution
        U = torch.rand_like(gumbel_noise[...,0]) # one per batch
        y_uniform = torch.rand_like(y_gumble[...,0])
        gumble_idxs = U > self.uniform_rate
        #print(U.size(), y_uniform.size())

        # zip together final product
        y = torch.zeros_like(y_gumble)
        #print("y:", y.size(), y_gumble.size(), gumble_idxs.size())
        y[...,0][gumble_idxs] = y_gumble[...,0][gumble_idxs]
        y[...,1][gumble_idxs] = y_gumble[...,1][gumble_idxs]
        y[...,0][~gumble_idxs] = y_uniform[~gumble_idxs]
        y[...,1][~gumble_idxs] = 1.0 - y_uniform[~gumble_idxs]
        return y

    """
    def sample(self, sample_shape=torch.Size()):
        # samples except moves to format consistant for action selection
        #print(f"Init  sample Memory: {torch.cuda.memory_allocated(torch.device('cuda'))}")
        with torch.no_grad():
            # mixture samples [n, B]
            probs = self.mixture_distribution.probs
            probs = probs.expand(sample_shape + probs.shape)
            comp_samples = self.component_distribution.rsample(sample_shape)
            
            # sample 0 or 1
            rand = torch.rand_like(probs[...,0])
            weights = (rand <= probs[...,0]).float()
            
            # uniform replacement
            uniform_mask = torch.rand_like(weights) < self.uniform_rate
            weights[uniform_mask] = torch.rand_like(weights)[uniform_mask]

            # create output
            out_samples = torch.cat( 
                (   
                    weights[...,:self.force_size], 
                    comp_samples[...,0:6,0] / self.p_w,
                    #comp_samples[...,3:6,0] / self.r_w,
                    comp_samples[...,0:self.force_size,1] / self.f_w
                ), dim=-1
            )
            
            out_samples[...,self.force_size+3:self.force_size+6] *= self.p_w / self.r_w
            if self.force_size > 3:
                out_samples[...,3:6,1] * self.f_w / self.t_w
                
            return out_samples
        
    def log_prob(self, action):
        #with torch.no_grad():
        log_z = (self.mixture_distribution.probs)#.log()
        # extract component inputs
        mean1 = action[..., self.force_size:self.force_size+6] * self.p_w         # (batch, 6)
        mean1[..., 3:] *= self.r_w / self.p_w
        
        mean2 = torch.zeros_like(mean1)
        mean2[...,:self.force_size] = action[..., self.force_size+6:6+2*self.force_size] * self.f_w
        if self.force_size > 3:
            mean2[...,3:] *= self.t_w / self.f_w
        else:
            mean2[...,3:] = mean1[...,3:]
        

        real_action = torch.ones_like(mean1)
        pos_action_idx = action[...,:self.force_size] > 0.5
        real_action[...,:self.force_size][...,pos_action_idx] = mean1[...,:self.force_size][...,pos_action_idx]
        real_action[...,:self.force_size][...,~pos_action_idx] = mean2[...,:self.force_size][...,~pos_action_idx]
        if self.force_size == 3:
            real_action[...,3:] = mean1[...,3:]
        
        #means = real_action.unsqueeze(1)
        
        # Combine means and expand to component axis
        means = torch.stack([mean1, mean2], dim=-1)  # (batch, 6 dims, 2 components)
        #means = torch.stack([real_action, real_action],dim=-1)
        
        component_logs = self.component_distribution.log_prob(means)#real_action)
        
        log_prob = torch.logsumexp(log_z + component_logs, dim=-1).sum(-1)
        
        return log_prob
    """    
        
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape

            # sample 0 or 1
            #uniform_mask = torch.rand_like(mix_sample) < self.uniform_rate
            #mix_sample[uniform_mask] = (torch.rand_like(mix_sample)[uniform_mask] > 0.5).float()

            # component samples [n, B, k, E] [n, 2]
            comp_samples = self.component_distribution.sample(sample_shape)
            #print("Pre cat shapes:", mix_sample.size(), comp_samples.size())
            out_samples = torch.cat(
                (
                    mix_sample[:,:self.force_size],
                    comp_samples[...,:,0],
                    #comp_samples[...,0,3:6] / self.r_w,
                    comp_samples[...,:self.force_size,1] 
                ), dim=-1
            )
            #print("pos_cat size:", out_samples.size())
            # scale everything
            out_samples[..., self.force_size:self.force_size+3] /= self.p_w
            out_samples[...,self.force_size+3:self.force_size+6] /= self.r_w
            out_samples[...,self.force_size+6:9+self.force_size] /= self.f_w # 12 or 15
            if self.force_size > 3:
                out_samples[...,self.force_size+9:] /= self.t_w
                
            return out_samples

    def log_prob(self, action):
        # extract samples
        print("Log Pos Actions:", torch.max(action[:,3:6]).item(), torch.min(action[:,3:6]).item(), torch.median(action[:,3:6]).item())
        print("Log Rot Actions:", torch.max(action[:,6:9]).item(), torch.min(action[:,6:9]).item(), torch.median(action[:,6:9]).item())
        print("Log For Actions:", torch.max(action[:,9:12]).item(), torch.min(action[:,9:12]).item(), torch.median(action[:,9:12]).item())
        print("Log Weights:", self.f_w, self.t_w, self.r_w, self.p_w)
        #print(f"Action:{action.size()}")
        mean1 = action[...,self.force_size:self.force_size+6]
        mean1[...,:3] *= self.p_w
        mean1[...,3:6] *= self.r_w
        print("Log Mean1 Pos Dist:", torch.max(mean1[:,:3]).item(), torch.min(mean1[:,:3]).item(), mean1[:,:3].median().item())
        print("Log Mean1 Rot Dist:", torch.max(mean1[:,3:]).item(), torch.min(mean1[:,3:]).item(), mean1[:,3:].median().item())
        
        mean2 = torch.zeros_like(mean1)
        mean2[...,:self.force_size] = action[...,self.force_size+6:6+2*self.force_size]
        mean2[...,:3] *= self.f_w
        if self.force_size > 3:
            mean2[...,3:] *= self.t_w
        else:
            mean2[...,3:] = mean1[...,3:]    
        print("Log Mean2 Pos Dist:", torch.max(mean2[:,:3]).item(), torch.min(mean2[:,:3]).item(), mean2[:,:3].median().item())
        print("Log Mean2 Rot Dist:", torch.max(mean2[:,3:]).item(), torch.min(mean2[:,3:]).item(), mean2[:,3:].median().item())

        means = torch.stack((mean1,mean2),dim=-1)
        
        comp_log_prob = self.component_distribution.log_prob(means)
        #force_log_prob = self.component_distribution.log_prob(means)
        log_z = (self.mixture_distribution.probs)
        print("Log Prob Mix:", torch.max(log_z).item(), torch.min(log_z).item(), log_z.median().item())
        log_z = log_z.log() #6,2
        
        # we use 6 instead of force_size so that we always include rotation
        # when force_size=3 pos and force log probs are equal for dims > 3 so it doesn't matter what is "selected"
        real_action = torch.where(action[...,:6] > 0.5, comp_log_prob[...,:6,1], comp_log_prob[...,:6,0])
        log_p_ra = torch.where(action[...,:6]  > 0.5, log_z[:,:,1], log_z[:,:,0])

        #log_prob = torch.logsumexp(log_p_ra + real_action, dim=0)
        log_prob = log_p_ra + real_action

        # when no torque control, there is no selection probability
        if self.force_size == 3:
            log_prob[:,self.force_size:] = real_action[:,self.force_size:]
        print("\n\n\n")
        return log_prob
    
    def entropy(self):
        #print("entropy check")
        samples = self.sample(sample_shape=(10000,))
        log_prob = self.log_prob(samples)
        return -log_prob.mean(dim=0)


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
        action = mean_actions
        print("Pos Actions:", torch.max(action[:,3:6]).item(), torch.min(action[:,3:6]).item(), torch.median(action[:,3:6]).item())
        print("Rot Actions:", torch.max(action[:,6:9]).item(), torch.min(action[:,6:9]).item(), torch.median(action[:,6:9]).item())
        print("For Actions:", torch.max(action[:,9:12]).item(), torch.min(action[:,9:12]).item(), torch.median(action[:,9:12]).item())
        #print("Mean Actions (act):", mean_actions.size())
        #print(inputs['policy'].size())
        # clamp log standard deviations
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)
            
        batch_size = mean_actions.shape[0]
        logits = mean_actions[:,:self.force_size]
        #print("Act logits:", logits)
        # Categorical logits need to be expanded to match components
        mix_logits = torch.zeros((batch_size, 6, 2), device=mean_actions.device)
        mix_logits[:, :self.force_size, 0] = logits[:,:]
        mix_logits[:, :self.force_size, 1] = 1-logits[:,:]
        if self.force_size < 6:
            mix_logits[:, self.force_size:, 0] = 1.0
            mix_logits[:, self.force_size:, 1] = 0.0

        print("Act dist:", torch.max(mix_logits).item(), torch.min(mix_logits).item(), mix_logits.median().item())
        mix_dist = Categorical(probs=mix_logits)

        
        #print("mix dist:", mix_dist.probs)
        mean1 = mean_actions[:, self.force_size:self.force_size+6]
        mean1[:,:3] *= self.pos_scale            # (batch, 6)
        mean1[:,3:6] *= self.rot_scale
        print("Mean1 Pos Dist:", torch.max(mean1[:,:3]).item(), torch.min(mean1[:,:3]).item(), mean1[:,:3].median().item())
        print("Mean1 Rot Dist:", torch.max(mean1[:,3:]).item(), torch.min(mean1[:,3:]).item(), mean1[:,3:].median().item())

        
        mean2 = torch.zeros_like(mean1)
        mean2[:,:self.force_size] = mean_actions[:, 6+self.force_size:6+2*self.force_size]
        mean2[:,:3] *= self.force_scale    # (batch, 6)
        if self.force_size > 3:
            mean2[:,3:6] *= self.torque_scale
        else:
            mean2[:,3:6] = mean1[:,3:6] #set to position values to make mixture work
        print("Mean2 Pos Dist:", torch.max(mean2[:,:3]).item(), torch.min(mean2[:,:3]).item(), mean2[:,:3].median().item())
        print("Mean2 Rot Dist:", torch.max(mean2[:,3:]).item(), torch.min(mean2[:,3:]).item(), mean2[:,3:].median().item())
            
        # Combine means and expand to component axis
        means = torch.stack([mean1, mean2], dim=2)  # (batch, 3 dims, 2 components)

        # std
        std = torch.ones_like(means)
        std[:, :3, 0] = log_std[:, 0:3].exp() * (self.pos_scale ** 2) #    36
        std[:, 3:, 0] = log_std[:, 3:6].exp() * (self.rot_scale ** 2) #   8.5
        std[:, :3, 1] = log_std[:, 6:9].exp() * (self.force_scale ** 2) # 1
        if self.force_size > 3:
            std[:, 3:, 1] = log_std[:, 9:].exp() * (self.torque_scale ** 2) #0.25
        else:
            std[:, 3:, 1] = std[:, 3:, 0]
        print("Stds:", std[0,:], log_std[0,:])
        # Create component Gaussians: Normal(loc, scale)
        components = Normal(loc=means, scale=std)  # (batch, 6, 2)
        print("Weights:", self.force_scale, self.torque_scale, self.rot_scale, self.pos_scale)
        self._g_distribution = HybridActionGMM(
            mix_dist, components,
            force_weight=self.force_scale,
            torque_weight = self.torque_scale,
            pos_weight=self.pos_scale,
            rot_weight = self.rot_scale,
            ctrl_torque = self.force_size > 3
        )
        actions = self._g_distribution.sample()
        print("Sampled Pos Actions:", torch.max(actions[:,3:6]).item(), torch.min(actions[:,3:6]).item(), torch.median(actions[:,3:6]).item())
        print("Sampled Rot Actions:", torch.max(actions[:,6:9]).item(), torch.min(actions[:,6:9]).item(), torch.median(actions[:,6:9]).item())
        print("Sampled For Actions:", torch.max(actions[:,9:12]).item(), torch.min(actions[:,9:12]).item(), torch.median(actions[:,9:12]).item())
        #print("Sampled Action:", actions.size())
    
        # clip actions
        if self._g_clip_actions:
            print("clipped actions")
            actions = torch.clamp(actions, min=self._g_clip_actions_min, max=self._g_clip_actions_max)

        # log of the probability density function
        log_prob = self._g_distribution.log_prob(inputs.get("taken_actions", actions))
        
        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)
        
        outputs["mean_actions"] = mean_actions
        assert 1==0
        return actions, log_prob, outputs
    
class HybridControlSimBaActor(HybridGMMMixin, Model):
    def __init__(
            self,
            observation_space,
            action_space,
            device, 
            pos_init_std = 0.60653066, # -0.5 value used in maniskill
            rot_init_std = 1.0,
            force_init_std = 1.0,
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
            ctrl_torque=False,
            sel_adj_types="none" #"bias_sel" "force_add", "zero_bias_weights", "scale_bias_pretan" 
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
        print("clipp std:", clip_log_std)
        #self.end_tanh = force_bias_type in ['none', 'bias_sel']

        
        self.actor_mean = SimBaNet(
            n=actor_n, 
            in_size=self.num_observations, 
            out_size=self.num_actions, 
            latent_size=actor_latent, 
            device=device,
            tan_out=False
        )
        self.force_size = 6 if ctrl_torque else 3
        print("force size:", self.force_size)

        self.sel_adj_types = sel_adj_types if type(sel_adj_types) == list else [sel_adj_types]
        print(self.sel_adj_types)
        self.init_bias = "init_bias" in self.sel_adj_types
        self.force_add = "force_add_zout" in self.sel_adj_types
        self.zero_weights = "zero_weights" in self.sel_adj_types
        self.scale_z = "scale_zout" in self.sel_adj_types
        stds = torch.ones(1, 6 + self.force_size)
        stds[0,:3] *= math.log(pos_init_std)
        stds[0,3:6] *= math.log(rot_init_std)
        stds[0,6:] *= math.log(force_init_std)
        print("init stds:", stds)
        self.actor_logstd = nn.Parameter(
            stds
            #torch.ones(1, 6 + self.force_size) #* math.log(act_init_std)
        )
        self.sig = torch.nn.Sigmoid()
        self.tan_h = torch.nn.Tanh()
        if self.force_add:
            print("prepped force_add")
            torch.autograd.set_detect_anomaly(True)
            self.force_bias_param = nn.Parameter( torch.ones(1, self.force_size) )
            self.tanh = nn.Tanh().to(device)

        if self.zero_weights:
            print("set zero weights")
            with torch.no_grad():
                self.actor_mean.output[-1].weight[:self.force_size,:] *= 0.0
                self.actor_mean.output[-1].bias[:self.force_size] *= 0.0
        
        if self.init_bias:
            print("set init bias")
            with torch.no_grad():
                #self.actor_mean.output[-2].weight *= 0.1 #1.0 #TODO FIX THIS TO 0.01
                self.actor_mean.output[-1].bias[:self.force_size] = 100 #-1.1 #-100.0

    def act(self, inputs, role):
        return HybridGMMMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        zout = self.action_gain * self.actor_mean(inputs['states'][:,:self.num_observations])
        # first force_size are sigma, remaining is tanh
        sels = zout[:,:self.force_size]
        not_sels = zout[:,self.force_size:]
        
        new_sel_def = False
        if self.force_add:
            # add force to sel
            f = torch.abs(inputs['states'][:, (self.num_observations - 6):self.num_observations - 6 + self.force_size])
            f_bias_s = F.softplus(self.force_bias_param.expand(zout.size()[0],-1)) * f 
            new_sels = sels + f_bias_s
            new_sel_def = True

        if self.scale_z:
            # multiply by factor
            print("scaling z")
            if new_sel_def:
                new_sels -= 100
            else:
                new_sels = sels - 100
            new_sel_def = True
        # tan sigma of sels
        if new_sel_def:
            final_sels = self.sig(new_sels)
        else:
            final_sels = self.sig(sels)
        #final_sels = (final_sels + 1) * 2.0
        
        final_not_sels = self.tan_h(not_sels)
        action_mean = torch.cat([final_sels, final_not_sels], dim=-1)
        return action_mean, self.actor_logstd.expand(action_mean.size()[0],-1), {}
    

  
