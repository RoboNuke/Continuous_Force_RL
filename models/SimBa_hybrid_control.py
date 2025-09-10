import torch
import torch.nn as nn
import numpy as np
from typing import Any, Mapping, Tuple, Union, Optional
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import math
from models.feature_net import NatureCNN, layer_init, he_layer_init
from torch.distributions import MixtureSameFamily, Normal, Bernoulli, Independent, Categorical
from models.SimBa import SimBaNet, ScaleLayer, MultiSimBaNet
from torch.distributions.distribution import Distribution
import torch.nn.functional as F
from models.block_simba import BlockSimBa

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
        self.sample_uniform = (self.uniform_rate > 0.0)
                
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            mix_sample = self.mixture_distribution.sample(sample_shape).float()
            mix_shape = mix_sample.shape
            # sample 0 or 1
            if self.sample_uniform:
                uniform_mask = torch.rand_like(mix_sample) < self.uniform_rate
                mix_sample[uniform_mask] = (torch.rand_like(mix_sample[uniform_mask]) > 0.5).float()
            # component samples [n, B, k, E] [n, 2]
            comp_samples = self.component_distribution.sample(sample_shape)
            out_samples = torch.cat(
                (
                    mix_sample[...,:self.force_size],
                    comp_samples[...,:,0],
                    comp_samples[...,:self.force_size,1] 
                ), dim=-1
            )
            # scale everything
            out_samples[..., self.force_size:self.force_size+3] /= self.p_w
            out_samples[...,self.force_size+3:self.force_size+6] /= self.r_w
            out_samples[...,self.force_size+6:9+self.force_size] /= self.f_w # 12 or 15
            if self.force_size > 3:
                out_samples[...,self.force_size+9:] /= self.t_w
            return out_samples

    def log_prob(self, action):
        # extract samples
        mean1 = action[...,self.force_size:self.force_size+6]
        mean1[...,:3] *= self.p_w
        mean1[...,3:6] *= self.r_w
        
        mean2 = torch.zeros_like(mean1)
        mean2[...,:self.force_size] = action[...,self.force_size+6:6+2*self.force_size]
        mean2[...,:3] *= self.f_w
        if self.force_size > 3:
            mean2[...,3:] *= self.t_w
        else:
            mean2[...,3:] = mean1[...,3:]    

        means = torch.stack((mean1,mean2),dim=-1)
        comp_log_prob = self.component_distribution.log_prob(means)

        if self.sample_uniform:
            log_z = (self.mixture_distribution.probs * (1 - self.uniform_rate) + self.uniform_rate/2.0).log()
        else:
            log_z = self.mixture_distribution.logits #(self.mixture_distribution.probs).log()
        
        log_prob = torch.logsumexp(log_z + comp_log_prob, dim=-1)
        return log_prob
    
    def entropy(self):
        samples = self.sample(sample_shape=(10000,))
        log_prob = self.log_prob(samples)
        return -(log_prob.exp() * log_prob).mean(dim=0)


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
            uniform_rate=0.0,
            ctrl_torque=False
    ) -> None:
        super().__init__(clip_actions, clip_log_std, min_log_std, max_log_std, reduction, role)
        self.force_size = 6 if ctrl_torque else 3
        self.force_scale = force_scale
        self.torque_scale= torque_scale
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.uniform_rate = uniform_rate
        
    def act(
        self, 
        inputs: Mapping[str, Union[torch.Tensor, Any]], 
        role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act stochastically in response to the state of the environment"""
        # map from states/observations to mean actions and log standard deviations
        mean_actions, log_std, outputs = self.compute(inputs, role)
        #print("Pos Actions:", torch.max(action[:,3:6]).item(), torch.min(action[:,3:6]).item(), torch.median(action[:,3:6]).item())
        
        # clamp log standard deviations
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)
            
        batch_size = mean_actions.shape[0]
        logits = mean_actions[:,:2*self.force_size].float()
        
        # Categorical logits need to be expanded to match components
        mix_logits = torch.zeros((batch_size, 6, 2), dtype=torch.float32, device=mean_actions.device)
        mix_logits[:,:self.force_size, :] = logits.view((batch_size, self.force_size, 2))
        #mix_logits[:, :self.force_size, 0] = (1.0 - logits.exp()).log()
        #mix_logits[:, :self.force_size, 1] = logits
        if self.force_size < 6:
            mix_logits[:, self.force_size:, 0] = -100.0 #0.0
            mix_logits[:, self.force_size:, 1] = 0.0 #1.0
            
        mix_dist = Categorical(probs=mix_logits)

        
        #print("mix dist:", mix_dist.probs)
        mean1 = mean_actions[:, 2*self.force_size:2*self.force_size+6]
        mean1[:,:3] *= self.pos_scale            # (batch, 6)
        mean1[:,3:6] *= self.rot_scale
        
        mean2 = torch.zeros_like(mean1)
        mean2[:,:self.force_size] = mean_actions[:, 6+2*self.force_size:6+3*self.force_size]
        mean2[:,:3] *= self.force_scale    # (batch, 6)
        if self.force_size > 3:
            mean2[:,3:6] *= self.torque_scale
        else:
            mean2[:,3:6] = mean1[:,3:6] #set to position values to make mixture work
            
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
            
        # Create component Gaussians: Normal(loc, scale)
        components = Normal(loc=means, scale=std)  # (batch, 6, 2)
        
        self._g_distribution = HybridActionGMM(
            mix_dist, components,
            force_weight=self.force_scale,
            torque_weight = self.torque_scale,
            pos_weight=self.pos_scale,
            rot_weight = self.rot_scale,
            ctrl_torque = self.force_size > 3,
            uniform_rate = self.uniform_rate
        )
        cols_to_keep = [0,2,4,5,6,7,8,9,10,11,12,13]
        mean_actions = mean_actions[:,cols_to_keep]
        actions = self._g_distribution.sample()
    
        # clip actions
        if self._g_clip_actions:
            actions = torch.clamp(actions, min=self._g_clip_actions_min, max=self._g_clip_actions_max)
        #ta = inputs.get("taken_actions", actions)
        
        # log of the probability density function
        log_prob = self._g_distribution.log_prob(inputs.get("taken_actions", actions).clone())
        
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
            hybrid_agent_parameters={},

            actor_n = 2,
            actor_latent=512,            
            clip_actions=False,
            clip_log_std=True, 
            min_log_std=-20, 
            max_log_std=2, 
            reduction="sum",
            num_agents=1
    ):

        # unpack cfg paramters
        pos_init_std = hybrid_agent_parameters['pos_init_std']
        rot_init_std = hybrid_agent_parameters['rot_init_std']
        force_init_std = hybrid_agent_parameters['force_init_std']

        pos_scale = hybrid_agent_parameters['pos_scale']
        rot_scale = hybrid_agent_parameters['rot_scale']
        force_scale = hybrid_agent_parameters['force_scale']
        torque_scale = hybrid_agent_parameters['torque_scale']
        ctrl_torque = hybrid_agent_parameters['ctrl_torque']
        uniform_rate = hybrid_agent_parameters['uniform_sampling_rate']
        
        sel_adj_types = hybrid_agent_parameters['selection_adjustment_types']
        
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
            ctrl_torque=ctrl_torque,
            uniform_rate=uniform_rate
        )
        print("[INFO]: Clipping Log STD" if clip_log_std else "[INFO]: No Lot STD Clipping")
        #self.end_tanh = force_bias_type in ['none', 'bias_sel']
        self.force_size = 6 if ctrl_torque else 3
        print("[INFO]: Controlling Torque" if self.force_size==6 else "[INFO]: Not Controlling Torque")

        self.num_agents = num_agents
        if self.num_agents <= 0:
            raise ValueError(f"Number of agents must be greater than zero: {self.num_agents} <= 0")
        print(f"[INFO]: Training {self.num_agents} agents with hybrid control with CLoP")
        
        self.sel_adj_types = sel_adj_types if type(sel_adj_types) == list else [sel_adj_types]
        print("[INFO]: Selection Adjustments:",self.sel_adj_types)
        self.init_bias = "init_bias" in self.sel_adj_types
        self.force_add = "force_add_zout" in self.sel_adj_types
        self.init_scale_weights = "init_scale_weights" in self.sel_adj_types
        self.scale_z = "scale_zin" in self.sel_adj_types

        print(f"[INFO]: {pos_init_std}\t{rot_init_std}\t{force_init_std}")
        stds = torch.ones(1, 6 + self.force_size)
        stds[0,:3] *= math.log(pos_init_std)
        stds[0,3:6] *= math.log(rot_init_std)
        stds[0,6:] *= math.log(force_init_std)
        self.actor_logstd = nn.ParameterList(
            [
                nn.Parameter(stds)
                for _ in range(self.num_agents)
            ]
            #torch.ones(1, 6 + self.force_size) #* math.log(act_init_std)
        )

        self.create_net(actor_n, actor_latent, hybrid_agent_parameters, device)

        self.apply_selection_adjustments(hybrid_agent_parameters)
    
    def create_net(self, actor_n, actor_latent, hybrid_agent_parameters, device):
        self.actor_mean = MultiSimBaNet(
            n=actor_n, 
            in_size=self.num_observations, 
            out_size=self.num_actions + self.force_size, 
            latent_size=actor_latent, 
            device=device,
            tan_out=False,
            num_nets=self.num_agents
        )
        
        if hybrid_agent_parameters['init_scale_last_layer']:
            with torch.no_grad():
                for net in self.actor_mean.pride_nets:
                    net.output[-1].weight *= hybrid_agent_parameters['init_layer_scale']
        
        self.selection_activation = nn.LogSigmoid().to(device) #torch.nn.Sigmoid().to(device)
        self.component_activation = nn.Tanh().to(device)

    def apply_selection_adjustments(self, hybrid_agent_parameters):
        if self.force_add:
            raise NotImplementedError("[ERROR]: Force add not updated for multi-net case")
            #print("[INFO]:\tAdding force to selection nodes")
            #torch.autograd.set_detect_anomaly(True)
            #self.force_bias_param = nn.ParameterList( [nn.Parameter( torch.ones(1, self.force_size) ) for _ in range(self.num_agents)])

        if self.init_scale_weights:
            print("[INFO]:\tDownscaling Initial Selection Weights")
            with torch.no_grad():
                scale_factor = hybrid_agent_parameters['init_scale_weights_factor']
                for net in self.actor_mean.pride_nets:
                    net.output[-1].weight[:self.force_size,:] *= scale_factor
                    net.output[-1].bias[:self.force_size] *= scale_factor
        
        if self.init_bias:
            print("[INFO]:\tSetting initial Selection Bias")
            with torch.no_grad():
                idxs = [0, 2, 4]
                for net in self.actor_mean.pride_nets:
                    net.output[-1].bias[idxs] = hybrid_agent_parameters['init_bias']  #-1.1 
                    net.output[-1].bias[idxs+1] = 0
        if self.scale_z:
            scale_factor = hybrid_agent_parameters['pre_layer_scale_factor']
            print(f"[INFO]:\tAdding Scale Layer before final linear layer (scale_factor={scale_factor}")
            for net in self.actor_mean.pride_nets:
                net.output.insert(-1, ScaleLayer(scale_factor=scale_factor))
        
    def act(self, inputs, role):
        return HybridGMMMixin.act(self, inputs, role)

    def get_zout(self, inputs):
        return self.actor_mean(inputs['states'][:,:self.num_observations])
    
    def compute(self, inputs, role):
        zout = self.get_zout(inputs)
        # first force_size are sigma, remaining is tanh
        sels = zout[:,:2*self.force_size]
        not_sels = zout[:,2*self.force_size:]
        
        new_sel_def = False
        if self.force_add:
            # add force to sel
            #f = torch.abs(inputs['states'][:, (self.num_observations - 6):self.num_observations - 6 + self.force_size])
            #f_bias_s = F.softplus(self.force_bias_param.expand(zout.size()[0],-1)) * f 
            #new_sels = sels + f_bias_s
            #new_sel_def = True
            raise NotImplementedError("[ERROR]: Force add not updated for multi-net case")
            
        # tan sigma of sels
        if new_sel_def:
            final_sels = self.selection_activation(new_sels)
        else:
            final_sels = self.selection_activation(sels)
        #final_sels = (final_sels + 1) * 2.0
        
        final_not_sels = self.component_activation(not_sels)
        action_mean = torch.cat([final_sels, final_not_sels], dim=-1)
        
        return action_mean, self.get_logstds(action_mean), {}

    def get_logstds(self, action_mean):

        # for this keep in mind the std is not the same size as the action space, because selection terms
        logstds = []
        batch_size = int(action_mean.size()[0] // self.num_agents)
        for  i, log_std in enumerate(self.actor_logstd):
            logstds.append(
                log_std.expand_as( action_mean[i*batch_size:(i+1)*batch_size,:6+self.force_size])
            )
        logstds = torch.cat(logstds, dim=0)
        return logstds
    
class HybridControlBlockSimBaActor(HybridControlSimBaActor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_net(self, actor_n, actor_latent, hybrid_agent_parameters, device):
        self.actor_mean = BlockSimBa(
            num_agents=self.num_agents,
            obs_dim=self.num_observations,
            hidden_dim=actor_latent,
            act_dim=self.num_actions + self.force_size,
            device=device,
            num_blocks=actor_n,
            tanh=False
        )

        if hybrid_agent_parameters['init_scale_last_layer']:
            with torch.no_grad():
                self.actor_mean.fc_out.weight *= hybrid_agent_parameters['init_layer_scale']
        
        self.selection_activation = nn.LogSigmoid().to(device) #torch.nn.Sigmoid().to(device)
        self.component_activation = nn.Tanh().to(device)


    def apply_selection_adjustments(self, hybrid_agent_parameters):
        if self.force_add:
            raise NotImplementedError("[ERROR]: Force add not updated for block SimBa networks")
        
        if self.init_scale_weights:
            print("[INFO]:\tDownscaling Initial Selection Weights")
            with torch.no_grad():
                scale_factor = hybrid_agent_parameters['init_scale_weights_factor']
                self.actor_mean.fc_out.weight[:,:self.force_size,:] *= scale_factor
                self.actor_mean.fc_out.bias[:, :self.force_size] *= scale_factor
        
        if self.init_bias:
            print("[INFO]:\tSetting initial Selection Bias")
            with torch.no_grad():
                # we have logits of size 2*force size 
                # were each is the chance of selecting force or not selecting force
                self.actor_mean.fc_out.bias[:, [0,2,4]] = hybrid_agent_parameters['init_bias']  #-1.1 
                self.actor_mean.fc_out.bias[:, [1,3,5]] = 0.0

        if self.scale_z:
            raise NotImplementedError("[ERROR]: Last layer input scaling not implemented for block SimBa networks")
    
    def get_zout(self, inputs):
        num_envs = inputs['states'].size()[0] // self.num_agents
        #print("Num envs:", num_envs)
        return self.actor_mean(inputs['states'][:,:self.num_observations], num_envs)
