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
        # Handle both (batch, action_dim) and (sample, batch, action_dim)
        original_shape = action.shape
        if action.ndim > 2:
            # Flatten sample dimensions with batch
            action = action.reshape(-1, action.shape[-1])

        # extract samples
        batch_size = action.shape[0]
        taken_action = torch.zeros((batch_size, 6, 2), dtype=torch.float32, device=action.device)
        use_force = torch.where(action[:,:self.force_size] > 0.5, True, False)      
        
        # Extract position/rotation and force/torque values
        pos_rot_values = action[:, self.force_size:self.force_size+6]  # (batch, 6)
        force_torque_values = action[:, self.force_size+6:2*self.force_size+6]  # (batch, force_size)

        # For the first force_size dimensions, choose based on use_force mask
        taken_action[:, :self.force_size, 0] = torch.where(
            use_force,
            force_torque_values,
            pos_rot_values[:, :self.force_size]
        )

        # For remaining dimensions (if force_size < 6), always use pos_rot_values
        if self.force_size < 6:
            taken_action[:, self.force_size:, 0] = pos_rot_values[:, self.force_size:]

        taken_action[:,:,1] = taken_action[:,:,0]
        
        comp_log_prob = self.component_distribution.log_prob(taken_action)

        if self.sample_uniform:
            log_z = (self.mixture_distribution.probs * (1 - self.uniform_rate) + self.uniform_rate/2.0).log()
        else:
            #log_z = self.mixture_distribution.logits #
            log_z = (self.mixture_distribution.probs).log()
        
        log_prob = torch.logsumexp(log_z + comp_log_prob, dim=-1)

        # Reshape back to original sample shape if needed
        if len(original_shape) > 2:
            log_prob = log_prob.reshape(original_shape[:-1])

        return log_prob
    
    #def entropy(self):
    #    return 0.0
    #    samples = self.sample(sample_shape=(10000,))
    #    log_prob = self.log_prob(samples)
    #    return -(log_prob.exp() * log_prob).mean(dim=0)


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
        probs = mean_actions[:,:self.force_size].float()

        # Create categorical distribution from sigmoid probabilities
        # probs[:, i] is P(force), so 1-probs[:, i] is P(position)
        mix_probs = torch.zeros((batch_size, 6, 2), dtype=torch.float32, device=mean_actions.device)
        # Stack [P(position), P(force)] for each dimension
        mix_probs[:, :self.force_size, 0] = 1.0 - probs  # P(position/rotation)
        mix_probs[:, :self.force_size, 1] = probs  # P(force)
        # For dimensions beyond force_size, set to always select position (index 0)
        if self.force_size < 6:
            mix_probs[:, self.force_size:, 0] = 1.0
            mix_probs[:, self.force_size:, 1] = 0.0
        mix_dist = Categorical(probs=mix_probs)
        mean1 = mean_actions[:, self.force_size:self.force_size+6]
        mean1[:,:3] *= self.pos_scale            # (batch, 6)
        mean1[:,3:6] *= self.rot_scale

        mean2 = torch.zeros_like(mean1)
        mean2[:,:self.force_size] = mean_actions[:, 6+self.force_size:6+2*self.force_size]
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
    

    def log_selection_probabilities(self, mix_logits, step):
        """
        Log the actual selection probabilities (after softmax)
        """
        if not (step % 15 == 0):
            return
        force_size = 3
        # mix_logits shape: (batch, 6, 2)
        probs = torch.softmax(mix_logits[:, :force_size, :], dim=-1)
        
        # Probability of selecting force (index 1)
        force_probs = probs[:, :, 1]  # (batch, 3)
        
        print(f"\n=== Step {step} Selection Probabilities ===")
        print(f"X force prob - mean: {force_probs[:, 0].mean().item():.4f}, std: {force_probs[:, 0].std().item():.4f}")
        print(f"Y force prob - mean: {force_probs[:, 1].mean().item():.4f}, std: {force_probs[:, 1].std().item():.4f}")
        print(f"Z force prob - mean: {force_probs[:, 2].mean().item():.4f}, std: {force_probs[:, 2].std().item():.4f}")
        
        # Check if they're actually different
        diff_xy = (force_probs[:, 0] - force_probs[:, 1]).abs().mean().item()
        diff_xz = (force_probs[:, 0] - force_probs[:, 2]).abs().mean().item()
        diff_yz = (force_probs[:, 1] - force_probs[:, 2]).abs().mean().item()
        
        print(f"Mean absolute differences: X-Y: {diff_xy:.4f}, X-Z: {diff_xz:.4f}, Y-Z: {diff_yz:.4f}")

    def log_selection_logits(self, mean_actions, step):
        """
        Log the actual selection logits to see if they're identical
        """
        if not (step % 15 == 0):
            return
        force_size = self.force_size
        logits = mean_actions[:, :2*force_size].float()  # (batch, 6)
        
        # Reshape to (batch, 3, 2) for easier analysis
        logits_reshaped = logits.view(-1, force_size, 2)
        
        print(f"\n=== Step {step} Selection Logits ===")
        # Look at first environment's logits
        print(f"X dimension logits: {logits_reshaped[0, 0, :].tolist()}")
        print(f"Y dimension logits: {logits_reshaped[0, 1, :].tolist()}")
        print(f"Z dimension logits: {logits_reshaped[0, 2, :].tolist()}")
        
        # Check if they're identical across batch
        for dim_name, dim_idx in [('X', 0), ('Y', 1), ('Z', 2)]:
            dim_logits = logits_reshaped[:, dim_idx, :]
            print(f"\n{dim_name} dimension stats across batch:")
            print(f"  Mean: {dim_logits.mean(dim=0).tolist()}")
            print(f"  Std: {dim_logits.std(dim=0).tolist()}")
        
        # Check correlation between dimensions
        x_logits = logits_reshaped[:, 0, 1].flatten()  # Force selection logit for X
        y_logits = logits_reshaped[:, 1, 1].flatten()  # Force selection logit for Y
        z_logits = logits_reshaped[:, 2, 1].flatten()  # Force selection logit for Z
        
        corr_xy = torch.corrcoef(torch.stack([x_logits, y_logits]))[0, 1].item()
        corr_xz = torch.corrcoef(torch.stack([x_logits, z_logits]))[0, 1].item()
        corr_yz = torch.corrcoef(torch.stack([y_logits, z_logits]))[0, 1].item()
        
        print(f"\nLogit correlations:")
        print(f"  X-Y: {corr_xy:.6f}, X-Z: {corr_xz:.6f}, Y-Z: {corr_yz:.6f}")




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
            out_size=self.num_actions,
            latent_size=actor_latent,
            device=device,
            tan_out=False,
            num_nets=self.num_agents
        )

        if hybrid_agent_parameters['init_scale_last_layer']:
            with torch.no_grad():
                for net in self.actor_mean.pride_nets:
                    net.output[-1].weight *= hybrid_agent_parameters['init_layer_scale']

        self.selection_activation = nn.Sigmoid().to(device)
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
                for net in self.actor_mean.pride_nets:
                    net.output[-1].bias[:self.force_size] -= hybrid_agent_parameters['init_bias']
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
        # first force_size are logits, remaining is tanh
        sels = zout[:,:self.force_size]
        not_sels = zout[:,self.force_size:]

        new_sel_def = False
        if self.force_add:
            raise NotImplementedError("[ERROR]: Force add not updated for multi-net case")

        # apply sigmoid to selection logits
        if new_sel_def:
            final_sels = self.selection_activation(new_sels)
        else:
            final_sels = self.selection_activation(sels)

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
            act_dim=self.num_actions,
            device=device,
            num_blocks=actor_n,
            tanh=False
        )

        if hybrid_agent_parameters['init_scale_last_layer']:
            with torch.no_grad():
                self.actor_mean.fc_out.weight *= hybrid_agent_parameters['init_layer_scale']

        self.selection_activation = nn.Sigmoid().to(device)
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
                self.actor_mean.fc_out.bias[:, :self.force_size] -= hybrid_agent_parameters['init_bias'] 

        if self.scale_z:
            raise NotImplementedError("[ERROR]: Last layer input scaling not implemented for block SimBa networks")
    
    def get_zout(self, inputs):
        num_envs = inputs['states'].size()[0] // self.num_agents
        #print("Num envs:", num_envs)
        return self.actor_mean(inputs['states'][:,:self.num_observations], num_envs)
    
