import torch
import torch.nn as nn
from models.SimBa import SimBaNet
from typing import Any, Mapping, Tuple, Union
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import math
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Independent


# -----------------------------
#  Squashed Gaussian Utilities
# -----------------------------
def squash_log_prob_correction(u):
    """Jacobian correction for tanh squashing: -sum(log(1 - tanh²(u))).

    Uses numerically stable formulation:
        log(1 - tanh²(u)) = 2*log(2) - 2*u - 2*softplus(-2*u)

    Args:
        u: Pre-squash samples from the Normal distribution.

    Returns:
        Correction term to subtract from Normal log_prob, summed over last dim.
    """
    return (2.0 * math.log(2.0) - 2.0 * u - 2.0 * F.softplus(-2.0 * u)).sum(dim=-1)


def safe_atanh(a, eps=1e-6):
    """Numerically stable inverse tanh.

    Clamps input to (-1+eps, 1-eps) to avoid inf at boundaries.

    Args:
        a: Tanh-squashed values in [-1, 1].
        eps: Clamping margin from boundaries.

    Returns:
        Pre-squash values u such that tanh(u) ≈ a.
    """
    return torch.atanh(torch.clamp(a, -1.0 + eps, 1.0 - eps))


# -----------------------------
#  Block Linear Layer
# -----------------------------
class BlockLinear(nn.Module):
    def __init__(self, num_blocks, in_features, out_features, name="linear"):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_blocks, out_features, in_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_blocks, out_features), requires_grad=True)
        # init each block separately
        for i in range(num_blocks):
            nn.init.kaiming_normal_(self.weight[i])
            nn.init.zeros_(self.bias[i])

    def forward(self, x):
        # x: (num_blocks, batch, in_features)
        out = torch.einsum("nbi,noi->nbo", x, self.weight) + self.bias[:, None, :]
        return out


# -----------------------------
#  Block LayerNorm (per-agent independent)
# -----------------------------
class BlockLayerNorm(nn.Module):
    def __init__(self, num_blocks, normalized_shape, eps=1e-5, affine=True, name="layer_norm"):
        super().__init__()
        self.eps = eps
        self.affine = affine
        shape = (num_blocks, normalized_shape)
        if affine:
            self.weight = nn.Parameter(torch.ones(*shape), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(*shape), requires_grad=True)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        

    def forward(self, x):
        # x: (num_blocks, batch, features)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)
        out = (x - mean) * inv_std
        if self.affine:
            out = out * self.weight[:, None, :] + self.bias[:, None, :]
        return out


# -----------------------------
#  Block MLP (2-layer with activation)
# -----------------------------
class BlockMLP(nn.Module):
    """2-layer MLP using block-parallel linear layers."""
    def __init__(self, num_blocks, in_dim, hidden_dim, out_dim, activation=None, name="block_mlp"):
        super().__init__()
        self.fc1 = BlockLinear(num_blocks, in_dim, hidden_dim, name=f"{name}.fc1")
        self.relu = nn.ReLU()
        self.fc2 = BlockLinear(num_blocks, hidden_dim, out_dim, name=f"{name}.fc2")
        self.activation = activation  # 'sigmoid', 'tanh', or None

    def forward(self, x):
        # x: (num_blocks, batch, in_dim)
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        if self.activation == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.activation == 'tanh':
            out = torch.tanh(out)
        return out


# -----------------------------
#  Block Residual Block
# -----------------------------
class BlockResidualBlock(nn.Module):
    def __init__(self, num_blocks, dim, name="resblock"):
        super().__init__()
        self.ln = BlockLayerNorm(num_blocks, dim, name=f"{name}.ln")
        self.fc1 = BlockLinear(num_blocks, dim, 4*dim, name=f"{name}.fc1")
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = BlockLinear(num_blocks, 4*dim, dim, name=f"{name}.fc2")

    def forward(self, x):
        res = x
        out = self.ln(x)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return res + out

# -----------------------------
#  Multi-Agent Block Policy
# -----------------------------
class BlockSimBa(nn.Module):
    def __init__(
            self,
            num_agents,
            obs_dim,
            hidden_dim,
            act_dim,
            device,
            num_blocks=2,
            tanh=False,
            use_separate_heads=False,
            selection_head_hidden_dim=64,
            component_head_hidden_dim=128,
            force_size=3,
            use_state_dependent_std=False,
            squash_actions=False,
        ):
        super().__init__()
        self.device=device
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim
        self.num_blocks = num_blocks
        self.use_tanh = tanh
        self.use_separate_heads = use_separate_heads
        self.force_size = force_size
        self.use_state_dependent_std = use_state_dependent_std

        # Calculate std_out_dim: number of continuous components (act_dim - force_size)
        self.std_out_dim = (act_dim - force_size) if use_state_dependent_std else 0

        # # DEBUG: Print configuration
        # print(f"[DEBUG BlockSimBa] use_state_dependent_std={use_state_dependent_std}")
        # print(f"[DEBUG BlockSimBa] act_dim={act_dim}, force_size={force_size}, std_out_dim={self.std_out_dim}")
        # print(f"[DEBUG BlockSimBa] use_separate_heads={use_separate_heads}")

        self.fc_in = BlockLinear(num_agents, obs_dim, hidden_dim, name="fc_in")
        self.resblocks = nn.ModuleList(
            [BlockResidualBlock(num_agents, hidden_dim, name=f"resblock_{i}") for i in range(num_blocks)]
        )
        self.ln_out = BlockLayerNorm(num_agents, hidden_dim, name="ln_out")

        if use_separate_heads:
            # When squashing, component heads output raw values (tanh applied after sampling)
            component_activation = None if squash_actions else 'tanh'
            # Create separate heads for each selection variable
            self.selection_heads = nn.ModuleList([
                BlockMLP(num_agents, hidden_dim, selection_head_hidden_dim, 1,
                        activation='sigmoid', name=f"sel_head_{i}")
                for i in range(force_size)
            ])
            # Position/rotation component head
            self.pos_rot_head = BlockMLP(num_agents, hidden_dim, component_head_hidden_dim, 6,
                                        activation=component_activation, name="pos_rot_head")
            # Force/torque component head
            self.force_torque_head = BlockMLP(num_agents, hidden_dim, component_head_hidden_dim, force_size,
                                             activation=component_activation, name="force_torque_head")
            # Optional std head for state-dependent std
            if self.std_out_dim > 0:
                self.std_head = BlockMLP(num_agents, hidden_dim, component_head_hidden_dim, self.std_out_dim,
                                        activation=None, name="std_head")
                # print(f"[DEBUG BlockSimBa] Created std_head with output_dim={self.std_out_dim}")
            else:
                self.std_head = None
                # print(f"[DEBUG BlockSimBa] No std_head (std_out_dim=0)")
        else:
            # fc_out outputs mean + std (if use_state_dependent_std)
            total_out_dim = act_dim + self.std_out_dim
            self.fc_out = BlockLinear(num_agents, hidden_dim, total_out_dim, name="fc_out")
            self.std_head = None
            # print(f"[DEBUG BlockSimBa] fc_out total_out_dim={total_out_dim} (act_dim={act_dim} + std_out_dim={self.std_out_dim})")
    # Debug counter to limit prints
    _debug_forward_count = 0

    def forward(self, obs, num_envs):
        """
        obs: (num_agents * num_envs, obs_dim)
        Returns: (actions, log_std) where log_std is None if use_state_dependent_std=False
        """
        num_agents = self.fc_in.weight.shape[0]
        obs = obs.view(num_agents, num_envs, -1)

        x = self.fc_in(obs)
        for block in self.resblocks:
            x = block(x)

        ln_out = self.ln_out(x)

        if self.use_separate_heads:
            # Apply each selection head and concatenate
            sel_outputs = [head(ln_out) for head in self.selection_heads]
            selections = torch.cat(sel_outputs, dim=-1)  # (num_agents, num_envs, force_size)

            # Apply component heads
            pos_rot_out = self.pos_rot_head(ln_out)  # (num_agents, num_envs, 6)
            force_torque_out = self.force_torque_head(ln_out)  # (num_agents, num_envs, force_size)

            # Concatenate all outputs: [selections, pos_rot, force_torque]
            actions = torch.cat([selections, pos_rot_out, force_torque_out], dim=-1)
            # Shape: (num_agents, num_envs, 2*force_size + 6)

            # Optional std head
            if self.std_head is not None:
                log_std = self.std_head(ln_out)
                log_std = log_std.view(-1, log_std.shape[-1])
            else:
                log_std = None
        else:
            raw_out = self.fc_out(ln_out)

            if self.std_out_dim > 0:
                # Split: first act_dim are mean, rest are log_std
                actions = raw_out[..., :self.act_dim]
                log_std = raw_out[..., self.act_dim:]
                log_std = log_std.view(-1, log_std.shape[-1])
            else:
                actions = raw_out
                log_std = None

            if self.use_tanh:
                actions = torch.tanh(actions)

        # # DEBUG: Print shapes and stats (only first 3 calls)
        # BlockSimBa._debug_forward_count += 1
        # if BlockSimBa._debug_forward_count <= 3:
        #     actions_flat = actions.view(-1, actions.shape[-1])
        #     print(f"[DEBUG BlockSimBa.forward #{BlockSimBa._debug_forward_count}]")
        #     print(f"  obs.shape={obs.shape}, num_envs={num_envs}")
        #     print(f"  actions.shape={actions_flat.shape}")
        #     if log_std is not None:
        #         print(f"  log_std.shape={log_std.shape}")
        #         print(f"  log_std min={log_std.min().item():.4f}, max={log_std.max().item():.4f}, mean={log_std.mean().item():.4f}")
        #         # Check if std varies across batch (state-dependent)
        #         std_variance = log_std.var(dim=0).mean().item()
        #         print(f"  log_std variance across batch (should be >0 if state-dependent): {std_variance:.6f}")
        #     else:
        #         print(f"  log_std=None (not state-dependent)")

        return actions.view(-1, actions.shape[-1]), log_std


class BlockSimBaActor(GaussianMixin, Model):
    def __init__(
            self,
            observation_space,
            action_space,
            device,
            num_agents=1,
            act_init_std = 0.60653066, # -0.5 value used in maniskill
            actor_n = 2,
            actor_latent=512,
            action_gain=1.0,
            last_layer_scale=1.0,
            init_sel_bias=0.0,

            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20,
            max_log_std=2,
            reduction="sum",
            sigma_idx=0,
            use_state_dependent_std=False,
            squash_actions=False,
            correct_squash_log_prob=False,
        ):
        print("[INFO]: \t\tInstantiating Block SimBa Actor")
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_agents = num_agents
        self.sigma_idx = sigma_idx
        self.use_state_dependent_std = use_state_dependent_std
        self.squash_actions = squash_actions
        self.correct_squash_log_prob = correct_squash_log_prob

        if squash_actions:
            print("[INFO]: \t\tSquashed Gaussian enabled (tanh applied after sampling)")
            if correct_squash_log_prob:
                print("[INFO]: \t\tJacobian log_prob correction enabled")
            else:
                print("[INFO]: \t\tJacobian log_prob correction DISABLED")

        # Compute number of component dimensions (exclude selection terms)
        # sigma_idx is the count of selection terms, Bernoulli doesn't need std dev
        num_component_dims = self.num_actions - self.sigma_idx

        # When squash_actions=True, network outputs raw (unbounded) values;
        # tanh is applied after sampling instead of on the network output.
        use_tanh_on_network = (self.sigma_idx == 0) and (not squash_actions)

        self.actor_mean = BlockSimBa(
            num_agents = self.num_agents,
            obs_dim = self.num_observations,
            hidden_dim = actor_latent,
            act_dim = self.num_actions,
            device=device,
            num_blocks = actor_n,
            tanh = use_tanh_on_network,
            force_size = self.sigma_idx,  # Pass sigma_idx as force_size for std_out_dim calculation
            use_state_dependent_std = use_state_dependent_std,
        ).to(device)

        if use_state_dependent_std:
            print(f"[INFO]: \t\tUsing STATE-DEPENDENT std (std_out_dim={self.actor_mean.std_out_dim})")
            # Initialize std portion of fc_out.bias
            with torch.no_grad():
                self.actor_mean.fc_out.bias[:, self.num_actions:] = math.log(act_init_std)
                # Scale std weights to 0.1: small enough for stable initial std values,
                # but large enough for meaningful state-dependent variation and faster learning.
                # Log-std clamping [-20, 2] provides safety bounds regardless of weight scale.
                self.actor_mean.fc_out.weight[:, self.num_actions:, :] *= 0.1

            # # DEBUG: Verify initialization
            # print(f"[DEBUG BlockSimBaActor] Initialized std bias to log({act_init_std})={math.log(act_init_std):.4f}")
            # print(f"[DEBUG BlockSimBaActor] fc_out.bias shape: {self.actor_mean.fc_out.bias.shape}")
            # print(f"[DEBUG BlockSimBaActor] fc_out.weight shape: {self.actor_mean.fc_out.weight.shape}")
            # print(f"[DEBUG BlockSimBaActor] std bias values: {self.actor_mean.fc_out.bias[:, self.num_actions:].mean().item():.4f}")
            # print(f"[DEBUG BlockSimBaActor] std weight magnitude: {self.actor_mean.fc_out.weight[:, self.num_actions:, :].abs().mean().item():.6f}")

            self.actor_logstd = None
        else:
            self.actor_logstd = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.ones(1, num_component_dims) * math.log(act_init_std), requires_grad=True
                    )
                    for _ in range(num_agents)
                ]
            ).to(device)

            print(f"[INFO]: \t\tInit Std Dev:{act_init_std}, num_component_dims:{num_component_dims}")
            for i in range(num_agents):
                self.actor_logstd[i]._agent_id = i
                self.actor_logstd[i]._name=f"logstd_{i}"

        self.component_activation = nn.Tanh().to(device)
        if self.sigma_idx > 0:
            self.selection_activation = nn.Sigmoid().to(device)

        with torch.no_grad():
            # Scale weights for action mean portion only
            self.actor_mean.fc_out.weight[:, :self.num_actions, :] *= last_layer_scale
            print(f"[INFO]: \t\tScaled model last layer (action portion) by {last_layer_scale}")
            if sigma_idx > 0:
                print(f"[INFO]: \t\tSigma Idx={sigma_idx} so last layer bias[:{sigma_idx}] -= {init_sel_bias}")
                # sigma_idx is count of selection terms, single logit per selection
                self.actor_mean.fc_out.bias[:, :sigma_idx] -= init_sel_bias


    def act(self, inputs, role):
        if self.sigma_idx == 0 and not self.squash_actions:
            return GaussianMixin.act(self, inputs, role)

        if self.sigma_idx == 0 and self.squash_actions:
            return self._act_squashed_gaussian(inputs, role)

        # sigma_idx > 0: Bernoulli selections + Gaussian components
        return self._act_bernoulli_gaussian(inputs, role)

    def _act_squashed_gaussian(self, inputs, role):
        """Squashed Gaussian for pure pose control (sigma_idx == 0).

        Flow: network outputs raw mean -> Normal(mean, std) -> sample u -> a = tanh(u)
        Log_prob: Normal.log_prob(u) - squash_correction(u) if correcting
        """
        mean_actions, log_std, outputs = self.compute(inputs, role)

        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)

        self._g_log_std = log_std
        self._g_num_samples = mean_actions.shape[0]

        # Create Normal distribution with raw (unbounded) mean
        dist = Normal(mean_actions, log_std.exp())
        self._g_distribution = dist

        # Sample u from Normal, then squash to get bounded action
        u = dist.rsample()
        actions = torch.tanh(u)

        # For log_prob: use taken_actions if replaying, otherwise use our samples
        taken_actions = inputs.get("taken_actions", actions)
        # Invert tanh to recover pre-squash values for log_prob evaluation
        u_for_log_prob = safe_atanh(taken_actions)

        log_prob = dist.log_prob(u_for_log_prob)
        if self.correct_squash_log_prob:
            log_prob = log_prob.sum(dim=-1) - squash_log_prob_correction(u_for_log_prob)
        else:
            log_prob = log_prob.sum(dim=-1)

        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        # Store for entropy calculation
        self._squash_u = u

        outputs["mean_actions"] = mean_actions
        return actions, log_prob, outputs

    def _act_bernoulli_gaussian(self, inputs, role):
        """Bernoulli selections + Gaussian components (sigma_idx > 0).

        When squash_actions=True: components are sampled from Normal then squashed via tanh.
        When squash_actions=False: components use tanh'd means directly (original behavior).
        """
        mean_actions, log_std, outputs = self.compute(inputs, role)

        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)

        # Split into selection probs and component means
        # sigma_idx is count of selection terms
        selection_probs = mean_actions[:, :self.sigma_idx]  # Already sigmoid-transformed
        component_means = mean_actions[:, self.sigma_idx:]

        # Selection: Bernoulli distribution
        selection_dist = Independent(Bernoulli(probs=selection_probs), 1)

        # Components: Gaussian distribution (only for non-selection dimensions)
        # log_std has shape (batch, num_components) - no selection std devs needed
        component_std = log_std.exp()
        component_dist = Normal(component_means, component_std)

        # Sample
        selection_samples = selection_dist.sample()
        component_samples = component_dist.rsample()

        if self.squash_actions:
            # Squash component samples to [-1, 1]
            component_actions = torch.tanh(component_samples)
        else:
            component_actions = component_samples

        actions = torch.cat([selection_samples, component_actions], dim=-1)

        # Compute log_prob for taken_actions or sampled actions
        taken_actions = inputs.get("taken_actions", actions)
        taken_selections = taken_actions[:, :self.sigma_idx]
        taken_components = taken_actions[:, self.sigma_idx:]

        # Bernoulli log_prob for selections
        sel_log_prob = selection_dist.log_prob(taken_selections)  # Shape: (batch,)

        # Gaussian log_prob for components
        if self.squash_actions:
            # Invert tanh to get pre-squash values for log_prob
            u_components = safe_atanh(taken_components)
            comp_log_prob = component_dist.log_prob(u_components).sum(dim=-1)
            if self.correct_squash_log_prob:
                comp_log_prob = comp_log_prob - squash_log_prob_correction(u_components)
        else:
            comp_log_prob = component_dist.log_prob(taken_components).sum(dim=-1)

        # Total log_prob
        log_prob = sel_log_prob + comp_log_prob

        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        # Store distribution for entropy calculation
        self._selection_dist = selection_dist
        self._component_dist = component_dist

        outputs["mean_actions"] = mean_actions
        return actions, log_prob, outputs

    # Debug counter for compute
    _debug_compute_count = 0

    def compute(self, inputs, role):
        num_envs = inputs['states'].size()[0] // self.num_agents

        # Backbone returns (action_mean, log_std) where log_std is None if not state-dependent
        action_mean, log_std = self.actor_mean(inputs['states'][:,:self.num_observations], num_envs)

        if self.sigma_idx > 0:
            # sigma_idx is count of selection terms
            selection_logits = action_mean[..., :self.sigma_idx]
            selection_probs = self.selection_activation(selection_logits)  # Sigmoid
            if self.squash_actions:
                # When squashing, leave components raw — tanh applied after sampling
                components = action_mean[..., self.sigma_idx:]
            else:
                components = self.component_activation(action_mean[..., self.sigma_idx:])
            action_mean = torch.cat([selection_probs, components], dim=-1)

        if not self.use_state_dependent_std:
            # Build log_std tensor from parameters - only for component dimensions (not selections)
            logstds = []
            batch_size = int(action_mean.size()[0] // self.num_agents)
            num_components = action_mean.size()[1] - self.sigma_idx if self.sigma_idx > 0 else action_mean.size()[1]
            for i, log_std_param in enumerate(self.actor_logstd):
                # log_std has shape (1, num_components), expand to (batch_size, num_components)
                logstds.append(log_std_param.expand(batch_size, num_components))
            log_std = torch.cat(logstds, dim=0)

        # # DEBUG: Print shapes and stats (only first 3 calls)
        # BlockSimBaActor._debug_compute_count += 1
        # if BlockSimBaActor._debug_compute_count <= 3:
        #     print(f"[DEBUG BlockSimBaActor.compute #{BlockSimBaActor._debug_compute_count}]")
        #     print(f"  use_state_dependent_std={self.use_state_dependent_std}")
        #     print(f"  action_mean.shape={action_mean.shape}")
        #     print(f"  log_std.shape={log_std.shape}")
        #     print(f"  log_std min={log_std.min().item():.4f}, max={log_std.max().item():.4f}, mean={log_std.mean().item():.4f}")
        #     if self.use_state_dependent_std:
        #         # Verify variance across batch (should be > 0 for state-dependent)
        #         std_variance = log_std.var(dim=0).mean().item()
        #         print(f"  log_std variance across batch: {std_variance:.6f}")
        #         if std_variance < 1e-8:
        #             print(f"  WARNING: log_std variance is very low - may not be state-dependent!")

        return action_mean, log_std, {}

    def get_entropy(self, role=""):
        if self.sigma_idx == 0 and not self.squash_actions:
            return GaussianMixin.get_entropy(self, role)

        if self.sigma_idx == 0 and self.squash_actions:
            # Use the Normal entropy as approximation — entropy is only used
            # as a regularizer, not for the policy gradient. The exact entropy
            # of a squashed Gaussian has no clean closed form.
            if self._g_distribution is None:
                return torch.tensor(0.0, device=self.device)
            return self._g_distribution.entropy().to(self.device)

        # Combined entropy from both distributions
        sel_entropy = self._selection_dist.entropy()  # Shape: (batch,)
        comp_entropy = self._component_dist.entropy().sum(dim=-1)  # Shape: (batch,)

        return (sel_entropy + comp_entropy).unsqueeze(-1)


class BlockSimBaCritic(DeterministicMixin, Model):
    def __init__(
            self, 
            state_space_size,
            device,
            num_agents=1,
            critic_output_init_mean = 0.0,
            critic_n = 1, 
            critic_latent=128,
            clip_actions=False,
        ):
        Model.__init__(self, state_space_size, 1, device)
        DeterministicMixin.__init__(self, clip_actions)

        in_size = self.num_observations

        self.critic = BlockSimBa(
            num_agents = num_agents,
            obs_dim = in_size,
            hidden_dim=critic_latent,
            act_dim = 1,
            device=device,
            num_blocks=critic_n,
            tanh = False
        ).to(device)
        
        torch.nn.init.constant_(self.critic.fc_out.bias, critic_output_init_mean)
        self.num_agents = num_agents

    def act(self, inputs, role):
        return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        num_envs = inputs['states'].size()[0] // self.num_agents
        value, _ = self.critic(inputs['states'][:,-self.num_observations:], num_envs)
        return value, {}

    
# -----------------------------
#  Helpers
# -----------------------------
"""
def export_policies(block_model, stds, save_paths):
    
    #Save each agent's parameters separately.
    #save_paths: list of file paths, length = num_agents
    
    num_agents = block_model.fc_in.weight.shape[0]
    hidden_dim = block_model.fc_in.weight.shape[1]
    obs_dim = block_model.fc_in.weight.shape[2]
    act_dim = block_model.fc_out.weight.shape[1]
    num_blocks = len(block_model.resblocks)

    for i in range(num_agents):
        agent = extract_agent(block_model, i)

        # save
        if stds is not None:
            # policy
            torch.save({
                'net_state_dict':agent.state_dict(),
                'log_std': stds[i]
            }, save_paths[i])
        else:
            # critic 
            torch.save(agent.state_dict(), save_paths[i])

"""            
def export_policies(block_model, stds, save_paths, preprocessor_states=None):
      """
      Save each agent's parameters separately with preprocessor states.
      save_paths: list of file paths, length = num_agents
      preprocessor_states: dict with state_preprocessor and value_preprocessor state_dicts
      stds: None for critic, ParameterList for non-state-dependent policy, or None for state-dependent policy
      """
      num_agents = block_model.fc_in.weight.shape[0]
      use_state_dependent_std = getattr(block_model, 'use_state_dependent_std', False)

      for i in range(num_agents):
          agent = extract_agent(block_model, i)

          # Create save dictionary
          if use_state_dependent_std:
              # State-dependent std: std is in the network, no separate log_std
              save_dict = {
                  'net_state_dict': agent.state_dict(),
                  'use_state_dependent_std': True
              }
          elif stds is not None:
              # policy with parameter-based log_std
              save_dict = {
                  'net_state_dict': agent.state_dict(),
                  'log_std': stds[i],
                  'use_state_dependent_std': False
              }
          else:
              # critic
              save_dict = {
                  'net_state_dict': agent.state_dict()
              }

          # Add per-agent preprocessor states if provided
          if preprocessor_states is not None:
              # Look up this agent's specific preprocessor states
              agent_state_key = f'agent_{i}_state_preprocessor'
              agent_value_key = f'agent_{i}_value_preprocessor'

              if agent_state_key in preprocessor_states:
                  save_dict['state_preprocessor'] = preprocessor_states[agent_state_key]
              if agent_value_key in preprocessor_states:
                  save_dict['value_preprocessor'] = preprocessor_states[agent_value_key]

          torch.save(save_dict, save_paths[i])

          
def extract_agent(block_model: nn.Module, agent_idx: int):
    """
    Extract a single-agent network from a block-parallel model.

    Args:
        block_model: block-parallel model (with tagged params)
        agent_idx: which agent to extract
        single_agent_class: class constructor for single-agent net (e.g., SimBaNet)
        *args, **kwargs: arguments to construct the single-agent network

    Returns:
        A single-agent model with weights/biases copied from block_model[agent_idx].
    """
    # Include std_out_dim in output size if state-dependent std
    std_out_dim = getattr(block_model, 'std_out_dim', 0)
    total_out_size = block_model.act_dim + std_out_dim

    # Build a fresh single-agent network
    agent_model = SimBaNet(
        n=len(block_model.resblocks),
        in_size=block_model.obs_dim,
        out_size=total_out_size,
        latent_size=block_model.hidden_dim,
        device=block_model.device,
        tan_out=block_model.use_tanh
    )
    # Copy over the parameters
    for (name, agent_param), (block_name, block_param) in zip(
        agent_model.named_parameters(), block_model.named_parameters()
    ):
        assert name.split('.')[-1] == block_name.split('.')[-1], f"Mismatch {name} vs {block_name}"
        # Take the slice for this agent
        if hasattr(block_param, "ndim") and block_param.ndim > 1 and block_param.shape[0] == block_model.num_agents:
            agent_param.data.copy_(block_param.data[agent_idx].clone())
        elif block_param.shape[0] == block_model.num_agents:  # vector param
            agent_param.data.copy_(block_param.data[agent_idx].clone())
        else:
            agent_param.data.copy_(block_param.data.clone())

        # Preserve tags
        agent_param._agent_id = agent_idx
        agent_param._name = getattr(block_param, "_name", name)
    
    return agent_model

def pack_agents_into_block(block_model: nn.Module, agent_models: dict):
    """
    Pack parameters from single-agent models back into a block-parallel model.
    
    Args:
        block_model: block-parallel model (with tagged params)
        agent_models: dict {agent_idx: single_agent_model}
    
    Notes:
        Overwrites block_model params for the agents provided.
    """
    for agent_idx, agent_model in agent_models.items():
        for (name, agent_param), (block_name, block_param) in zip(
            agent_model.named_parameters(), block_model.named_parameters()
        ):
            assert name.split('.')[-1] == block_name.split('.')[-1], f"Mismatch {name} vs {block_name}"

            if hasattr(block_param, "ndim") and block_param.ndim > 1 and block_param.shape[0] == block_model.num_agents:
                block_param.data[agent_idx].copy_(agent_param.data)
            elif block_param.shape[0] == block_model.num_agents:  # vector param
                block_param.data[agent_idx].copy_(agent_param.data)
            else:
                block_param.data.copy_(agent_param.data)

            # Restore tags
            block_param._agent_id = agent_idx
            block_param._name = getattr(block_param, "_name", name)


import torch.optim as optim

def make_agent_optimizer(
        block_policy,
        block_critic,
        policy_lr,
        critic_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        debug=False
):
    """
    Create a single Adam optimizer with parameter groups for each agent's
    policy and critic, with different base learning rates.
    """
    param_groups = [
        {"params": list(block_policy.parameters()), "lr": policy_lr, "role": "policy"},
        {"params": list(block_critic.parameters()), "lr": critic_lr, "role": "critic"},
    ]
    #if debug:
    #    from learning.launch_utils import attach_grad_debug_hooks
    #    attach_grad_debug_hooks(block_policy, "policy")
    #    attach_grad_debug_hooks(block_critic, "critic")
        
    return torch.optim.Adam(param_groups, betas=betas, eps=eps, weight_decay=weight_decay)

    
