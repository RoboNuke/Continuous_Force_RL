from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl import config
from skrl.memories.torch import Memory
from skrl.models.torch import Model

from typing import Any, Mapping, Optional, Tuple, Union
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium
import itertools
from filelock import FileLock
from models.block_simba import export_policies

class BlockPPO(PPO):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        state_size=-1,
        track_ckpt_paths = False,
        task="Isaac-Factory-PegInsert-Local-v0",
        num_agents: int = 1,
        num_envs: int = 256,
        env = None,  ## NEW REQUIRED PARAMETER FOR WRAPPER ACCESS ##
    ) -> None:
        ## VALIDATE REQUIRED ENVIRONMENT PARAMETER ##
        if env is None:
            raise ValueError("env parameter is required for BlockPPO. Must pass environment with wrapper system.")

        ## VALIDATE WRAPPER INTEGRATION ##
        wandb_wrapper = self._find_wandb_wrapper(env)
        if wandb_wrapper is None:
            raise ValueError("Environment must have GenericWandbLoggingWrapper with add_metrics method.")

        self.env = env  ## STORE ENVIRONMENT FOR WRAPPER ACCESS ##

        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg
        )

        self.global_step = 0
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.envs_per_agent = int(self.num_envs // self.num_agents)
        self.tracker_path = cfg['ckpt_tracker_path']
        self.track_ckpt_paths = track_ckpt_paths or cfg['track_ckpts']
        if self.track_ckpt_paths:
            lock = FileLock(self.tracker_path + ".lock")
            with lock:
                if not os.path.exists(self.tracker_path):
                    with open(self.tracker_path, "w") as f:
                        f.write("")

        self.task_name = task
        if state_size == -1:
            state_size=observation_space
        self.state_size = state_size
        
        self.value_update_ratio = cfg['value_update_ratio']
        self.huber_value_loss = cfg['use_huber_value_loss']
        self.loggers = []
        self._random_value_timesteps = cfg['random_value_timesteps']

        # Initialize agent experiment configs
        self.agent_exp_cfgs = []
        for i in range(self.num_agents):
            if f'agent_{i}' in cfg:
                self.agent_exp_cfgs.append(cfg[f'agent_{i}'])
            else:
                # Create default config if not provided
                self.agent_exp_cfgs.append({
                    'experiment': {
                        'directory': cfg.get('experiment', {}).get('directory', './experiments'),
                        'experiment_name': cfg.get('experiment', {}).get('experiment_name', 'default_experiment')
                    }
                })

        # Initialize secondary memories list
        if type(memory) is list:
            self.memory = memory[0]
            self.secondary_memories = memory[1:]
        else:
            self.memory = memory
            self.secondary_memories = []

    def _find_wandb_wrapper(self, env):
        """Find GenericWandbLoggingWrapper in the environment wrapper chain."""
        current_env = env
        while hasattr(current_env, 'env'):
            if hasattr(current_env, 'add_metrics'):
                return current_env
            current_env = current_env.env

        # Check the final environment too
        if hasattr(current_env, 'add_metrics'):
            return current_env

        return None

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:

        trainer_cfg = trainer_cfg if trainer_cfg is not None else {}
        # main entry to log data for consumption and visualization by TensorBoard
        if self.write_interval == "auto":
            self.write_interval = int(trainer_cfg.get("timesteps", 0) / 100)
        #if self.write_interval > 0:
        #    self.writer = SummaryWriter(log_dir=self.experiment_dir)

        if self.checkpoint_interval == "auto":
            self.checkpoint_interval = int(trainer_cfg.get("timesteps", 0) / 10)
        if self.checkpoint_interval > 0:
            for i in range(self.num_agents):
                os.makedirs(os.path.join(
                    self.cfg[f'agent_{i}']['experiment']['directory'],
                    self.cfg[f'agent_{i}']['experiment']['experiment_name'],
                    "checkpoints"
                ), exist_ok=True)

        self.set_mode("eval")

        ## NEW PER-AGENT PREPROCESSOR SETUP ##
        self._setup_per_agent_preprocessors()

        ## VALIDATE WRAPPER INTEGRATION FOR LOGGING ##
        self._validate_wrapper_integration()

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.state_size, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            # tensors sampled during training
            self._tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages"]


        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

    def write_checkpoint(self, timestep: int, timesteps: int):
        ckpt_paths = []
        critic_paths = []
        vid_paths = []
        for i in range(self.num_agents):
            exp_args = self.agent_exp_cfgs[i]['experiment'] #self.cfg[f'agent_{i}']['experiment']
            ckpt_paths.append(os.path.join(exp_args['directory'], exp_args['experiment_name'], "checkpoints", f"agent_{i}_{timestep*self.num_envs // self.num_agents}.pt"))
            vid_paths.append( os.path.join(exp_args['directory'], exp_args['experiment_name'], "eval_videos", f"agent_{i}_{timestep* self.num_envs // self.num_agents}.gif"))
            critic_paths.append(os.path.join(exp_args['directory'], exp_args['experiment_name'], "checkpoints", f"critic_{i}_{timestep*self.num_envs // self.num_agents}.pt"))

        ## ORIGINAL SHARED PREPROCESSOR STATES ##
        # preprocessor_states = {
        #     'state_preprocessor': self._state_preprocessor.state_dict() if hasattr(self, '_state_preprocessor') and self._state_preprocessor is not None else None,
        #     'value_preprocessor': self._value_preprocessor.state_dict() if hasattr(self, '_value_preprocessor') and self._value_preprocessor is not None else None,
        # }

        ## NEW PER-AGENT PREPROCESSOR STATES ##
        # Collect all per-agent preprocessor states for saving
        all_preprocessor_states = {}
        if hasattr(self, '_per_agent_state_preprocessors'):
            for i, preprocessor in enumerate(self._per_agent_state_preprocessors):
                if preprocessor is not None:
                    all_preprocessor_states[f'agent_{i}_state_preprocessor'] = preprocessor.state_dict()
        if hasattr(self, '_per_agent_value_preprocessors'):
            for i, preprocessor in enumerate(self._per_agent_value_preprocessors):
                if preprocessor is not None:
                    all_preprocessor_states[f'agent_{i}_value_preprocessor'] = preprocessor.state_dict()

        # For backward compatibility, also include primary preprocessor states
        if hasattr(self, '_state_preprocessor') and self._state_preprocessor is not None and hasattr(self._state_preprocessor, 'state_dict'):
            all_preprocessor_states['state_preprocessor'] = self._state_preprocessor.state_dict()
        if hasattr(self, '_value_preprocessor') and self._value_preprocessor is not None and hasattr(self._value_preprocessor, 'state_dict'):
            all_preprocessor_states['value_preprocessor'] = self._value_preprocessor.state_dict()

        export_policies(self.models['policy'].actor_mean, self.models['policy'].actor_logstd, ckpt_paths, all_preprocessor_states)
        export_policies(self.models['value'].critic, None, critic_paths, all_preprocessor_states)
        
        #self.track_data("ckpt_video", (timestep, vid_path) )
        if self.track_ckpt_paths:
            lock = FileLock(self.tracker_path + ".lock")
            with lock:
                with open(self.tracker_path, "a") as f:
                    for i in range(self.num_agents):
                        f.write(f'{ckpt_paths[i]} {self.task_name} {vid_paths[i]} {self.loggers[i].wandb_cfg["project"]} {self.loggers[i].wandb_cfg["run_id"]}\n')

    def load(self, path: str, **kwargs):
        """Load single agent checkpoint with policy, critic, and preprocessor states."""
        checkpoint = torch.load(path, map_location=self.device)

        # Load policy network parameters
        if 'net_state_dict' in checkpoint:
            # Standard checkpoint format
            self.models['policy'].actor_mean.load_state_dict(checkpoint['net_state_dict'])
            if 'log_std' in checkpoint:
                self.models['policy'].actor_logstd = checkpoint['log_std']
            print("Loaded policy network")
        else:
            # Fallback: assume entire checkpoint is the policy state dict
            self.models['policy'].actor_mean.load_state_dict(checkpoint)
            print("Loaded policy network (fallback format)")

        # Load critic network - need to construct critic checkpoint path
        # Policy path: "agent_0_12345.pt" -> Critic path: "critic_0_12345.pt"
        critic_path = path.replace("agent_", "critic_")
        if os.path.exists(critic_path):
            try:
                critic_checkpoint = torch.load(critic_path, map_location=self.device)
                
                if 'net_state_dict' in critic_checkpoint:
                    # Standard format
                    self.models['value'].critic.load_state_dict(critic_checkpoint['net_state_dict'])
                else:
                    # Fallback: assume entire checkpoint is the critic state dict
                    self.models['value'].critic.load_state_dict(critic_checkpoint)
                print("Loaded critic network")
            except Exception as e:
                print(f"Warning: Failed to load critic from {critic_path}: {e}")
        else:
            print(f"Warning: Critic checkpoint not found at {critic_path}")

        ## ORIGINAL SHARED PREPROCESSOR LOADING ##
        # for ckpt_name, ckpt in [("policy", checkpoint), ("critic", critic_checkpoint if 'critic_checkpoint' in locals() else {})]:
        #     if isinstance(ckpt, dict):
        #         if 'state_preprocessor' in ckpt and ckpt['state_preprocessor'] is not None:
        #             if hasattr(self, '_state_preprocessor') and self._state_preprocessor is not None:
        #                 self._state_preprocessor.load_state_dict(ckpt['state_preprocessor'])
        #                 print(f"Loaded state preprocessor from {ckpt_name} checkpoint")
        #             else:
        #                 print(f"Warning: {ckpt_name} checkpoint contains state_preprocessor but agent doesn't have one configured")
        #
        #         if 'value_preprocessor' in ckpt and ckpt['value_preprocessor'] is not None:
        #             if hasattr(self, '_value_preprocessor') and self._value_preprocessor is not None:
        #                 self._value_preprocessor.load_state_dict(ckpt['value_preprocessor'])
        #                 print(f"Loaded value preprocessor from {ckpt_name} checkpoint")
        #             else:
        #                 print(f"Warning: {ckpt_name} checkpoint contains value_preprocessor but agent doesn't have one configured")

        ## NEW PER-AGENT PREPROCESSOR LOADING ##
        # Load per-agent preprocessor states from both policy and critic checkpoints
        for ckpt_name, ckpt in [("policy", checkpoint), ("critic", critic_checkpoint if 'critic_checkpoint' in locals() else {})]:
            if isinstance(ckpt, dict):
                # Load per-agent state preprocessors
                if hasattr(self, '_per_agent_state_preprocessors'):
                    for i, preprocessor in enumerate(self._per_agent_state_preprocessors):
                        key = f'agent_{i}_state_preprocessor'
                        if key in ckpt and ckpt[key] is not None and preprocessor is not None:
                            preprocessor.load_state_dict(ckpt[key])
                            print(f"Loaded agent {i} state preprocessor from {ckpt_name} checkpoint")

                # Load per-agent value preprocessors
                if hasattr(self, '_per_agent_value_preprocessors'):
                    for i, preprocessor in enumerate(self._per_agent_value_preprocessors):
                        key = f'agent_{i}_value_preprocessor'
                        if key in ckpt and ckpt[key] is not None and preprocessor is not None:
                            preprocessor.load_state_dict(ckpt[key])
                            print(f"Loaded agent {i} value preprocessor from {ckpt_name} checkpoint")

                # Backward compatibility: Load shared preprocessor states
                if 'state_preprocessor' in ckpt and ckpt['state_preprocessor'] is not None:
                    if hasattr(self, '_state_preprocessor') and self._state_preprocessor is not None:
                        self._state_preprocessor.load_state_dict(ckpt['state_preprocessor'])
                        print(f"Loaded shared state preprocessor from {ckpt_name} checkpoint")

                if 'value_preprocessor' in ckpt and ckpt['value_preprocessor'] is not None:
                    if hasattr(self, '_value_preprocessor') and self._value_preprocessor is not None:
                        self._value_preprocessor.load_state_dict(ckpt['value_preprocessor'])
                        print(f"Loaded shared value preprocessor from {ckpt_name} checkpoint")

        print(f"Loaded checkpoint from {path}")

    def add_sample_to_memory(self, **tensors: torch.Tensor) -> None:
        self.memory.add_samples( **tensors )
        for memory in self.secondary_memories:
                memory.add_samples( **tensors )

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        if self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                try:
                    rewards = self._rewards_shaper(rewards, timestep, timesteps)
                except:
                    rewards = self._rewards_shaper(rewards)

            # compute values with per-agent preprocessing
            processed_states = self._apply_per_agent_preprocessing(states, self._per_agent_state_preprocessors)
            values, _, _ = self.value.act({"states": processed_states}, role="value")
            values = self._apply_per_agent_preprocessing(values, self._per_agent_value_preprocessors, inverse=True)
            #print("Applied value preprocessor in record transition")

            # time-limit (truncation) boostrapping
            if self._time_limit_bootstrap:
                # Ensure values matches rewards shape to prevent broadcasting issues
                if values.dim() > rewards.dim():
                    values_reshaped = values.squeeze(-1)
                else:
                    values_reshaped = values
                rewards += self._discount_factor * values_reshaped * truncated
            
            
            # alternative approach to deal with termination, see ppo but no matter what it 
            # goes through every sample in the memory (even if not filled), this way we set 
            # the actions to zero on termination, and then pass the fixed state in basically
            #copy_state = {'policy':states['policy'].clone(), 'critic':states['critic'].clone()}
            #copy_next_state = {'policy':next_states['policy'].clone(), 'critic':next_states['critic'].clone()}
            #print("Wandb Log Prob Size:", self._current_log_prob.size())
            #ta = actions
            #print("Wandb Action Pos:", torch.max(ta[:,3:6]).item(), torch.min(ta[:,3:6]).item(), torch.median(ta[:,3:6]).item())
            self.add_sample_to_memory(
                states=states.clone(), 
                actions=actions.clone(), 
                rewards=rewards.clone(), 
                next_states=next_states.clone(),
                terminated=terminated.clone(),
                truncated=truncated.clone(), 
                log_prob=self._current_log_prob.clone(), 
                values=values.clone()
            )   
    
    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """   
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            #print("Return:", np.mean(np.array(self._track_rewards)))
            self._update(timestep, timesteps)
            
            self.set_mode("eval")

        # Store original timestep for checkpoint/tracking checks
        original_timestep = timestep
        timestep += 1
        self.global_step+= self.num_envs

        # update best models and write checkpoints
        if original_timestep > 0 and self.checkpoint_interval > 0 and not original_timestep % self.checkpoint_interval:
            # write checkpoints
            self.write_checkpoint(original_timestep, timesteps)

        # write wandb
        if original_timestep > 0 and self.write_interval > 0 and not original_timestep % self.write_interval:
            self.write_tracking_data(original_timestep, timesteps)                     

    def _update(self, timestep: int, timesteps: int):
        #super()._update(timestep, timesteps) def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            processed_next_states = self._apply_per_agent_preprocessing(self._current_next_states.float(), self._per_agent_state_preprocessors)
            last_values, _, _ = self.value.act(
                {"states": processed_next_states}, role="value"
            )
            self.value.train(True)
            last_values = self._apply_per_agent_preprocessing(last_values, self._per_agent_value_preprocessors, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )
        
        self.memory.set_tensor_by_name("values", self._apply_per_agent_preprocessing(values, self._per_agent_value_preprocessors, train=True))
        self.memory.set_tensor_by_name("returns", self._apply_per_agent_preprocessing(returns, self._per_agent_value_preprocessors, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)
        sample_size = self._rollouts // self._mini_batches
        
        # learning epochs
        for epoch in range(self._learning_epochs):
            mini_batch = 0
            #print(f"Epoch:{epoch}")
            # mini-batches loop
            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:
                #print(f"Mini Batch:{mini_batch}")
                if mini_batch > 5:
                    continue
                keep_mask = torch.ones((self.num_agents,), dtype=bool, device=self.device)
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states = self._apply_per_agent_preprocessing(sampled_states, self._per_agent_state_preprocessors, train=not epoch)
                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )
                    sampled_states = sampled_states.view(sample_size, self.num_agents, self.envs_per_agent, -1)
                    next_log_prob = next_log_prob.view(sample_size, self.num_agents, self.envs_per_agent)
                    sampled_log_prob = sampled_log_prob.view(sample_size, self.num_agents, self.envs_per_agent)
                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        agent_kls = ((torch.exp(ratio) - 1) - ratio)

                        agent_kls = torch.mean(agent_kls, (0,2)) # assumes sample_size x num_agents x num_envs_per_agent x 1
                        if self._kl_threshold:
                            keep_mask = ~torch.logical_or( ~keep_mask, agent_kls > self._kl_threshold)

                    entropys = self.policy.get_entropy(role="policy")
                    entropys = entropys.view(sample_size, self.num_agents, self.envs_per_agent,-1)
                    entropys[:,~keep_mask,:] = 0.0

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * entropys[:,keep_mask,:].mean()
                    else:
                        entropy_loss = 0
                    

                    # compute policy loss
                    sampled_advantages = sampled_advantages.view(sample_size, self.num_agents, self.envs_per_agent)
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_losses = -torch.min(surrogate, surrogate_clipped)
                    policy_losses[:,~keep_mask,:]=0.0
                    policy_loss = policy_losses[:,keep_mask,:].mean()

                    # compute value loss
                    batch_total_envs = sample_size * self.num_agents * self.envs_per_agent
                    predicted_values, _, _ = self.value.act({"states": sampled_states.view(batch_total_envs,-1)}, role="value")
                    predicted_values = predicted_values.view( sample_size, self.num_agents, self.envs_per_agent)
                    sampled_values = sampled_values.view(sample_size, self.num_agents, self.envs_per_agent)
                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )

                    sampled_returns = sampled_returns.view(sample_size, self.num_agents, self.envs_per_agent)
                    
                    value_loss, value_losses, predicted_values = self.calc_value_loss(sampled_states, sampled_values, sampled_returns, keep_mask, sample_size)

                # optimization step
                # zero out losses from cancelled

                # COLLECT NETWORK STATES BEFORE ANY OPTIMIZER STEP (Problem 1 Fix)
                # Store policy and critic states while gradients still exist
                self._log_minibatch_update(
                    advantages = sampled_advantages,
                    old_log_probs = sampled_log_prob,
                    new_log_probs = next_log_prob,
                    entropies = entropys,
                    policy_losses = policy_losses,
                    store_policy_state = True,
                    store_critic_state = True
                )

                self.optimizer.zero_grad()
                if timestep < self._random_value_timesteps:
                    self.update_nets(value_loss)
                else:
                    self.update_nets(policy_loss + entropy_loss + value_loss)

                if self.value_update_ratio > 1:
                    for i in range(self.value_update_ratio-1): #minus one because we already updated critic once
                        value_loss, vls, predicted_values = self.calc_value_loss(
                            sampled_states,
                            sampled_values,
                            sampled_returns,
                            keep_mask,
                            sample_size
                        )
                        value_losses += vls
                        self.update_nets(value_loss)

                    value_losses /= self.value_update_ratio

                    # Log final results - network states already collected
                    self._log_minibatch_update(
                        returns = sampled_returns,
                        values = predicted_values,
                        value_losses = value_losses,
                        store_critic_state = False  # Already collected before optimization
                    )
                else:
                    # Log all metrics - network states already collected
                    self._log_minibatch_update(
                        returns=sampled_returns,
                        values=predicted_values,
                        advantages = sampled_advantages,
                        old_log_probs = sampled_log_prob,
                        new_log_probs = next_log_prob,
                        entropies = entropys,
                        policy_losses = policy_losses,
                        value_losses=value_losses,
                        store_policy_state = False,  # Already collected before optimization
                        store_critic_state = False   # Already collected before optimization
                    )
                mini_batch += 1

        # Publish accumulated learning metrics to wandb after all minibatches complete
        wrapper = self._get_logging_wrapper()
        if wrapper:
            wrapper.publish()

    def update_nets(self, loss):
                        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        if config.torch.is_distributed:
            self.policy.reduce_parameters()
            if self.policy is not self.value:
                self.value.reduce_parameters()

        if self._grad_norm_clip > 0:
            self.scaler.unscale_(self.optimizer)
            if self.policy is self.value:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
            else:
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                )             
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
    def adaptive_huber_delta(self, predicted, sampled, k=1.35):
        e = (sampled - predicted)
        med = e.median()
        mad = (e - med).abs().median() + 1e-8
        return float(k * mad)
    
    def calc_value_loss(self, sampled_states, sampled_values, sampled_returns, keep_mask, sample_size):

        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            # compute value loss
            batch_total_envs = sample_size * self.num_agents * self.envs_per_agent
            predicted_values, _, _ = self.value.act({"states": sampled_states.view(batch_total_envs,-1)}, role="value")
            predicted_values = predicted_values.view( sample_size, self.num_agents, self.envs_per_agent)

            if self._clip_predicted_values:
                predicted_values = sampled_values + torch.clip(
                    predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                )

            sampled_returns = sampled_returns.view(sample_size, self.num_agents, self.envs_per_agent)

            if self.huber_value_loss:
                vls = self._value_loss_scale * F.huber_loss(
                    predicted_values,
                    sampled_returns,
                    reduction='none',
                    delta = self.adaptive_huber_delta(predicted_values, sampled_values)
                )
            else:
                vls = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values, reduction='none')
            vls[:,~keep_mask,:] = 0.0

            # make sure we get average losses for each update
            #value_losses += vls

            value_loss = vls[:,keep_mask,:].mean()

            return value_loss, vls, predicted_values


    def _get_network_state(self, agent_idx):
        """
        Extract optimizer state for one agent, returning separate dicts for
        policy and critic. Assumes optimizer param_groups were created with
        make_agent_optimizer (policy first, critic second per agent).
        """
        print(f"[DEBUG] _get_network_state called for agent {agent_idx}")
        state = {
            "policy": {"gradients":[], "weight_norms":{}, "optimizer_state":{}},
            "critic":{"gradients":[], "weight_norms":{}, "optimizer_state":{}}
        }

        # Collect all gradients and states first WITHOUT clearing
        for role in ['critic','policy']:
            net = self.value if role=='critic' else self.policy
            params_with_grad = 0
            total_params = 0
            for pname, p in net.named_parameters():
                total_params += 1

                if p.grad is not None:
                    params_with_grad += 1

                    # Problem 3 Fix: Handle different parameter architectures
                    if hasattr(p, '_agent_id'):
                        # ParameterList parameter - already agent-specific
                        if p._agent_id == agent_idx:
                            grad_norm = p.grad.detach().norm(2)
                            state[role]['gradients'].append(grad_norm)
                            print(f"[DEBUG] ParameterList param {pname}: agent_id={p._agent_id}, grad_norm={grad_norm.item():.6f}")
                        # Skip parameters for other agents
                    else:
                        # Block parameter - has agent dimension
                        try:
                            # For 3D tensors: (num_agents, out_features, in_features)
                            grad_norm = p.grad.detach()[agent_idx,:,:].norm(2)
                            state[role]['gradients'].append(grad_norm)
                            print(f"[DEBUG] Block param {pname}: 3D indexing, grad_norm={grad_norm.item():.6f}")
                        except (IndexError, RuntimeError):
                            try:
                                # For 2D tensors: (num_agents, out_features)
                                grad_norm = p.grad.detach()[agent_idx,:].norm(2)
                                state[role]['gradients'].append(grad_norm)
                                print(f"[DEBUG] Block param {pname}: 2D indexing, grad_norm={grad_norm.item():.6f}")
                            except (IndexError, RuntimeError):
                                # For 1D tensors or other cases: (num_agents,)
                                try:
                                    grad_norm = abs(p.grad.detach()[agent_idx])
                                    state[role]['gradients'].append(grad_norm)
                                    print(f"[DEBUG] Block param {pname}: 1D indexing, grad_norm={grad_norm.item():.6f}")
                                except (IndexError, RuntimeError) as e:
                                    print(f"[DEBUG] Failed to index gradient for {pname}: {e}, shape: {p.grad.shape}")
                                    # Skip this parameter

                    # Collect optimizer state if available (Problem 2 Fix - use parameter object as key)
                    # Only collect for parameters that belong to this agent
                    should_collect_optimizer_state = False
                    if hasattr(p, '_agent_id'):
                        # ParameterList parameter - only collect for matching agent
                        should_collect_optimizer_state = (p._agent_id == agent_idx)
                    else:
                        # Block parameter - collect for all agents (shared optimizer state)
                        should_collect_optimizer_state = True

                    if should_collect_optimizer_state and p in self.optimizer.state:
                        op_state = self.optimizer.state[p]
                        state[role]['optimizer_state'][pname] = {
                            "exp_avg": op_state.get("exp_avg", torch.tensor(0.0)),
                            "exp_avg_sq": op_state.get("exp_avg_sq", torch.tensor(0.0)),
                            "step": op_state.get("step", 0),
                        }
                        print(f"[DEBUG] Collected optimizer state for {pname}: step={op_state.get('step', 0)}")
                    else:
                        if not should_collect_optimizer_state:
                            print(f"[DEBUG] Skipping optimizer state for {pname} (different agent: {getattr(p, '_agent_id', 'N/A')})")
                        else:
                            print(f"[DEBUG] No optimizer state found for parameter {pname}")

                # Weight norms (always collect)
                state[role]['weight_norms'][pname] = p.norm().item()

            print(f"[DEBUG] {role} network: {params_with_grad}/{total_params} parameters have gradients")

        return state

    def _log_minibatch_update(
            self,
            returns=None,
            values=None,
            advantages=None,
            old_log_probs=None,
            new_log_probs=None,
            entropies=None,
            policy_losses=None,
            value_losses=None,
            store_policy_state=False,
            store_critic_state=False
    ):
        """Log minibatch update metrics through wrapper system with per-agent support.

        All metrics are computed per-agent and passed as lists to the wrapper system.
        Tensors are assumed to be shaped as (sample_size, num_agents, envs_per_agent).

        Args:
            store_policy_state: If True, collect and log policy network state metrics
            store_critic_state: If True, collect and log critic network state metrics
        """
        wrapper = self._get_logging_wrapper()
        if wrapper is None:
            print("[INFO] Unable to log minibatch due to no logging wrapper!")
            return  # No wrapper system available, skip logging

        try:
            stats = {}

            # Calculate index ranges for each agent
            idx_ranges = [(i * self.envs_per_agent, (i + 1) * self.envs_per_agent)
                         for i in range(self.num_agents)]

            with torch.no_grad():
                # --- Policy stats (per-agent) ---
                if new_log_probs is not None and old_log_probs is not None:
                    ratio = (new_log_probs - old_log_probs).exp()
                    clip_mask = (ratio < 1 - self._ratio_clip) | (ratio > 1 + self._ratio_clip)
                    kl = old_log_probs - new_log_probs

                    # Calculate per-agent metrics (as tensors)
                    stats["Policy/KL_Divergence_Avg"] = torch.tensor([kl[:, i, :].mean().item() for i in range(self.num_agents)], device=self.device)
                    stats["Policy/KL_Divergence_95_Quantile"] = torch.tensor([kl[:, i, :].quantile(0.95).item() for i in range(self.num_agents)], device=self.device)
                    stats["Policy/Clip_Fraction"] = torch.tensor([clip_mask[:, i, :].float().mean().item() for i in range(self.num_agents)], device=self.device)

                if entropies is not None:
                    entropy_values = torch.tensor([entropies[:, i, :].mean().item() for i in range(self.num_agents)], device=self.device)
                    stats["Policy/Entropy_Avg"] = entropy_values
                    print(f"[DEBUG] Policy entropy metrics created: {entropy_values}")
                else:
                    print(f"[DEBUG] No entropies provided for logging")

                if policy_losses is not None:
                    policy_loss_values = torch.tensor([policy_losses[:, i, :].mean().item() for i in range(self.num_agents)], device=self.device)
                    stats["Policy/Loss_Avg"] = policy_loss_values
                    print(f"[DEBUG] Policy loss metrics created: {policy_loss_values}")
                else:
                    print(f"[DEBUG] No policy_losses provided for logging")

                # --- Value stats (per-agent) ---
                if value_losses is not None and values is not None and returns is not None:
                    stats["Critic/Loss_Avg"] = torch.tensor([value_losses[:, i, :].mean().item() for i in range(self.num_agents)], device=self.device)
                    stats["Critic/Loss_Median"] = torch.tensor([value_losses[:, i, :].median().item() for i in range(self.num_agents)], device=self.device)
                    stats["Critic/Loss_95_Quantile"] = torch.tensor([value_losses[:, i, :].quantile(0.95).item() for i in range(self.num_agents)], device=self.device)
                    stats["Critic/Loss_90_Quantile"] = torch.tensor([value_losses[:, i, :].quantile(0.90).item() for i in range(self.num_agents)], device=self.device)
                    stats["Critic/Predicted_Values_Avg"] = torch.tensor([values[:, i, :].mean().item() for i in range(self.num_agents)], device=self.device)
                    stats["Critic/Predicted_Values_Std"] = torch.tensor([values[:, i, :].std().item() for i in range(self.num_agents)], device=self.device)

                    # Explained variance per agent
                    explained_var_list = []
                    for i in range(self.num_agents):
                        agent_returns = returns[:, i, :]
                        agent_values = values[:, i, :]
                        explained_var = (1 - ((agent_returns - agent_values).var(unbiased=False) /
                                            agent_returns.var(unbiased=False).clamp(min=1e-8))).item()
                        explained_var_list.append(explained_var)
                    stats["Critic/Explained_Variance"] = torch.tensor(explained_var_list, device=self.device)

                # --- Advantage diagnostics (per-agent) ---
                if advantages is not None:
                    stats["Advantage/Mean"] = torch.tensor([advantages[:, i, :].mean().item() for i in range(self.num_agents)], device=self.device)
                    stats["Advantage/Std_Dev"] = torch.tensor([advantages[:, i, :].std().item() for i in range(self.num_agents)], device=self.device)

                    # Skewness per agent
                    skew_list = []
                    for i in range(self.num_agents):
                        agent_adv = advantages[:, i, :]
                        skew = (((agent_adv - agent_adv.mean()) ** 3).mean() / (agent_adv.std() ** 3 + 1e-8)).item()
                        skew_list.append(skew)
                    stats["Advantage/Skew"] = torch.tensor(skew_list, device=self.device)

                # --- Network state diagnostics (per-agent) ---
                def grad_norm_per_agent(agent_gradients):
                    """Calculate gradient norm for a single agent's gradients."""
                    print(f"[DEBUG] grad_norm_per_agent called with {len(agent_gradients)} gradients")
                    if len(agent_gradients) == 0:
                        print(f"[DEBUG] No gradients available, returning 0.0")
                        return 0.0

                    # Check individual gradient values
                    grad_values = [g.item() for g in agent_gradients]
                    print(f"[DEBUG] Individual gradient norms: {grad_values[:5]}...")  # First 5 values

                    # Calculate total norm
                    result = torch.norm(torch.stack(agent_gradients), 2).item()
                    print(f"[DEBUG] Calculated gradient norm: {result}")
                    return result

                def adam_step_size_per_agent(agent_optimizer_state):
                    """Calculate Adam step size for a single agent's optimizer state."""
                    print(f"[DEBUG] adam_step_size_per_agent called with {len(agent_optimizer_state)} optimizer states")

                    if len(agent_optimizer_state) == 0:
                        print(f"[DEBUG] No optimizer states available, returning 0.0")
                        return 0.0

                    step_sizes = []
                    for name, state in agent_optimizer_state.items():
                        print(f"[DEBUG] Processing optimizer state for {name}, keys: {list(state.keys()) if isinstance(state, dict) else 'Not dict'}")

                        # Check if state has required fields
                        if isinstance(state, dict) and 'exp_avg' in state and 'exp_avg_sq' in state and 'step' in state:
                            try:
                                # Get learning rate from appropriate param group
                                lr = None
                                for group in self.optimizer.param_groups:
                                    if group.get('role') == 'policy':
                                        lr = group['lr']
                                        break
                                if lr is None:
                                    lr = self.optimizer.param_groups[0]['lr']  # Fallback

                                beta1, beta2 = self.optimizer.param_groups[0]['betas']
                                eps = self.optimizer.param_groups[0]['eps']
                                step_count = state['step']

                                # Get the actual tensor values
                                exp_avg = state['exp_avg']
                                exp_avg_sq = state['exp_avg_sq']

                                print(f"[DEBUG] {name}: step={step_count}, exp_avg_norm={torch.norm(exp_avg).item():.6f}, exp_avg_sq_norm={torch.norm(exp_avg_sq).item():.6f}")

                                if step_count > 0 and torch.norm(exp_avg_sq).item() > 1e-8:
                                    bias_correction1 = 1 - beta1 ** step_count
                                    bias_correction2 = 1 - beta2 ** step_count

                                    # Calculate effective step size
                                    effective_step_direction = (exp_avg / bias_correction1) / (torch.sqrt(exp_avg_sq / bias_correction2) + eps)
                                    step_size = lr * torch.norm(effective_step_direction).item()
                                    step_sizes.append(step_size)
                                    print(f"[DEBUG] {name}: calculated step_size={step_size:.6f}")
                                else:
                                    print(f"[DEBUG] {name}: step count is {step_count} or exp_avg_sq norm too small, skipping")

                            except Exception as e:
                                print(f"[DEBUG] Error calculating step size for {name}: {e}")
                        else:
                            print(f"[DEBUG] {name}: missing required optimizer state fields")

                    result = sum(step_sizes) / max(len(step_sizes), 1) if step_sizes else 0.0
                    print(f"[DEBUG] Final average step size: {result:.6f}")
                    return result

                # Collect network states if requested
                network_states = None
                if store_policy_state or store_critic_state:
                    network_states = [self._get_network_state(i) for i in range(self.num_agents)]
                    print(f"[DEBUG] Network states collected: {network_states is not None}, count: {len(network_states) if network_states else 0}")
                    if network_states:
                        print(f"[DEBUG] Sample network state keys: {list(network_states[0].keys()) if network_states[0] else 'None'}")

                # Policy network diagnostics
                if store_policy_state and network_states is not None:
                    print(f"[DEBUG] Generating Policy/Step_Size metrics for {len(network_states)} agents")
                    grad_norms = torch.tensor([grad_norm_per_agent(state['policy']['gradients'])
                                                   for state in network_states], device=self.device)
                    step_sizes = torch.tensor([adam_step_size_per_agent(state['policy']['optimizer_state'])
                                                for state in network_states], device=self.device)
                    stats['Policy/Gradient_Norm'] = grad_norms
                    stats['Policy/Step_Size'] = step_sizes
                    print(f"[DEBUG] Policy metrics created - Gradient_Norm: {grad_norms}, Step_Size: {step_sizes}")

                # Critic network diagnostics
                if store_critic_state and network_states is not None:
                    print(f"[DEBUG] Generating Critic/Step_Size metrics for {len(network_states)} agents")
                    critic_grad_norms = torch.tensor([grad_norm_per_agent(state['critic']['gradients'])
                                                    for state in network_states], device=self.device)
                    critic_step_sizes = torch.tensor([adam_step_size_per_agent(state['critic']['optimizer_state'])
                                                for state in network_states], device=self.device)
                    stats['Critic/Gradient_Norm'] = critic_grad_norms
                    stats['Critic/Step_Size'] = critic_step_sizes
                    print(f"[DEBUG] Critic metrics created - Gradient_Norm: {critic_grad_norms}, Step_Size: {critic_step_sizes}")

            # Pass metrics to wrapper system
            print(f"[DEBUG] Sending {len(stats)} metrics to wrapper: {list(stats.keys())}")
            print(f"[DEBUG] Step size present in stats: {'Policy/Step_Size' in stats}, {'Critic/Step_Size' in stats}")
            wrapper.add_metrics(stats)

        except Exception as e:
            # Silently fail if wrapper system not available
            pass


    def _setup_per_agent_preprocessors(self):
        """Set up per-agent independent preprocessors.

        This creates separate preprocessor instances for each agent using the same
        preprocessor type/config specified in the main config. Much simpler than
        requiring per-agent config entries.
        """
        print(f"  - Setting up per-agent preprocessors for {self.num_agents} agents")

        # Initialize per-agent preprocessor lists
        self._per_agent_state_preprocessors = []
        self._per_agent_value_preprocessors = []

        # Check for standard preprocessor configs (same as old shared system)
        state_preprocessor_class = self.cfg.get("state_preprocessor")
        state_preprocessor_kwargs = self.cfg.get("state_preprocessor_kwargs", {})
        value_preprocessor_class = self.cfg.get("value_preprocessor")
        value_preprocessor_kwargs = self.cfg.get("value_preprocessor_kwargs", {})

        for i in range(self.num_agents):
            # Create independent state preprocessor for this agent
            if state_preprocessor_class is not None:
                # Create separate instance for this agent
                state_preprocessor = state_preprocessor_class(**state_preprocessor_kwargs)
                self._per_agent_state_preprocessors.append(state_preprocessor)
                print(f"    - Agent {i}: State preprocessor enabled ({state_preprocessor_class.__name__})")
            else:
                self._per_agent_state_preprocessors.append(None)
                print(f"    - Agent {i}: No state preprocessor")

            # Create independent value preprocessor for this agent
            if value_preprocessor_class is not None:
                # Create separate instance for this agent
                value_preprocessor = value_preprocessor_class(**value_preprocessor_kwargs)
                self._per_agent_value_preprocessors.append(value_preprocessor)
                print(f"    - Agent {i}: Value preprocessor enabled ({value_preprocessor_class.__name__})")
            else:
                self._per_agent_value_preprocessors.append(None)
                print(f"    - Agent {i}: No value preprocessor")

        # Set the primary preprocessors to the first agent's for SKRL compatibility
        self._state_preprocessor = self._per_agent_state_preprocessors[0] if self._per_agent_state_preprocessors[0] is not None else None
        self._value_preprocessor = self._per_agent_value_preprocessors[0] if self._per_agent_value_preprocessors[0] is not None else None

        print(f"  - Per-agent preprocessor setup complete")

    def _apply_per_agent_preprocessing(self, tensor_input, preprocessor_list, train=False, inverse=False):
        """Apply per-agent preprocessing to input tensor.

        Args:
            tensor_input: Input tensor of shape [total_envs, features] or [batch_size, total_envs, features]
            preprocessor_list: List of preprocessors, one per agent
            train: Whether to run in training mode (update statistics)
            inverse: Whether to apply inverse transform

        Returns:
            Processed tensor with same shape as input
        """
        if not preprocessor_list or all(p is None for p in preprocessor_list):
            return tensor_input

        # Handle different input shapes
        original_shape = tensor_input.shape
        if len(original_shape) == 2:
            # Shape: [total_envs, features]
            batch_size = None
            total_envs, features = original_shape
        elif len(original_shape) == 3:
            # Shape: [batch_size, total_envs, features]
            batch_size, total_envs, features = original_shape
            tensor_input = tensor_input.view(-1, features)  # Flatten to [batch_size * total_envs, features]
        else:
            raise ValueError(f"Unsupported tensor shape: {original_shape}")

        # Split input by agent environments
        processed_chunks = []
        for agent_id in range(self.num_agents):
            start_idx = agent_id * self.envs_per_agent
            end_idx = (agent_id + 1) * self.envs_per_agent

            # Extract this agent's portion
            if batch_size is None:
                agent_chunk = tensor_input[start_idx:end_idx, :]
            else:
                # For batched input, we need to extract envs for this agent across all batches
                agent_chunk = tensor_input.view(batch_size, total_envs, features)[:, start_idx:end_idx, :].contiguous().view(-1, features)

            # Apply this agent's preprocessor
            preprocessor = preprocessor_list[agent_id]
            if preprocessor is not None:
                if inverse:
                    agent_chunk = preprocessor(agent_chunk, inverse=True)
                else:
                    agent_chunk = preprocessor(agent_chunk, train=train)

            processed_chunks.append(agent_chunk)

        # Concatenate all agent chunks back together
        if batch_size is None:
            # Simple concatenation for 2D case
            result = torch.cat(processed_chunks, dim=0)
        else:
            # Reshape and concatenate for 3D case
            for i, chunk in enumerate(processed_chunks):
                processed_chunks[i] = chunk.view(batch_size, self.envs_per_agent, features)
            result = torch.cat(processed_chunks, dim=1)  # Concatenate along env dimension
            result = result.view(batch_size * total_envs, features)  # Flatten back

        # Restore original shape
        return result.view(original_shape)

    def _validate_wrapper_integration(self):
        """Validate that the wrapper system is properly integrated for logging.

        This checks if GenericWandbLoggingWrapper or compatible wrapper is available
        for the logging methods to use. Searches through the wrapper chain thoroughly.
        """
        # Method 1: Check direct unwrapped access
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'add_metrics'):
            print(f"  - Wrapper integration validated: Found logging wrapper at self.env.unwrapped")
            return True

        # Method 2: Check if SKRL wrapper has the base environment with add_metrics
        if hasattr(self.env, '_env') and hasattr(self.env._env, 'add_metrics'):
            print(f"  - Wrapper integration validated: Found logging wrapper at self.env._env")
            return True

        # Method 3: Search through wrapper chain manually
        current_env = self.env
        search_depth = 0
        max_depth = 10  # Prevent infinite loops

        while current_env is not None and search_depth < max_depth:
            if hasattr(current_env, 'add_metrics'):
                print(f"  - Wrapper integration validated: Found logging wrapper at depth {search_depth}")
                return True

            # Try different wrapper access patterns
            next_env = None
            for attr in ['env', '_env', 'unwrapped']:
                if hasattr(current_env, attr):
                    next_env = getattr(current_env, attr)
                    break

            current_env = next_env
            search_depth += 1

        # If we get here, no logging wrapper was found
        raise ValueError(
            "No compatible logging wrapper found. Data collection is required for experiments.\n"
            "Expected: GenericWandbLoggingWrapper with add_metrics method.\n"
            "Ensure the GenericWandbLoggingWrapper is properly applied to the environment.\n"
            f"Searched through {search_depth} wrapper layers without finding add_metrics method."
        )

    def _get_logging_wrapper(self):
        """Get the logging wrapper for metrics reporting.

        Returns the wrapper object if found, None otherwise.
        Uses the same thorough search as _validate_wrapper_integration.
        """
        # Method 1: Check direct unwrapped access
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'add_metrics'):
            return self.env.unwrapped

        # Method 2: Check if SKRL wrapper has the base environment with add_metrics
        if hasattr(self.env, '_env') and hasattr(self.env._env, 'add_metrics'):
            return self.env._env

        # Method 3: Search through wrapper chain manually
        current_env = self.env
        search_depth = 0
        max_depth = 10  # Prevent infinite loops

        while current_env is not None and search_depth < max_depth:
            if hasattr(current_env, 'add_metrics'):
                return current_env

            # Try different wrapper access patterns
            next_env = None
            for attr in ['env', '_env', 'unwrapped']:
                if hasattr(current_env, attr):
                    next_env = getattr(current_env, attr)
                    break

            current_env = next_env
            search_depth += 1

        return None

