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
from dataclasses import asdict

class PerAgentPreprocessorWrapper:
    """Wrapper that makes per-agent preprocessing compatible with SKRL's single preprocessor interface."""

    def __init__(self, agent, preprocessor_list):
        self.agent = agent
        self.preprocessor_list = preprocessor_list

    def __call__(self, tensor_input, train=False, inverse=False):
        """Called by SKRL as if it's a regular preprocessor."""
        return self.agent._apply_per_agent_preprocessing(
            tensor_input,
            self.preprocessor_list,
            train=train,
            inverse=inverse
        )

    def __bool__(self):
        """Return True if preprocessors exist (for SKRL's if self._state_preprocessor checks)."""
        return any(p is not None for p in self.preprocessor_list)

    def state_dict(self):
        """Return state dicts for all per-agent preprocessors (for checkpointing)."""
        state_dicts = {}
        for i, preprocessor in enumerate(self.preprocessor_list):
            if preprocessor is not None and hasattr(preprocessor, 'state_dict'):
                state_dicts[f'agent_{i}'] = preprocessor.state_dict()
        return state_dicts

    def load_state_dict(self, state_dict):
        """Load state dicts for all per-agent preprocessors (from checkpoint)."""
        for i, preprocessor in enumerate(self.preprocessor_list):
            key = f'agent_{i}'
            if key in state_dict and preprocessor is not None and hasattr(preprocessor, 'load_state_dict'):
                preprocessor.load_state_dict(state_dict[key])

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

        self.env = env  ## STORE ENVIRONMENT FOR WRAPPER ACCESS ##
        print(cfg["state_preprocessor_kwargs"])

        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg
        )

        # FIX: Override parent's auto values with our explicit config values
        # Parent Agent.__init__ expects these in cfg["experiment"]["key"] but we provide them at top level
        # This is because we have per-agent experiment configs in agent_exp_cfgs instead
        self.write_interval = cfg['write_interval']
        self.checkpoint_interval = cfg['checkpoint_interval']
        self.checkpoint_store_separately = False  # We handle per-agent checkpoints ourselves

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
        self.upload_ckpts_to_wandb = cfg.get('upload_ckpts_to_wandb', False)

        # Supervised selection loss parameters
        self.force_size = cfg.get('force_size', 3)
        self.supervised_selection_loss_weight = cfg.get('supervised_selection_loss_weight', 0.0)

        # Initialize agent experiment configs
        self.agent_exp_cfgs = cfg['agent_exp_cfgs']
        """for i in range(self.num_agents):
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
        """
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
                    self.agent_exp_cfgs[i]['experiment']['directory'],
                    self.agent_exp_cfgs[i]['experiment']['experiment_name'],
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
            self.memory.create_tensor(name="in-contact", size=self.force_size, dtype=torch.float32)

            # NOTE: All float tensors are filled with NaN by SKRL's create_tensor implementation.
            # This is expected behavior - NaN values will be overwritten during record_transition().

            # tensors sampled during training
            self._tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages", "in-contact"]


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

        # Log preprocessor statistics to WandB for verification
        wrapper = self._get_logging_wrapper()
        if wrapper:
            preprocessor_metrics = {}

            # Log state preprocessor metrics for each agent
            if hasattr(self, '_per_agent_state_preprocessors'):
                for i, preprocessor in enumerate(self._per_agent_state_preprocessors):
                    if preprocessor is not None and hasattr(preprocessor, 'running_mean'):
                        # Per-dimension statistics
                        running_mean = preprocessor.running_mean  # Shape: (obs_dim,)
                        running_var = preprocessor.running_variance  # Shape: (obs_dim,)
                        count = preprocessor.current_count.item()

                        # Aggregate statistics
                        mean_avg = running_mean.mean().item()
                        mean_std = running_mean.std().item()
                        var_avg = running_var.mean().item()
                        var_std = running_var.std().item()

                        print(f"    Agent {i} state preprocessor: mean_avg={mean_avg:.4f}, mean_std={mean_std:.4f}, var_avg={var_avg:.4f}, count={count}")

                        # Create per-agent tensors for WandB metrics
                        if 'state preprocessor / mean_avg' not in preprocessor_metrics:
                            preprocessor_metrics['state preprocessor / mean_avg'] = torch.full((self.num_agents,), float('nan'), device=self.device)
                            preprocessor_metrics['state preprocessor / mean_std'] = torch.full((self.num_agents,), float('nan'), device=self.device)
                            preprocessor_metrics['state preprocessor / var_avg'] = torch.full((self.num_agents,), float('nan'), device=self.device)
                            preprocessor_metrics['state preprocessor / var_std'] = torch.full((self.num_agents,), float('nan'), device=self.device)
                            preprocessor_metrics['state preprocessor / count'] = torch.full((self.num_agents,), float('nan'), device=self.device)

                        preprocessor_metrics['state preprocessor / mean_avg'][i] = mean_avg
                        preprocessor_metrics['state preprocessor / mean_std'][i] = mean_std
                        preprocessor_metrics['state preprocessor / var_avg'][i] = var_avg
                        preprocessor_metrics['state preprocessor / var_std'][i] = var_std
                        preprocessor_metrics['state preprocessor / count'][i] = float(count)

            # Log value preprocessor metrics for each agent
            if hasattr(self, '_per_agent_value_preprocessors'):
                for i, preprocessor in enumerate(self._per_agent_value_preprocessors):
                    if preprocessor is not None and hasattr(preprocessor, 'running_mean'):
                        # Scalar statistics (size=1)
                        mean_val = preprocessor.running_mean.item()
                        var_val = preprocessor.running_variance.item()
                        count = preprocessor.current_count.item()

                        print(f"    Agent {i} value preprocessor: mean={mean_val:.4f}, var={var_val:.4f}, count={count}")

                        # Create per-agent tensors for WandB metrics
                        if 'value preprocessor / mean' not in preprocessor_metrics:
                            preprocessor_metrics['value preprocessor / mean'] = torch.full((self.num_agents,), float('nan'), device=self.device)
                            preprocessor_metrics['value preprocessor / var'] = torch.full((self.num_agents,), float('nan'), device=self.device)
                            preprocessor_metrics['value preprocessor / count'] = torch.full((self.num_agents,), float('nan'), device=self.device)

                        preprocessor_metrics['value preprocessor / mean'][i] = mean_val
                        preprocessor_metrics['value preprocessor / var'][i] = var_val
                        preprocessor_metrics['value preprocessor / count'][i] = float(count)

            if preprocessor_metrics:
                wrapper.add_metrics(preprocessor_metrics)

        # Upload checkpoints to WandB if enabled
        if self.upload_ckpts_to_wandb:
            wrapper = self._get_logging_wrapper()
            if wrapper is None:
                raise RuntimeError("upload_ckpts_to_wandb is enabled but no logging wrapper found")

            for i in range(self.num_agents):
                # Upload policy checkpoint
                policy_success = wrapper.upload_checkpoint(i, ckpt_paths[i], 'policy')
                # Upload critic checkpoint
                critic_success = wrapper.upload_checkpoint(i, critic_paths[i], 'critic')
                print(f"Added agent {i} ckpt from path {ckpt_paths[i]}")
                # Only delete local files if both uploads succeeded
                """if policy_success and critic_success:
                    os.remove(ckpt_paths[i])
                    os.remove(critic_paths[i])
                else:
                    raise RuntimeError(f"Checkpoint upload failed for agent {i}")
                """
        #self.track_data("ckpt_video", (timestep, vid_path) )
        if self.track_ckpt_paths:
            lock = FileLock(self.tracker_path + ".lock")
            with lock:
                with open(self.tracker_path, "a") as f:
                    # Get WandB wrapper to access tracker data
                    wrapper = self._get_logging_wrapper()
                    for i in range(self.num_agents):
                        # Extract project and run_id from WandB wrapper trackers
                        if wrapper and hasattr(wrapper, 'trackers') and i < len(wrapper.trackers):
                            project = wrapper.trackers[i].run.project
                            run_id = wrapper.trackers[i].run.id
                        else:
                            # Fallback values if wrapper not available
                            project = "unknown_project"
                            run_id = "unknown_run_id"
                        f.write(f'{ckpt_paths[i]} {self.task_name} {vid_paths[i]} {project} {run_id}\n')

    def load(self, path: str, **kwargs):
        """Load single agent checkpoint with policy, critic, and preprocessor states."""
        checkpoint = torch.load(path, map_location=self.device)

        # Load policy network parameters
        if 'net_state_dict' in checkpoint:
            # Standard checkpoint format
            self.models['policy'].actor_mean.load_state_dict(checkpoint['net_state_dict'])
            if 'log_std' in checkpoint:
                self.models['policy'].actor_logstd = checkpoint['log_std']
        else:
            # Fallback: assume entire checkpoint is the policy state dict
            self.models['policy'].actor_mean.load_state_dict(checkpoint)

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
            except Exception as e:
                pass
        else:
            pass

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

                # Load per-agent value preprocessors
                if hasattr(self, '_per_agent_value_preprocessors'):
                    for i, preprocessor in enumerate(self._per_agent_value_preprocessors):
                        key = f'agent_{i}_value_preprocessor'
                        if key in ckpt and ckpt[key] is not None and preprocessor is not None:
                            preprocessor.load_state_dict(ckpt[key])

                # Backward compatibility: Load shared preprocessor states
                if 'state_preprocessor' in ckpt and ckpt['state_preprocessor'] is not None:
                    if hasattr(self, '_state_preprocessor') and self._state_preprocessor is not None:
                        self._state_preprocessor.load_state_dict(ckpt['state_preprocessor'])

                if 'value_preprocessor' in ckpt and ckpt['value_preprocessor'] is not None:
                    if hasattr(self, '_value_preprocessor') and self._value_preprocessor is not None:
                        self._value_preprocessor.load_state_dict(ckpt['value_preprocessor'])


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
            processed_states = self._state_preprocessor(states)
            values, _, _ = self.value.act({"states": processed_states}, role="value")
            values = self._value_preprocessor(values, inverse=True)
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

            # Get in-contact state from ForceTorqueWrapper
            ft_wrapper = self._get_force_torque_wrapper()
            if ft_wrapper is not None and hasattr(ft_wrapper.unwrapped, 'in_contact'):
                # Extract first force_size contact flags (no aggregation)
                # in_contact shape: (num_envs, 6) -> extract (num_envs, force_size)
                raw_in_contact = ft_wrapper.unwrapped.in_contact[:, :self.force_size]

                # Convert to float if boolean
                if raw_in_contact.dtype == torch.bool:
                    in_contact_state = raw_in_contact.float()
                else:
                    in_contact_state = raw_in_contact.clone()

                # Validate: fail fast if wrapper returns invalid data
                if torch.isnan(in_contact_state).any():
                    raise RuntimeError("ForceTorqueWrapper in_contact tensor contains NaN values!")
                if torch.isinf(in_contact_state).any():
                    raise RuntimeError("ForceTorqueWrapper in_contact tensor contains Inf values!")
                if (in_contact_state < 0.0).any() or (in_contact_state > 1.0).any():
                    min_val = in_contact_state.min().item()
                    max_val = in_contact_state.max().item()
                    raise RuntimeError(f"ForceTorqueWrapper in_contact has invalid values: min={min_val}, max={max_val}")
            else:
                # Error if wrapper not found - fail fast and loud
                raise RuntimeError(
                    "ForceTorqueWrapper not found in environment wrapper chain or in_contact attribute missing. "
                    "Ensure ForceTorqueWrapper is properly applied to the environment."
                )

            self.add_sample_to_memory(
                states=states.clone(),
                actions=actions.clone(),
                rewards=rewards.clone(),
                next_states=next_states.clone(),
                terminated=terminated.clone(),
                truncated=truncated.clone(),
                log_prob=self._current_log_prob.clone(),
                values=values.clone(),
                **{"in-contact": in_contact_state.clone()}
            )   
    
    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # update best models and write checkpoints
        if timestep > 0 and self.checkpoint_interval > 0 and not timestep % self.checkpoint_interval:
            # write checkpoints
            self.write_checkpoint(timestep, timesteps)

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

        # write wandb
        #if original_timestep > 0 and self.write_interval > 0 and not original_timestep % self.write_interval:
        #    self.write_tracking_data(original_timestep, timesteps)

        # Publish accumulated metrics to wandb after all operations complete
        # This ensures checkpoint metrics are published with the correct timestep
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            wrapper = self._get_logging_wrapper()
            if wrapper:
                wrapper.publish()

        # Debug: Print log_std values periodically to check divergence
        if original_timestep > 0 and self.write_interval > 0 and not original_timestep % self.write_interval:
            print("\n=== Log Std Values After Training ===")
            for i in range(self.num_agents):
                print(f"Agent {i}: {self.policy.actor_logstd[i].data}")
            print("=" * 50)

    def _update(self, timestep: int, timesteps: int):
        #super()._update(timestep, timesteps) def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # Note: timestep parameter available for use in update logic
        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
            num_agents: int = 1,
            envs_per_agent: int = 16
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
            num_envs = num_agents * envs_per_agent
            #print(memory_size, num_envs)
            #rollout_length = memory_size // num_envs
            #print(advantages.size())
            # Reshape to (rollout_steps, num_agents, envs_per_agent, 1)
            advantages_reshaped = advantages.view(memory_size, num_agents, envs_per_agent, -1)
            #print(advantages_reshaped.size())
            
            # Compute mean and std per agent (over rollout_steps and envs_per_agent dimensions)
            # Result shape: (1, num_agents, 1, 1)
            agent_means = advantages_reshaped.mean(dim=(0, 2), keepdim=True)
            agent_stds = advantages_reshaped.std(dim=(0, 2), keepdim=True)
            
            # Normalize (broadcasting handles per-agent normalization automatically)
            advantages_normalized = (advantages_reshaped - agent_means) / (agent_stds + 1e-8)
            
            # Flatten back to original shape (memory_size, 1)
            advantages = advantages_normalized.view(memory_size, num_envs, -1)
            
            return returns, advantages

        # log std:
        self._log_policy_std()

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            processed_next_states = self._state_preprocessor(self._current_next_states.float())
            last_values, _, _ = self.value.act(
                {"states": processed_next_states}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
            num_agents=self.num_agents,
            envs_per_agent=self.envs_per_agent
        )


        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # Log batch-level advantage metrics (once per update, before minibatch loop)
        in_contact = self.memory.get_tensor_by_name("in-contact")
        self._log_batch_advantage_metrics(advantages, in_contact)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)
        sample_size = self._rollouts // self._mini_batches
        # Initialize keep_mask at epoch start - persists across mini-batches
        keep_mask = torch.ones((self.num_agents,), dtype=bool, device=self.device)
        # Track how many minibatches each agent completed before KL violation
        minibatch_count = torch.zeros((self.num_agents,), dtype=torch.long, device=self.device)
        # learning epochs
        for epoch in range(self._learning_epochs):
            #print(f"Epoch:{epoch+1}:{keep_mask}")
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
                sampled_in_contact,
            ) in sampled_batches:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)
                    _, next_log_prob, policy_outputs = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )
                    mean_actions = policy_outputs["mean_actions"]  # Extract mean actions for supervised loss
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

                        # Increment minibatch count for agents still active (not yet violated KL)
                        minibatch_count += keep_mask.long()

                    # compute entropy loss
                    #print(self._entropy_loss_scale)
                    entropys = self.policy.get_entropy(role="policy")
                    entropys = entropys.view(sample_size, self.num_agents, self.envs_per_agent,-1)
                    if self._entropy_loss_scale > 0.0:
                        entropy_loss = -self._entropy_loss_scale * entropys[:, keep_mask,:].mean() 
                    else:
                        entropy_loss = 0
                        #entropys = None
                    

                    # compute policy loss
                    sampled_advantages = sampled_advantages.view(sample_size, self.num_agents, self.envs_per_agent)
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_losses = -torch.min(surrogate, surrogate_clipped)
                    policy_loss = policy_losses[:,keep_mask,:].mean()

                    # Compute supervised selection loss
                    if self.supervised_selection_loss_weight > 1e-8:  # Check if enabled
                        # Extract selection probabilities from mean_actions (already computed)
                        # mean_actions shape: (sample_size * num_agents * envs_per_agent, action_dim)
                        # First force_size values are selection probabilities (after sigmoid)
                        selection_probs = mean_actions[:, :self.force_size]

                        # Reshape both to match minibatch structure
                        selection_probs = selection_probs.view(sample_size, self.num_agents, self.envs_per_agent, self.force_size)

                        # sampled_in_contact shape: (total_num_envs, force_size) where total_num_envs = sample_size * num_agents * envs_per_agent
                        # Reshape to: (sample_size, num_agents, envs_per_agent, force_size)
                        sampled_in_contact_reshaped = sampled_in_contact.view(sample_size, self.num_agents, self.envs_per_agent, self.force_size)

                        # Ensure targets are valid binary labels [0, 1]
                        # Clamp to be safe in case of numerical issues during conversion
                        target_contact = torch.clamp(sampled_in_contact_reshaped, min=0.0, max=1.0)

                        # Clamp selection_probs to valid range [eps, 1-eps] to avoid log(0) or log(1) issues
                        selection_probs_clamped = torch.clamp(selection_probs, min=1e-7, max=1.0 - 1e-7)

                        # Binary cross-entropy per sample: compare predicted probs vs ground truth contact
                        # Shape: (sample_size, num_agents, envs_per_agent, force_size)
                        supervised_losses_per_dim = F.binary_cross_entropy(
                            selection_probs_clamped,
                            target_contact,
                            reduction='none'
                        )
                        # Average across force dimensions to get per-sample loss
                        # Shape: (sample_size, num_agents, envs_per_agent)
                        supervised_losses = supervised_losses_per_dim.mean(dim=-1)

                        # Compute total loss for backprop (with weight and keep_mask filtering)
                        supervised_loss = self.supervised_selection_loss_weight * supervised_losses[:, keep_mask, :].mean()
                    else:
                        supervised_loss = 0
                        supervised_losses = None

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
                    
                    value_loss, value_losses, predicted_values = self.calc_value_loss(sampled_states, sampled_values, sampled_returns, sample_size)

                # optimization step
                # zero out losses from cancelled

                # COLLECT NETWORK STATES BEFORE ANY OPTIMIZER STEP (Problem 1 Fix)
                # Store policy and critic states while gradients still exist
                self._log_minibatch_update(
                    keep_mask = keep_mask,
                    advantages = sampled_advantages,
                    old_log_probs = sampled_log_prob,
                    new_log_probs = next_log_prob,
                    entropies = entropys,
                    policy_losses = policy_losses,
                    supervised_losses = supervised_losses
                )

                self.optimizer.zero_grad()
                #self.timestep = timestep
                #self.log_policy_loss_components(self.policy, sampled_states, sampled_actions, sampled_advantages, sampled_log_prob, self.timestep)
                
                if timestep < self._random_value_timesteps:
                    self.update_nets(value_loss, update_policy=False, update_critic=True)  # Value-only training
                else:
                    self.update_nets(policy_loss + entropy_loss + value_loss + supervised_loss, update_policy=True, update_critic=True)  # Both networks


                if self.value_update_ratio > 1:
                    for i in range(self.value_update_ratio-1): #minus one because we already updated critic once
                        value_loss, vls, predicted_values = self.calc_value_loss(
                            sampled_states,
                            sampled_values,
                            sampled_returns,
                            sample_size
                        )
                        value_losses += vls
                        self.update_nets(value_loss, update_policy=False, update_critic=True)

                    value_losses /= self.value_update_ratio

                    # Log final results - network states already collected
                    self._log_minibatch_update(
                        keep_mask = keep_mask,
                        returns = sampled_returns,
                        values = predicted_values,
                        value_losses = value_losses,
                        supervised_losses = supervised_losses
                    )
                else:
                    # Log all metrics - network states already collected
                    self._log_minibatch_update(
                        keep_mask = keep_mask,
                        returns=sampled_returns,
                        values=predicted_values,
                        advantages = sampled_advantages,
                        old_log_probs = sampled_log_prob,
                        new_log_probs = next_log_prob,
                        entropies = entropys,
                        policy_losses = policy_losses,
                        value_losses=value_losses,
                        supervised_losses = supervised_losses
                    )
                mini_batch += 1

        # Add minibatch counts as onetime metric (will be published in post_interaction)
        wrapper = self._get_logging_wrapper()
        if wrapper:
            # Log minibatch counts before KL violation as onetime metric
            onetime_metrics = {
                "Policy/Minibatches_Before_KL_Violation": minibatch_count.float()
            }
            wrapper.add_metrics(onetime_metrics)

    def log_policy_loss_components(self, policy, states, actions, advantages, old_log_probs, step):
        """
        Log individual components of the PPO loss
        """
        with torch.no_grad():
            # Get current policy output
            _, new_log_probs, _ = policy.act(
                        {"states": states, "taken_actions": actions}, role="policy"
                    )
            
            # Compute ratio
            ratio = (new_log_probs - old_log_probs).exp()
            
            print(f"\n=== Step {step} Policy Loss Components ===")
            print(f"Ratio - mean: {ratio.mean().item():.4f}, std: {ratio.std().item():.4f}, min: {ratio.min().item():.4f}, max: {ratio.max().item():.4f}")
            print(f"Advantages - mean: {advantages.mean().item():.4f}, std: {advantages.std().item():.4f}")
            print(f"Old log probs - mean: {old_log_probs.mean().item():.4f}, std: {old_log_probs.std().item():.4f}")
            print(f"New log probs - mean: {new_log_probs.mean().item():.4f}, std: {new_log_probs.std().item():.4f}")

    def update_nets(self, loss, update_policy=True, update_critic=True):
        """
        Update networks and collect gradients for logging.

        Args:
            loss: Combined loss tensor
            update_policy: Whether policy network is being updated
            update_critic: Whether critic network is being updated
        """
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        #self.log_selection_gradients(self.policy, self.timestep)
        # Collect gradients immediately after backward pass while they exist
        if update_policy or update_critic:
            self._collect_and_store_gradients(
                store_policy_state=update_policy,
                store_critic_state=update_critic
            )

        if config.torch.is_distributed:
            self.policy.reduce_parameters()
            if self.policy is not self.value:
                self.value.reduce_parameters()

        if self._grad_norm_clip > 0:
            self.scaler.unscale_(self.optimizer)

            # NEW: Per-agent gradient clipping for improved performance
            if not hasattr(self, '_param_classification'):
                # Cache parameter classification for efficiency - only compute once
                self._param_classification = self.classify_model_parameters()

            block_params, regular_params = self._param_classification
            agent_grad_norms = self.compute_per_agent_grad_norms(block_params, regular_params)
            self.apply_per_agent_gradient_clipping(
                block_params, regular_params, agent_grad_norms, self._grad_norm_clip
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def classify_model_parameters(self):
        """Classify parameters into block vs regular for efficient per-agent processing.

        Returns:
            tuple: (block_params, regular_params) where:
                - block_params: List of (name, param) with agent dimension shape[0] == num_agents
                - regular_params: List of (name, param) for ParameterList items with _agent_id
        """
        block_params = []  # Parameters with agent dimension (shape[0] == num_agents)
        regular_params = []  # Regular parameters (like actor_logstd from ParameterList)

        for name, param in itertools.chain(
            self.policy.named_parameters(),
            self.value.named_parameters()
        ):
            if param.dim() > 0 and param.shape[0] == self.num_agents:
                # Block parameter: (num_agents, ...)
                block_params.append((name, param))
            elif hasattr(param, '_agent_id'):
                # ParameterList item with agent ID
                regular_params.append((name, param))
            # Skip other parameters (shouldn't be any in this architecture)

        return block_params, regular_params

    def compute_per_agent_grad_norms(self, block_params, regular_params):
        """Compute gradient norms separately for each agent.

        Args:
            block_params: List of (name, param) tuples for block parameters
            regular_params: List of (name, param) tuples for regular parameters

        Returns:
            torch.Tensor: Gradient norms for each agent, shape (num_agents,)
        """
        agent_grad_norms = torch.zeros(self.num_agents, device=self.device)

        for agent_idx in range(self.num_agents):
            grad_squares = []

            # Process block parameters (slice by agent)
            for name, param in block_params:
                if param.grad is not None:
                    # Slice agent's portion of the gradient
                    agent_grad = param.grad[agent_idx]  # Shape: varies by parameter
                    grad_squares.append(agent_grad.pow(2).sum())

            # Process regular parameters (filter by _agent_id)
            for name, param in regular_params:
                if param.grad is not None and param._agent_id == agent_idx:
                    grad_squares.append(param.grad.pow(2).sum())

            # Compute total norm for this agent
            if grad_squares:
                agent_grad_norms[agent_idx] = torch.sqrt(sum(grad_squares))

        return agent_grad_norms

    def apply_per_agent_gradient_clipping(self, block_params, regular_params, grad_norms, clip_value):
        """Apply gradient clipping separately for each agent.

        Args:
            block_params: List of (name, param) tuples for block parameters
            regular_params: List of (name, param) tuples for regular parameters
            grad_norms: Agent gradient norms, shape (num_agents,)
            clip_value: Maximum allowed gradient norm
        """
        for agent_idx in range(self.num_agents):
            if grad_norms[agent_idx] > clip_value:
                scale_factor = clip_value / grad_norms[agent_idx]

                # Clip block parameters
                for name, param in block_params:
                    if param.grad is not None:
                        param.grad[agent_idx].mul_(scale_factor)

                # Clip regular parameters
                for name, param in regular_params:
                    if param.grad is not None and param._agent_id == agent_idx:
                        param.grad.mul_(scale_factor)

    def adaptive_huber_delta(self, predicted, sampled, k=1.35):
        """Compute adaptive Huber delta with minimal GPU-CPU synchronization.

        F.huber_loss() requires delta as a Python float, so we still need .item()
        but we can optimize the computation to minimize the synchronization cost.
        """
        e = (sampled - predicted)
        med = e.median()
        mad = (e - med).abs().median() + 1e-8
        # F.huber_loss requires float, so we need .item() but only once at the end
        return float(k * mad)
    
    def calc_value_loss(self, sampled_states, sampled_values, sampled_returns, sample_size):

        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            # compute value loss
            batch_total_envs = sample_size * self.num_agents * self.envs_per_agent
            predicted_values, _, _ = self.value.act({"states": sampled_states.view(batch_total_envs,-1)}, role="value")
            predicted_values = predicted_values.view( sample_size, self.num_agents, self.envs_per_agent)

            if self._clip_predicted_values:
                # Optimized value clipping - use clamp directly, avoid intermediate tensors
                # Original: sampled_values + torch.clip(predicted_values - sampled_values, ...)
                # Optimized: torch.clamp(predicted_values, min=sampled_values-clip, max=sampled_values+clip)
                predicted_values = torch.clamp(
                    predicted_values,
                    min=sampled_values - self._value_clip,
                    max=sampled_values + self._value_clip
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

            value_loss = vls.mean()

            return value_loss, vls, predicted_values


    def _get_network_state(self, agent_idx):
        """
        Extract optimizer state for one agent, returning separate dicts for
        policy and critic. Assumes optimizer param_groups were created with
        make_agent_optimizer (policy first, critic second per agent).
        """
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
                        # Skip parameters for other agents
                    else:
                        # Block parameter - has agent dimension
                        try:
                            # For 3D tensors: (num_agents, out_features, in_features)
                            grad_norm = p.grad.detach()[agent_idx,:,:].norm(2)
                            state[role]['gradients'].append(grad_norm)
                        except (IndexError, RuntimeError):
                            try:
                                # For 2D tensors: (num_agents, out_features)
                                grad_norm = p.grad.detach()[agent_idx,:].norm(2)
                                state[role]['gradients'].append(grad_norm)
                            except (IndexError, RuntimeError):
                                # For 1D tensors or other cases: (num_agents,)
                                try:
                                    grad_norm = abs(p.grad.detach()[agent_idx])
                                    state[role]['gradients'].append(grad_norm)
                                except (IndexError, RuntimeError) as e:
                                    # Skip this parameter
                                    pass

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

                # Weight norms (always collect)
                state[role]['weight_norms'][pname] = p.norm().item()


        return state

    def _collect_and_store_gradients(self, store_policy_state=False, store_critic_state=False):
        """Optimized gradient collection using vectorized operations."""
        wrapper = self._get_logging_wrapper()
        if not wrapper:
            return

        # Use cached parameter classification for efficiency
        if not hasattr(self, '_param_classification'):
            self._param_classification = self.classify_model_parameters()

        block_params, regular_params = self._param_classification
        gradient_metrics = {}

        with torch.no_grad():
            if store_policy_state:
                policy_grad_norms, policy_step_sizes = self._compute_vectorized_metrics(
                    self.policy, block_params, regular_params, 'policy'
                )
                gradient_metrics['Policy/Gradient_Norm'] = policy_grad_norms
                gradient_metrics['Policy/Step_Size'] = policy_step_sizes

            if store_critic_state:
                critic_grad_norms, critic_step_sizes = self._compute_vectorized_metrics(
                    self.value, block_params, regular_params, 'critic'
                )
                gradient_metrics['Critic/Gradient_Norm'] = critic_grad_norms
                gradient_metrics['Critic/Step_Size'] = critic_step_sizes

        if gradient_metrics:
            wrapper.add_metrics(gradient_metrics)

    def _compute_vectorized_metrics(self, network, block_params, regular_params, role):
        """Compute gradient norms and step sizes for all agents in one vectorized operation."""
        # Pre-allocate tensors for all agents
        grad_norms = torch.zeros(self.num_agents, device=self.device)
        step_sizes = torch.zeros(self.num_agents, device=self.device)

        # Collect all gradients for block parameters (vectorized)
        block_grad_squares = torch.zeros(self.num_agents, device=self.device)
        for name, param in block_params:
            if param.grad is not None and self._param_belongs_to_network(param, network):
                # Compute per-agent gradient norms in one operation
                # Shape: (num_agents, ...) -> (num_agents,)
                agent_grads = param.grad.view(self.num_agents, -1)  # Flatten non-agent dimensions
                block_grad_squares += torch.sum(agent_grads ** 2, dim=1)

        # Collect gradients for regular parameters (agent-specific)
        regular_grad_squares = torch.zeros(self.num_agents, device=self.device)
        for name, param in regular_params:
            if param.grad is not None and hasattr(param, '_agent_id') and self._param_belongs_to_network(param, network):
                agent_id = param._agent_id
                regular_grad_squares[agent_id] += torch.sum(param.grad ** 2)

        # Compute final gradient norms
        grad_norms = torch.sqrt(block_grad_squares + regular_grad_squares)

        # Vectorized step size computation
        step_sizes = self._compute_vectorized_step_sizes(network, block_params, regular_params)

        return grad_norms, step_sizes

    def _compute_vectorized_step_sizes(self, network, block_params, regular_params):
        """Compute Adam step sizes for all agents using vectorized operations."""
        step_sizes = torch.zeros(self.num_agents, device=self.device)

        # Get optimizer parameters once
        lr = self._get_learning_rate()
        beta1, beta2 = self.optimizer.param_groups[0]['betas']
        eps = self.optimizer.param_groups[0]['eps']

        # Process block parameters
        for name, param in block_params:
            if (param in self.optimizer.state and
                param.grad is not None and
                self._param_belongs_to_network(param, network)):

                opt_state = self.optimizer.state[param]
                if all(key in opt_state for key in ['exp_avg', 'exp_avg_sq', 'step']):
                    step_count = opt_state['step']
                    if step_count > 0:
                        # Vectorized bias correction
                        bias_correction1 = 1 - beta1 ** step_count
                        bias_correction2 = 1 - beta2 ** step_count

                        # Vectorized step size calculation for all agents
                        exp_avg = opt_state['exp_avg'].view(self.num_agents, -1)
                        exp_avg_sq = opt_state['exp_avg_sq'].view(self.num_agents, -1)

                        corrected_exp_avg = exp_avg / bias_correction1
                        corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                        effective_step = corrected_exp_avg / (torch.sqrt(corrected_exp_avg_sq) + eps)
                        param_step_sizes = lr * torch.norm(effective_step, dim=1)
                        step_sizes += param_step_sizes

        # Process regular parameters
        for name, param in regular_params:
            if (param in self.optimizer.state and
                param.grad is not None and
                hasattr(param, '_agent_id') and
                self._param_belongs_to_network(param, network)):

                agent_id = param._agent_id
                opt_state = self.optimizer.state[param]

                if all(key in opt_state for key in ['exp_avg', 'exp_avg_sq', 'step']):
                    step_count = opt_state['step']
                    if step_count > 0:
                        bias_correction1 = 1 - beta1 ** step_count
                        bias_correction2 = 1 - beta2 ** step_count

                        exp_avg = opt_state['exp_avg']
                        exp_avg_sq = opt_state['exp_avg_sq']

                        corrected_exp_avg = exp_avg / bias_correction1
                        corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                        effective_step = corrected_exp_avg / (torch.sqrt(corrected_exp_avg_sq) + eps)
                        step_sizes[agent_id] += lr * torch.norm(effective_step)

        return step_sizes

    def _param_belongs_to_network(self, param, network):
        """Check if parameter belongs to the specified network."""
        # Simple check based on parameter being in network's parameters
        return any(p is param for p in network.parameters())

    def _get_learning_rate(self):
        """Get learning rate from optimizer."""
        for group in self.optimizer.param_groups:
            if group.get('role') == 'policy':
                return group['lr']
        return self.optimizer.param_groups[0]['lr']  # Fallback

    def _grad_norm_per_agent(self, agent_gradients):
        """Calculate gradient norm for a single agent's gradients."""
        if len(agent_gradients) == 0:
            return 0.0

        # Check individual gradient values
        grad_values = [g.item() for g in agent_gradients]

        # Calculate total norm
        result = torch.norm(torch.stack(agent_gradients), 2).item()
        return result

    def _adam_step_size_per_agent(self, agent_optimizer_state):
        """Calculate Adam step size for a single agent's optimizer state."""

        if len(agent_optimizer_state) == 0:
            return 0.0

        step_sizes = []
        for name, state in agent_optimizer_state.items():

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


                    if step_count > 0 and torch.norm(exp_avg_sq).item() > 1e-8:
                        bias_correction1 = 1 - beta1 ** step_count
                        bias_correction2 = 1 - beta2 ** step_count

                        # Calculate effective step size
                        effective_step_direction = (exp_avg / bias_correction1) / (torch.sqrt(exp_avg_sq / bias_correction2) + eps)
                        step_size = lr * torch.norm(effective_step_direction).item()
                        step_sizes.append(step_size)

                except Exception as e:
                    pass

        result = sum(step_sizes) / max(len(step_sizes), 1) if step_sizes else 0.0
        return result

    def _log_policy_std(self):
        """Log per-agent standard deviations for all action components by direction.

        Std structure depends on policy type:
        - BlockSimBaActor with sigma_idx > 0 (hybrid control with selection):
          - Indices 0:sigma_idx: Selection std (logged by direction)
          - Indices sigma_idx:sigma_idx+3: Position std X, Y, Z
          - Indices sigma_idx+3:sigma_idx+6: Rotation std (skipped)
          - Indices sigma_idx+6:sigma_idx+9: Force std X, Y, Z (if available)
        - Standard BlockSimBaActor or HybridControlBlockSimBaActor:
          - Indices 0-2: Position std X, Y, Z
          - Indices 3-5: Rotation std (skipped)
          - Indices 6-8: Force std X, Y, Z (if available)
          - Indices 9-11: Torque std X, Y, Z (if available)
        """
        wrapper = self._get_logging_wrapper()
        if not wrapper:
            return

        # Detect if using BlockSimBaActor with hybrid control (has sigma_idx)
        sigma_offset = 0
        if hasattr(self.policy, 'sigma_idx') and self.policy.sigma_idx > 0:
            sigma_offset = self.policy.sigma_idx

        # Pre-allocate per-agent tensors for all metrics
        std_metrics = {}

        # Initialize metric tensors based on what's available
        sample_log_std = self.policy.actor_logstd[0]
        std_dim = sample_log_std.shape[1]

        # Selection stds (if applicable)
        if sigma_offset >= 3:
            std_metrics['Network Output / Selection Std X'] = torch.zeros(self.num_agents, device=self.device)
            std_metrics['Network Output / Selection Std Y'] = torch.zeros(self.num_agents, device=self.device)
            std_metrics['Network Output / Selection Std Z'] = torch.zeros(self.num_agents, device=self.device)
        if sigma_offset >= 6:
            std_metrics['Network Output / Selection Std RX'] = torch.zeros(self.num_agents, device=self.device)
            std_metrics['Network Output / Selection Std RY'] = torch.zeros(self.num_agents, device=self.device)
            std_metrics['Network Output / Selection Std RZ'] = torch.zeros(self.num_agents, device=self.device)

        # Position stds (always present)
        std_metrics['Network Output / Pos Std X'] = torch.zeros(self.num_agents, device=self.device)
        std_metrics['Network Output / Pos Std Y'] = torch.zeros(self.num_agents, device=self.device)
        std_metrics['Network Output / Pos Std Z'] = torch.zeros(self.num_agents, device=self.device)

        # Force stds (if available)
        force_start_idx = sigma_offset + 6
        if std_dim >= force_start_idx + 3:
            std_metrics['Network Output / Force Std X'] = torch.zeros(self.num_agents, device=self.device)
            std_metrics['Network Output / Force Std Y'] = torch.zeros(self.num_agents, device=self.device)
            std_metrics['Network Output / Force Std Z'] = torch.zeros(self.num_agents, device=self.device)

        # Torque stds (if available)
        torque_start_idx = sigma_offset + 9
        if std_dim >= torque_start_idx + 3:
            std_metrics['Network Output / Torque Std X'] = torch.zeros(self.num_agents, device=self.device)
            std_metrics['Network Output / Torque Std Y'] = torch.zeros(self.num_agents, device=self.device)
            std_metrics['Network Output / Torque Std Z'] = torch.zeros(self.num_agents, device=self.device)

        # Fill in values for each agent
        for i, log_std in enumerate(self.policy.actor_logstd):
            # log_std shape: (1, std_dim) where std_dim depends on action space
            std = log_std.exp()[0]  # Shape: (std_dim,)

            # Selection std by direction (if sigma_offset > 0, meaning hybrid control)
            if sigma_offset >= 3:
                std_metrics['Network Output / Selection Std X'][i] = std[0]
                std_metrics['Network Output / Selection Std Y'][i] = std[1]
                std_metrics['Network Output / Selection Std Z'][i] = std[2]
            if sigma_offset >= 6:
                std_metrics['Network Output / Selection Std RX'][i] = std[3]
                std_metrics['Network Output / Selection Std RY'][i] = std[4]
                std_metrics['Network Output / Selection Std RZ'][i] = std[5]

            # Position std by direction (always present after sigma_offset)
            std_metrics['Network Output / Pos Std X'][i] = std[sigma_offset + 0]
            std_metrics['Network Output / Pos Std Y'][i] = std[sigma_offset + 1]
            std_metrics['Network Output / Pos Std Z'][i] = std[sigma_offset + 2]

            # Force std by direction (if std has enough elements)
            if std_dim >= force_start_idx + 3:
                std_metrics['Network Output / Force Std X'][i] = std[force_start_idx + 0]
                std_metrics['Network Output / Force Std Y'][i] = std[force_start_idx + 1]
                std_metrics['Network Output / Force Std Z'][i] = std[force_start_idx + 2]

            # Torque std by direction (if std has enough elements)
            if std_dim >= torque_start_idx + 3:
                std_metrics['Network Output / Torque Std X'][i] = std[torque_start_idx + 0]
                std_metrics['Network Output / Torque Std Y'][i] = std[torque_start_idx + 1]
                std_metrics['Network Output / Torque Std Z'][i] = std[torque_start_idx + 2]

        # All metrics are per-agent tensors (shape: num_agents)
        wrapper.add_metrics(std_metrics)

    def _log_minibatch_update(
            self,
            keep_mask=None,
            returns=None,
            values=None,
            advantages=None,
            old_log_probs=None,
            new_log_probs=None,
            entropies=None,
            policy_losses=None,
            value_losses=None,
            supervised_losses=None
    ):
        """Log minibatch update metrics through wrapper system with per-agent support.

        All metrics are computed per-agent using vectorized operations for efficiency.
        Tensors are assumed to be shaped as (sample_size, num_agents, envs_per_agent).

        Optimized version that eliminates GPU-CPU synchronization and uses vectorized operations
        instead of per-agent loops. All operations stay on GPU until passed to wrapper.

        Args:
            returns: Return values tensor (sample_size, num_agents, envs_per_agent)
            values: Value predictions tensor (sample_size, num_agents, envs_per_agent)
            advantages: Advantage values tensor (sample_size, num_agents, envs_per_agent)
            old_log_probs: Old log probabilities tensor (sample_size, num_agents, envs_per_agent)
            new_log_probs: New log probabilities tensor (sample_size, num_agents, envs_per_agent)
            entropies: Entropy values tensor (sample_size, num_agents, envs_per_agent)
            policy_losses: Policy loss values tensor (sample_size, num_agents, envs_per_agent)
            value_losses: Value loss values tensor (sample_size, num_agents, envs_per_agent)
            supervised_losses: Supervised selection loss tensor (sample_size, num_agents, envs_per_agent) or None
        """
        wrapper = self._get_logging_wrapper()
        if wrapper is None:
            pass
            return  # No wrapper system available, skip logging

        try:
            stats = {}

            with torch.no_grad():
                # --- Policy stats (vectorized) ---
                if new_log_probs is not None and old_log_probs is not None:
                    ratio = (new_log_probs - old_log_probs).exp()
                    clip_mask = (ratio < 1 - self._ratio_clip) | (ratio > 1 + self._ratio_clip)
                    kl = old_log_probs - new_log_probs

                    # Vectorized means across samples and environments, keep agents dimension
                    kl_avg = kl.mean(dim=(0, 2))  # Shape: (num_agents,)
                    if keep_mask is not None:
                        kl_avg[~keep_mask] = float('nan')
                    stats["Policy/KL_Divergence_Avg"] = kl_avg

                    clip_fraction = clip_mask.float().mean(dim=(0, 2))  # Shape: (num_agents,)
                    if keep_mask is not None:
                        clip_fraction[~keep_mask] = float('nan')
                    stats["Policy/Clip_Fraction"] = clip_fraction

                    # Vectorized quantiles - reshape to (sample_size * envs_per_agent, num_agents)
                    kl_per_agent = kl.permute(1, 0, 2).reshape(self.num_agents, -1).T
                    kl_95 = kl_per_agent.quantile(0.95, dim=0)  # Shape: (num_agents,)
                    if keep_mask is not None:
                        kl_95[~keep_mask] = float('nan')
                    stats["Policy/KL_Divergence_95_Quantile"] = kl_95

                # Vectorized entropy and policy loss
                if entropies is not None:
                    entropy_avg = entropies.mean(dim=(0, 2))  # Shape: (num_agents,)
                    if keep_mask is not None:
                        entropy_avg[~keep_mask] = float('nan')
                    stats["Policy/Entropy_Avg"] = entropy_avg

                if policy_losses is not None:
                    policy_loss_avg = policy_losses.mean(dim=(0, 2))  # Shape: (num_agents,)
                    if keep_mask is not None:
                        policy_loss_avg[~keep_mask] = float('nan')
                    stats["Policy/Loss_Avg"] = policy_loss_avg

                # --- Value stats (vectorized) ---
                if value_losses is not None and values is not None and returns is not None:
                    #inverse_values = self._value_preprocessor(values, inverse=True)
                    sample_size = values.shape[0]
                    batch_total = sample_size * self.num_agents * self.envs_per_agent
                    inverse_values = self._value_preprocessor(values.view(batch_total, 1), inverse=True).view(sample_size, self.num_agents, self.envs_per_agent)
                    #print(f"Inverse values - has NaN: {torch.isnan(inverse_values).any()}, has Inf: {torch.isinf(inverse_values).any()}")

                    # Vectorized basic statistics
                    stats["Critic/Loss_Avg"] = value_losses.mean(dim=(0, 2))  # Shape: (num_agents,)
                    stats["Critic/Predicted_Values_Avg"] = inverse_values.mean(dim=(0, 2))  # Shape: (num_agents,)
                    stats["Critic/Predicted_Values_Std"] = inverse_values.std(dim=(0, 2))  # Shape: (num_agents,)

                    # Inverse transform returns to original scale
                    inverse_returns = self._value_preprocessor(returns.view(batch_total, 1), inverse=True).view(sample_size, self.num_agents, self.envs_per_agent)

                    # Returns statistics
                    stats["Critic/Returns_Avg"] = inverse_returns.mean(dim=(0, 2))  # Shape: (num_agents,)
                    stats["Critic/Returns_Std"] = inverse_returns.std(dim=(0, 2))  # Shape: (num_agents,)

                    # Value error statistics (returns - predicted_values)
                    value_error = inverse_returns - inverse_values  # Both in original scale
                    stats["Critic/Value_Error_Mean"] = value_error.mean(dim=(0, 2))  # Shape: (num_agents,)
                    stats["Critic/Value_Error_Std"] = value_error.std(dim=(0, 2))  # Shape: (num_agents,)

                    # Vectorized quantiles - reshape to (sample_size * envs_per_agent, num_agents)
                    value_losses_per_agent = value_losses.permute(1, 0, 2).reshape(self.num_agents, -1).T
                    stats["Critic/Loss_Median"] = value_losses_per_agent.median(dim=0).values  # Shape: (num_agents,)
                    stats["Critic/Loss_95_Quantile"] = value_losses_per_agent.quantile(0.95, dim=0)  # Shape: (num_agents,)
                    stats["Critic/Loss_90_Quantile"] = value_losses_per_agent.quantile(0.90, dim=0)  # Shape: (num_agents,)

                    # Vectorized explained variance - all operations stay on GPU
                    returns_var = returns.var(dim=(0, 2), unbiased=False).clamp(min=1e-8)  # Shape: (num_agents,)
                    residual_var = (returns - values).var(dim=(0, 2), unbiased=False)  # Shape: (num_agents,)
                    stats["Critic/Explained_Variance"] = 1 - (residual_var / returns_var)  # Shape: (num_agents,)

                # --- Advantage diagnostics (vectorized) ---
                if advantages is not None:
                    # Vectorized mean and standard deviation
                    adv_mean_stat = advantages.mean(dim=(0, 2))  # Shape: (num_agents,)
                    if keep_mask is not None:
                        adv_mean_stat[~keep_mask] = float('nan')
                    stats["Advantage/Minibatch_Mean"] = adv_mean_stat

                    adv_std_stat = advantages.std(dim=(0, 2))  # Shape: (num_agents,)
                    if keep_mask is not None:
                        adv_std_stat[~keep_mask] = float('nan')
                    stats["Advantage/Minibatch_Std"] = adv_std_stat

                    # Vectorized skewness calculation
                    adv_mean = advantages.mean(dim=(0, 2), keepdim=True)  # Shape: (1, num_agents, 1)
                    adv_std = advantages.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)  # Shape: (1, num_agents, 1)
                    adv_centered = advantages - adv_mean  # Shape: (sample_size, num_agents, envs_per_agent)
                    adv_skew = (adv_centered ** 3).mean(dim=(0, 2)) / (adv_std.squeeze() ** 3)  # Shape: (num_agents,)
                    if keep_mask is not None:
                        adv_skew[~keep_mask] = float('nan')
                    stats["Advantage/Minibatch_Skew"] = adv_skew

                # --- Supervised Selection Loss stats (vectorized) ---
                if supervised_losses is not None:
                    # Vectorized mean and quantiles (same as policy_losses processing)
                    supervised_loss_avg = supervised_losses.mean(dim=(0, 2))  # Shape: (num_agents,)
                    if keep_mask is not None:
                        supervised_loss_avg[~keep_mask] = float('nan')
                    stats["Policy/Supervised_Loss_Avg"] = supervised_loss_avg

            # Pass vectorized tensors to wrapper system (no .item() calls!)
            wrapper.add_metrics(stats)

        except Exception as e:
            # Silently fail if wrapper system not available
            print(e)
            raise RuntimeError

    def _log_batch_advantage_metrics(self, advantages, in_contact):
        """Log batch-level advantage metrics including contact-conditioned stats.

        Called once per update before minibatch loop. All metrics are per-agent.

        Args:
            advantages: Full advantage tensor from memory, shape (memory_size, num_envs, 1)
            in_contact: Full contact tensor from memory, shape (memory_size, num_envs, force_size)
        """
        wrapper = self._get_logging_wrapper()
        if wrapper is None:
            return

        with torch.no_grad():
            # Reshape to (memory_size, num_agents, envs_per_agent)
            memory_size = advantages.shape[0]
            adv = advantages.view(memory_size, self.num_agents, self.envs_per_agent)
            contact = in_contact.view(memory_size, self.num_agents, self.envs_per_agent, -1)

            stats = {}

            # Batch-level stats (per agent)
            stats['Advantage/Batch_Mean'] = adv.mean(dim=(0, 2))
            stats['Advantage/Batch_Std'] = adv.std(dim=(0, 2))
            # Skew calculation
            adv_mean = adv.mean(dim=(0, 2), keepdim=True)
            adv_std = adv.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
            adv_centered = adv - adv_mean
            stats['Advantage/Batch_Skew'] = (adv_centered ** 3).mean(dim=(0, 2)) / (adv_std.squeeze() ** 3)

            # Contact masks
            any_contact = contact.any(dim=-1)  # (memory_size, num_agents, envs_per_agent)
            no_contact = ~any_contact
            contact_x = contact[..., 0] > 0.5
            contact_y = contact[..., 1] > 0.5
            contact_z = contact[..., 2] > 0.5

            # Helper to compute masked stats per agent
            def masked_stats(mask, prefix):
                means = torch.full((self.num_agents,), float('nan'), device=self.device)
                stds = torch.full((self.num_agents,), float('nan'), device=self.device)
                skews = torch.full((self.num_agents,), float('nan'), device=self.device)

                for i in range(self.num_agents):
                    agent_adv = adv[:, i, :]
                    agent_mask = mask[:, i, :]
                    masked = agent_adv[agent_mask]

                    if masked.numel() > 1:
                        m = masked.mean()
                        s = masked.std().clamp(min=1e-8)
                        means[i] = m
                        stds[i] = s
                        skews[i] = ((masked - m) ** 3).mean() / (s ** 3)

                stats[f'{prefix}_Mean'] = means
                stats[f'{prefix}_Std'] = stds
                stats[f'{prefix}_Skew'] = skews

            # Compute all contact-conditioned metrics
            masked_stats(any_contact, 'Advantage/Any_Contact')
            masked_stats(no_contact, 'Advantage/No_Contact')
            masked_stats(contact_x, 'Advantage/Contact_X')
            masked_stats(~contact_x, 'Advantage/No_Contact_X')
            masked_stats(contact_y, 'Advantage/Contact_Y')
            masked_stats(~contact_y, 'Advantage/No_Contact_Y')
            masked_stats(contact_z, 'Advantage/Contact_Z')
            masked_stats(~contact_z, 'Advantage/No_Contact_Z')

            wrapper.add_metrics(stats)

    def _setup_per_agent_preprocessors(self):
        """Set up per-agent independent preprocessors.

        This creates separate preprocessor instances for each agent using the same
        preprocessor type/config specified in the main config. Much simpler than
        requiring per-agent config entries.
        """

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
            else:
                self._per_agent_state_preprocessors.append(None)

            # Create independent value preprocessor for this agent
            if value_preprocessor_class is not None:
                # Create separate instance for this agent
                value_preprocessor = value_preprocessor_class(**value_preprocessor_kwargs)
                self._per_agent_value_preprocessors.append(value_preprocessor)
            else:
                self._per_agent_value_preprocessors.append(None)

        # Set the primary preprocessors using wrapper for SKRL compatibility
        self._state_preprocessor = PerAgentPreprocessorWrapper(self, self._per_agent_state_preprocessors)
        self._value_preprocessor = PerAgentPreprocessorWrapper(self, self._per_agent_value_preprocessors)


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
            return True

        # Method 2: Check if SKRL wrapper has the base environment with add_metrics
        if hasattr(self.env, '_env') and hasattr(self.env._env, 'add_metrics'):
            return True

        # Method 3: Search through wrapper chain manually
        current_env = self.env
        search_depth = 0
        max_depth = 10  # Prevent infinite loops

        while current_env is not None and search_depth < max_depth:
            if hasattr(current_env, 'add_metrics'):
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

    def log_selection_gradients(self, model, step):
        """
        Log gradients of selection parameters to see if they're coupled
        """
        
        # Access the final output layer that produces selection logits
        fc_out = model.actor_mean.fc_out
        
        # Get gradients for selection parameters (first 2*force_size outputs)
        if fc_out.weight.grad is not None:
            # Shape: (num_agents, act_dim, hidden_dim)
            # Selection weights are first 2*force_size rows
            sel_weight_grad = fc_out.weight.grad[:, :6, :]  # (num_agents, 6, hidden_dim)
            
            # Check if gradients for different dimensions are identical
            print(f"\n=== Step {step} Selection Weight Gradients ===")
            for agent_idx in range(model.num_agents):
                # Get gradients for this agent
                grad_x = sel_weight_grad[agent_idx, 0:2, :]  # X dimension (pos & force logits)
                grad_y = sel_weight_grad[agent_idx, 2:4, :]  # Y dimension
                grad_z = sel_weight_grad[agent_idx, 4:6, :]  # Z dimension
                
                # Compute norms
                norm_x = grad_x.norm().item()
                norm_y = grad_y.norm().item()
                norm_z = grad_z.norm().item()
                
                # Check similarity (cosine similarity between flattened gradients)
                grad_x_flat = grad_x.flatten()
                grad_y_flat = grad_y.flatten()
                grad_z_flat = grad_z.flatten()
                
                cos_xy = torch.nn.functional.cosine_similarity(
                    grad_x_flat.unsqueeze(0), grad_y_flat.unsqueeze(0)
                ).item()
                cos_xz = torch.nn.functional.cosine_similarity(
                    grad_x_flat.unsqueeze(0), grad_z_flat.unsqueeze(0)
                ).item()
                cos_yz = torch.nn.functional.cosine_similarity(
                    grad_y_flat.unsqueeze(0), grad_z_flat.unsqueeze(0)
                ).item()
                
                print(f"Agent {agent_idx}:")
                print(f"  Gradient norms - X: {norm_x:.6f}, Y: {norm_y:.6f}, Z: {norm_z:.6f}")
                print(f"  Cosine similarity - X-Y: {cos_xy:.6f}, X-Z: {cos_xz:.6f}, Y-Z: {cos_yz:.6f}")
                
                # Check if they're nearly identical (similarity > 0.99)
                if cos_xy > 0.99 and cos_xz > 0.99 and cos_yz > 0.99:
                    print(f"  ⚠️  WARNING: Gradients are nearly IDENTICAL across dimensions!")
                
        # Also check bias gradients
        if fc_out.bias.grad is not None:
            sel_bias_grad = fc_out.bias.grad[:, :6]  # (num_agents, 6)
            print(f"\nSelection Bias Gradients:")
            for agent_idx in range(model.num_agents):
                bias_grads = sel_bias_grad[agent_idx]
                print(f"Agent {agent_idx}: {bias_grads.tolist()}")


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

    def _get_force_torque_wrapper(self):
        """Get the ForceTorqueWrapper for accessing contact state.

        Returns the wrapper object if found, None otherwise.
        Searches through the wrapper chain to find ForceTorqueWrapper.
        """
        current_env = self.env
        search_depth = 0
        max_depth = 10  # Prevent infinite loops

        while current_env is not None and search_depth < max_depth:
            # Check if this is the ForceTorqueWrapper
            if hasattr(current_env, '__class__') and 'ForceTorqueWrapper' in str(current_env.__class__):
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

