from typing import Any, Mapping, Optional, Tuple, Union
import copy
import os

import gym
import gymnasium
import skrl
import torch
import numpy as np
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl import config, logger

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from data_processing.data_manager import DataManager

import wandb

import collections
class WandbLoggerPPO(PPO):
    def __del__(self):
        self.data_manager.finish()

    def __init__(
            self,
            models: Mapping[str, Model],
            memory: Optional[Union[Memory, Tuple[Memory]]] = None,
            observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
            action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
            num_envs: int = 256,
            device: Optional[Union[str, torch.device]] = None,
            cfg: Optional[dict] = None
    ) -> None:
        super().__init__(models, memory, observation_space, action_space, device, cfg)
        self.global_step = 0
        self.num_envs = num_envs
        
        self._track_rewards = collections.deque(maxlen=1000)
        self._track_timesteps = collections.deque(maxlen=1000)

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent

        This method should be called before the agent is used.
        It will initialize the TensorBoard writer (and optionally Weights & Biases) and create the checkpoints directory

        :param trainer_cfg: Trainer configuration
        :type trainer_cfg: dict, optional
        """

        #super().init(trainer_cfg)
        #return
        trainer_cfg = trainer_cfg if trainer_cfg is not None else {}

        # update agent configuration to avoid duplicated logging/checking in distributed runs
        if config.torch.is_distributed and config.torch.rank:
            self.write_interval = 0
            self.checkpoint_interval = 0
            

        # main entry to log data for consumption and visualization by TensorBoard
        if self.write_interval == "auto":
            self.write_interval = int(trainer_cfg.get("timesteps", 0) / 100)
        #if self.write_interval > 0:
        #    self.writer = SummaryWriter(log_dir=self.experiment_dir)

        if self.checkpoint_interval == "auto":
            self.checkpoint_interval = int(trainer_cfg.get("timesteps", 0) / 10)
        if self.checkpoint_interval > 0:
            os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)

        self.set_mode("eval")
        
        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
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

        # setup Weights & Biases
        self.log_wandb = False
        if self.cfg.get("experiment", {}).get("wandb", False):
            # save experiment configuration

            try:
                models_cfg = {k: v.net._modules for (k, v) in self.models.items()}
            except AttributeError:
                models_cfg = {k: v._modules for (k, v) in self.models.items()}
            wandb_config={**self.cfg, **trainer_cfg, **models_cfg, "num_envs":self.num_envs}
            # set default values
            wandb_kwargs = copy.deepcopy(self.cfg.get("experiment", {}).get("wandb_kwargs", {}))
            wandb_kwargs.setdefault("name", os.path.split(self.experiment_dir)[-1])
            wandb_kwargs.setdefault("config", {})
            wandb_kwargs["config"].update(wandb_config)
            # init Weights & Biases
            
            
            if wandb_kwargs['api_key'] == "-1":
                self.data_manager = DataManager(
                    project=wandb_kwargs['project'],
                    entity=wandb_kwargs['entity']
                )
            else:
                self.data_manager = DataManager(
                    project=wandb_kwargs['project'],
                    entity=wandb_kwargs['entity'],
                    api_key=wandb_kwargs['api_key']
                )
            
            self.data_manager.init_new_run(
                run_name=wandb_kwargs['run_name'],
                config=wandb_config,
                tags=wandb_kwargs['tags'],
                group=wandb_kwargs['group']
            )
            self.log_wandb = True

    def track_video_path(self, tag: str, value: str, timestep)-> None:
        self.tracking_data[tag + "_video"].append(value)
        self.tracking_data[tag + "_video"].append(timestep * self.num_envs)
        #self.data_manager.add_mp4(value, step= timestep * self.num_envs, cap=tag, fps=30)

    def track_hist(self, tag: str, value: torch.Tensor, timestep: int) -> None:
        self.data_manager.add_histogram(tag, value, timestep * self.num_envs)

    def track_data(self, tag: str, value: float) -> None:
        """Track data to TensorBoard

        Currently only scalar data are supported

        :param tag: Data identifier (e.g. 'Loss / policy loss')
        :type tag: str
        :param value: Value to track
        :type value: float
        """
        self.tracking_data[tag].append(value)

    def write_tracking_data(self, timestep: int, timesteps: int, eval=False) -> None:
        """Write tracking data to TensorBoard

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if self.log_wandb:
            prefix = "Eval_" if eval else "Training_"

            # handle cumulative rewards
            if len(self._track_rewards):
                track_rewards = np.array(self._track_rewards)
                track_timesteps = np.array(self._track_timesteps)
                
                self.tracking_data[prefix + "Reward / Return (max)"].append(np.max(track_rewards))
                self.tracking_data[prefix + "Reward / Return (min)"].append(np.min(track_rewards))
                self.tracking_data[prefix + "Reward / Return (mean)"].append(np.mean(track_rewards))

                self.tracking_data[prefix + "Episode / Total timesteps (max)"].append(np.max(track_timesteps))
                self.tracking_data[prefix + "Episode / Total timesteps (min)"].append(np.min(track_timesteps))
                self.tracking_data[prefix + "Episode / Total timesteps (mean)"].append(np.mean(track_timesteps))

                for rew_type in self.rew_types:
                    #print(f"final {rew_type}: { np.mean(self.rews['_track_'+rew_type]) }")
                    self.track_data(prefix + "Reward / " + rew_type, np.mean(self.rews["_track_"+rew_type]))

            keep = {}
            for k, v in self.tracking_data.items():
                if k.endswith("_video"):
                    try:
                        self.data_manager.add_mp4(
                            v[0], # path
                            k[:-6], #tag
                            timestep * self.num_envs, # step
                            f"At timestep {v[1]}" # caption
                        )
                    except FileNotFoundError:
                        keep[k] = v
                elif k.endswith("(min)"):
                    self.data_manager.add_scalar({k:np.min(v)}, timestep * self.num_envs)
                elif k.endswith("(max)"):
                    self.data_manager.add_scalar({k:np.max(v)}, timestep * self.num_envs)
                else:
                    self.data_manager.add_scalar({k:np.mean(v)}, timestep * self.num_envs)
            
            tot = 0
            for name in self.totals:
                self.data_manager.add_scalar(
                    {prefix + "Termination / " + name: self.totals[name]}, 
                    timestep*self.num_envs
                )
                tot += 0 if 'engage' in name else self.totals[name]
                self.totals[name] = 0

            timeouts = max(0, self.num_envs - tot)
            self.curr_engaged[:] = False
            self.data_manager.add_scalar(
                {prefix + "Termination / time_out" : timeouts}, 
                timestep*self.num_envs
            )

            for name in self.once:
                self.data_manager.add_scalar(
                    {prefix + " Termination / " + name + " once": torch.sum(self.once[name])}, 
                    timestep*self.num_envs
                )
                self.once[name][:] = False
            #self.data_manager.add_scalar({prefix + 'Termination / Engaged':torch.sum(self.engaged_once).item()}, timestep * self.num_envs)
        # reset data containers for next iteration
        self._track_rewards.clear()
        self._track_timesteps.clear()
        self.tracking_data.clear()
        for rew_type in self.rew_types:
            self.rews[rew_type]*=0
            self.rews["_track_"+rew_type].clear()
        
        self.reset_tracking()
        if self.log_wandb:
            for k, v in keep.items():
                self.tracking_data[k] = v

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        #print(f'Rollouts-Count {self._rollout}\t{self._rollouts}\t{not self._rollout % self._rollouts}\t{timestep >= self._learning_starts}')
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        timestep += 1
        self.global_step+= self.num_envs

        # update best models and write checkpoints
        if timestep > 1 and self.checkpoint_interval > 0 and not timestep % self.checkpoint_interval:
            # update best models
            reward = np.mean(self.tracking_data.get("Reward / Total reward (mean)", -2 ** 31))
            if reward > self.checkpoint_best_modules["reward"]:
                self.checkpoint_best_modules["timestep"] = timestep
                self.checkpoint_best_modules["reward"] = reward
                self.checkpoint_best_modules["saved"] = False
                self.checkpoint_best_modules["modules"] = {k: copy.deepcopy(self._get_internal_value(v)) for k, v in self.checkpoint_modules.items()}
            # write checkpoints
            self.write_checkpoint(timestep, timesteps)

        # write wandb
        if timestep > 1 and self.write_interval > 0 and not timestep % self.write_interval:
            self.write_tracking_data(timestep, timesteps)


    def add_sample_to_memory(self, **tensors: torch.Tensor) -> None:
        self.memory.add_samples( **tensors )
        for memory in self.secondary_memories:
                memory.add_samples( **tensors )


    def record_transition(self,
                        states: torch.Tensor,
                        actions: torch.Tensor,
                        rewards: torch.Tensor,
                        next_states: torch.Tensor,
                        terminated: torch.Tensor,
                        truncated: torch.Tensor,
                        infos: Any,
                        timestep: int,
                        timesteps: int,
                        reward_dist: dict,
                        term_dist: dict,
                        alive_mask: torch.Tensor = None) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # note this is where I can add tracking stuff
        eval_mode = alive_mask is not None
        #print("Logger eval mode:", eval_mode)
        if not eval_mode and self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
            #print("Avg Raw Values:", torch.mean(values))
            values = self._value_preprocessor(values, inverse=True)
            #print("Avg Processed Values:", torch.mean(values))

            # time-limit (truncation) boostrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated
            
            
            # alternative approach to deal with termination, see ppo but no matter what it 
            # goes through every sample in the memory (even if not filled), this way we set 
            # the actions to zero on termination, and then pass the fixed state in basically
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



        #print('eval mode:', 'on' if eval_mode else 'off')
        if self.write_interval > 0 or eval_mode:
            # compute the cumulative sum of the rewards and timesteps
            if self._cumulative_rewards is None:
                self._cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
                self._cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)
                #self.engaged_once = torch.zeros((self.num_envs, ), dtype=torch.bool, device = self.device)
                self.totals = {}
                for name in infos['my_log_data']["count_step"]:
                    self.totals[name] = 0

                self.once = {}
                for name in infos['my_log_data']['once']:
                    self.once[name] = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                self.curr_engaged = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                self.rews = {}
                self.rew_types = [name for name in infos['my_log_data']['step_rew']]
                for name in self.rew_types:
                    self.rews[name] = torch.zeros_like(rewards, dtype=torch.float32)
                    self.rews["_track_"+name] = collections.deque(maxlen=1000)

            # handle reward 
            prefix = "Training "
            if eval_mode:
                prefix = "Eval "

            if 'smoothness' in infos.keys():
                for skey in infos['smoothness']:
                    self.track_data(
                        prefix + skey + " (mean)", 
                        torch.mean( infos['smoothness'][skey] ).item()
                    )
                    self.track_data(
                        prefix + skey + " (max)", 
                        torch.max( infos['smoothness'][skey] ).item()
                    )

            for key in infos['log']:
                if 'Curriculum/init_height_sampling' in key:
                    self.track_data(key, infos['log'][key])

            # record step reward data
            self.tracking_data[prefix + "Reward / Instantaneous reward (max)"].append(torch.max(rewards).item())
            self.tracking_data[prefix + "Reward / Instantaneous reward (min)"].append(torch.min(rewards).item())
            self.tracking_data[prefix + "Reward / Instantaneous reward (mean)"].append(torch.mean(rewards).item())

            for name in infos['my_log_data']["once"]:
                self.once[name] = torch.logical_or(self.once[name], infos['my_log_data']['once'][name])
                if 'engage' in name: # count step is the current status, once is the rollout status
                    self.curr_engaged = torch.logical_or(self.curr_engaged, infos['my_log_data']['count_step'][name])
                    
            if eval_mode:
                self._cumulative_rewards += alive_mask * rewards

                for rew_type in self.rew_types:
                    #self.rews[rew_type] += alive_mask * infos['my_log_data']['step_rew'][rew_type]
                    #self.rews["_track_" + rew_type].extend(
                    #    (alive_mask * self.rews[rew_type][:,0]).reshape(-1).tolist()
                    #)
                    rew = infos['my_log_data']['step_rew'][rew_type]
                    #print(rew_type, rew.size(), self.rews[rew_type].size())
                    self.rews[rew_type].add_(alive_mask * rew[:,None])
                    #self.rews[rew_type][finished_episodes] = 0

                self._cumulative_timesteps[alive_mask] += 1
                
                mask_update = ~torch.logical_or(terminated, truncated) # one that didn't term
                finished_episodes = torch.logical_and(~mask_update, alive_mask)[:,0]
                # term this step and alive overall

                alive_mask *= mask_update

            else:
                self._cumulative_rewards.add_(rewards)
                self._cumulative_timesteps.add_(1)

                for rew_type in self.rew_types:
                    rew = infos['my_log_data']['step_rew'][rew_type]
                    #print(rew_type, rew.size(), self.rews[rew_type].size())
                    self.rews[rew_type].add_(rew[:,None])

                finished_episodes = (terminated + truncated).nonzero(as_tuple=False)

            if finished_episodes.numel():
                # storage cumulative rewards and timesteps
                self._track_rewards.extend(self._cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
                for rew_type in self.rew_types:
                    #print(f'rew_type logging: {self.rews[rew_type][finished_episodes][:,0].reshape(-1).tolist()[0]}')
                    self.rews["_track_" + rew_type].extend(self.rews[rew_type][finished_episodes][:,0].reshape(-1).tolist())
                    self.rews[rew_type][finished_episodes] = 0

                self._track_timesteps.extend(self._cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())

                # reset the cumulative rewards and timesteps
                self._cumulative_rewards[finished_episodes] = 0
                self._cumulative_timesteps[finished_episodes] = 0

                for name in infos['my_log_data']["count_step"]:
                    if 'engage' in name: 
                        # engage is special because it doesn't have to be true at the end of rollout to count
                        self.totals[name] += torch.sum(self.curr_engaged[finished_episodes].float())
                        self.curr_engaged[finished_episodes] = False
                    else:
                        self.totals[name] += torch.sum(infos['my_log_data']['count_step'][name][finished_episodes].float())
                    infos['my_log_data']['count_step'][name][finished_episodes] = False

                    
        return alive_mask
    
    def reset_tracking(self):
        if self._cumulative_rewards is None:
            return
        self._cumulative_rewards *= 0
        self._cumulative_timesteps *= 0
        for rew_type in self.rew_types:
            self.rews[rew_type] *= 0.0
            self.rews["_track_"+rew_type].clear()
        
        for name in self.totals:
            self.totals[name] *= 0
            self.once[name][:] = False
        self.curr_engaged[:] = False
        self._track_rewards.clear()
        self._track_timesteps.clear() 

    def set_running_mode(self, mode):
        super().set_running_mode(mode)
        self.reset_tracking() 

    #def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
    #    acts, log_probs, outputs = super().act(states, timestep, timesteps)
    #    acts[self.finished_envs,:] *= 0.0
    #    return acts, log_probs, outputs
    

    def _update(self, timestep: int, timesteps: int):
        super()._update(timestep, timesteps)
        # reset optimizer step
        self.resetAdamOptimizerTime(self.optimizer)
        
        if self.cfg['track_layernorms']:
            self.track_data(
                "Layer Weight Norm / Output", 
                torch.linalg.norm(self.value.critic.output[-2].weight, dim=None, ord=None).item()
            )
            self.track_data(
                "Layer Weight Norm / Input", 
                torch.linalg.norm(self.value.critic.input[0].weight, dim=None, ord=None).item()
            )

            for layer_idx, layer in enumerate(self.value.critic.layers):
                self.track_data(
                    f"Layer Weight Norm / Layer {layer_idx}-1", 
                    torch.linalg.norm(layer.path[1].weight, dim=None, ord=None).item()
                )
                self.track_data(
                    f"Layer Weight Norm / Layer {layer_idx}-2", 
                    torch.linalg.norm(layer.path[3].weight, dim=None, ord=None).item()
                )
        

    def resetAdamOptimizerTime(self, opt):
        for p,v in opt.state_dict()['state'].items():
            v["step"]=torch.Tensor([0])            
    #def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
    """Process the environment's states to make a decision (actions) using the main policy

    :param states: Environment's states
    :type states: torch.Tensor
    :param timestep: Current timestep
    :type timestep: int
    :param timesteps: Number of timesteps
    :type timesteps: int

    :return: Actions
    :rtype: torch.Tensor
    """
    """
    # sample random actions
    # TODO, check for stochasticity
    if timestep < self._random_timesteps:
        return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

    # sample stochastic actions
    with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
        actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")
        self._current_log_prob = log_prob

    return actions, log_prob, outputs"
    """

