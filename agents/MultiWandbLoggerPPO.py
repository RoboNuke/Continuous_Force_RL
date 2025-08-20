from typing import Any, Mapping, Optional, Tuple, Union
import copy
import os
from skrl.models.torch import Model
from skrl import config, logger
from skrl.memories.torch import Memory

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from agents.wandb_logger_ppo_agent import WandbLoggerPPO
from typing import Any, Mapping, Optional, Tuple, Union

#import gym
import gymnasium as gym
import skrl
import torch
import torch.nn as nn
import numpy as np
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl import config, logger

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from data_processing.data_manager import DataManager

import wandb

import collections
import subprocess
from filelock import FileLock

import itertools
from packaging import version

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.resources.schedulers.torch import KLAdaptiveLR
import statistics

class EpisodeTracker:
    def __init__(
            self,
            cfg,
            log_cfg,
            device,
            clip_eps= 0.2 # should get from cfg
    ):
        self.num_envs = log_cfg['num_envs']
        self.device = device
        self.cfg = log_cfg
        self.reset_all()
        self.run = wandb.init(
            entity=cfg['entity'],
            project=cfg['project'],
            name=cfg['run_name'],
            reinit="create_new",
            config=log_cfg,
            group=cfg['group'],
            tags=cfg['tags']
        )
        self.wandb_cfg = cfg
        self.wandb_cfg['run_id'] = self.run.id
        self.metrics = {}
        if self.cfg['ctrl_type'] == 'pose_ctrl':
            self.force_size = 0
        else:
            self.force_size = 3 + (3 if self.cfg['hybrid_agent']['ctrl_torque'] else 0)
        self.clip_eps = self.cfg['ratio_clip']

    def publish(self):
        # get episode data into logging dict
        episode_dict = self.pop_finished()
        self._finalize_metrics(episode_dict)
        
        agged_lms = self.aggregate_learning_metrics()
        self._finalize_metrics(agged_lms, mean_only=True) # we are averaging higher order stats accross minibatches

        # log everything at once
        self.run.log(self.metrics)
        self.metrics = {}
        self.learning_dict = {}


    def _finalize_metrics(self, mets, mean_only=False):
        for key, val in mets.items():
            tag = key.lower()
            if mean_only:
                self.metrics[key] = val.mean().item()
            elif "median" in tag:
                self.metrics[key] = val.median().item()
            elif "std" in tag:
                self.metrics[key] = val.std().item()
            elif "total" in tag:
                self.metrics[key] = torch.sum(val).item()
            else:
                self.metrics[key] = val.mean().item()
        
    def add_metric(self, tag, val):
        self.metrics[tag] = val
        
    def reset_all(self):
        self.env_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.env_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.engaged_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.succeeded_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # engagement tracking
        self.current_engaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.engagement_start = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.engagement_lengths = defaultdict(list)  # env_id -> list of engagement durations

        # success tracking
        self.steps_to_first_success = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        self.comp_sums = {}
        self.finished_metrics = []
        self.learning_metrics = []

    def reset_envs(self, done_mask: torch.Tensor):
        if done_mask.any():
            metrics = self._gather_metrics(done_mask)
            self.finished_metrics.append(metrics)

        # reset selected envs
        idxs = torch.nonzero(done_mask, as_tuple=False).flatten()
        self.env_returns[idxs] = 0.0
        self.env_steps[idxs] = 0
        self.engaged_any[idxs] = False
        self.succeeded_any[idxs] = False
        self.current_engaged[idxs] = False
        self.engagement_start[idxs] = -1
        for i in idxs.tolist():
            self.engagement_lengths[i] = []
        self.steps_to_first_success[idxs] = -1
        for k in self.comp_sums.keys():
            self.comp_sums[k][idxs] = 0.0

    def step(
            self,
            reward,
            reward_components: dict,
            engaged: torch.Tensor,
            success: torch.Tensor,
            done: torch.Tensor
    ):
        reward = reward.detach().squeeze()
        engaged = engaged.detach().bool()
        success = success.detach().bool()
        done = done.detach()
        
        # update returns & steps
        self.env_returns += reward
        self.env_steps += 1

        # component rewards
        for k, v in reward_components.items():
            v = v.detach()
            #print(v)
            if k not in self.comp_sums:
                self.comp_sums[k] = torch.zeros_like(v, dtype=torch.float32, device=self.device)
            self.comp_sums[k] += v
            

        # engagement transitions
        just_started = engaged & ~self.current_engaged
        just_ended = ~engaged & self.current_engaged

        self.engagement_start[just_started] = self.env_steps[just_started]
        ended_idxs = torch.nonzero(just_ended, as_tuple=False).flatten()
        for i in ended_idxs.tolist():
            start = self.engagement_start[i].item()
            if start >= 0:
                length = self.env_steps[i].item() - start
                self.engagement_lengths[i].append(length)

        self.current_engaged = engaged
        self.engaged_any |= engaged

        # success tracking
        new_success = success & ~self.succeeded_any
        self.steps_to_first_success[new_success] = self.env_steps[new_success]
        self.succeeded_any |= success

        # handle completed episodes
        if done.any():
            self.reset_envs(done)

    def _gather_metrics(self, mask: torch.Tensor):
        idxs = torch.nonzero(mask, as_tuple=False).flatten()
        metrics = {
            "Episode / Return (Avg)": self.env_returns[idxs], #.mean().item(),
            "Episode / Return (Median)": self.env_returns[idxs], #mean().item(),
            "Episode / Episode Length": self.env_steps[idxs].float(),
            "Engagement / Engaged Rate": self.engaged_any[idxs].float(),
            "Success / Success Rate": self.succeeded_any[idxs].float(),
        }

        # component rewards (averaged per step)
        for k, total in self.comp_sums.items():
            avg = total[idxs] / torch.clamp(self.env_steps[idxs], min=1)
            tag = k
            if "kp_" in k:
                # most pythonic line I have ever written
                tag = "".join([word.capitalize() for word in k.replace("kp_", "").split()]).replace("/", " / ") + " Keypoint Reward"
                
            metrics[tag] = avg

        # engagement stats: average length + count per episode
        avg_lengths, counts = [], []
        for i in idxs.tolist():
            if self.engagement_lengths[i]:
                avg_lengths.append(sum(self.engagement_lengths[i]) / len(self.engagement_lengths[i]))
                counts.append(len(self.engagement_lengths[i]))
            else:
                avg_lengths.append(None)
                counts.append(0)

        if any(x is not None for x in avg_lengths):
            metrics["Engagement / Engagement Length (Avg)"] = torch.tensor([x for x in avg_lengths if x is not None], device=self.device, dtype=torch.float32)
        if any(c > 0 for c in counts):
            metrics["Engagement / Total Engagements"] = torch.tensor([c for c in counts if c > 0], device=self.device, dtype=torch.float32)

        # steps to first success
        success_mask = self.succeeded_any[idxs]
        if success_mask.any():
            metrics["Success / Steps to Success (Avg)"] = self.steps_to_first_success[idxs][success_mask].float()

        return metrics

    def pop_finished(self):
        if not self.finished_metrics:
            return {}
        merged = {}
        for m in self.finished_metrics:
            for k, v in m.items():
                merged.setdefault(k, [])
                merged[k].append(v)
        for k,v in merged.items():
            merged[k] = torch.cat(v, 0)
        self.finished_metrics = []
        return merged

    def log_minibatch_update(
            self,
            returns,
            values,
            advantages,
            old_log_probs,
            new_log_probs,
            entropies,
            policy_losses,
            value_losses,
            policy_model = None,
            critic_model = None
    ):
        """
        Compute PPO diagnostic metrics. All args are tensors collected over minibatch/update.
        Returns a dictionary of scalars.
        """
        stats = {}
        with torch.no_grad():
            # --- Policy stats ---
            ratio = (new_log_probs - old_log_probs).exp()
            clip_mask = (ratio < 1-self.clip_eps) | (ratio > 1+self.clip_eps)
            kl = old_log_probs - new_log_probs
            stats["Policy / KL-Divergence (Avg)"] = kl.mean().item()
            stats["Policy / KL-Divergence (0.95 Quantile)"] = kl.quantile(0.95).item()
            stats["Policy / Clip Fraction"] = clip_mask.float().mean().item()

            
            stats["Policy / Entropy (Avg)"] = entropies.mean().item()
            stats["Policy / Loss (Avg)"] = policy_losses.mean().item()

            # --- Value stats ---
            stats["Critic / Loss (Avg)"] = value_losses.mean().item()
            stats["Critic / Loss (0.95 Quantile)"] = value_losses.quantile(0.95).item()
            stats["Critic / Predicted Values (Avg)"] = values.mean().item()
                        

            # --- Advantage diagnostics ---
            stats["Advantage / Mean"] = advantages.mean().item()
            stats["Advantage / Std Dev"] = advantages.std().item()
            stats["Advantage / Skew"] = (
                (((advantages - advantages.mean()) ** 3).mean() / (advantages.std() ** 3 + 1e-8))
                .item()
            )
            
            stats["Critic / Explained Variance"] = (
                1 - ((returns - values).var(unbiased=False) / returns.var(unbiased=False).clamp(min=1e-8))
            ).item()

            # Gradient norms (if models given)
            def grad_norm(model):
                norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
                return torch.norm(torch.stack(norms), 2) if norms else torch.tensor(0.)

            policy_grad_norm = grad_norm(policy_model) if policy_model is not None else torch.tensor(0.)
            critic_grad_norm = grad_norm(critic_model) if critic_model is not None else torch.tensor(0.)
            
            stats['Policy / Gradient Norm'] = policy_grad_norm
            stats['Critic / Gradient Norm'] = critic_grad_norm

            """
            # Optimizer step size estimate (Adam effective step)
            def adam_step_size(optimizer, grad_norm):
                step_sizes = []
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is None: 
                            continue
                        state = optimizer.state[p]
                        if 'exp_avg_sq' not in state: 
                            continue
                        denom = (state['exp_avg_sq'].sqrt() / group['lr']).add_(group['eps'])
                        step_sizes.append((p.grad / denom).norm(2))
                return torch.norm(torch.stack(step_sizes), 2) if step_sizes else torch.tensor(0.)

            policy_step_size = adam_step_size(policy_optimizer, policy_grad_norm) if policy_optimizer else torch.tensor(0.)
            value_step_size = adam_step_size(value_optimizer, value_grad_norm) if value_optimizer else torch.tensor(0.)

            stats['policy_step_size'] = policy_step_size
            stats['value_step_size'] = value_step_size
            """
        self.learning_metrics.append(stats)

    
    def aggregate_learning_metrics(self):
        """Aggregate minibatches → dict of means."""
        #if not self.learning_metrics:
        #    return {}
        out = {}
        keys = self.learning_metrics[0].keys()
        for k in keys:
            vals = torch.tensor([m[k] for m in self.learning_metrics], device=self.device, dtype=torch.float32)
            #if isinstance(vals[0], float):
            #out[k] = sum(vals) / len(vals)
            #else:  # numpy arrays
            #    out[k] = sum(vals) / len(vals)
            out[k] = vals
        self.learning_metrics = []
        return out

    def one_time_learning_metrics(
            self,
            actions,
            returns= None,
            global_step=-1
    ):
        with torch.no_grad():
            # --- Action stats ---
            groups = [ [0, 'Position', (self.force_size, self.force_size+3), (-1, 1)],
                       [0, 'Rotation', (self.force_size+3, self.force_size+6), (-1, 1)],
                       [3, 'Selection', (0, self.force_size), (0,1)],
                       [3, 'Force', (self.force_size+6, self.force_size+9), (-1, 1)],
                       [6, 'Torque', (self.force_size+9, self.force_size+12), (-1,1)] ]
            names = ['X','Y','Z','RX','RY','RZ']
            for group in groups:
                if self.force_size >= group[0]:
                    a = group[2][0]
                    b = group[2][1]
                    for i, idx in enumerate(range( a, b )):
                        tag = f"Action / {names[i]} {group[1]} "
                        acts = actions[...,idx]
                        self.metrics[tag + ' (Avg)'] = acts.mean().item()
                        self.metrics[tag + ' (Std)'] = acts.std().item()
                        self.metrics[tag + ' Saturation'] = torch.logical_or(
                            acts <= group[3][0] + 1e-2,
                            acts >= group[3][1] - 1e-2
                        ).float().mean().item()
            if global_step > 0:
                self.metrics['Total Steps'] = global_step
                self.metrics['Env Steps'] = global_step // self.num_envs
            # --- Returns ---
            #self.metrics["return_mean"] = returns.mean().item()
            #self.metrics["return_median"] = returns.median().item()

class MultiWandbLoggerPPO(WandbLoggerPPO):
    def __init__(
            self,
            models: Mapping[str, Model],
            memory: Optional[Union[Memory, Tuple[Memory]]] = None,
            observation_space: Optional[Union[int, Tuple[int], gym.Space, gym.Space]] = None,
            action_space: Optional[Union[int, Tuple[int], gym.Space, gym.Space]] = None,
            num_agents: int = 1,
            num_envs: int = 256,
            device: Optional[Union[str, torch.device]] = None,
            state_size=-1,
            cfg: Optional[dict] = None,
            track_ckpt_paths = False,
            task="Isaac-Factory-PegInsert-Local-v0"
    ) -> None:
        super().__init__(models, memory, observation_space, action_space, num_envs, device, state_size, cfg, track_ckpt_paths, task)
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
        
        self.track_input_histogram = cfg['track_input']
        self.track_action_histogram = cfg['track_action_hists']
        self.loggers = []
        

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        #DONE
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
            for i in range(self.num_agents):
                os.makedirs(os.path.join(
                    self.cfg[f'agent_{i}']['experiment']['directory'],
                    self.cfg[f'agent_{i}']['experiment']['experiment_name'],
                    "checkpoints"
                ), exist_ok=True)

        self.set_mode("eval")
        
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

        # setup Weights & Biases
        self.log_wandb = False
        if self.cfg.get("experiment", {}).get("wandb", False):
            # save experiment configuration
            for i in range(self.num_agents):
                try:
                    models_cfg = {k: v._modules for (k, v) in self.models.items()}
                except AttributeError:
                    print("[INFO]: Extracting model config had Attribute Error...")
                    models_cfg = {k: v[i]._modules for (k, v) in self.models.items()}
                    print("[INFO]: Attribute Error Handled")
                self.cfg['experiment'] = self.cfg[f'agent_{i}']['experiment']
                wandb_config={**self.cfg, **trainer_cfg, **models_cfg, "num_envs":self.num_envs // self.num_agents}
                # set default values
                exp_dir = os.path.join(self.cfg['experiment']['directory'], self.cfg['experiment']['experiment_name'])
                print("logger exp dir:", exp_dir)
                wandb_kwargs = copy.deepcopy(self.cfg[f'agent_{i}'].get("experiment", {}).get("wandb_kwargs", {}))
                wandb_kwargs.setdefault("name", exp_dir)
                wandb_kwargs.setdefault("config", {})
                wandb_kwargs["config"].update(wandb_config)
                
                # init loggers
                self.loggers.append(EpisodeTracker(wandb_kwargs, wandb_config,self.device))
                self.log_wandb = True
        print("[INFO]: Multilogging WandB Initialized")

    def write_checkpoint(self, timestep: int, timesteps: int):
        #DONE
        # save the network parameters
        actor_model = self.models['policy']
        if actor_model.num_agents > 1:
            for i, (net, log_std) in enumerate(zip(actor_model.actor_mean.pride_nets, actor_model.actor_logstd)):
                exp_args = self.cfg[f'agent_{i}']['experiment']
                ckpt_path = os.path.join(exp_args['directory'], exp_args['experiment_name'], "checkpoints", f"agent_{i}_{timestep*self.num_envs // self.num_agents}.pt")
                torch.save({
                    'net_state_dict': net.state_dict(),
                    'log_std': log_std
                }, ckpt_path)
                print(f"Saved network {i} to {ckpt_path}")
        else:
            exp_args = self.cfg['experiment']
            ckpt_path = os.path.join(exp_args['directory'], exp_args['experiment_name'], "checkpoints", f"agent_{timestep*self.num_envs // self.num_agents}.pt")
            torch.save({
                'net_state_dict': actor_model.actor_mean.state_dict(),
                'log_std': actor_model.actor_logstd
            }, ckpt_path)
            print(f"Saved single network to {ckpt_path}")
            
        vid_path = os.path.join(exp_args['directory'], exp_args['experiment_name'], "eval_videos", f"agent_{i}_{timestep* self.num_envs // self.num_agents}.gif")
        #self.track_data("ckpt_video", (timestep, vid_path) )
        if self.track_ckpt_paths:
            lock = FileLock(self.tracker_path + ".lock")
            with lock:
                with open(self.tracker_path, "a") as f:
                    f.write(f'{ckpt_path} {self.task_name} {vid_path} {self.loggers[i].wandb_cfg["project"]} {self.loggers[i].wandb_cfg["run_id"]}\n')
                        
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
        eval_mode = alive_mask is not None
        
        if not eval_mode and self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
            values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) boostrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated
            
            
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

        ##### TODO ADD SMOOTHNESS HERE OR ABOVE ######
        if eval_mode:
            mask_update = torch.logical_or(terminated, truncated).view((self.num_envs,)) # one that didn't term
            alive_mask[mask_update] = False
            finished_episodes = torch.logical_and(mask_update, alive_mask).view(-1,)
        else:
            finished_episodes = torch.logical_or(terminated, truncated).view(-1,)

        if torch.any(finished_episodes):
            if 'smoothness' in infos.keys():
                for key, rew in infos['smoothness'].items():
                    pass
        
        if self.track_input_histogram:
            # pos input
            for i, dir in enumerate(['x','y','z']):
                self.track_hist(f"obs_pos_{dir}", states[:,i])
                self.track_hist(f"obs_force_{dir}", states[:,-6+i])

        if self.track_action_histogram and actions.size()[1] >= 12:
            for i, dir in enumerate(['x','y','z']):
                self.track_hist(f"act_sel_{dir}",actions[:,i])
                self.track_hist(f"act_pos_{dir}", actions[:,i+3])
                self.track_hist(f"act_force_{dir}", actions[:,i+9])
                    
        for i, logger in enumerate(self.loggers):
            a = i * self.envs_per_agent
            b = (i+1) * self.envs_per_agent
            rew = rewards[a:b]
            comps = {}
            for key,v in infos.items():
                if 'Reward /' in key:
                    comps[key] = v[a:b]
            engaged = infos['current_engagements'][a:b]
            success = infos['current_successes'][a:b]
            dones = finished_episodes[a:b]
            logger.step(rew, comps, engaged, success, dones)
        
        return alive_mask
            
    def write_tracking_data(self, timestep: int, timesteps: int, eval=False) -> None:
        #DONE?
        for logger in self.loggers:
            logger.publish()

    def track_data(self, tag: str, values: torch.tensor) -> None:
        #DONE
        if type(values) == float:
            self.loggers[0].add_metric(tag, values)
            return
        
        batch_size = values.size()[0] // self.num_agents
        for i, logger in enumerate(self.loggers):
            logger.add_metric(tag, values[i*batch_size,(i+1)*batch_size]).detach().mean().item()

    def track_hist(self, key: str, value: torch.Tensor) -> None:
        #DONE
        batch_size = value.size()[0] // self.num_agents
        prefix = "Training" #TODO##############################
        for i, logger in enumerate(self.loggers):
            hist_data = value[i*batch_size:(i+1)*batch_size].detach().cpu().numpy()
            if "act_sel_" in key:
                dim = key.replace("act_sel_", "")
                hist = np.histogram(hist_data, range=(0.0, 1.0), bins=20)
                logger.add_metric(f"{prefix} Hybrid Controller / Force Selection {dim} (hist)", wandb.Histogram(np_histogram=hist))
            elif "act_pos_" in key:
                dim = key.replace("act_pos_","")
                hist = np.histogram(hist_data,range=(-1,1), bins=40)
                logger.add_metric(f"{prefix} Hybrid Controller / Pos Action {dim} (hist)", wandb.Histogram(np_histogram=hist))
            elif "act_force_" in key:
                dim = key.replace("act_force_","")
                hist = np.histogram(hist_data,range=(-1,1), bins=40)
                logger.add_metric(f"{prefix} Hybrid Controller / Force Action {dim} (hist)", wandb.Histogram(np_histogram=hist))
            elif "obs_pos_" in key:
                dim = key.replace("obs_pos_","")
                hist = np.histogram(hist_data, range=(-1,1), bins=40)
                logger.add_metric(f"Observation / {prefix} {dim}-Position (hist)", wandb.Histogram(np_histogram=hist))
            elif "obs_force_" in key:
                dim = key.replace("obs_force_","")
                hist = np.histogram(hist_data, range=(-1,1), bins=40)
                logger.add_metric(f"Observation / {prefix} {dim}-Force (hist)", wandb.Histogram(np_histogram=hist))
            elif "adv" in key:
                hist =  np.histogram(hist_data, range=(-5.0, 5.0), bins=200)
                logger.add_metric(f"Learning Performance / Advantage Distribution (hist)", wandb.Histogram(np_histogram=hist))

    def post_interaction(self, timestep: int, timesteps: int):
        # DONE
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            #print("Return:", np.mean(np.array(self._track_rewards)))
            self._update(timestep, timesteps)
            
            self.set_mode("eval")

        timestep += 1
        self.global_step+= self.num_envs

        # update best models and write checkpoints
        if timestep > 1 and self.checkpoint_interval > 0 and not timestep % self.checkpoint_interval:
            # write checkpoints
            self.write_checkpoint(timestep, timesteps)

        # write wandb
        if timestep > 1 and self.write_interval > 0 and not timestep % self.write_interval:
            self.write_tracking_data(timestep, timesteps)

    def _log_minibatch_update(
            self,
            returns,
            values,
            advantages,
            old_log_probs,
            new_log_probs,
            entropies,
            policy_losses,
            value_losses,
            policy_model = None,
            critic_model = None
    ):
        batch_size = returns.size()[0]
        agent_batch = batch_size // self.num_agents
        for i, logger in enumerate(self.loggers):
            a = i * agent_batch
            b = a + agent_batch
            logger.log_minibatch_update(
                returns[a:b],
                values[a:b],
                advantages[a:b],
                old_log_probs[a:b],
                new_log_probs[a:b],
                entropies[a:b],
                policy_losses[a:b],
                value_losses[a:b]
            )

            
    def _one_time_metrics(self,
        actions,
        returns
    ):
        agent_batch = actions.size()[0] // self.num_agents
        for i, logger in enumerate(self.loggers):
            a = i * agent_batch
            b = a + agent_batch
            logger.one_time_learning_metrics(
                actions[a:b],
                returns[a:b],
                self.global_step // self.num_agents
            )
                
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
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
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
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)
        
        # learning epochs
        for epoch in range(self._learning_epochs):
            mini_batch = 0
            # mini-batches loop
            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:

                sample_size = sampled_states.size()[0]
                keep_mask = torch.ones((sample_size,), dtype=bool, device=self.device)
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        agent_kls = ((torch.exp(ratio) - 1) - ratio)
                        
                        for i in range(self.num_agents):
                            a = i * sample_size // self.num_agents
                            b = a + sample_size // self.num_agents
                            agent_kl = agent_kls[a:b].mean() 

                            # zero out gradients for agents with high approximate kls
                            if self._kl_threshold and agent_kl > self._kl_threshold:
                                keep_mask[a:b] = False
                            
                            #print(f"\tAgent {i} Approx KL:", agent_kl)

                    entropys = self.policy.get_entropy(role="policy")
                    

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * entropys[keep_mask].mean()
                    else:
                        entropy_loss = 0
                    

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_losses = -torch.min(surrogate, surrogate_clipped)
                    
                    policy_loss = policy_losses[keep_mask].mean()

                    # compute value loss
                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                        
                    value_losses = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values, reduction='none')
                    value_loss = value_losses[keep_mask].mean()

                # optimization step
                # zero out losses from cancelled
                
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

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
                self._log_minibatch_update(
                    sampled_returns,
                    predicted_values,
                    sampled_advantages,
                    sampled_log_prob,
                    next_log_prob,
                    entropys,
                    policy_losses,
                    value_losses
                )


            self._one_time_metrics(
                self.memory.get_tensor_by_name("actions"),
                self.memory.get_tensor_by_name("returns")
            )
                
            # TODO: ########################################################################
            # update learning rate
            """if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()
            """







        """
        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
        """





