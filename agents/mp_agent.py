import torch
import torch.multiprocessing as mp
from typing import List, Optional, Union, Any

def fn_processor(process_index, *args):
    #print(f"[INFO] Processor {process_index}: started")

    pipe = args[0][process_index]
    queue = args[1][process_index]
    barrier = args[2]
    scope = args[3][process_index]

    agent = None
    _states = None
    _actions = None
    _log_prob = None
    _outputs = None
    _rew_dist = {}
    _term_dist = {}

    # wait for the main process to start all the workers
    barrier.wait()

    while True:
        msg = pipe.recv()
        task = msg["task"]
        
        # terminate process
        if task == "terminate":
            break

        # initialize agent
        elif task == "init":
            print("about to pull agent from queue")
            agent = queue.get()
            print("got agent")
            agent.init(trainer_cfg=queue.get())
            print(f"[INFO] Processor {process_index}: init agent {type(agent).__name__} with scope {scope} on {agent.num_envs} envs")
            barrier.wait()

        # execute agent's pre-interaction step
        elif task == "pre_interaction":
            agent.pre_interaction(timestep=msg["timestep"], timesteps=msg["timesteps"])
            barrier.wait()

        # get agent's actions
        elif task == "act": #TODO deal with stochastic eval
            raw = queue.get()
            if raw is None:
                raw = queue.get()
            #_states = queue.get()[scope[0] : scope[1]]
            _states = raw[scope[0] : scope[1]]
            with torch.no_grad():
                stochastic_evaluation = agent.training #msg["stochastic_evaluation"]
                #_outputs = agent.act(_states, timestep=msg["timestep"], timesteps=msg["timesteps"])
                #_actions = _outputs[0] if not stochastic_evaluation else _outputs[-1].get("mean_actions", _outputs[0])
                _actions, _log_prob, _outputs = agent.act(_states, timestep=msg["timestep"], timesteps=msg["timesteps"])
                if not _actions.is_cuda:
                    _actions.share_memory_()
                if _log_prob is not None and not _log_prob.is_cuda:
                    _log_prob.share_memory_()
                if not _outputs['mean_actions'].is_cuda:
                    _outputs['mean_actions'].share_memory_()
                queue.put(_actions)
                queue.put(_log_prob)
                queue.put(_outputs['mean_actions'])
                barrier.wait()

        # record agent's experience
        elif task == "record_transition":
            alive_mask_is_none = queue.get()

            with torch.no_grad():
                r = queue.get()[scope[0] : scope[1]]
                ns = queue.get()[scope[0] : scope[1]]
                ter = queue.get()[scope[0] : scope[1]]
                tru = queue.get()[scope[0] : scope[1]]
                info = queue.get()
                red_info = {} # copy everything but 'smoothness
                for key in info.keys():
                    #print("key:", key)
                    if key == 'smoothness':
                        red_info['smoothness'] = {}
                        for s_key in info['smoothness'].keys():
                            red_info['smoothness'][s_key] = info['smoothness'][s_key][scope[0] : scope[1]]
                    else:
                        if type(info[key]) == dict:
                            red_info[key] = {}
                            for small_key in info[key]:
                                red_info[key][small_key] = info[key][small_key][scope[0] : scope[1]]
                        else:
                            red_info[key] = info[key][scope[0] : scope[1]]

                #print("Sent Info:", red_info['log'])
                alive_mask = None if alive_mask_is_none else queue.get()[scope[0] : scope[1]]
                alive_mask_out = agent.record_transition(
                    states=_states,
                    actions=_actions,
                    rewards=r,
                    next_states=ns,
                    terminated=ter,
                    truncated=tru,
                    infos=red_info,
                    timestep=msg["timestep"],
                    timesteps=msg["timesteps"],
                    alive_mask= alive_mask,
                )
                if not alive_mask_is_none:
                    alive_mask = alive_mask_out

                if alive_mask_is_none:
                    queue.get()
                barrier.wait()

        # execute agent's post-interaction step
        elif task == "post_interaction":
            agent.post_interaction(
                timestep=msg["timestep"], 
                timesteps=msg["timesteps"]
            )
            barrier.wait()
        
        elif task == "reset_memory":
            agent.memory.reset()
            barrier.wait()

        #elif task == "track_hists":
        #    agent.track_hists(tag = queue.get(), value=queue.get()[scope[0]:scope[1]])
        #    barrier.wait()
            
        elif task == "track_data":
            agent.track_data(tag = queue.get(), value = queue.get())
            barrier.wait()


        elif task == "track_video_path":
            agent.track_video_path(tag=queue.get(), value=queue.get())
            barrier.wait()

        elif task == "write_tracking_data":
            agent.write_tracking_data(
                timestep=msg["timestep"], 
                timesteps=msg["timesteps"],
                eval=queue.get()
            )
            agent.reset_tracking()
            barrier.wait()

        elif task == "write_checkpoint":
            agent.write_checkpoint(
                timestep=msg["timestep"], 
                timesteps=msg["timesteps"]
            )
            barrier.wait()

        elif task == "set_running_mode":
            agent.set_running_mode(queue.get())
            barrier.wait()

        elif task == "reset_tracking":
            agent.reset_tracking()
            barrier.wait()
        
        #queue.get()



class MPAgent():
    def __del__(self):
        for pipe in self.producer_pipes:
            pipe.send({'task':"terminate"})
        for process in self.processes:
            process.join()
        print("Destroyed MPAgent")

    def __init__(self, num_agents, agents_scope=None):
        self.num_agents = num_agents
        self.agents_scope = agents_scope
        self.queues = []
        self.producer_pipes = []
        self.consumer_pipes = []
        self.barrier = mp.Barrier(self.num_agents + 1)
        self.processes = []


        for i in range(self.num_agents):
            pipe_read, pipe_write = mp.Pipe(duplex=False)
            self.producer_pipes.append(pipe_write)
            self.consumer_pipes.append(pipe_read)
            self.queues.append(mp.Queue())

        # spawn and wait for all processes to start
        for i in range(self.num_agents):
            process = mp.Process(
                target=fn_processor, 
                args= (
                    i, 
                    self.consumer_pipes, 
                    self.queues, 
                    self.barrier, 
                    self.agents_scope
                ), daemon=True
            )
            self.processes.append(process)
            process.start()
            
        self.barrier.wait()
        
        print("Multiprocess Agents Created!")

    def set_agents(self, agents):
        self.agents = agents
        
        # move tensors to shared memory
        for agent in self.agents:
            if agent.memory is not None:
                agent.memory.share_memory()
            for model in agent.models.values():
                try:
                    model.share_memory()
                except RuntimeError:
                    print("Failed to share model memory")
                    pass

    def init(self, trainer_cfg):
        # initialize agents
        for pipe, queue, agent in zip(self.producer_pipes, self.queues, self.agents):
            pipe.send({"task": "init"})
            queue.put(agent)
            queue.put(trainer_cfg)
        self.barrier.wait()

    def reset_memory(self)-> None:
        self.send({"task":"reset_memory"})

    def send(self, pipe_data: dict = {}, que: list = [])->None:
        """ pipe: task_name, timestep, timesteps (whatever is required)
            que: task specific data as an ordered list
        """
        for pipe, queue in zip(self.producer_pipes, self.queues):
            pipe.send(pipe_data)
            for val in que:
                queue.put(val)
        self.barrier.wait()

    def track_video_path(self, tag: str, value: str, timestep)-> None:
        # TODO figure out how to send video paths
        self.send({"task":"track_video_path", "timestep":timestep}, [tag, value])

    def track_hist(self, tag: str, value: torch.Tensor) -> None:
        if not value.is_cuda:
            value.share_memory_()
        #print("sending mp agent")
        self.send({"task":"track_hist"}, [tag, value])
        
    def track_data(self, tag: str, value: float) -> None:
        self.send({"task":"track_data"}, [tag, value])

    def write_tracking_data(self, timestep: int, timesteps: int, eval=False) -> None:
        self.send({"task":"write_tracking_data", "timestep":timestep, "timesteps":timesteps}, [eval])

    def act(self,
            states: torch.Tensor,
            timestep: int,
            timesteps: int) -> torch.Tensor:
        if not states.is_cuda:
            states.share_memory_()
        self.send({"task":"act", "timestep":timestep, "timesteps":timesteps}, [states])
        action = torch.vstack([queue.get() for queue in self.queues])
        
        # SAC agent returns None instead of log prob, so this...
        log_prob = []
        for queue in self.queues:
            log_prob.append(queue.get())
        if log_prob[0] is not None:
            log_prob = torch.vstack(log_prob)
        
        outputs = torch.vstack([queue.get() for queue in self.queues]) # we assume we only care about mean actions, works for me :)
        return action, log_prob, {'mean_actions':outputs}

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
        if not rewards.is_cuda:
            rewards.share_memory_()

        if not next_states.is_cuda:
            next_states.share_memory_()

        if not terminated.is_cuda:
            terminated.share_memory_()

        if not truncated.is_cuda:
            truncated.share_memory_()

        if alive_mask is not None and not alive_mask.is_cuda:
            alive_mask.share_memory_()

        for big_key in infos.keys():
            if type(infos[big_key]) == dict:
                for key in infos[big_key].keys():
                    if type(infos[big_key][key]) == torch.Tensor and not infos[big_key][key].is_cuda:
                        infos[big_key][key].share_memory_()
                    elif type(infos[big_key][key]) == dict:
                        for small_key in infos[big_key][key].keys():
                            if type(infos[big_key][key][small_key]) == torch.Tensor and not infos[big_key][key][small_key].is_cuda:
                                infos[big_key][key][small_key].share_memory_()
            else:
                infos[big_key].share_memory_()

        #print("sending record trans") 
        self.send(
            {
                "task":"record_transition", "timestep":timestep, "timesteps":timesteps
            }, [
                alive_mask is None,
                rewards,
                next_states,
                terminated,
                truncated,
                infos,
                alive_mask
            ]
        )
        if not alive_mask is None:
            return alive_mask

    def set_running_mode(self, mode: str) -> None:
        self.send({"task":"set_running_mode"}, [mode])

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        self.send({"task":"pre_interaction", "timestep":timestep, "timesteps":timesteps} )

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        self.send({"task":"post_interaction", "timestep":timestep, "timesteps":timesteps})

    def reset_tracking(self) -> None:
        self.send({"task":"reset_tracking"} )

    def write_checkpoint(self, timestep: int, timesteps: int) -> None:
        self.send({"task":"write_checkpoint", "timestep":timestep, "timesteps":timesteps})
