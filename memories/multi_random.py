

from typing import List, Optional, Tuple, Union

import torch

from skrl.memories.torch import Memory, RandomMemory

class MultiRandomMemory(Memory):
    def __init__(
            self,
            memory_size: int,
            num_envs: int = 1,
            num_agents: int = 1,
            device: Optional[Union[str, torch.device]] = None,
            export: bool = False,
            export_format: str = "pt",
            export_directory: str = "",
            replacement=True,
    ) -> None:
        super().__init__(memory_size, num_envs, device, export, export_format, export_directory)

        self.num_agents = num_agents
        self.envs_per_agent = (num_envs // num_agents)


    def sample_all(
        self, names: Tuple[str], mini_batches: int = 1, sequence_length: int = 1, shuffle: bool = False #True
    ) -> List[List[torch.Tensor]]:
        """Sample all data from memory

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``) unused, only kept for inheritance reasons
        :type sequence_length: int, optional
        :param shuffle: Whether to shuffle temporal order (default: ``True``)
        :type shuffle: bool, optional

        :return: Sampled data from memory.
                 Data in each batch will be ordered based on number of agents
                 The sampled tensors will have the following shape: (memory size * number of environments, data size)
        :rtype: list of torch.Tensor list
        """
        agent_batch_size = self.memory_size * self.envs_per_agent // mini_batches

        # Generate shuffled or sequential timestep indices per environment
        if shuffle:
            # Each environment gets its own independent temporal shuffle
            # Shape: (num_envs, memory_size)
            timestep_indices = torch.stack([torch.randperm(self.memory_size, device=self.device)
                                           for _ in range(self.num_envs)])
        else:
            # All environments use sequential order
            # Shape: (num_envs, memory_size)
            timestep_indices = torch.arange(self.memory_size, device=self.device).unsqueeze(0).expand(self.num_envs, -1)

        # Generate indices for each agent with vectorized operations
        agent_data_idxs = []
        for a_idx in range(self.num_agents):
            # Get shuffles for this agent's environments
            # Shape: (envs_per_agent, memory_size)
            agent_env_start = a_idx * self.envs_per_agent
            agent_env_end = agent_env_start + self.envs_per_agent
            agent_shuffles = timestep_indices[agent_env_start:agent_env_end, :]

            # For each environment in this agent, compute indices
            # env_indices[i, j] = index for env i at its shuffled timestep j
            env_indices_list = []
            for env_offset in range(self.envs_per_agent):
                # Global environment index
                global_env_idx = agent_env_start + env_offset
                # Shuffled timesteps for this specific environment
                env_timesteps = agent_shuffles[env_offset, :]  # Shape: (memory_size,)
                # Compute memory indices: timestep * num_envs + global_env_idx
                env_memory_indices = env_timesteps * self.num_envs + global_env_idx
                env_indices_list.append(env_memory_indices)

            # Stack and interleave: we want [env0_t0, env1_t0, ..., env0_t1, env1_t1, ...]
            # Shape after stack: (envs_per_agent, memory_size)
            stacked = torch.stack(env_indices_list, dim=0)
            # Transpose to (memory_size, envs_per_agent) then flatten
            indices = stacked.t().flatten()

            agent_data_idxs.append(indices)
            
        idxs = [[] for  i in range(mini_batches)]
        for b_idx in range(mini_batches):
            for a_idx in range(self.num_agents):
                a = b_idx * agent_batch_size
                b = a + agent_batch_size
                idxs[b_idx].append(agent_data_idxs[a_idx][a:b])
            #print(idxs[b_idx])
            idxs[b_idx] = torch.cat(idxs[b_idx], dim=0)

        #print("Indexs:")
        #print(idxs)
        
        #print(self.sample_by_index(names, torch.tensor([0,5]))[0][2])
        # sample by idxs: (0 x len(names) x agent_batch_size
        #print("Sample:", self.sample_by_index(names, idxs[0]) )
        return [self.sample_by_index(names, idxs[j])[0] for j in range(mini_batches)]
        
    


def test_sample_all():
    envs_per_agent = 4
    num_agents = 3
    rollout_len = 3
    tot_envs = envs_per_agent * num_agents
    
    memory = MultiRandomMemory(#MultiRandomMemory(
        memory_size=rollout_len,
        num_envs=envs_per_agent * num_agents,
        num_agents = num_agents,
        replacement=True
    )
    
    names = ['obs', 'act'] #, 'rew', 'done']
    for name in names:
        #print(name, 2 if name in ['obs','next_obs'] else 1)
        memory.create_tensor(name=name, size = 2 if name in ['obs','next_obs'] else 1, dtype=torch.float32)

    # Fake trajectory data
    for i in range(rollout_len):
        obs = torch.tensor(
            [[i/10+k//envs_per_agent, i/10+k//envs_per_agent+0.5] for k in range(tot_envs)]
        ) 
        act = obs[:,0] + 10 
        rew = obs[:,0] + 100
        #print(rew)
        done = obs[:,0] + 1000
        print(obs)
        act=act.unsqueeze(-1)
        print(act.size())
        print(act)
        memory.add_samples(
            obs=obs,
            act=act#,
            #rew=rew,
            #done=done
        )
    #print(memory.tensors['rew'])
    # Call sample_all
    print(memory.get_tensor_by_name("obs"))
    print(memory.get_tensor_by_name("act"))
    batchs = memory.sample_all(names=names, mini_batches= 3)
    for batch in range(3):
        for a in range(num_agents):
            print(f"Batch {batch}:{a+1} obs:")
            print(batchs[batch][0][a*envs_per_agent:(a+1)*envs_per_agent])
            print(f"Batch {batch}:{a+1} act:")
            print( batchs[batch][1][a*envs_per_agent:(a+1)*envs_per_agent])
    #print("=== Sample All Test ===")
    #print(batch)
    #print(len(batch), len(batch[0]), batch[0][0].size())

if __name__ == "__main__":
    test_sample_all()
