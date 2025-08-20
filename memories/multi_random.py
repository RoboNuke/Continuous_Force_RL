

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
        self, names: Tuple[str], mini_batches: int = 1, sequence_length: int = 1
    ) -> List[List[torch.Tensor]]:
        """Sample all data from memory

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``) unused, only kept for inheritance reasons
        :type sequence_length: int, optional

        :return: Sampled data from memory.
                 Data in each batch will be ordered based on number of agents
                 The sampled tensors will have the following shape: (memory size * number of environments, data size)
        :rtype: list of torch.Tensor list
        """
        agent_batch_size = self.memory_size * self.envs_per_agent // mini_batches
        #print("Agent Batch Size:", agent_batch_size)
        
        # generate idxs
        agent_data_idxs = [ [] for i in range(self.num_agents)]
        for a_idx in range(self.num_agents):
            a = a_idx * self.envs_per_agent
            b = a + self.envs_per_agent
            for m_idx in range(self.memory_size):
                agent_data_idxs[a_idx].append(torch.arange(a + m_idx * self.num_envs, b + m_idx * self.num_envs))
            agent_data_idxs[a_idx] = torch.cat(agent_data_idxs[a_idx], dim=0)
        #print("Agent Idxs:", agent_data_idxs)
            
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
    envs_per_agent = 3
    num_agents = 4
    rollout_len = 2
    tot_envs = envs_per_agent * num_agents
    
    memory = MultiRandomMemory(#MultiRandomMemory(
        memory_size=rollout_len,
        num_envs=envs_per_agent * num_agents,
        num_agents = num_agents,
        replacement=True
    )
    
    names = ['obs', 'act', 'rew', 'next_obs', 'done']
    for name in names:
        #print(name, 2 if name in ['obs','next_obs'] else 1)
        memory.create_tensor(name=name, size = 2 if name in ['obs','next_obs'] else 1, dtype=torch.float32)

    # Fake trajectory data
    for i in range(rollout_len):
        obs = torch.tensor([[i*k, (i + 0.1)*k] for k in range(tot_envs)])
        act = torch.tensor([[i * 0.5 * k] for k in range(tot_envs)])
        rew = torch.tensor([[i * tot_envs + k] for k in range(tot_envs)])
        next_obs = torch.tensor([[i + 1*k, i + 1.1*k] for k in range(tot_envs)])
        done = torch.tensor([(i == (num_agents - k // envs_per_agent)) for k in range(tot_envs)]).unsqueeze(-1)
        memory.add_samples(
            obs=obs,
            act=act,
            rew=rew,
            next_obs=next_obs,
            done=done
        )
    #print(memory.tensors['rew'])
    # Call sample_all
    batch = memory.sample_all(names=names, mini_batches= 3)
    #print("=== Sample All Test ===")
    #print(batch)
    #print(len(batch), len(batch[0]), batch[0][0].size())

if __name__ == "__main__":
    test_sample_all()
