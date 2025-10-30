

from typing import List, Optional, Tuple, Union

import torch

from memories.multi_random import MultiRandomMemory


class ImportanceSamplingMemory(MultiRandomMemory):
    """Memory that performs importance sampling based on a boolean 'upsample' tensor.

    Inherits multi-agent temporal shuffling from MultiRandomMemory and adds:
    - Upsampling of samples marked True in the 'upsample' tensor
    - Importance weight adjustment to log_prob values
    - Dynamic mini-batch count to ensure all False samples used at least once
    """

    def __init__(
        self,
        memory_size: int,
        num_envs: int = 1,
        num_agents: int = 1,
        target_true_ratio: float = 0.5,
        min_true_percentage: float = 0.05,
        device: Optional[Union[str, torch.device]] = None,
        export: bool = False,
        export_format: str = "pt",
        export_directory: str = "",
        replacement=True,
    ) -> None:
        """Initialize ImportanceSamplingMemory.

        :param memory_size: Number of timesteps to store
        :param num_envs: Total number of environments
        :param num_agents: Number of agents (envs divided equally among agents)
        :param target_true_ratio: Desired ratio of True samples in each batch (default: 0.5)
        :param min_true_percentage: Minimum percentage of True samples needed to enable upsampling (default: 0.1)
        :param device: Device to store tensors
        :param export: Whether to export memory
        :param export_format: Format for export
        :param export_directory: Directory for export
        :param replacement: Whether to sample with replacement (kept for compatibility)
        """
        super().__init__(
            memory_size=memory_size,
            num_envs=num_envs,
            num_agents=num_agents,
            device=device,
            export=export,
            export_format=export_format,
            export_directory=export_directory,
            replacement=replacement,
        )

        self.target_true_ratio = target_true_ratio
        self.min_true_percentage = min_true_percentage

    def sample_all(
        self, names: Tuple[str], mini_batches: int = 1, sequence_length: int = 1, shuffle: bool = True
    ) -> List[List[torch.Tensor]]:
        """Sample all data from memory with importance sampling.

        The mini_batches parameter defines the TARGET batch size. The actual number of
        mini-batches returned may be larger to accommodate upsampled data.

        :param names: Tensor names from which to obtain samples
        :param mini_batches: Number used to calculate target batch size (actual returned may be more)
        :param sequence_length: Length of each sequence (unused, kept for inheritance)
        :param shuffle: Whether to shuffle temporal order

        :return: List of mini-batches, each containing list of tensors ordered by agent.
                 May return more mini-batches than requested to accommodate upsampling.
        :rtype: list of list of torch.Tensor
        """

        # Check that upsample tensor exists
        if 'upsample' not in self.tensors:
            raise ValueError(
                "'upsample' tensor must be created before sampling with ImportanceSamplingMemory. "
                "Use memory.create_tensor(name='upsample', size=1, dtype=torch.bool)"
            )

        # Get the upsample boolean tensor directly from storage
        # Tensor is stored as [memory_size, num_envs, feature_size], so flatten it
        upsample_tensor = self.tensors['upsample']  # Shape: (memory_size, num_envs, 1)
        upsample_flat = upsample_tensor.reshape(-1, upsample_tensor.shape[-1])  # Shape: (memory_size * num_envs, 1)
        upsample_bool = upsample_flat.squeeze(-1).bool()  # Shape: (memory_size * num_envs,)

        # Calculate target batch size based on normal (non-upsampled) data
        normal_total_samples = self.memory_size * self.num_envs
        target_batch_size = normal_total_samples // mini_batches

        # Generate shuffled or sequential timestep indices per environment (same as MultiRandomMemory)
        if shuffle:
            timestep_indices = torch.stack([
                torch.randperm(self.memory_size, device=self.device)
                for _ in range(self.num_envs)
            ])
        else:
            timestep_indices = torch.arange(self.memory_size, device=self.device).unsqueeze(0).expand(self.num_envs, -1)

        # Process each agent independently
        all_sampled_indices = []
        all_importance_weights = []

        for a_idx in range(self.num_agents):
            # Get this agent's environment range
            agent_env_start = a_idx * self.envs_per_agent
            agent_env_end = agent_env_start + self.envs_per_agent
            agent_shuffles = timestep_indices[agent_env_start:agent_env_end, :]

            # Compute all memory indices for this agent (same logic as MultiRandomMemory)
            env_indices_list = []
            for env_offset in range(self.envs_per_agent):
                global_env_idx = agent_env_start + env_offset
                env_timesteps = agent_shuffles[env_offset, :]  # Shape: (memory_size,)
                env_memory_indices = env_timesteps * self.num_envs + global_env_idx
                env_indices_list.append(env_memory_indices)

            # Stack and flatten to get all indices for this agent
            # Shape: (memory_size * envs_per_agent,)
            stacked = torch.stack(env_indices_list, dim=0)
            agent_indices = stacked.t().flatten()

            # Get upsample mask for this agent's indices
            agent_upsample_mask = upsample_bool[agent_indices]  # Shape: (memory_size * envs_per_agent,)

            # Count True and False samples
            n_true = agent_upsample_mask.sum().item()
            n_total = len(agent_indices)
            n_false = n_total - n_true
            true_pct = n_true / n_total if n_total > 0 else 0.0

            # Check if we have enough True samples to do importance sampling
            # Use <= so that exactly at threshold, we use fallback
            if true_pct <= self.min_true_percentage:
                # FALLBACK MODE: Not enough True samples, use all samples once
                sampled_indices = agent_indices
                # No importance weighting needed - samples used at natural frequency
                importance_weights = torch.zeros(len(sampled_indices), device=self.device)
            else:
                # UPSAMPLING MODE: We have enough True samples, do importance sampling

                # Separate into True and False indices
                true_indices = agent_indices[agent_upsample_mask]
                false_indices = agent_indices[~agent_upsample_mask]

                # Compute how many True samples we need to achieve target ratio
                # We use all False samples once, so:
                # batch_size = n_false / (1 - target_true_ratio)
                # n_true_needed = batch_size * target_true_ratio
                if n_false > 0:
                    agent_batch_size = n_false / (1.0 - self.target_true_ratio)
                    n_true_needed = int(agent_batch_size * self.target_true_ratio)
                else:
                    # Edge case: no false samples, just use all true samples
                    n_true_needed = n_true

                # Sample False indices (all of them, once each)
                sampled_false = false_indices  # Already shuffled from timestep_indices

                # Sample True indices WITH replacement
                if n_true_needed > 0 and len(true_indices) > 0:
                    # Sample with replacement from true_indices
                    true_sample_idxs = torch.randint(
                        0, len(true_indices), (n_true_needed,), device=self.device
                    )
                    sampled_true = true_indices[true_sample_idxs]
                else:
                    sampled_true = torch.tensor([], dtype=torch.long, device=self.device)

                # Concatenate: [all false samples, sampled true samples]
                sampled_indices = torch.cat([sampled_false, sampled_true], dim=0)

                # Compute importance sampling weights (in log space)
                # IS weight = (actual probability) / (sampling probability)
                actual_prob_true = true_pct
                actual_prob_false = 1.0 - true_pct
                sampling_prob_true = self.target_true_ratio
                sampling_prob_false = 1.0 - self.target_true_ratio

                # Avoid division by zero
                if sampling_prob_true > 0 and sampling_prob_false > 0:
                    log_weight_true = torch.log(
                        torch.tensor(actual_prob_true / sampling_prob_true, device=self.device)
                    )
                    log_weight_false = torch.log(
                        torch.tensor(actual_prob_false / sampling_prob_false, device=self.device)
                    )
                else:
                    # Edge case: degenerate sampling probabilities
                    log_weight_true = torch.tensor(0.0, device=self.device)
                    log_weight_false = torch.tensor(0.0, device=self.device)

                # Create weight tensor: false samples get log_weight_false, true samples get log_weight_true
                importance_weights = torch.cat([
                    log_weight_false.expand(len(sampled_false)),
                    log_weight_true.expand(len(sampled_true))
                ], dim=0)

            all_sampled_indices.append(sampled_indices)
            all_importance_weights.append(importance_weights)

        # Pool all samples from all agents
        pooled_indices = torch.cat(all_sampled_indices, dim=0)
        pooled_weights = torch.cat(all_importance_weights, dim=0)

        # Calculate actual number of mini-batches needed
        total_samples = len(pooled_indices)
        actual_mini_batches = (total_samples + target_batch_size - 1) // target_batch_size  # Ceiling division

        # Split into mini-batches
        batches = []
        for b_idx in range(actual_mini_batches):
            start = b_idx * target_batch_size
            end = min(start + target_batch_size, total_samples)

            batch_indices = pooled_indices[start:end]
            batch_weights = pooled_weights[start:end]

            # Sample the actual data using indices
            batch_data = self.sample_by_index(names, batch_indices)[0]

            # Apply importance weights to log_prob if it exists in the batch
            if 'log_prob' in names:
                log_prob_idx = names.index('log_prob')
                # Add importance weights to log_prob (in log space)
                batch_data[log_prob_idx] = batch_data[log_prob_idx] + batch_weights.unsqueeze(-1)

            batches.append(batch_data)

        return batches
