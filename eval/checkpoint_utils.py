"""
Checkpoint Loading Utilities

Functions for querying WandB runs, downloading checkpoints, and loading them into agents.
These are extracted from wandb_eval.py to allow importing without triggering argparse.
"""

import os
import time
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
import wandb

from configs.config_manager_v3 import ConfigManagerV3


def query_runs_by_tag(tag: str, entity: str, project: str, run_id: Optional[str] = None) -> List[wandb.Run]:
    """
    Query WandB for runs with specified tag.

    Args:
        tag: Experiment tag (format: group_name:YYYY-MM-DD_HH:MM)
        entity: WandB entity (username or team name)
        project: WandB project name
        run_id: Optional specific run ID to filter by

    Returns:
        List of WandB runs

    Raises:
        RuntimeError: If no runs found or API query fails
    """
    print(f"Querying WandB for runs with tag: {tag}")
    print(f"  Entity: {entity}")
    print(f"  Project: {project}")

    api = wandb.Api(timeout=60)
    max_retries = 5
    retry_delay = 2.0

    for attempt in range(max_retries):
        try:
            # Query all runs with the tag in the specified project
            project_path = f"{entity}/{project}"
            runs = api.runs(project_path, filters={"tags": {"$in": [tag]}})

            # Convert to list to actually execute query
            runs_list = list(runs)

            if len(runs_list) == 0:
                raise RuntimeError(f"No runs found with tag: {tag}")

            # Filter out evaluation runs (those with "Eval_" in the name)
            original_count = len(runs_list)
            runs_list = [r for r in runs_list if "Eval_" not in r.name]
            if original_count != len(runs_list):
                print(f"  Filtered out {original_count - len(runs_list)} evaluation run(s)")

            if len(runs_list) == 0:
                raise RuntimeError(f"No training runs found with tag: {tag} (all runs were evaluation runs)")

            # Filter by run_id if specified
            if run_id is not None:
                runs_list = [r for r in runs_list if r.id == run_id]
                if len(runs_list) == 0:
                    raise RuntimeError(f"No runs found with tag '{tag}' and run_id '{run_id}'")

            print(f"  Found {len(runs_list)} run(s) with tag '{tag}'")
            for r in runs_list:
                print(f"    - {r.project}/{r.id} ({r.name})")

            return runs_list

        except wandb.errors.CommError as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < max_retries - 1:
                    print(f"    Rate limit hit, waiting {retry_delay:.1f}s before retry {attempt+1}/{max_retries-1}...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise RuntimeError(
                        f"Failed to query runs after {max_retries} attempts due to rate limiting. "
                        f"Please wait a few minutes before running again."
                    )
            else:
                raise RuntimeError(f"Failed to query WandB runs: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to query WandB runs: {e}")

    raise RuntimeError(f"Failed to query WandB runs after all retries")


def reconstruct_config_from_wandb(run: wandb.Run) -> Dict[str, Any]:
    """Reconstruct configuration from WandB run using config files.

    Downloads config YAML files and uses ConfigManagerV3 to load them.

    Args:
        run: WandB run object

    Returns:
        Dictionary with config instances compatible with setup_environment_once

    Raises:
        RuntimeError: If config files missing or loading fails
    """
    print(f"Reconstructing config from WandB run: {run.project}/{run.id}")

    # Use ConfigManagerV3's new config_from_wandb method
    config_manager = ConfigManagerV3()
    configs = config_manager.config_from_wandb(run)

    print(f"  Successfully reconstructed {len(configs)} config sections")
    return configs


def download_checkpoint_pair(run: wandb.Run, step: int) -> Tuple[str, str]:
    """
    Download policy and critic checkpoints from WandB.

    Args:
        run: WandB run object
        step: Checkpoint step number

    Returns:
        Tuple of (policy_path, critic_path)

    Raises:
        RuntimeError: If download fails
    """
    print(f"  Downloading checkpoint pair for step {step}...")

    # Construct file paths on WandB
    policy_filename = f"ckpts/policies/{step}.pt"
    critic_filename = f"ckpts/critics/{step}.pt"

    # Create temporary download directory
    download_dir = tempfile.mkdtemp(prefix="wandb_ckpt_")

    try:
        # Download policy checkpoint
        policy_file = run.file(policy_filename)
        policy_path = policy_file.download(root=download_dir, replace=True).name
        print(f"    Downloaded policy: {policy_path}")

        # Download critic checkpoint
        critic_file = run.file(critic_filename)
        critic_path = critic_file.download(root=download_dir, replace=True).name
        print(f"    Downloaded critic: {critic_path}")

        return policy_path, critic_path

    except Exception as e:
        # Clean up temp directory on failure
        shutil.rmtree(download_dir, ignore_errors=True)
        raise RuntimeError(
            f"Failed to download checkpoint files from WandB run {run.project}/{run.id} at step {step}: {e}"
        )


def load_checkpoints_parallel(
    runs: List[wandb.Run],
    step: int,
    env: Any,
    agent: Any,
) -> List[str]:
    """
    Download and load checkpoints from all runs into their respective agent slots.

    Args:
        runs: List of WandB run objects
        step: Checkpoint step number
        env: Environment instance
        agent: Agent instance with num_agents = len(runs)

    Returns:
        List of download directories (for cleanup after evaluation)

    Raises:
        RuntimeError: If any checkpoint fails to load (fail fast)
    """
    from models.SimBa import SimBaNet
    from models.block_simba import pack_agents_into_block
    from agents.block_ppo import PerAgentPreprocessorWrapper
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    print(f"  Loading {len(runs)} checkpoints in parallel for step {step}...")

    # Download all checkpoints first
    checkpoint_pairs = []
    download_dirs = []
    for run_idx, run in enumerate(runs):
        print(f"    Downloading checkpoint for run {run_idx}: {run.id}")
        policy_path, critic_path = download_checkpoint_pair(run, step)
        checkpoint_pairs.append((policy_path, critic_path))
        # Track download directory for cleanup
        download_dirs.append(os.path.dirname(os.path.dirname(policy_path)))

    # Initialize per-agent preprocessor lists if not already present
    if not hasattr(agent, '_per_agent_state_preprocessors'):
        agent._per_agent_state_preprocessors = [None] * agent.num_agents
    if not hasattr(agent, '_per_agent_value_preprocessors'):
        agent._per_agent_value_preprocessors = [None] * agent.num_agents

    # Load each checkpoint into its agent slot
    for agent_idx, (run, (policy_path, critic_path)) in enumerate(zip(runs, checkpoint_pairs)):
        print(f"    Loading run {run.id} into agent slot {agent_idx}...")

        # Load checkpoint files
        policy_checkpoint = torch.load(policy_path, map_location=env.unwrapped.device, weights_only=False)
        critic_checkpoint = torch.load(critic_path, map_location=env.unwrapped.device, weights_only=False)

        # Validate checkpoint contents
        if 'net_state_dict' not in policy_checkpoint:
            raise RuntimeError(f"Policy checkpoint for run {run.id} missing 'net_state_dict'")
        if 'net_state_dict' not in critic_checkpoint:
            raise RuntimeError(f"Critic checkpoint for run {run.id} missing 'net_state_dict'")
        if 'state_preprocessor' not in policy_checkpoint:
            raise RuntimeError(f"Policy checkpoint for run {run.id} missing 'state_preprocessor'")
        if 'value_preprocessor' not in critic_checkpoint:
            raise RuntimeError(f"Critic checkpoint for run {run.id} missing 'value_preprocessor'")
        use_state_dependent_std = policy_checkpoint.get('use_state_dependent_std', False)
        if not use_state_dependent_std and 'log_std' not in policy_checkpoint:
            raise RuntimeError(f"Policy checkpoint for run {run.id} missing 'log_std'")

        # Create single-agent SimBaNet for policy
        std_out_dim = getattr(agent.policy.actor_mean, 'std_out_dim', 0)
        policy_agent = SimBaNet(
            n=len(agent.policy.actor_mean.resblocks),
            in_size=agent.policy.actor_mean.obs_dim,
            out_size=agent.policy.actor_mean.act_dim + std_out_dim,
            latent_size=agent.policy.actor_mean.hidden_dim,
            device=agent.device,
            tan_out=agent.policy.actor_mean.use_tanh
        )
        policy_agent.load_state_dict(policy_checkpoint['net_state_dict'])

        # Create single-agent SimBaNet for critic
        critic_agent = SimBaNet(
            n=len(agent.value.critic.resblocks),
            in_size=agent.value.critic.obs_dim,
            out_size=agent.value.critic.act_dim,
            latent_size=agent.value.critic.hidden_dim,
            device=agent.device,
            tan_out=agent.value.critic.use_tanh
        )
        critic_agent.load_state_dict(critic_checkpoint['net_state_dict'])

        # Pack into block models at this agent's index
        pack_agents_into_block(agent.policy.actor_mean, {agent_idx: policy_agent})
        pack_agents_into_block(agent.value.critic, {agent_idx: critic_agent})

        # Load log_std (only for non-state-dependent std)
        if not use_state_dependent_std:
            agent.policy.actor_logstd[agent_idx].data.copy_(policy_checkpoint['log_std'].data)

        # Load state preprocessor for this agent
        obs_size = policy_checkpoint['state_preprocessor']['running_mean'].shape[0]
        if agent._per_agent_state_preprocessors[agent_idx] is None:
            agent._per_agent_state_preprocessors[agent_idx] = RunningStandardScaler(
                size=obs_size, device=agent.device
            )
        agent._per_agent_state_preprocessors[agent_idx].load_state_dict(
            policy_checkpoint['state_preprocessor']
        )

        # Load value preprocessor for this agent
        if agent._per_agent_value_preprocessors[agent_idx] is None:
            agent._per_agent_value_preprocessors[agent_idx] = RunningStandardScaler(
                size=1, device=agent.device
            )
        agent._per_agent_value_preprocessors[agent_idx].load_state_dict(
            critic_checkpoint['value_preprocessor']
        )

        print(f"      Loaded run {run.id} -> agent {agent_idx}")

    # Wrap preprocessors for SKRL compatibility
    agent._state_preprocessor = PerAgentPreprocessorWrapper(agent, agent._per_agent_state_preprocessors)
    agent._value_preprocessor = PerAgentPreprocessorWrapper(agent, agent._per_agent_value_preprocessors)

    # Set agent to eval mode
    agent.set_running_mode("eval")
    print(f"  All {len(runs)} checkpoints loaded successfully!")

    return download_dirs
