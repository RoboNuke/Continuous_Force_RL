"""
Checkpoint Loading Utilities

Functions for querying WandB runs, downloading checkpoints, and loading them into agents.
These are extracted from wandb_eval.py to allow importing without triggering argparse.
"""

import os
import time
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

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


def reconstruct_config_from_wandb(run: wandb.Run) -> Tuple[Dict[str, Any], str]:
    """Reconstruct configuration from WandB run using config files.

    Downloads config YAML files and uses ConfigManagerV3 to load them.

    Args:
        run: WandB run object

    Returns:
        Tuple of (configs, temp_dir) where:
        - configs: Dictionary with config instances compatible with setup_environment_once
        - temp_dir: Path to temporary directory containing downloaded config files.
                   Caller is responsible for cleanup after configs are no longer needed.

    Raises:
        RuntimeError: If config files missing or loading fails
    """
    print(f"Reconstructing config from WandB run: {run.project}/{run.id}")

    # Use ConfigManagerV3's new config_from_wandb method
    config_manager = ConfigManagerV3()
    configs, temp_dir = config_manager.config_from_wandb(run)

    print(f"  Successfully reconstructed {len(configs)} config sections")
    return configs, temp_dir


def get_available_checkpoint_steps(run: wandb.Run) -> List[int]:
    """Query WandB for all available checkpoint steps in a run.

    Lists files matching 'ckpts/policies/*.pt' and extracts step numbers
    from filenames.

    Args:
        run: WandB run object

    Returns:
        Sorted list of available checkpoint step numbers (ascending order)

    Raises:
        RuntimeError: If query fails or no checkpoints found
    """
    print(f"  Querying available checkpoints for run {run.id}...")

    try:
        # List all files in the run
        files = run.files()

        # Filter for policy checkpoint files and extract step numbers
        checkpoint_steps = []
        for f in files:
            # Match pattern: ckpts/policies/{step}.pt
            if f.name.startswith('ckpts/policies/') and f.name.endswith('.pt'):
                # Extract step number from filename
                filename = os.path.basename(f.name)
                step_str = filename[:-3]  # Remove '.pt'
                try:
                    step = int(step_str)
                    checkpoint_steps.append(step)
                except ValueError:
                    # Skip files that don't have numeric step names
                    continue

        if not checkpoint_steps:
            raise RuntimeError(f"No checkpoints found in run {run.id}")

        checkpoint_steps.sort()
        print(f"    Found {len(checkpoint_steps)} checkpoints: {checkpoint_steps[0]} to {checkpoint_steps[-1]}")
        return checkpoint_steps

    except Exception as e:
        if "No checkpoints found" in str(e):
            raise
        raise RuntimeError(f"Failed to query checkpoints for run {run.id}: {e}")


def get_latest_common_checkpoint_step(runs: List[wandb.Run]) -> int:
    """Find the highest checkpoint step that exists in ALL runs.

    Queries each run for available checkpoints, finds the intersection,
    and returns the maximum step from that intersection.

    Args:
        runs: List of WandB run objects

    Returns:
        The highest checkpoint step available in all runs

    Raises:
        RuntimeError: If no common checkpoint exists across all runs
    """
    if not runs:
        raise RuntimeError("No runs provided to get_latest_common_checkpoint_step")

    print(f"Finding latest common checkpoint across {len(runs)} runs...")

    # Get available steps for each run
    all_steps = []
    for run in runs:
        steps = get_available_checkpoint_steps(run)
        all_steps.append(set(steps))

    # Find intersection of all step sets
    common_steps = all_steps[0]
    for steps in all_steps[1:]:
        common_steps = common_steps.intersection(steps)

    if not common_steps:
        raise RuntimeError(
            f"No common checkpoint step found across {len(runs)} runs. "
            f"Each run may have different checkpoint steps."
        )

    latest_step = max(common_steps)
    print(f"  Latest common checkpoint step: {latest_step}")
    return latest_step


def sort_runs_by_agent_index(runs: List[wandb.Run]) -> List[wandb.Run]:
    """Sort WandB runs by agent index extracted from run name.

    Run names are expected to follow the format:
    {exp_name}_f({break_force})_{agent_idx}

    This ensures runs are loaded into the correct agent slots.

    Args:
        runs: List of WandB run objects

    Returns:
        List of runs sorted by agent index (ascending)

    Raises:
        RuntimeError: If agent index cannot be extracted from any run name
    """
    import re

    def extract_agent_index(run: wandb.Run) -> int:
        """Extract agent index from run name."""
        # Pattern: anything_f(number)_agentidx
        # The agent index is the last number after the last underscore
        match = re.search(r'_(\d+)$', run.name)
        if match:
            return int(match.group(1))

        # Fallback: try to find pattern _f(...)_N
        match = re.search(r'_f\([^)]+\)_(\d+)', run.name)
        if match:
            return int(match.group(1))

        raise RuntimeError(
            f"Could not extract agent index from run name: '{run.name}'. "
            f"Expected format: '{{exp_name}}_f({{break_force}})_{{agent_idx}}'"
        )

    # Extract indices and sort
    try:
        runs_with_indices = [(run, extract_agent_index(run)) for run in runs]
        runs_with_indices.sort(key=lambda x: x[1])

        sorted_runs = [run for run, _ in runs_with_indices]

        print(f"  Sorted {len(runs)} runs by agent index:")
        for run, idx in runs_with_indices:
            print(f"    Agent {idx}: {run.name} ({run.id})")

        return sorted_runs

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to sort runs by agent index: {e}")


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
    for_training: bool = False,
) -> List[str]:
    """
    Download and load checkpoints from all runs into their respective agent slots.

    Args:
        runs: List of WandB run objects
        step: Checkpoint step number
        env: Environment instance
        agent: Agent instance with num_agents = len(runs)
        for_training: If True, leave agent in training mode after loading.
                     If False (default), set agent to eval mode for evaluation.

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

    # Set agent mode based on intended use
    if for_training:
        print(f"  All {len(runs)} checkpoints loaded successfully! (training mode)")
    else:
        agent.set_running_mode("eval")
        print(f"  All {len(runs)} checkpoints loaded successfully! (eval mode)")

    return download_dirs


def get_best_checkpoints_for_runs(
    api: wandb.Api,
    runs: List[wandb.Run],
    method_tag: str,
    entity: str,
    project: str
) -> Tuple[Dict[str, int], Dict[str, dict]]:
    """
    Determine the best checkpoint for each run based on eval_performance data.

    Uses the same logic as analysis_utils.get_best_checkpoint_per_run():
    score = num_successful_completions - num_breaks

    Args:
        api: WandB API instance
        runs: List of training runs to find best checkpoints for
        method_tag: Tag identifying the method/experiment (e.g., "MATCH:2024-01-15_10:00")
        entity: WandB entity name
        project: WandB project name

    Returns:
        Tuple of (best_checkpoints, best_scores) where:
        - best_checkpoints: Dict mapping source_run_id -> best_checkpoint_step
        - best_scores: Dict mapping source_run_id -> {'score': int, 'successes': int, 'breaks': int}

    Raises:
        RuntimeError: If no eval_performance data found for any run
    """
    print(f"\n{'=' * 80}")
    print("BEST CHECKPOINT DISCOVERY")
    print(f"{'=' * 80}")
    print(f"Querying eval_performance runs with tags: '{method_tag}' AND 'eval_performance'")

    # Query for eval_performance runs with both tags
    eval_perf_runs = api.runs(
        f"{entity}/{project}",
        filters={"$and": [{"tags": method_tag}, {"tags": "eval_performance"}]}
    )
    eval_perf_runs = list(eval_perf_runs)

    if len(eval_perf_runs) == 0:
        raise RuntimeError(
            f"No eval_performance runs found with tags '{method_tag}' AND 'eval_performance'. "
            f"You must run eval_mode=performance first before using --eval_best_policies."
        )

    print(f"Found {len(eval_perf_runs)} eval_performance runs")

    # Build mapping from eval run to source training run
    # Eval runs have tag "source_run:{source_run_id}"
    eval_run_to_source = {}
    for eval_run in eval_perf_runs:
        for tag in eval_run.tags:
            if tag.startswith("source_run:"):
                source_id = tag.split(":", 1)[1]
                eval_run_to_source[eval_run.id] = source_id
                break

    # Build set of training run IDs we need best checkpoints for
    training_run_ids = {run.id for run in runs}

    # Find best checkpoint for each training run
    best_checkpoints = {}
    best_scores = {}
    missing_runs = []

    for training_run in runs:
        # Find the eval_performance run for this training run
        matching_eval_run = None
        for eval_run in eval_perf_runs:
            if eval_run_to_source.get(eval_run.id) == training_run.id:
                matching_eval_run = eval_run
                break

        if matching_eval_run is None:
            missing_runs.append(training_run)
            continue

        # Get history and find best checkpoint
        history = matching_eval_run.history()
        if history.empty:
            missing_runs.append(training_run)
            continue

        # Check required columns exist
        success_col = "Eval_Core/num_successful_completions"
        breaks_col = "Eval_Core/num_breaks"
        steps_col = "total_steps"

        if success_col not in history.columns or breaks_col not in history.columns:
            raise RuntimeError(
                f"eval_performance run {matching_eval_run.id} missing required columns. "
                f"Expected '{success_col}' and '{breaks_col}'. "
                f"Available columns: {list(history.columns)}"
            )

        if steps_col not in history.columns:
            raise RuntimeError(
                f"eval_performance run {matching_eval_run.id} missing '{steps_col}' column."
            )

        # Calculate score: successes - breaks
        history["_score"] = history[success_col] - history[breaks_col]
        best_idx = history["_score"].idxmax()
        best_step = int(history.loc[best_idx, steps_col])
        best_score = history.loc[best_idx, "_score"]

        best_successes = int(history.loc[best_idx, success_col])
        best_breaks = int(history.loc[best_idx, breaks_col])

        best_checkpoints[training_run.id] = best_step
        best_scores[training_run.id] = {
            'score': int(best_score),
            'successes': best_successes,
            'breaks': best_breaks,
        }
        print(f"  {training_run.name}: best checkpoint at step {best_step} (score: {best_score:.0f})")

    # Error if any runs are missing eval_performance data
    if missing_runs:
        missing_names = [r.name for r in missing_runs]
        raise RuntimeError(
            f"No eval_performance data found for {len(missing_runs)} run(s): {missing_names}. "
            f"All runs must have eval_performance data before using --eval_best_policies."
        )

    print(f"\nFound best checkpoints for all {len(best_checkpoints)} runs")
    return best_checkpoints, best_scores
