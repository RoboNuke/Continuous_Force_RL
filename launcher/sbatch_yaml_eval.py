#!/usr/bin/env python3
"""
YAML-based WandB Eval Launcher
Reads a training sets YAML and launches eval jobs based on group mappings.

Usage:
    python launcher/sbatch_yaml_eval.py <yaml_file> [--video/--no-video] [--job_per_run]

Example:
    python launcher/sbatch_yaml_eval.py configs/experiments/req_trainning_sets/set1_group1.yaml --video
    python launcher/sbatch_yaml_eval.py configs/experiments/req_trainning_sets/set1_group1.yaml --job_per_run
"""

import argparse
import subprocess
import sys
from pathlib import Path

import wandb
import yaml

# ============================================================
# CONFIGURATION: Map group names to eval modes
# Modify this dictionary to change which eval modes run for each group
# ============================================================
GROUP_EVAL_MAPPING = {
    "base": ["performance", "noise", "dynamics"],
    "fragile": ["performance"],
    "full_noise": ["performance", "noise"],
}
# ============================================================


def get_run_ids_by_tag(tag: str) -> list[str]:
    """Query WandB for run IDs matching the given tag."""
    api = wandb.Api(timeout=60)
    runs = api.runs("hur/Peg_in_Hole", filters={"tags": {"$in": [tag]}})
    return [r.id for r in runs]


def extract_leaf_tags(data: dict | str) -> list[str]:
    """
    Recursively traverse nested dicts to extract all leaf string values (tags).

    Args:
        data: Either a dict (continue recursing) or a string (leaf tag value)

    Returns:
        List of all leaf string values found
    """
    tags = []

    if isinstance(data, str):
        tags.append(data)
    elif isinstance(data, dict):
        for value in data.values():
            tags.extend(extract_leaf_tags(value))
    else:
        raise ValueError(f"Unexpected data type in YAML: {type(data)}. Expected dict or str.")

    return tags


def main():
    parser = argparse.ArgumentParser(
        description="Launch WandB eval jobs from a training sets YAML file"
    )
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to the training sets YAML file"
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Enable video recording for evals"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        default=False,
        help="Disable video recording for evals (default)"
    )
    parser.add_argument(
        "--job_per_run",
        action="store_true",
        default=False,
        help="Launch one job per run ID instead of per tag"
    )

    args = parser.parse_args()

    # Resolve video flag (--no-video takes precedence if both specified)
    video_enabled = args.video and not args.no_video

    # Validate YAML file exists
    yaml_path = Path(args.yaml_file)
    if not yaml_path.exists():
        print(f"Error: YAML file not found: {yaml_path}")
        sys.exit(1)

    # Load YAML file
    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file: {e}")
        sys.exit(1)

    if not isinstance(config, dict):
        print(f"Error: YAML file must contain a dictionary at the top level")
        sys.exit(1)

    print("=== YAML Eval Launcher ===")
    print(f"Config: {yaml_path}")
    print(f"Video: {video_enabled}")
    print(f"Job per run: {args.job_per_run}")
    print()

    # Get the directory where this script lives (to find sbatch_wandb_eval.bash)
    script_dir = Path(__file__).parent
    wandb_eval_script = script_dir / "sbatch_wandb_eval.bash"
    wandb_eval_by_run_script = script_dir / "sbatch_wandb_eval_by_run.bash"

    if not wandb_eval_script.exists():
        print(f"Error: sbatch_wandb_eval.bash not found at {wandb_eval_script}")
        sys.exit(1)

    if args.job_per_run and not wandb_eval_by_run_script.exists():
        print(f"Error: sbatch_wandb_eval_by_run.bash not found at {wandb_eval_by_run_script}")
        sys.exit(1)

    # Process each top-level group
    for group_name, group_data in config.items():
        print(f"Processing group: {group_name}")

        # Check if group has a mapping
        if group_name not in GROUP_EVAL_MAPPING:
            print(f"  Warning: No eval mapping defined for group '{group_name}', skipping")
            print()
            continue

        eval_modes = GROUP_EVAL_MAPPING[group_name]
        eval_modes_str = " ".join(eval_modes)

        # Extract all leaf tags from this group
        tags = extract_leaf_tags(group_data)

        if not tags:
            print(f"  Warning: No tags found in group '{group_name}', skipping")
            print()
            continue

        tags_str = " ".join(tags)

        print(f"  Eval modes: {eval_modes_str}")
        print(f"  Tags ({len(tags)}): {tags_str[:80]}{'...' if len(tags_str) > 80 else ''}")

        if args.job_per_run:
            # Launch one job per (tag, eval_mode, run_id) combination
            for tag in tags:
                run_ids = get_run_ids_by_tag(tag)
                if not run_ids:
                    print(f"    Warning: No runs found for tag '{tag}'")
                    continue

                print(f"    Tag '{tag}' -> {len(run_ids)} runs: {run_ids}")

                for eval_mode in eval_modes:
                    for run_id in run_ids:
                        cmd = [
                            str(wandb_eval_by_run_script),
                            "--run_ids", run_id,
                            "--tag", tag,
                            "--eval_mode", eval_mode,
                        ]
                        if video_enabled:
                            cmd.append("--video")
                        else:
                            cmd.append("--no-video")

                        print(f"    Running: {wandb_eval_by_run_script.name} --run_ids {run_id} --tag {tag} --eval_mode {eval_mode}")

                        result = subprocess.run(cmd, capture_output=False)

                        if result.returncode != 0:
                            print(f"    Error: sbatch_wandb_eval_by_run.bash failed with return code {result.returncode}")
                            sys.exit(1)
        else:
            # Original behavior: launch sbatch_wandb_eval.bash with all tags
            cmd = [
                str(wandb_eval_script),
                "--tags", tags_str,
                "--eval_modes", eval_modes_str,
            ]

            if video_enabled:
                cmd.append("--video")
            else:
                cmd.append("--no-video")

            print(f"  Running: {wandb_eval_script.name} --tags \"...\" --eval_modes \"{eval_modes_str}\"")

            result = subprocess.run(cmd, capture_output=False)

            if result.returncode != 0:
                print(f"  Error: sbatch_wandb_eval.bash failed with return code {result.returncode}")
                sys.exit(1)

        print()

    print("=== All groups processed ===")


if __name__ == "__main__":
    main()
