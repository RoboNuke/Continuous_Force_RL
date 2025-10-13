#!/usr/bin/env python3
"""
Checkpoint Tracker Filter Script

Extracts specific run IDs from a checkpoint tracker file to a new file,
and removes them from the original tracker.
"""

import argparse
import os
import sys


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Filter checkpoint tracker by run IDs")

    parser.add_argument(
        "--runs_to_filter",
        type=str,
        required=True,
        help="Comma-separated list of run IDs to extract (e.g., 'run123,run456,run789')"
    )
    parser.add_argument(
        "--tracker_to_filter",
        type=str,
        required=True,
        help="Path to the checkpoint tracker file to filter"
    )
    parser.add_argument(
        "--new_tracker_path",
        type=str,
        default=None,
        help="Output path for extracted run IDs (default: {original_path}_filtered.txt)"
    )

    return parser.parse_args()


def main():
    """Main function to filter checkpoint tracker."""
    args = parse_arguments()

    # Parse run IDs to filter
    runs_to_filter = set(run_id.strip() for run_id in args.runs_to_filter.split(',') if run_id.strip())

    if len(runs_to_filter) == 0:
        print("Error: No run IDs provided to filter")
        sys.exit(1)

    print(f"Extracting {len(runs_to_filter)} run ID(s): {', '.join(sorted(runs_to_filter))}")

    # Check if tracker file exists
    if not os.path.exists(args.tracker_to_filter):
        print(f"Error: Tracker file not found: {args.tracker_to_filter}")
        sys.exit(1)

    # Determine output path
    if args.new_tracker_path is None:
        # Create default output path: {directory}/{basename}_filtered.txt
        directory = os.path.dirname(args.tracker_to_filter)
        basename = os.path.basename(args.tracker_to_filter)

        # Remove extension and add _filtered
        name_without_ext = os.path.splitext(basename)[0]
        new_tracker_path = os.path.join(directory, f"{name_without_ext}_filtered.txt")
    else:
        new_tracker_path = args.new_tracker_path

    print(f"Input tracker: {args.tracker_to_filter}")
    print(f"Output tracker: {new_tracker_path}")

    # Read and filter tracker file
    remaining_lines = []  # Lines to keep in original tracker
    extracted_lines = []  # Lines to extract to new file
    total_lines = 0

    with open(args.tracker_to_filter, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            total_lines += 1

            # Parse line: <ckpt_path> <task> <vid_path> <project> <run_id>
            parts = line.split()

            if len(parts) != 5:
                print(f"Warning: Skipping malformed line (expected 5 fields, got {len(parts)}): {line}")
                continue

            ckpt_path, task, vid_path, project, run_id = parts

            # Check if run_id should be extracted
            if run_id in runs_to_filter:
                extracted_lines.append(line)
                print(f"  Extracting: {run_id} (checkpoint: {os.path.basename(ckpt_path)})")
            else:
                remaining_lines.append(line)

    # Write extracted lines to new tracker
    with open(new_tracker_path, 'w') as f:
        for line in extracted_lines:
            f.write(line + '\n')

    # Overwrite original tracker with remaining lines
    with open(args.tracker_to_filter, 'w') as f:
        for line in remaining_lines:
            f.write(line + '\n')

    # Print summary
    print("\n" + "=" * 80)
    print("Filtering Summary:")
    print(f"  Total lines read: {total_lines}")
    print(f"  Lines extracted: {len(extracted_lines)}")
    print(f"  Lines remaining in original: {len(remaining_lines)}")
    print(f"  Extracted lines written to: {new_tracker_path}")
    print(f"  Original tracker updated: {args.tracker_to_filter}")
    print("=" * 80)


if __name__ == "__main__":
    main()
