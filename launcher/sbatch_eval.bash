#!/bin/bash

# Eval Launch Script for HPC
# Usage: ./launcher/sbatch_eval.bash --configs "config1 config2" --experiment_tag "EXP_TAG" [--video/--no-video]
# Or:    ./launcher/sbatch_eval.bash --folder "folder_name" --experiment_tag "EXP_TAG" [--video/--no-video]

# Default values
CONFIGS=""
FOLDER=""
EXPERIMENT_TAG=""
VIDEO_ENABLED="true"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --configs)
            CONFIGS="$2"
            shift 2
            ;;
        --folder)
            FOLDER="$2"
            shift 2
            ;;
        --experiment_tag)
            EXPERIMENT_TAG="$2"
            shift 2
            ;;
        --video)
            VIDEO_ENABLED="true"
            shift 1
            ;;
        --no-video)
            VIDEO_ENABLED="false"
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --configs \"config1 config2\" --experiment_tag EXP_TAG [--video/--no-video]"
            echo "   Or: $0 --folder \"folder_name\" --experiment_tag EXP_TAG [--video/--no-video]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -n "$CONFIGS" && -n "$FOLDER" ]]; then
    echo "Error: --configs and --folder are mutually exclusive. Use one or the other."
    echo "Usage: $0 --configs \"config1 config2\" --experiment_tag EXP_TAG [--video/--no-video]"
    echo "   Or: $0 --folder \"folder_name\" --experiment_tag EXP_TAG [--video/--no-video]"
    exit 1
fi

if [[ -z "$CONFIGS" && -z "$FOLDER" ]]; then
    echo "Error: Either --configs or --folder is required"
    echo "Usage: $0 --configs \"config1 config2\" --experiment_tag EXP_TAG [--video/--no-video]"
    echo "   Or: $0 --folder \"folder_name\" --experiment_tag EXP_TAG [--video/--no-video]"
    exit 1
fi

if [[ -z "$EXPERIMENT_TAG" ]]; then
    echo "Error: --experiment_tag is required"
    echo "Usage: $0 --configs \"config1 config2\" --experiment_tag EXP_TAG [--video/--no-video]"
    echo "   Or: $0 --folder \"folder_name\" --experiment_tag EXP_TAG [--video/--no-video]"
    exit 1
fi

echo "=== Eval Launch Script ==="
echo "Configs: $CONFIGS"
echo "Folder: $FOLDER"
echo "Experiment Tag: $EXPERIMENT_TAG"
echo "Video Enabled: $VIDEO_ENABLED"
echo ""

# Handle folder mode or individual config mode
if [[ -n "$FOLDER" ]]; then
    # Folder mode: discover all .yaml files in the specified folder
    folder_path="configs/experiments/${FOLDER}"

    # Check if folder exists
    if [[ ! -d "$folder_path" ]]; then
        echo "Error: Folder not found: $folder_path"
        exit 1
    fi

    # Find all .yaml files in the folder
    CONFIG_ARRAY=()
    while IFS= read -r -d '' file; do
        # Extract just the filename without extension
        config_name=$(basename "$file" .yaml)
        CONFIG_ARRAY+=("${FOLDER}/${config_name}")
    done < <(find "$folder_path" -name "*.yaml" -type f -print0)

    # Check if any configs were found
    if [[ ${#CONFIG_ARRAY[@]} -eq 0 ]]; then
        echo "Error: No .yaml files found in folder: $folder_path"
        exit 1
    fi

    echo "Found ${#CONFIG_ARRAY[@]} config(s) in folder '$FOLDER': ${CONFIG_ARRAY[*]}"
    echo ""
else
    # Individual config mode: convert configs string to array
    read -ra CONFIG_ARRAY <<< "$CONFIGS"
fi

# Python helper script to extract tracker path from YAML
read -r -d '' PYTHON_EXTRACT_TRACKER << 'EOF'
import yaml
import sys

config_path = sys.argv[1]
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    tracker_path = config.get('primary', {}).get('ckpt_tracker_path', '')

    if not tracker_path:
        print(f"ERROR: primary.ckpt_tracker_path not found in {config_path}", file=sys.stderr)
        sys.exit(1)

    print(tracker_path)
except Exception as e:
    print(f"ERROR: Failed to parse config {config_path}: {e}", file=sys.stderr)
    sys.exit(1)
EOF

# Process each config
for config_name in "${CONFIG_ARRAY[@]}"; do
    # Construct full config path
    config_path="configs/experiments/${config_name}.yaml"

    # Check if config file exists
    if [[ ! -f "$config_path" ]]; then
        echo "Error: Config file not found: $config_path"
        continue
    fi

    # Extract tracker path from config using Python
    tracker_path=$(python3 -c "$PYTHON_EXTRACT_TRACKER" "$config_path" 2>&1)
    extract_status=$?

    if [[ $extract_status -ne 0 ]]; then
        echo "Error: $tracker_path"
        continue
    fi

    # Create job name (replace slashes with underscores for valid job names)
    job_name="EVAL_${EXPERIMENT_TAG}_$(echo "${config_name}" | tr '/' '_')"

    echo "Submitting eval job: $job_name"
    echo "  Config: $config_path"
    echo "  Tracker: $tracker_path"
    echo "  Experiment Tag: $EXPERIMENT_TAG"
    echo "  Video: $VIDEO_ENABLED"

    # Submit SLURM job with dynamic output file paths (replace slashes with underscores for file names)
    output_name="eval_$(echo "${config_name}" | tr '/' '_')"
    sbatch -J "$job_name" \
           -o "exp_logs/${EXPERIMENT_TAG}/${output_name}_%j.out" \
           -e "exp_logs/${EXPERIMENT_TAG}/${output_name}_%j.err" \
           launcher/hpc_eval_batch.bash "$config_path" "$EXPERIMENT_TAG" "$tracker_path" "$VIDEO_ENABLED"

    echo "  Job submitted successfully"
    echo ""
done

echo "All eval jobs submitted!"
