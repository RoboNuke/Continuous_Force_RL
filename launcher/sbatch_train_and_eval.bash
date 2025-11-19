#!/bin/bash

# Combined Training + Eval Launch Script for HPC
# Usage: ./launcher/sbatch_train_and_eval.bash --configs "config1 config2" --experiment_tag "EXP_TAG" [--overrides "key=value"] [--video/--no-video]
# Or:    ./launcher/sbatch_train_and_eval.bash --folder "folder_name" --experiment_tag "EXP_TAG" [--overrides "key=value"] [--video/--no-video]

# Default values
CONFIGS=""
FOLDER=""
EXPERIMENT_TAG=""
OVERRIDES=""
VIDEO_ENABLED="false"

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
        --overrides)
            OVERRIDES="$2"
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
            echo "Usage: $0 --configs \"config1 config2\" --experiment_tag EXP_TAG [--overrides \"key=value\"] [--video/--no-video]"
            echo "   Or: $0 --folder \"folder_name\" --experiment_tag EXP_TAG [--overrides \"key=value\"] [--video/--no-video]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -n "$CONFIGS" && -n "$FOLDER" ]]; then
    echo "Error: --configs and --folder are mutually exclusive. Use one or the other."
    echo "Usage: $0 --configs \"config1 config2\" --experiment_tag EXP_TAG [--overrides \"key=value\"] [--video/--no-video]"
    echo "   Or: $0 --folder \"folder_name\" --experiment_tag EXP_TAG [--overrides \"key=value\"] [--video/--no-video]"
    exit 1
fi

if [[ -z "$CONFIGS" && -z "$FOLDER" ]]; then
    echo "Error: Either --configs or --folder is required"
    echo "Usage: $0 --configs \"config1 config2\" --experiment_tag EXP_TAG [--overrides \"key=value\"] [--video/--no-video]"
    echo "   Or: $0 --folder \"folder_name\" --experiment_tag EXP_TAG [--overrides \"key=value\"] [--video/--no-video]"
    exit 1
fi

if [[ -z "$EXPERIMENT_TAG" ]]; then
    echo "Error: --experiment_tag is required"
    echo "Usage: $0 --configs \"config1 config2\" --experiment_tag EXP_TAG [--overrides \"key=value\"] [--video/--no-video]"
    echo "   Or: $0 --folder \"folder_name\" --experiment_tag EXP_TAG [--overrides \"key=value\"] [--video/--no-video]"
    exit 1
fi

# Generate base timestamp for eval_tags
TIMESTAMP=$(date +"%Y-%m-%d_%H:%M")

echo "=== Combined Training + Eval Launch Script ==="
echo "Configs: $CONFIGS"
echo "Folder: $FOLDER"
echo "Experiment Tag: $EXPERIMENT_TAG"
echo "Overrides: $OVERRIDES"
echo "Video Enabled: $VIDEO_ENABLED"
echo "Base Timestamp: $TIMESTAMP"
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

# Arrays to store training job IDs and eval tags
TRAINING_JOB_IDS=()
EVAL_TAGS=()
OUTPUT_NAMES=()

# Submit training jobs first
echo "=== Submitting Training Jobs ==="
JOB_IDX=0
for config_name in "${CONFIG_ARRAY[@]}"; do
    # Construct full config path
    config_path="configs/experiments/${config_name}.yaml"

    # Check if config file exists
    if [[ ! -f "$config_path" ]]; then
        echo "Error: Config file not found: $config_path"
        continue
    fi

    # Generate unique eval_tag for this training job
    eval_tag="${EXPERIMENT_TAG}:${JOB_IDX}:${TIMESTAMP}"

    # Create job name (replace slashes with underscores for valid job names)
    job_name="${EXPERIMENT_TAG}_$(echo "${config_name}" | tr '/' '_')"

    echo "Submitting training job: $job_name"
    echo "  Config: $config_path"
    echo "  Experiment Tag: $EXPERIMENT_TAG"
    echo "  Eval Tag: $eval_tag"
    if [[ -n "$OVERRIDES" ]]; then
        echo "  Overrides: $OVERRIDES"
    fi

    # Submit SLURM job with dynamic output file paths (replace slashes with underscores for file names)
    output_name="$(echo "${config_name}" | tr '/' '_')"
    job_output=$(sbatch -J "$job_name" \
           -o "exp_logs/${EXPERIMENT_TAG}/${output_name}_%j.out" \
           -e "exp_logs/${EXPERIMENT_TAG}/${output_name}_%j.err" \
           launcher/hpc_batch.bash "$config_path" "$EXPERIMENT_TAG" "$OVERRIDES" "$eval_tag")

    # Extract job ID from sbatch output
    job_id=$(echo "$job_output" | grep -oP 'Submitted batch job \K\d+')

    if [[ -n "$job_id" ]]; then
        echo "  Training job submitted successfully with ID: $job_id"
        TRAINING_JOB_IDS+=("$job_id")
        EVAL_TAGS+=("$eval_tag")
        OUTPUT_NAMES+=("$output_name")
    else
        echo "  ERROR: Failed to extract job ID from: $job_output"
    fi

    # Increment job index for next training job
    JOB_IDX=$((JOB_IDX + 1))
    echo ""
done

# Check if any training jobs were submitted
if [[ ${#TRAINING_JOB_IDS[@]} -eq 0 ]]; then
    echo "Error: No training jobs were successfully submitted"
    exit 1
fi

# Build dependency string for all training jobs (used by eval jobs)
ALL_TRAINING_IDS=$(IFS=:; echo "${TRAINING_JOB_IDS[*]}")
echo "All training jobs submitted: ${TRAINING_JOB_IDS[*]}"
echo ""

# Submit evaluation jobs
echo "=== Submitting Evaluation Jobs ==="
EVAL_MODES=("performance" "noise" "rotation")
TOTAL_EVAL_JOBS=0

# Iterate through each training job
for i in "${!TRAINING_JOB_IDS[@]}"; do
    training_job_id="${TRAINING_JOB_IDS[$i]}"
    eval_tag="${EVAL_TAGS[$i]}"
    output_name="${OUTPUT_NAMES[$i]}"

    echo "Submitting eval jobs for training job $training_job_id (eval_tag: $eval_tag):"

    # Submit 3 eval jobs (performance, noise, rotation) for this training job
    for eval_mode in "${EVAL_MODES[@]}"; do
        # Create job name
        eval_job_name="EVAL_${eval_mode}_${EXPERIMENT_TAG}_idx${i}"

        # Build dependency string:
        # - after:{this_training_job_id} - run after this specific training job completes successfully
        # - afterany:{all_training_jobs} - ensure doesn't run until all training jobs finish (regardless of success)
        dependency_str="after:${training_job_id},afterany:${ALL_TRAINING_IDS}"

        echo "  - Eval Mode: $eval_mode"
        echo "    Dependency: $dependency_str"

        # Submit SLURM job with dependency
        eval_output_name="eval_${eval_mode}_${output_name}"
        eval_job_output=$(sbatch -J "$eval_job_name" \
               --dependency="$dependency_str" \
               -o "exp_logs/wandb_eval/${eval_output_name}_%j.out" \
               -e "exp_logs/wandb_eval/${eval_output_name}_%j.err" \
               launcher/hpc_wandb_eval_batch.bash "$eval_tag" "$VIDEO_ENABLED" "$eval_mode")

        # Extract eval job ID
        eval_job_id=$(echo "$eval_job_output" | grep -oP 'Submitted batch job \K\d+')
        if [[ -n "$eval_job_id" ]]; then
            TOTAL_EVAL_JOBS=$((TOTAL_EVAL_JOBS + 1))
            echo "    Eval job submitted with ID: $eval_job_id"
        else
            echo "    ERROR: Failed to extract eval job ID"
        fi
    done
    echo ""
done

echo "All jobs submitted!"
echo "Training jobs: ${#TRAINING_JOB_IDS[@]} (IDs: ${TRAINING_JOB_IDS[*]})"
echo "Eval jobs: $TOTAL_EVAL_JOBS"
echo "Total jobs: $((${#TRAINING_JOB_IDS[@]} + TOTAL_EVAL_JOBS))"
