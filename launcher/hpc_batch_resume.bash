#!/bin/bash
#SBATCH -J RESUME_JOB		        # job name (will be overridden by sbatch -J)
#SBATCH -A virl-grp	                # sponsored account name
#SBATCH -p dgxh,dgx2,tiamat,gpu,eecs2     # partition names
#SBATCH --time=0-09:00:00           # time limit: 9 hours
#SBATCH --gres=gpu:1                # number of GPUs to request
#SBATCH --mem=32G                   # request 32 gigabytes memory
#SBATCH -c 12                       # number of cores/threads per task
#SBATCH --signal=TERM@300           # send SIGTERM 300 seconds (5 min) before time limit
# Note: Output files will be set dynamically based on tag name

# Parse named arguments
CHECKPOINT_TAG=""
CHECKPOINT_STEP=""
OVERRIDES=""
NEW_PROJECT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_tag)
            CHECKPOINT_TAG="$2"
            shift 2
            ;;
        --checkpoint_step)
            CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --overrides)
            OVERRIDES="$2"
            shift 2
            ;;
        --new_project)
            NEW_PROJECT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$CHECKPOINT_TAG" ]]; then
    echo "Error: --checkpoint_tag is required"
    exit 1
fi

echo "=== HPC Resume Batch Script Started ==="
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Checkpoint Tag: $CHECKPOINT_TAG"
echo "Checkpoint Step: ${CHECKPOINT_STEP:-"(auto-discover)"}"
echo "New Project: ${NEW_PROJECT:-"(inherit from checkpoint)"}"
echo "Overrides: $OVERRIDES"
echo ""

# Create log directory if it doesn't exist
mkdir -p "../exp_logs/resume"

# Load environment
echo "Loading conda environment..."
source ~/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate isaaclab_drail

echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Build the python command
python_cmd="python -m learning.factory_runnerv3"
python_cmd="$python_cmd --checkpoint_tag $CHECKPOINT_TAG"

# Add checkpoint_step if provided
if [[ -n "$CHECKPOINT_STEP" ]]; then
    python_cmd="$python_cmd --checkpoint_step $CHECKPOINT_STEP"
fi

# Add new_project if provided
if [[ -n "$NEW_PROJECT" ]]; then
    python_cmd="$python_cmd --new_project $NEW_PROJECT"
fi

# Add additional overrides if provided
if [[ -n "$OVERRIDES" ]]; then
    # Split overrides by space and add each as a separate --override
    read -ra OVERRIDE_ARRAY <<< "$OVERRIDES"
    for override in "${OVERRIDE_ARRAY[@]}"; do
        python_cmd="$python_cmd --override $override"
    done
fi

# Add headless flag
python_cmd="$python_cmd --headless"

echo "Executing command:"
echo "$python_cmd"
echo ""

# Force unbuffered Python output so logs are written immediately
export PYTHONUNBUFFERED=1

# Use exec to replace bash process with Python process
# This ensures SLURM's SIGTERM goes directly to Python (no forwarding needed)
# Python's signal handler will handle cleanup on SIGTERM
exec $python_cmd
