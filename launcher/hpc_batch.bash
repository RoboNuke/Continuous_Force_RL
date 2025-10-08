#!/bin/bash
#SBATCH -J LAUNCHER_JOB		        # job name (will be overridden by sbatch -J)
#SBATCH -A virl-grp	                # sponsored account name
#SBATCH -p tiamat,gpu,eecs2,dgx2,dgxh               # partition names
#SBATCH --time=0-09:59:59           # time limit: 1 day, 23 hours, 59 minutes
#SBATCH --gres=gpu:1                # number of GPUs to request
#SBATCH --mem=32G                   # request 32 gigabytes memory
#SBATCH -c 12                       # number of cores/threads per task
# Note: Output files will be set dynamically based on EXPERIMENT_TAG parameter

# Script arguments
CONFIG_PATH=$1
EXPERIMENT_TAG=$2
OVERRIDES=$3

echo "=== HPC Batch Script Started ==="
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Config Path: $CONFIG_PATH"
echo "Experiment Tag: $EXPERIMENT_TAG"
echo "Overrides: $OVERRIDES"
echo ""

# Create log directory if it doesn't exist
mkdir -p "../exp_logs/$EXPERIMENT_TAG"

# Load environment
echo "Loading conda environment..."
source ~/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate isaaclab_drail

echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Build the python command
python_cmd="python -m learning.factory_runnerv3"
python_cmd="$python_cmd --config $CONFIG_PATH"
python_cmd="$python_cmd --override experiment.tags=$EXPERIMENT_TAG"

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

# Execute the training
eval $python_cmd

echo ""
echo "=== HPC Batch Script Completed ==="