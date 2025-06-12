#!/bin/bash

# Define the minimum free memory threshold in MiB
MIN_FREE_MEMORY_MIB=10300
#LOGFILE="/home/hunter/hpc_share/current_jobs.json"
LOGFILE="/nfs/stak/users/brownhun/hpc_share/current_jobs.json"
# Specify the path to the script you want to run
EXPERIMENT_SCRIPT="exp_control/run_exp.bash"

# --- Function to get free GPU memory ---
get_free_gpu_memory() {
    # Use nvidia-smi to query free GPU memory for GPU 0 (change --id accordingly if needed)
    # Parse the output to get the numeric value in MiB
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=0)

    # Check if the output is empty or contains non-numeric characters
    if [[ -z "$free_mem" || ! "$free_mem" =~ ^[0-9]+$ ]]; then
        echo "Error: Unable to retrieve or parse free GPU memory from nvidia-smi." >&2
        return 1
    fi

    echo "$free_mem"
}


free_gpu_memory=$(get_free_gpu_memory)
exps_to_launch=$((free_gpu_memory / MIN_FREE_MEMORY_MIB))
echo "Launching $exps_to_launch exps"
job_id=$1
WIN_NUM=0


tmux new-session -d -s "EXP_TEST"

RAW_OUTPUT=$(python3 -m exp_control.HPC_utils.jobid_utils pop_by_id "$LOGFILE" "$job_id")
echo "Loaded config: $RAW_OUTPUT"

# Convert JSON string to command-line args using jq (or Python inline if you prefer)
ARGS=$(python3 -c "import json, shlex #data = json.loads($RAW_OUTPUT)
data=$RAW_OUTPUT
base_arg=' '.join(f'--{k} {shlex.quote(str(v))}' for k, v in data.items() if k != 'job_id')
result=base_arg.replace(\" '' \", \" \")
print(result)
")

# --- Launch args for current job ---
tmux send-keys -t $WIN_NUM "conda activate isaaclab_242" C-m
tmux send-keys -t $WIN_NUM "bash $EXPERIMENT_SCRIPT $ARGS" C-m

# --- Main script logic ---
for WIN_NUM in $(seq 1 $((exps_to_launch - 1))); do
    echo "Launching experiment:$WIN_NUM"
    # Pop the last experimental config from the log
    RAW_OUTPUT=$(python3 -m exp_control.HPC_utils.jobid_utils pop "$LOGFILE")

    # Exit if no data returned
    if [ -z "$RAW_OUTPUT" ] || [[ "$RAW_OUTPUT" == *"File is empty."* ]]; then
        echo "No experimental conditions found."
        tmux select-layout even-vertical
        tmux attach-session -t "EXP_TEST"
        exit 0
    fi

    echo "Loaded config: $RAW_OUTPUT"

    # Convert JSON string to command-line args using jq (or Python inline if you prefer)
    ARGS=$(python3 -c "import json, shlex #data = json.loads($RAW_OUTPUT)
    data=$RAW_OUTPUT
    base_arg=' '.join(f'--{k} {shlex.quote(str(v))}' for k, v in data.items() if k != 'job_id')
    result=base_arg.replace(\" '' \", \" \")
    print(result)
    ")

    JOB_ID=$(python3 -c "import json; print(json.loads('$DICT_JSON').get('job_id', 'UNKNOWN'))")

    # Run the experiment
    echo "Creating a new tmux pane and running: $EXPERIMENT_SCRIPT"

    # Create a new tmux pane and run the script
    # Adjust session_uuid and target based on your tmux setup
    # tmux new-window -d -n "gpu_task" "$SCRIPT_TO_RUN" # Option to create a new window
    # tmux split-window -d "$SCRIPT_TO_RUN" # Option to split the current window vertically and run in the new pane
    #tmux select-pane -t $WIN_NUM
    tmux split-window -v #-d "$SCRIPT_TO_RUN"  # Splits the current window horizontally and runs the script in the new pane

    tmux send-keys -t $WIN_NUM "conda activate isaaclab_242" C-m
    echo "Running: python3 $EXPERIMENT_SCRIPT $ARGS"
    tmux send-keys -t $WIN_NUM "bash $EXPERIMENT_SCRIPT $ARGS" C-m
    echo "Script executed in a new tmux pane."

    echo "Canceling $JOB_ID"
    scancel $JOB_ID

    # TODO: Create the evaluation script here TODO
        
done


tmux select-layout even-vertical
tmux attach-session -t "EXP_TEST"
echo "Script complete"
