#!/bin/bash

# Define the minimum free memory threshold in MiB
MIN_FREE_MEMORY_MIB=10000

# Specify the path to the script you want to run
SCRIPT_TO_RUN="exp_control/test_exp.bash"

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

# --- Main script logic ---
main() {
    # Get the amount of free GPU memory
    free_gpu_memory=$(get_free_gpu_memory)

    # Check for errors in retrieving memory information
    if [[ "$?" -ne 0 ]]; then
        exit 1
    fi

    # Check if free GPU memory is greater than the defined threshold
    if [[ "$free_gpu_memory" -gt "$MIN_FREE_MEMORY_MIB" ]]; then
        echo "Free GPU memory ($free_gpu_memory MiB) is greater than $MIN_FREE_MEMORY_MIB MiB."
        echo "Creating a new tmux pane and running: $SCRIPT_TO_RUN"

        # Create a new tmux pane and run the script
        # Adjust session_uuid and target based on your tmux setup
        # tmux new-window -d -n "gpu_task" "$SCRIPT_TO_RUN" # Option to create a new window
        # tmux split-window -d "$SCRIPT_TO_RUN" # Option to split the current window vertically and run in the new pane
        tmux split-window -h #-d "$SCRIPT_TO_RUN"  # Splits the current window horizontally and runs the script in the new pane
        tmux send-keys -t 1 "cd ~/hpc-share/Continuous_Force_RL/ && bash $script_to_run 0 0" C-m
        echo "Script executed in a new tmux pane."
    else
        echo "Free GPU memory ($free_gpu_memory MiB) is not greater than $MIN_FREE_MEMORY_MIB MiB."
        echo "Script will not be executed."
    fi
}

tmux new-session -d -s "EXP_TEST"
tmux attach-session -t "EXP_TEST"


# Run the main function
main