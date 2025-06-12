#!/bin/bash

# Configuration]
#LOGFILE="/home/hunter/hpc_share/current_jobs.json"
LOGFILE="/nfs/stak/users/brownhun/hpc_share/current_jobs.json"

NUM_AGENTS=10

num_tasks=1
num_obs=3

NICK_NAMES=("PiH") # "Gear" "Nut")
tasks=( "PegInsert" "GearMesh")
obs_modes=( "Local" "HistoryObs" "DMPObs")
BREAK_FORCE=-1
USE_FT_SENSOR=1
EXP_TAG="jun12"
WANDB_GROUP_PREFIX="Baseline"
HISTORY_SAMPLE_SIZE=16

# Iterate over all condition lists
for use_ft_sensor in $(seq 0 1); do
    for task_idx in "${NICK_NAMES[@]}"; do
        for obs_idx in $(seq 0 $((num_obs - 1))); do
            echo "Processing task_idx=$task_idx, obs_idx=$obs_idx"
            task_name="Isaac-Factory-TaskType-ObsType-v0"
            echo "    Task Type: ${tasks[$task_idx]}"
            echo "    Obs Mode: ${obs_modes[$obs_idx]}"

            task_name="${task_name/ObsType/${obs_modes[obs_idx]}}" 
            task_name="${task_name/TaskType/${tasks[task_idx]}}"
            # Submit job and capture job ID
            JOB_OUTPUT=$( sbatch -J "${NICK_NAMES[$task_idx]}_$1" exp_control/HPC_utils/dynamic_hpc_batch.bash )
            echo "    Job output:$JOB_OUTPUT"
            JOB_ID=$(echo "$JOB_OUTPUT" | cut -d';' -f1) #$((JOB_ID+1)) #
            echo "    Job id:$JOB_ID"
            if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
                echo "    Submitted job with ID $JOB_ID (task=$task_name)"

                # Create JSON dictionary string
                DICT="{\"job_id\": $JOB_ID, \
                    \"task\": \"$task_name\", \
                    \"headless\": \"\", \
                    \"no_vids\": \"\", \
                    \"max_steps\": \"50000000\", \
                    \"decimation\": \"16\", \
                    \"num_envs\": \"$((16 * $NUM_AGENTS))\", \
                    \"wandb_project\": \"Continuous_Force_RL\", \
                    \"num_agents\": \"$NUM_AGENTS\", \
                    \"exp_name\": \"$1\", \
                    \"history_sample_size\": \"$HISTORY_SAMPLE_SIZE\", \
                    \"break_force\": \"$BREAK_FORCE\", \
                    \"use_ft_sensor\": \"$use_ft_sensor\", \
                    \"exp_tag\": \"$EXP_TAG\", \
                    \"wandb_group_prefix\": \"$WANDB_GROUP_PREFIX\"}"

                # Log it using Python script
                python3 -m exp_control.HPC_utils.jobid_utils write "$LOGFILE" "$DICT"
            else
                echo "    Failed to submit job with ID $JOB_ID (task=$task_name)"
            fi
        done
    done
done