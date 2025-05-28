#!/bin/bash
tasks=( "PegInsert" "GearMesh")
obs_modes=( "Local" "HistoryObs" "DMPObs")

task_idx=$1
obs_idx=$2
#if [ "$exp_idx" -gt 4 ]; then
#  exp_idx=0
#fi
task_name="Isaac-Factory-TaskType-ObsType-v0"
echo "$task_name"
echo "Task Type: ${tasks[$task_idx]}"
echo "Obs Mode: ${obs_modes[$obs_idx]}"


task_name="${task_name/ObsType/${obs_modes[obs_idx]}}" 
task_name="${task_name/TaskType/${tasks[task_idx]}}"
echo "Env name: $task_name"
#CUDA_LAUNCH_BLOCKING=1 
HYDRA_FULL_ERROR=1 python -m learning.ppo_factory_trainer \
    --task=$task_name \
    --max_steps=50000000 \
    --num_envs=4 \
    --num_agents=1 \
    --exp_name=$3  \
    --seed=1 \
    --no_vids \
    --decimation=16
