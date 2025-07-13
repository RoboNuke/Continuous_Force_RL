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

use_ft_sensor=1
break_force=-1
parallel_control=0
hybrid_control=1
hybrid_agent=1

python -m exp_control.record_ckpts.single_task_ckpt_recorder \
       --headless \
       --decimation=8 \
       --task=$task_name \
       --ckpt_record_path=$3 \
       --break_force=$break_force \
       --use_ft_sensor=$use_ft_sensor \
       --parallel_control=$parallel_control \
       --hybrid_control=$hybrid_control \
       --hybrid_agent=$hybrid_agent
       #--log_smoothness_metrics=0
