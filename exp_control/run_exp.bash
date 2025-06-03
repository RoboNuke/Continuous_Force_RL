#!/bin/bash
tasks=( "PegInsert" "GearMesh")
obs_modes=( "Local" "HistoryObs" "DMPObs")

task_idx=$1
obs_idx=$2
num_agents=$3
num_history_samples=$5
break_force=$6
#if [ "$exp_idx" -gt 4 ]; then
#  exp_idx=0
#fi
task_name="Isaac-Factory-TaskType-ObsType-v0"
echo "Task Type: ${tasks[$task_idx]}"
echo "Obs Mode: ${obs_modes[$obs_idx]}"


task_name="${task_name/ObsType/${obs_modes[obs_idx]}}" 
task_name="${task_name/TaskType/${tasks[task_idx]}}"
echo "Env name: $task_name"

echo "Num Agents: $num_agents"


exp_name="$4_${tasks[task_idx]}_${obs_modes[obs_idx]}"
echo $task_name
python -m learning.ppo_factory_trainer \
    --headless \
    --task=$task_name \
    --max_steps=50000000 \
    --num_envs=$((256 * $num_agents)) \
    --num_agents=$num_agents \
    --exp_name="$4_$num_history_samples" \
    --no_vids \
    --decimation=16 \
    --history_sample_size=$num_history_samples \
    --use_ft_sensor \
    --break_force=$break_force
    




# "DMP_Observation_Testing" \
#python -m learning.single_agent_train --task TB2-Factor-PiH-v0 --exp_name basic_PiH_baseline --headless --max_steps 50000000 --no_vids --num_agents 5 --num_envs 1280 --wandb_tags multi_agent_tests basic_obs
## "${names[$exp_idx]}"  \
#    --wandb_project="DMP_Observation_Testing" \
#    --wandb_tags="rl_update","no_curriculum","pih_apr11" \
    #--seed=1 \
    #--log_smoothness_metrics \