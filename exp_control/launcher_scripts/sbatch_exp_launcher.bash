#!/bin/bash

num_agents=8
num_exp_per=1

num_tasks=1
num_obs=1

nick_names=("PiH" "Gear" "Nut")
break_force=-1
use_ft_sensor=1
exp_tag="debug"
wandb_group_prefix="SizeProblems"

for task_idx in $(seq 0 $((num_tasks - 1)))
do
    for obs_idx in $(seq 0 $((num_obs - 1)))
    do
        exp_condition
        sbatch -J "${nick_names[$task_idx]}_$1" -a 1-$num_exp_per exp_control/HPC_utils/hpc_batch.bash \
                $task_idx \
                $obs_idx \
                $num_agents \
                $1 \
                16 \
                $break_force \
                $use_ft_sensor \
                $exp_tag \
                $wandb_group_prefix
    done
done