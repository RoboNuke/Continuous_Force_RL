#!/bin/bash

num_agents=4
num_exp_per=2

num_tasks=1
num_obs=3

nick_names=("PiH" "Gear" "Nut")

for task_idx in $(seq 1 $((num_tasks - 1)))
do
    for obs_idx in $(seq 1 $((num_obs - 1)))
    do
        sbatch -J "${nick_names[$task_idx]}_$1" -a 1-$num_exp_per exp_control/hpc_batch.bash \
                $task_idx \
                $obs_idx \
                $num_agents \
                $1
    done
done