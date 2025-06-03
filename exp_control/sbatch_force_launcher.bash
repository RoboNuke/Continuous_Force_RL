#!/bin/bash

num_agents=4
num_exp_per=2

nick_names=("PiH" "Gear" "Nut")

task_idx=0

num_forces=1
forces=(1 5 25 100 -1)
for force_idx in $(seq 0 $((num_forces - 1)))
do
    for obs_idx in $(seq 0 0)
    do 
        sbatch -J "${nick_names[$task_idx]}_$1_${samples[$sample_idx]}" -a 1-$num_exp_per exp_control/hpc_batch.bash \
                $task_idx \
                $obs_idx \
                $num_agents \
                $1 \
                16 \
                ${forces[$force_idx]}
    done
done