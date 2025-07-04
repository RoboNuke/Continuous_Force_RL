#!/bin/bash

num_agents=10
num_exp_per=1

nick_names=("PiH" "Gear" "Nut")

task_idx=0

num_forces=2
forces=(-1 10 5 25 50)
obs_idx=0
exp_tag="jul01_hybrid_control"
wandb_group_prefix="Hybrid_Controller2_JUL01"

for force_idx in $(seq 0 $((num_forces - 1)))
do
    for use_ft in $(seq 0 1) # only doing local and history
    do 
        sbatch -J "${nick_names[$task_idx]}_$1_${samples[$sample_idx]}" -a 1-$num_exp_per exp_control/HPC_utils/hpc_batch.bash \
                $task_idx \
                $obs_idx \
                $num_agents \
                $1 \
                16 \
                "${forces[$force_idx]}" \
                $use_ft \
                $exp_tag \
                $wandb_group_prefix \
                "/nfs/stak/users/brownhun/ckpt_trackers" \
                1
    done
done