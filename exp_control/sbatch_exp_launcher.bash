#!/bin/bash

num_agents=4
num_exp_per_env=2
num_exp_env=3
nick_names=("PiH" "Gear_Mesh" "Nut_Thread")

for env_idx in $(seq 0 $((num_exp_env - 1)))
do
    sbatch -J "${nick_names[$env_idx]}_$1" -a 1-$num_exp_per_env exp_control/hpc_batch.bash $env_idx $num_agents "$1_${nick_names[$env_idx]}"
    #sbatch -a 1-$num_exp_per_env exp_control/hpc_batch.bash $env_idx $num_agents "$1_$env_idx"
done