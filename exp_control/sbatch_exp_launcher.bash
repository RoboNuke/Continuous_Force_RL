#!/bin/bash

num_agents=4
num_exp_per_env=2
num_exp_env=3
names=("PiH" "Gear_Mesh" "Nut_Thread" "Force_only_DMP" "Force_only_Hist")

for env_idx in $(seq 0 $((num_exp_env - 1)))
do
    sbatch -J "${names[$exp_idx]}_$1" -a 1-$num_exp_per_env exp_control/hpc_batch.bash $env_idx $num_agents "$1_${names[$exp_idx]}"
    #sbatch -a 1-$num_exp_per_env exp_control/hpc_batch.bash $env_idx $num_agents "$1_$env_idx"
done