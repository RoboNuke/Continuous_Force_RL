#!/bin/bash

nick_names=("PiH" "Gear" "Nut")

task_idx=0
obs_idx=0

rew_types=( "base" ) # "pos_simp" "delta" "dirs" ) #"base" ) "wrench_norm"
exp_tag="simp_env" #"long_baseline" #"exps_AUG15" #"hybrid_baseline_llr"
hybrid_agent=0
hybrid_control=0
ctrl_torque=0
use_ft_sensor=1
break_forces="-1,5,10,15,20,25"
sel_adjs="init_bias" #,force_add_zout,init_scale_weights,scale_zin" # "init_bias", "force_add_zout", "init_scale_weights", "scale_zin"
#for use_ft_sensor in {0..1};do
    for rew_type in "${rew_types[@]}";
    do

	
	#sbatch -J "${nick_names[$task_idx]}_$1_${use_ft_sensor}" exp_control/HPC_utils/hpc_batch_hybrid.bash \
	    bash exp_control/run_hybrid_exp.bash \
		 $task_idx \
		 $obs_idx \
		 "$1_$use_ft_sensor" \
		 $exp_tag \
		 $break_forces \
		 $hybrid_agent \
		 $ctrl_torque \
		 $rew_type \
		 $sel_adjs \
		 $use_ft_sensor \
		 $hybrid_control
	done
    #done $exp_tag \
	#	 
