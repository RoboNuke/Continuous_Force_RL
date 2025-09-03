#!/bin/bash

nick_names=("PiH" "Gear" "Nut")

task_idx=0
obs_idx=0

rew_types=( "base" ) # "pos_simp" "delta" "dirs" ) #"base" ) "wrench_norm"
exp_tag="FPiH_SEP03" #"long_baseline" #"exps_AUG15" #"hybrid_baseline_llr"
project="SEP03_Exps" #"AUG31_Exps"
hybrid_agent=0
#hybrid_control=0
ctrl_torque=0
#use_ft_sensor=1
use_ft_sensors=(0 1)
#break_force="25"
break_forces=( "-1,5" "10,15" "20,25" )
#break_forces=( "-1,5,10" ) #"15,20,25" )

sel_adjs="init_bias" #,force_add_zout,init_scale_weights,scale_zin" # "init_bias", "force_add_zout", "init_scale_weights", "scale_zin"

for use_ft_sensor in "${use_ft_sensors[@]}";
do
    for hybrid_control in {0..1};
    do		   
	#    for rew_type in "${rew_types[@]}";
		if [[ "$use_ft_sensor" -eq 1 ]]; then
			name="$1_FT(ON)"
		# Check if use_ft_sensor is 0
		elif [[ "$use_ft_sensor" -eq 0 ]]; then
			name="$1_FT(OFF)"
		else
			name="$1"
		fi
		
		if [[ "$hybrid_control" -eq 1 ]]; then
			name="hybrid_$name"
		elif [[ "$hybrid_control" -eq 0 ]]; then
			name="pose_$name"
		else
			echo "Failed to get name correct"
		fi


		for break_force in "${break_forces[@]}";
		do
			echo "  ${nick_names[$task_idx]}_$name_${break_force}"
			
			#bash exp_control/run_hybrid_exp.bash \
			sbatch -J "${nick_names[$task_idx]}_$name_${break_force}" exp_control/HPC_utils/hpc_batch_hybrid.bash \
				$task_idx \
				$obs_idx \
				$name \
				$exp_tag \
				$break_force \
				$hybrid_agent \
				$ctrl_torque \
				"base" \
				$sel_adjs \
				$use_ft_sensor \
				$hybrid_control \
				$project
		done
    done
done

#$exp_tag \
	#	 
#
#	 
