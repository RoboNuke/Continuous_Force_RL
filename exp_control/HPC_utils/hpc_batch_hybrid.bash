#!/bin/bash
##SBATCH --array=1-100             # set up the array
#SBATCH -J RLUPDATE			    # name of job
#SBATCH -A virl-grp	            # name of my sponsored account, e.g. class or research group, NOT ONID!
##SBATCH -p gpu,eecs2,tiamat,dgxh,dgx2,ampere		# name of partition or queue
#SBATCH -p tiamat,gpu,dgx2,dgxh
##SBATCH -p tiamat
#SBATCH --time=1-06:00:00        # time limit on job: 2 days, 12 hours, 30 minutes (default 12 hours)
##SBATCH -N 1                   # number of nodes (default 1)
#SBATCH --gres=gpu:1            # number of GPUs to request (default 0)
#SBATCH --mem=32G               # request 10 gigabytes memory (per node, default depends on node)
#SBATCH -c 8                   # number of cores/threads per task (default 1)
#SBATCH -o ../hybrid_baseline/%A_%a.out		# name of output file for this submission script
#SBATCH -e ../hybrid_baseline/%A_%a.err		# name of error file for this submission script
# load any software environment module required for app (e.g. matlab, gcc, cuda)

#echo "Job Name:" $SLURM_JOB_NAME
#echo "Array:" $SLURM_ARRAY_TASK_COUNT
##module load cuda/10.1
#source ~/.bashrc
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate isaaclab_242

echo "Conda env: $CONDA_DEFAULT_ENV"


free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
numeric_string="${free_memory//[^0-9]/}"
int_free=$((numeric_string))
echo "Free memory:$int_free"
#if (( int_free > 8000 )); then
#    echo "8 parallel it is"
#    #bash "exp_control/run_exp.bash" $1 8 $2
#else
#    echo "two 4 parallel"
#    #bash "exp_control/run_exp.bash" $1 4 $2
#    #   bash "exp_control/run_exp.bash" $1 4 $2
#fi

#echo "call run_exp.bash"
echo $6

bash "exp_control/run_hybrid_exp.bash" $1 $2 "$3_$SLURM_ARRAY_TASK_ID" $4 $5 $6 $7 $8
