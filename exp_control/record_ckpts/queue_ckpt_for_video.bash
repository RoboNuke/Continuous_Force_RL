

# $1 is the absolute file path to checkpoint file
# $2 is the step for the checkpoint
# $3 is the task id
# $4 is the destination filepath
sbatch -J "$2_ckpt_vid" exp_control/record_ckpts/batch_ckpt_video.bash $1 $2 $3 $4
    