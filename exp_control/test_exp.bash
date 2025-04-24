names=( "PiH" "GearMesh" )
envs=( "Isaac-Factory-PegInsert-Local-v0" "Isaac-Factory-GearMesh-Local-v0" )

exp_idx=$1
if [ "$exp_idx" -gt 4 ]; then
  exp_idx=0
fi

echo "Exp: ${names[$exp_idx]}"

#CUDA_LAUNCH_BLOCKING=1 
HYDRA_FULL_ERROR=1 python -m learning.ppo_factory_trainer \
    --task=${envs[$exp_idx]} \
    --max_steps=50000000 \
    --num_envs=256 \
    --num_agents=1 \
    --exp_name=$2  \
    --seed=1 \
    --init_eval \
    --no_vids 

    #
    #
    #
    #--init_eval \
    #--log_smoothness_metrics \
    #
    #
    #
    # 
    #
    #
    #
    #
   #
   # 
   # 
    #
    #
   # 
#
    #--wandb_tags="debug_Apr11" \#
    #
    #-
    #