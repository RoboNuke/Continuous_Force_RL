
names=("Std_Obs" "DMP_Obs" "Hist_Obs" "Force_only_DMP" "Force_only_Hist")
envs=("Isaac-Factory-PegInsert-Local-v0" "Isaac-Factory-GearMesh-Local-v0" "TIsaac-Factory-NutThread-Local-v0" )

# default num_agents is 2
if [ -z "$2" ]; then
    num_agents=2
else
    num_agents=$2
fi

exp_idx=$1
if [ "$exp_idx" -gt 4 ]; then
  exp_idx=0
fi


echo "Exp: ${names[$exp_idx]}"
echo "Num Agents: $num_agents"
echo "Learning Method: $learning_method"
python -m learning.ppo_factory_trainer \
    --headless \
    --task=${envs[$exp_idx]} \
    --max_steps=50000000 \
    --num_envs=$((256 * $num_agents)) \
    --num_agents $num_agents \
    --exp_name=$3 \
    --no_vids 
# "DMP_Observation_Testing" \
#python -m learning.single_agent_train --task TB2-Factor-PiH-v0 --exp_name basic_PiH_baseline --headless --max_steps 50000000 --no_vids --num_agents 5 --num_envs 1280 --wandb_tags multi_agent_tests basic_obs
## "${names[$exp_idx]}"  \
#    --wandb_project="DMP_Observation_Testing" \
#    --wandb_tags="rl_update","no_curriculum","pih_apr11" \
    #--seed=1 \
    #--log_smoothness_metrics \