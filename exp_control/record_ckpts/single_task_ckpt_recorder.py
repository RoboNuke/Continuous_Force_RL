import argparse
import sys
try:
    from isaaclab.app import AppLauncher
except:
    from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
# exp
parser.add_argument("--seed", type=int, default=-1, help="Seed used for the environment")
parser.add_argument("--decimation", type=int, default=8, help="How many simulation steps between policy observations")
parser.add_argument("--policy_hz", type=int, default=15, help="Rate in hz that the policy should get new observations")
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Local-v0")
parser.add_argument("--ckpt_record_path", type=str, default="/nfs/stak/users/brownhun/ckpt_tracker.txt")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

args_cli.video = True
args_cli.enable_cameras = True


# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import torchvision.transforms.functional as F
from PIL import Image
import gymnasium as gym
import os
import random
from datetime import datetime

import skrl
from packaging import version

from skrl.utils import set_seed
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

#from wrappers.smoothness_obs_wrapper import SmoothnessObservationWrapper
from wrappers.close_gripper_action_wrapper import GripperCloseEnv
from models.default_mixin import Shared
from wrappers.smoothness_obs_wrapper import SmoothnessObservationWrapper
from wrappers.hybrid_control_action_wrapper import HybridControlActionWrapper
#from models.bro_model import BroAgent
#from wrappers.DMP_observation_wrapper import DMPObservationWrapper

try:
    import isaaclab.sim as sim_utils
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
    )
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_pickle, dump_yaml

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from isaaclab.sensors import TiledCameraCfg, ImuCfg
except:
    from omni.isaac.lab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
    )
    from omni.isaac.lab.utils.dict import print_dict
    from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

    import omni.isaac.lab_tasks  # noqa: F401
    from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
    from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.sensors import TiledCameraCfg, ImuCfg
import time
#import envs.FPiH.config.franka
import envs.factory

from agents.wandb_logger_ppo_agent import WandbLoggerPPO
from models.SimBa import SimBaActor, SimBaCritic
from skrl.resources.schedulers.torch import KLAdaptiveLR
import tqdm

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from filelock import FileLock
import wandb



wandb.login(
    key='a593b534585893ad93cf62243228a866c9053247',
    force=True
)

def getNextCkpt(
        task, 
        #ckpt_tracker_path="/home/hunter/temp_ckpt_holder.txt"
        ckpt_tracker_path="/nfs/stak/users/brownhun/ckpt_tracker.txt"
    ):
    lock = FileLock(ckpt_tracker_path + ".lock")
    with lock:
        try:
            #found_next = False
            with open(ckpt_tracker_path, 'r') as file:
                lines = file.readlines()

            next_line = None
            with open(ckpt_tracker_path, 'w+') as file:
                for line in reversed(lines): # get last example
                    if task in line:
                        next_line = line
                        break
                for line in lines: # rewrite file
                    if line == next_line:
                        continue
                    else:
                        file.write(line)

            ckpt_path, task, gif_path, wandb_project, run_id = next_line.split() # order: ckpt_path, task, gif_path, wandb_project, run_id
            return ckpt_path, gif_path, wandb_project, run_id
        except FileNotFoundError:
            return False, False, False, False


def save_tensor_as_gif(tensor_list, filename, tot_rew, vals, succ_step, engaged_mask, duration=100, loop=0):
    """
    Saves a list of PyTorch tensors as a GIF image.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to be saved as frames.
        filename (str): Output filename for the GIF.
        tot_rew (tensor): Return for each environment
        vals (any): List to write onto corner of images
        duration (int, optional): Duration of each frame in milliseconds. Defaults to 100.
        loop (int, optional): Number of times the GIF should loop. 0 means infinite loop. Defaults to 0.
    """
    images = []
    ordered_rew_idxs = torch.squeeze(torch.argsort(tot_rew.T))
    n = ordered_rew_idxs.size()[0]
    img_idxs_display = torch.cat((
        ordered_rew_idxs[:4], # worst 4
        ordered_rew_idxs[ n//2 -2:n//2+2], # mid 4 
        ordered_rew_idxs[-4:] # best 4
    ), dim=-1)
    vals = vals[img_idxs_display, :]
    succ_step = succ_step[img_idxs_display]
    engaged_mask = engaged_mask[img_idxs_display,:]

    # edit tensor_list down
    new_tensor_list = torch.zeros((engaged_mask.size()[1], 180 * 3, 240 * 4, 3))
    for i, idx in enumerate(img_idxs_display):
        y = i // 4
        x = i % 4
        new_tensor_list[:, y*180:(y+1)*180, x*240:(x+1)*240, :] = tensor_list[:, idx*180:(idx+1)*180, 0:240, :]

    tensor_list =  new_tensor_list
    tensor_list = tensor_list.permute(0, 3, 1, 2)
    tensor_list = 1.0 - tensor_list
    #print(torch.max(tensor_list), torch.min(tensor_list))
    for i in range(engaged_mask.size()[1]):#tensor_list:
        # Ensure the tensor is in CPU and convert it to a PIL Image
        tensor = tensor_list[i,:,:,:] 
        img = F.to_pil_image(tensor.to("cpu"))
        for img_idx, val in enumerate(vals[:,i]):
            x = img_idx % 4
            y = img_idx // 4
            draw = ImageDraw.Draw(img)
            #font = ImageFont.truetype("sans-serif.ttf", 16)
            #font = ImageFont.truetype("UbuntuMono-R.ttf", 20)            
            font = ImageFont.truetype("DejaVuSans.ttf",20)
            if engaged_mask[img_idx, i]:
                draw.rectangle( 
                    (
                        (x*240, y*180),
                        ((x+1)*240, (y+1)*180)
                    ), 
                    outline="orange", 
                    width=3
                )

            if i >= succ_step[img_idx]:
                draw.rectangle( 
                    (
                        (x*240, y*180),
                        ((x+1)*240, (y+1)*180)
                    ), 
                    outline="green", 
                    width=3
                )
            # draw.text((x, y),"Sample Text",(r,g,b))
            draw.text((x * 240, y*180+160),f"Value Est={round(val.item(),2)}",(0,255,0),font=font)

        images.append(img)
        #plt.plot(img)
        #plt.show()

    # Save the list of PIL Images as a GIF
    folder_path = os.path.dirname(filename)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    images[0].save(filename, save_all=True, append_images=images[1:], duration=duration, loop=loop)

agent_cfg_entry_point = f"SimBaNet_ppo_cfg_entry_point"

class Img2InfoWrapperclass(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)
        infos['img'] = self.unwrapped.scene['tiled_camera'].data.output['rgb']
        return observations, rewards, terminateds, truncateds, infos 
    
    def reset(self, **kwargs):
        """Reset the environment using kwargs and then starts recording if video enabled."""
        observations, info = super().reset(**kwargs)
        info['img'] = self.unwrapped.scene['tiled_camera'].data.output['rgb'] #observations['info']['img']
        return observations, info

def getEnv(env_cfg, task):
    env = gym.make(
            task, 
            cfg=env_cfg, 
            render_mode="rgb_array"
        )
    
    env = Img2InfoWrapperclass(env)

    env = SkrlVecEnvWrapper(
        env, 
        ml_framework="torch"
    )  # same as: `wrap_env(env, wrapper="auto")    
    #env._reset_once = False
    #env = GripperCloseEnv(env)
    return env

def getAgent(env, agent_cfg, task_name, device):
    models = {}
    agent_cfg['agent']['track_ckpts'] = False
    models['policy'] = SimBaActor( #BroAgent(
        observation_space=env.cfg.observation_space, 
        action_space=env.action_space,
        #action_gain=0.05,
        device=device,
        act_init_std = agent_cfg['models']['act_init_std'],
        actor_n = agent_cfg['models']['actor']['n'],
        actor_latent = agent_cfg['models']['actor']['latent_size']
    ) 

    models["value"] = SimBaCritic( #BroAgent(
        state_space_size=env.cfg.state_space, 
        device=device,
        critic_output_init_mean = agent_cfg['models']['critic_output_init_mean'],
        critic_n = agent_cfg['models']['critic']['n'],
        critic_latent = agent_cfg['models']['critic']['latent_size']
    )

    agent_cfg['agent']['experiment']['wandb'] = False
    agent = WandbLoggerPPO(
        models=models,
        memory=None,
        cfg=agent_cfg['agent'],
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_envs=env.num_envs,
        device=device,
        state_size=env.cfg.observation_space+env.cfg.state_space,
        track_ckpt_paths=False,
        task=task_name
    )
    return agent
if args_cli.seed == -1:
    args_cli.seed = random.randint(0, 10000)
seed = args_cli.seed
print("Seed:", seed)
set_seed(seed)

@hydra_task_config("Isaac-Factory-PegInsert-Local-v0", agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, 
    agent_cfg: dict
):    
    
    env_cfg.filter_collisions = False
    task = args_cli.task
    print("task:", task)
    print("initialize single ckpt video")
    
    # create env
    num_envs = 100
    env_cfg.decimation = args_cli.decimation
    env_cfg.sim.dt = (1/args_cli.policy_hz) / args_cli.decimation
    print(f"Time scale config parameters\n\tDec: {env_cfg.decimation}\n\tSim_dt:{1/env_cfg.sim.dt}\n\tPolicy_Hz:{args_cli.policy_hz}")
    
    """Train with skrl agent."""
    max_rollout_steps = int((1/env_cfg.sim.dt) / env_cfg.decimation * env_cfg.episode_length_s)
    agent_cfg['agent']['rollouts'] = max_rollout_steps
    agent_cfg['agent']['experiment']['write_interval'] = max_rollout_steps
    agent_cfg['agent']['experiment']['checkpoint_interval'] = max_rollout_steps * 10
    agent_cfg['agent']['experiment']['tags'].append(env_cfg.task_name)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = num_envs 
    #env_cfg.scene.replicate_physics = True
    env_cfg.sim.device = env_cfg.sim.device

    env_cfg.num_agents = 1
    if 'learning_rate_scheduler' in agent_cfg['agent'].keys():
        # yaml doesn't read it as a class, but as a string idk
        agent_cfg['agent']['learning_rate_scheduler'] = KLAdaptiveLR

    env_cfg.scene.tiled_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0, 0.0, 0.35), 
            rot=(0.6123724, 0.3535534, 0.3535534, 0.6123724,), 
            convention="opengl"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=0.05, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=240,
        height=180,
        debug_vis = False,
    )

    # create env
    env = gym.make(
            task, 
            cfg=env_cfg, 
            render_mode="rgb_array"
        )
    
    env = Img2InfoWrapperclass(env)

    if args_cli.hybrid_control:
        print("\n\n[INFO] Using Hybrid Control Wrapper.\n\n")
        env = HybridControlActionWrapper(env)


    if args_cli.log_smoothness_metrics:
        print("\n\n[INFO] Recording Smoothness Metrics in info.\n\n")
        env = SmoothnessObservationWrapper(env)

    env = SkrlVecEnvWrapper(
        env, 
        ml_framework="torch"
    )  # same as: `wrap_env(env, wrapper="auto")  
    
    print("env ready...")
    env.cfg.recording = True
    device = env.device
    images = torch.zeros((max_rollout_steps, num_envs*180, 240, 3), device = env.device)

    agent = getAgent(env, agent_cfg, task, device)

    print("configured...")
    while True:
        ckpt_path, gif_path, wandb_project, run_id = getNextCkpt(args_cli.task, ckpt_tracker_path=args_cli.ckpt_record_path)
        if ckpt_path == False:
            print("Waiting 2 minutes")
            time.sleep(120) # wait 5 min
            continue # try again
        print(f"Filming {ckpt_path}")
        # load wandb run
        
        with torch.no_grad():
            #   load agent
            agent.load(ckpt_path)
            print("agent loaded")
            # reset env
            agent.set_running_mode("eval")
            states, infos = env.unwrapped.reset()
            states, infos = env.reset()
            states = env.unwrapped._get_observations()
            states = torch.cat( (states['policy'], states['critic']),dim=1)
            
            env.unwrapped.evaluating = True
            alive_mask = torch.ones(size=(states.shape[0], 1), device=states.device, dtype=bool)
            vals = []
            success_mask = torch.zeros(size=(states.shape[0], 1), device=states.device, dtype=bool)
            succ_step = torch.ones(size=(states.shape[0],), device=states.device, dtype=torch.int32) * max_rollout_steps
            tot_rew = torch.zeros(size=(states.shape[0], 1), device=states.device, dtype=torch.float32)
            engaged_mask = torch.zeros(size=(states.shape[0], max_rollout_steps), dtype=bool, device=device)
            for i in tqdm.tqdm(range(max_rollout_steps), file=sys.stdout):
                # get action
                actions = agent.act(
                    states, 
                    timestep=i, 
                    timesteps=max_rollout_steps
                )[-1]['mean_actions']

                vals.append(agent.value.act({"states": states}, role="value")[0])
                actions[~alive_mask[:,0],:] *= 0.0
                actions[success_mask[:,0],:] *= 0.0
                
                # take action
                next_states, step_rew, terminated, truncated, infos = env.step(actions)
                next_states = torch.cat( (env.unwrapped.obs_buf['policy'], env.unwrapped.obs_buf['critic']),dim=1)
                env.cfg.recording = True

                for k in range(env.num_envs):
                    images[i, k*180:(k+1)*180, 0:240, :] = infos['img'][k,:,:,:]

                tot_rew[alive_mask] += step_rew[alive_mask]

                mask_update = ~torch.logical_or(terminated, truncated)
                alive_mask *= mask_update
                curr_successes = env.unwrapped._get_curr_successes(
                    success_threshold=env.cfg_task.success_threshold, 
                    check_rot = env.cfg_task.name == "nut_thread"
                )
                engaged_mask[:,i] = env.unwrapped._get_curr_successes(
                    success_threshold=env.cfg_task.engage_threshold,
                    check_rot= env.cfg_task.name == "nut_thread"
                )


                if (curr_successes).any():
                    succ_step[
                        torch.logical_and(
                            succ_step == max_rollout_steps,
                            curr_successes
                        )
                    ] = i
                    success_mask[curr_successes,0] = True

                # reset environments
                if env.num_envs > 1:
                    states = next_states
                else:
                    if terminated.any() or truncated.any():
                        with torch.no_grad():
                            states, infos = env.reset()
                    else:
                        states = next_states
            # draw eval est + actions on image
            # make imgs into gif
            save_tensor_as_gif(images, gif_path, tot_rew, torch.cat(vals, dim=1), succ_step, engaged_mask)
            step_num = int(ckpt_path.split("/")[-1][6:-3])
            
            wandb.init(
                project=wandb_project,  # Specify your project
                id=run_id, 
                resume="must"
            )
            wandb.log(
                { 
                    'Checkpoint videos':wandb.Video(
                        data_or_path=gif_path,
                        caption=f'Agent at Checkpoint {step_num}',
                        fps=args_cli.policy_hz,
                        format='gif'
                    ),
                    "env_step":step_num, 
                    "exp_step":step_num // 256
                }
            )
            wandb.finish()


if __name__ == "__main__":
    main()
    simulation_app.close()