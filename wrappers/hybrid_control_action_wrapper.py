from gym import Env, spaces
import gym
from gym.spaces import Box
import torch
import envs.factory.factory_control as fc
try:
    import isaacsim.core.utils.torch as torch_utils
except:
    import omni.isaac.core.utils.torch as torch_utils
import numpy as np

class HybridControlActionWrapper(gym.ActionWrapper):
    """
    Use this wrapper changes the task to using a hybrid control env
        in order to do this within the IsaacLab + Factory framework
        we override the _apply_action function which is called by IsaacLab
        env at each physics step and we overwrite (if we want to) generate_ctrl_signals
        which is called in factory env 

        This is simple hybrid control, we take the error in force and multiply it by a gain
        that theoretically converts it to the equivalent of pose error. This is then added 
        to the change in pose and fed to the initial env.  The logical inverse of the selection
        "matrix" (really a vector) is used to select between force control and pose control in 
        a given dimension 
    """

    def __init__(
            self, 
            env,
            history=1):
        super().__init__(env)
        # get these for factory env
        self.old_randomize_initial_state = self.env.unwrapped.randomize_initial_state
        self.env.unwrapped.randomize_initial_state = self.randomize_initial_state

        #self.old_pre_physics_step = self.env.unwrapped._pre_physics_step
        self.env.unwrapped._pre_physics_step = self._pre_physics_step
        self.env.unwrapped._apply_action = self._apply_action
        
        self.act_size = 12 * history

        # previous action is now larger
        #self.unwrapped.cfg.state_space += self.act_size - self.unwrapped.cfg.action_space
        #self.unwrapped.cfg.observation_space += self.act_size - self.unwrapped.cfg.action_space
        self.unwrapped.cfg.action_space = self.act_size

        # reconfigure spaces to match
        self.unwrapped._configure_gym_env_spaces()

        self.new_action = torch.zeros((self.unwrapped.num_envs, 6), device = self.unwrapped.device)
        self.unwrapped.prev_actions = torch.zeros_like(self.new_action)
        self.unwrapped.actions = torch.zeros_like(self.new_action)
        self.sel_matrix = torch.zeros((self.unwrapped.num_envs, 3), dtype=bool, device=self.unwrapped.device)
        self.force_action = torch.zeros((self.unwrapped.num_envs, 3), device=self.unwrapped.device)
        self.pos_action = torch.zeros_like(self.force_action)
        
        self.kp = torch.tensor(self.unwrapped.cfg.ctrl.default_task_force_gains, device=self.unwrapped.device).repeat(
            (self.unwrapped.num_envs, 1)
        )
        self.force_threshold = torch.tensor(self.unwrapped.cfg.ctrl.force_action_threshold, device=self.unwrapped.device).repeat(
            (self.unwrapped.num_envs, 1)
        )

        self.force_action_bounds = torch.tensor(self.unwrapped.cfg.ctrl.force_action_bounds, device=self.unwrapped.device)

    def randomize_initial_state(self, env_ids):
        self.old_randomize_initial_state(env_ids)
        # set a random force gain
        # this is setup so if we want to vary the gain we can add it here

    def _calc_ctrl_force(self, min_idx=9, max_idx=12):
        self.force_action = self.unwrapped.actions[:,min_idx:max_idx] * self.force_threshold
        self.force_action = torch.clip(
            self.force_action, -self.force_action_bounds, self.force_action_bounds
        )  

    def _pre_physics_step(self, action):
        self.sel_matrix = action[:,:3] #torch.where(action[:,:3] > 0, True, False)
        
        self.unwrapped.extras[f'Hybrid Controller / Force Control X'] = action[:,0]#.mean()
        self.unwrapped.extras[f'Hybrid Controller / Force Control Y'] = action[:,1]#.mean()
        self.unwrapped.extras[f'Hybrid Controller / Force Control Z'] = action[:,2]#.mean()
        #new_action = self.pose_action 
        # we inturrpret force action as a difference in force from current force reading
        # this means that goal_force = force_action + current_force
        # the error term is error = goal_force - current_force => error = force_action
        #new_action[:,0:3] += self.kp * ~self.sel_matrix * self.force_action
        self.unwrapped.prev_action = self.unwrapped.actions
        self.unwrapped.actions = action

        env_ids = self.unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.env.unwrapped._reset_buffers(env_ids)

        self.unwrapped.actions = (
            self.unwrapped.cfg.ctrl.ema_factor * action.clone().to(self.unwrapped.device) + (1 - self.unwrapped.cfg.ctrl.ema_factor) * self.unwrapped.actions
        )
        
        # set goal 1 time
        self.env.unwrapped._calc_ctrl_pos(min_idx=3, max_idx=6)
        self.pos_action = self.unwrapped.ctrl_target_fingertip_midpoint_pos.clone()
        self.env.unwrapped._calc_ctrl_quat(min_idx=6, max_idx=9)
        self._calc_ctrl_force()

        # update current values
        #self._compute_intermediate_values(dt=self.physics_dt) 

        # smoothness metrics
        self.unwrapped.ep_ssv += (torch.linalg.norm(self.unwrapped.ee_linvel_fd, axis=1))
        if self.unwrapped.use_ft or self.unwrapped.fragile:
            self.unwrapped.ep_sum_force += torch.linalg.norm(self.unwrapped.robot_force_torque[:,:3], axis=1)
            self.unwrapped.ep_sum_torque += torch.linalg.norm(self.unwrapped.robot_force_torque[:,3:], axis=1)
            self.unwrapped.ep_max_force = torch.max(self.unwrapped.ep_max_force, torch.linalg.norm(self.unwrapped.robot_force_torque[:,:3], axis=1 ))
            self.unwrapped.ep_max_torque = torch.max(self.unwrapped.ep_max_torque, torch.linalg.norm(self.unwrapped.robot_force_torque[:,3:]))

    def action(self, action):
        return action

    def _apply_action(self):
        # Get current yaw for success checking.
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.unwrapped.fingertip_midpoint_quat)
        self.unwrapped.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.unwrapped.last_update_timestamp < self.unwrapped._robot._data._sim_timestamp:
            self.env.unwrapped._compute_intermediate_values(dt=self.unwrapped.physics_dt)

        self.unwrapped.ctrl_target_gripper_dof_pos = 0.0
        self.env.unwrapped.generate_ctrl_signals()
        force_error =  self.sel_matrix * ( self.unwrapped.robot_force_torque[:,:3] - self.force_action ) 

        # set control target
        self.unwrapped.ctrl_target_fingertip_midpoint_pos = self.pos_action + self.kp * force_error
        
        self.env.unwrapped.generate_ctrl_signals()