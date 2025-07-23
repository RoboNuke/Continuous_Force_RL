import torch
import gym
import envs.factory.factor_control as fc

class VICActionWrapper(gym.Wrapper):
    """
    Use this wrapper to perform VIC under the following control law:
        u = x = K_p * (x_d - x) - K_d * x_dot

    Where K_p and x_d are always part of the action space, and K_d can also be icluded
    """

    def __init__(
            self,
            env,
            ctrl_damping=False
    ):
        super().__init__(env)

        
        # have to fix function references
        self.old_pre_physics_step = self.env.unwrapped._pre_physics_step
        self.env.unwrapped._pre_physics_step = self._pre_physics_step

        self.old_apply_action = self.env.unwrapped._apply_action
        self.env.unwrapped._apply_action = self._apply_action

        # define action space
        self.unwrapped.cfg.action_space = 2 * 6 + (6 if ctrl_damping else 0)
        self.unwrapped._configure_gym_env_spaces()

        # get cfg refs
        self.cfg_task = self.unwrapped.cfg_task
        self.cfg = self.unwrapped.cfg

        # get useful constants
        self.ctrl_damping= ctrl_damping
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device

        # allocate memory
        self.kp = torch.zeros((self.num_envs, 6), device=self.device)
        self.prop_thresh = torch.tensor(self.cfg.ctrl.vic_prop_action_threshold, device=self.device)
        self.prop_bounds = torch.tensor(self.cfg.ctrl.vic_prop_action_bounds, device=self.device)
        
        self.kd = torch.zeros_like(self.kp)
        self.damp_thresh = torch.tensor(self.cfg.ctrl.vic_damp_action_threshold, device=self.device)
        self.damp_bounds = torch.tensor(self.cfg.ctrl.vic_damp_action_bounds, device=self.device)


    def _pre_physics_step(self, action):

        self.old_pre_physics_step(action)

        # unwrap control gains
        self.kp = action[:,6:12] * self.prop_thresh[None,:]

        if self.ctrl_damping:
            self.kd = action[:,12:] * self.damp_thresh[None,:]
        else:
            # assuming a 1kg virtual mass at EE, then critically damped is as follows:
            self.kd = 2 * torch.sqrt(self.kp)

        # bound them
        self.kp = torch.clamp(
            self.kp,
            min=-self.prop_bounds,
            max=self.prop_bounds
        )

        self.kd = torch.clamp(
            self.kd,
            min=-self.damp_bounds,
            max=self.damp_bounds
        )

        # track gains
        for axis in ['x','y','z','rx','ry','rz']:
            self.unwrapped.extras[f'Controller / {axis} Gain'] = self.kp[:,0]
            self.unwrapped.extras[f'Controller / {axis} Damping'] = self.kd[:,0]

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Get current yaw for success checking.
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.unwrapped.fingertip_midpoint_quat)
        self.unwrapped.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.unwrapped.last_update_timestamp < self.unwrapped._robot._data._sim_timestamp:
            self.unwrapped._compute_intermediate_values(dt=self.unwrapped.physics_dt)

        self.unwrapped.ctrl_target_gripper_dof_pos = 0.0

        
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        #print(self.arm_mass_matrix)
        # joint_torque is the null space control terms
        self.unwrapped.joint_torque, self.unwrapped.applied_wrench = fc.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.unwrapped.joint_pos,
            dof_vel=self.unwrapped.joint_vel_fd,
            fingertip_midpoint_pos=self.unwrapped.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.unwrapped.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.unwrapped.ee_linvel_fd,
            fingertip_midpoint_angvel=self.unwrapped.ee_angvel_fd,
            jacobian=self.unwrapped.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.unwrapped.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=self.unwrapped.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.unwrapped.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.kp,
            task_deriv_gains=self.kd,
            device=self.device,
        ) # this calculates osc on goal pose and biases with null space

        # set target for gripper joints to use physx's PD controller
        self.unwrapped.ctrl_target_joint_pos[:, 7:9] = self.unwrapped.ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self.unwrapped._robot.set_joint_position_target(self.unwrapped.ctrl_target_joint_pos) # this is not used since actuator gains = 0
        self.unwrapped._robot.set_joint_effort_target(self.joint_torque)

        

        
        
        
