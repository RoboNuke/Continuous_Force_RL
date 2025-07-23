





class ImpedanceActionGMM(Distribution):
    def __init__(
            self,
            pose_dist: Distribution,
            kp_dist: Distribution,
            kd_dist: Distribution = None,
            pos_scale: float = 1.0,
            rot_scale: float = 1.0,
            prop_scale: float=1.0,
            damp_scale: float=1.0,
            validate_args: Optional[bool] = None
    ):
        super().__init__(validate_args=validate_args)
        
        self.pose_dist = pose_dist
        self.kp_dist = kp_dist
        self.w_pos = pos_scale
        self.w_rot = rot_scale
        self.w_prop = prop_scale
        self.kd_dist = kd_dist
        if kd_dist is not None:
            self.kd_dist = kd_dist
            self.w_damp = damp_scale
        
    def rsample(self, sample_shape=torch.size()):

        pose = self.pose_dist.rsample(sample_shape)
        kp = self.pose_dist.rsample(sample_shape)

        if self.kd_dist is None:

            outsamples = torch.cat(
                (
                    pose[:,0:3] / self.w_pos,
                    pose[:,3:6] / self.w_rot,
                    kp / self.w_prop
                ), dim=1
            )
        else:
            kd = self.kd_dist.rsample(sample_shape)

            outsamples = torch.cat(
                (
                    pose[:,0:3] / self.w_pos,
                    pose[:,3:6] / self.w_rot,
                    kp / self.w_prop,
                    kd / self.w_damp
                ), dim=1
            )

        return outsamples

    def log_prob(self, action):

        if self.use_jnt_prob:
            self.log_prob_jnt_dist(action)

    def scale_action(self, action):
        # scale correctly
        pose = action[:,:6] * self.w_pos
        pose[:,3:] *= self.w_rot / self.w_pos
        kp = self.action[:,6:12] * self.w_prop

        if self.kd_dist is None:
            return pose, kp, None
        else:
            kd = action[:,12:18] * self.w_damp
            return pose, kp, kd
            
    def log_prob_jnt_dist(self, action):
        pose, kp, kd = self.scale_action(action)
        
        # add probs
        log_prob = self.pos_dist.log_prob(pose) + self.kp_dist.log_prob(kp)

        if self.kd_dist is not None:
            log_prob += self.kd_dist.log_prob(kd)

        return log_prob

    def log_prob_dist(self, action):
        
    

class ImpedanceMixin(GaussianMixin):
    def __init__(
            self,
            clip_actions: bool = False,
            clip_log_std: bool = True,
            min_log_std: float = -20,
            max_log_std: float = 2,
            reduction: str = "sum",
            role: str = "",
            pos_scale=1.0,
            rot_scale=1.0,
            prop_scale=1.0,
            damp_scale=1.0,
            ctrl_damp=False
    ) -> None:
        super().__init__(clip_actions, clip_log_std, min_log_std, max_log_std, reduction, role)
        self.w_pos = pos_scale
        self.w_rot = rot_scale
        self.w_prop = prop_scale
        self.ctrl_damp = ctrl_damp
        self.w_damp = damp_scale

    def act(
            self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str= ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:

        mean_actions, log_std, outputs = self.compute(inputs, role)

        batch_size = mean_actions.shape[0]

        # create standard normal distribution for pose
        pose_mean = mean_actions[:,:6] * self.w_pos
        pose_mean = mean_actions[:,3:6] * self.w_rot / self.w_pos

        pose_std = log_std[:,:6].exp()

        pose_dist = Normal(loc=pose_mean, scale=pose_std)

        # create a log-normal distribution for kp + kd (if required)
        kp_mean = mean_actions[:,6:12] * self.w_prop
        kp_std = log_std[:,6:12].exp()

        kp_dist = LogNormal(loc = kp_mean, scale = kp_std)

        kd_dist = None
        if self.ctrl_damp:
            kd_mean = mean_actions[:,12:18] * self.w_damp
            kd_std = log_std[:,12:18].exp()

            kd_dist = LogNormal(loc=kd_mean, scale=kd_std)

        self._g_distribution = ImpedanceActionGMM(
            pose_dist,
            kp_dist,
            kd_dist,
            pos_scale=self.w_pos,
            rot_scale=self.w_rot,
            prop_scale=self.w_prop,
            damp_scale=self.w_damp,
        )

        actions = self._g_distribution.sample()
    
        # clip actions
        if self._g_clip_actions:
            actions = torch.clamp(actions, min=self._g_clip_actions_min, max=self._g_clip_actions_max)

        # log of the probability density function
        log_prob = self._g_distribution.log_prob(inputs.get("taken_actions", actions))
        
        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)
        
        outputs["mean_actions"] = mean_actions
        return actions, log_prob, outputs


    
class ImpedanceControlSimBaActor(HybridGMMMixin, Model):
    def __init__(
            self,
            observation_space,
            action_space,
            device, 
            act_init_std = 0.60653066, # -0.5 value used in maniskill 
            actor_n = 2,
            actor_latent=512,
            action_gain=1.0,
            
            clip_actions=False,
            clip_log_std=True, 
            min_log_std=-20, 
            max_log_std=2, 
            reduction="sum",
            prop_scale=1.0,
            damp_scale=1.0,
            pos_scale=1.0,
            rot_scale=1.0,
            ctrl_damp=False
    ):
        Model.__init__(self, observation_space, action_space, device)
        ImpedanceMixin.__init__(
            self,
            clip_actions,
            clip_log_std,
            min_log_std,
            max_log_std,
            reduction,
            pos_scale=pos_scale,
            rot_scale=rot_scale,
            prop_scale=prop_scale,
            kd_scale=kd__scale,
            ctrl_damp=ctrl_damp
        )
        self.action_gain = action_gain
        self.actor_mean = SimBaNet(
            n=actor_n, 
            in_size=self.num_observations, 
            out_size=self.num_actions, 
            latent_size=actor_latent, 
            device=device,
            tan_out=True
        )
        with torch.no_grad():
            #self.actor_mean.output[-2].weight *= 0.1 #1.0 #TODO FIX THIS TO 0.01
            #self.actor_mean.output[-2].bias[:self.force_size] = -1.1
            pass
        
        self.actor_logstd = nn.Parameter(
            torch.ones(1, self.num_actions) * math.log(act_init_std)
        )
        

    def act(self, inputs, role):
        return HybridGMMMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        action_mean = self.action_gain * self.actor_mean(inputs['states'][:,:self.num_observations])
        return action_mean, self.actor_logstd.expand(action_mean.size()[0],-1), {}
    
            
    
