import numpy as np
import json
import os
from grid2op.Reward.baseReward import BaseReward
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction
from grid2op.dtypes import dt_float
from l2rpn_baselines.utils import GymEnvWithHeuristics, GymEnvWithRecoWithDN
from l2rpn_baselines.PPO_SB3 import evaluate
from typing import List, Dict, Tuple
from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace, GymEnv
from l2rpn_baselines.PPO_SB3.utils import SB3Agent
import gymnasium

with open("preprocess_obs.json", "r", encoding="utf-8") as f:
    obs_space_kwargs = json.load(f)
with open("preprocess_act.json", "r", encoding="utf-8") as f:
    act_space_kwargs = json.load(f) 

class CustomGymEnv(GymEnvWithRecoWithDN):
    """ 
    
    For this, you might want to have a look at: 
      - https://grid2op.readthedocs.io/en/latest/parameters.html#grid2op.Parameters.Parameters.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION
      - https://grid2op.readthedocs.io/en/latest/action.html#grid2op.Action.BaseAction.limit_curtail_storage
    
    This really helps the training, but you cannot change
    this parameter when you evaluate your agent, so you need to rely
    on act.limit_curtail_storage(...) before you give your action to the
    environment
    """
    def __init__(self, env_init, *args, reward_cumul="init", safe_max_rho=0.9, curtail_margin=30, **kwargs):
        super().__init__(env_init, reward_cumul=reward_cumul, safe_max_rho=safe_max_rho, *args, **kwargs)
        self._curtail_margin = curtail_margin
        
    def fix_action(self, grid2op_action, g2op_obs):
        # We try to limit to end up with a "game over" because actions on curtailment or storage units.
        # this is "required" because we use curtailment and action on storage units
        # but the main goal is to 
        grid2op_action.limit_curtail_storage(g2op_obs, margin=self._curtail_margin)
        return grid2op_action

class PenalizeSetpointPosReward(BaseReward):
    """
    Reward based on lines capacity usage
    Returns max reward if no current is flowing in the lines
    Returns min reward if all lines are used at max capacity

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import LinesCapacityReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "rte_case14_realistic"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=LinesCapacityReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the LinesCapacityReward class

    """

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        obs = env.get_obs(_do_copy=False)
        n_connected = dt_float(obs.line_status.sum())
        usage = obs.rho[obs.line_status == True].sum()
        usage = np.clip(usage, 0.0, float(n_connected))
        rew_lines_capacity = np.interp(
            n_connected - usage,
            [dt_float(0.0), float(n_connected)],
            [self.reward_min, self.reward_max],
        )
        
        # rew_curtail = np.sum(np.abs(1 - obs.curtailment_limit[env.gen_renewable])) / env.gen_renewable.sum()
        rew_curtail = np.sum(obs.curtailment_limit[env.gen_renewable]) / env.gen_renewable.sum()
        rew_storage = 0
        
        reward = 0.5 * rew_lines_capacity + 0.5 * rew_curtail # empirically rew_lines_capacity is around 0.7 - 0.9, rew_curtail is around 0.3 - 0.4
        
        return reward
    
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class GymEnvWithSetPoint(GymEnvWithHeuristics): 
    def __init__(self, env_init, *args, reward_cumul="init", safe_max_rho=0.9, curtail_margin=30, very_safe_max_rho=0.9, ratio_max_storage_power=1., weight_penalization_storage=0.25, ind=1, 
                 init_storage_capa_ratio=(0.2, 0.8),
                 ratio_setpoint=(0.1, 3.),
                 smooth=12,
                 penality_storage_bound = None,
                 **kwargs):
        super().__init__(env_init=env_init, *args, reward_cumul=reward_cumul, **kwargs)
        self._safe_max_rho = safe_max_rho
        self._curtail_margin = curtail_margin
        self._very_safe_max_rho = very_safe_max_rho if (very_safe_max_rho < safe_max_rho) else safe_max_rho
        self.ind = ind
        self.smooth = smooth
        self._ratio_max_storage_power = ratio_max_storage_power
        self.init_storage_capa_ratio = np.clip(init_storage_capa_ratio, 0, 1)
        self.ratio_setpoint = ratio_setpoint
        self.weight_penalization_storage = weight_penalization_storage
        self.penality_storage_bound = penality_storage_bound
        self.storage_setpoint = np.zeros((self.init_env.max_episode_duration()+1, self.init_env.n_storage))
        
        
    def fix_action(self, grid2op_action, g2op_obs):
        # We try to limit to end up with a "game over" because actions on curtailment or storage units.
        # this is "required" because we use curtailment and action on storage units
        # but the main goal is to 
        grid2op_action.limit_curtail_storage(g2op_obs, margin=self._curtail_margin)
        return grid2op_action

    def reset(self, seed=None, return_info=True, options=None, return_all=False):
        # gym_obs, reward, done, info = super().reset(seed=seed, options=options, return_all=True)
        
        if hasattr(type(self), "_gymnasium") and type(self)._gymnasium:
            return_info = True
            
        done = True
        info = {}  # no extra information provided !
        while done:
            # super()._aux_reset(seed, return_info, options)  # reset the scenario
            super()._aux_reset_new(seed, options)
            g2op_obs = self.init_env.get_obs()  # retrieve the observation
            reward = self.init_env.reward_range[0]  # the reward at first step is always minimal
            
            # perform the "heuristics" steps
            g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, False, info)
            
            # convert back the observation to gym
            if not done:
                self._previous_obs = g2op_obs
                gym_obs = self.observation_space.to_gym(g2op_obs)
                break
                
        self.storage_setpoint = self._get_target_random_walk()
        gym_obs = self._update_gym_obs(gym_obs)
        if return_all:
            return gym_obs, reward, done, info
        elif return_info:
            return gym_obs, info
        else:
            return gym_obs

    def _update_gym_obs(self, gym_obs):
        # The "storage_setpoint" part of the observation is 0 by default. 
        # This method replaces the zeros with the correct values.
        try:
            # If gym_obs is a dictionary
            gym_obs["storage_setpoint"] = self.storage_setpoint[self.init_env.nb_time_step, :] 
            return gym_obs
        except Exception:
            # If gym_obs is an array
            dims = [0] + self._observation_space._dims 
            idx = np.where(np.array(self._observation_space._attr_to_keep)=="storage_setpoint")[0][0]
            gym_obs[dims[idx]:dims[idx + 1]] = self.storage_setpoint[self.init_env.nb_time_step, :]  
            return gym_obs
    
    def _update_reward(self, g2op_obs, reward):
        # The "PenalizationStorage" term in the reward depends on the storage setpoint of the previous step.
        # The later is available only in the gymenv, that's why we can't compute it with a classic grid2op reward class.
        # This method computes the "PenalizationStorage" term and substracts it to the grid2op reward in input.
        if self.init_env.nb_time_step == 0:
            raise ValueError("You are still at timestep 0, hence there is no previous setpoint to update the reward")
        Emin, Emax = self.init_env.storage_Emin, self.init_env.storage_Emax
        storage_charge = g2op_obs.storage_charge
        storage_setpoint_mwh = self.storage_setpoint[self.init_env.nb_time_step - 1, :] * (Emax - Emin) + Emin
       
        ind = self.ind
        k = 1/(0.5)**ind
        difference = k * np.mean(np.power(np.abs(storage_setpoint_mwh - storage_charge)/(Emax-Emin), ind)) # is around 1 in average if acting randomly
        
        penality = self.weight_penalization_storage * difference
        if self.penality_storage_bound is not None:
            penality = min(penality, self.penality_storage_bound)
        reward = reward - penality
        
        return reward
        
    
    def apply_heuristics_actions(self,
                                 g2op_obs: BaseObservation,
                                 reward: float,
                                 done: bool,
                                 info: Dict ) -> Tuple[BaseObservation, float, bool, Dict]:
        """This function implements the "logic" behind the heuristic part. Unless you have a particular reason too, you
        probably should not modify this function.
        
        If you modify it, you should also modify the way the agent implements it (remember: this function is used 
        at training time, the "GymAgent" part is used at inference time. Both behaviour should match for the best
        performance).

        While there are "heuristics" / "expert rules" / etc. this function should perform steps in the underlying grid2op
        environment.
        
        It is expected to return when:
        
        - either the flag `done` is ``True`` 
        - or the neural network agent is asked to perform action on the grid
        
        The neural network agent will receive the outpout of this function. 
        
        Parameters
        ----------
        g2op_obs : BaseObservation
            The grid2op observation.
            
        reward : ``float``
            The reward
            
        done : ``bool``
            The flag that indicates whether the environment is over or not.
            
        info : Dict
            Other information flags

        Returns
        -------
        Tuple[BaseObservation, float, bool, Dict]
            It should return `obs, reward, done, info`(same as a single call to `grid2op_env.step(grid2op_act)`)
            
            Then, this will be transmitted to the neural network agent (but before the observation will be 
            transformed to a gym observation thanks to the observation space.)
            
        """
        need_action = True
        res_reward = reward
        
        tmp_reward = reward
        tmp_info = info
        while need_action:
            need_action = False
            g2op_actions = self.heuristic_actions(g2op_obs, tmp_reward, done, tmp_info)
            for g2op_act in g2op_actions:
                need_action = True
                tmp_obs, tmp_reward, tmp_done, tmp_info = self.init_env.step(g2op_act)
                tmp_reward =  self._update_reward(tmp_obs, tmp_reward)
                g2op_obs = tmp_obs
                done = tmp_done
                
                if self._reward_cumul == "max":
                    res_reward = max(tmp_reward, res_reward)
                    if "rewards" in tmp_info.keys() and not tmp_done:
                        if "rewards" in info.keys():
                            dict_rewards = info["rewards"]
                            tmp_dict_rewards = tmp_info["rewards"]
                            info["rewards"] = {rew_name: max(rew, tmp_dict_rewards[rew_name]) for rew_name, rew in dict_rewards.items()}
                        else:
                            info["rewards"] = tmp_info["rewards"]
                elif self._reward_cumul == "sum":
                    res_reward += tmp_reward
                    if "rewards" in tmp_info.keys() and not tmp_done:
                        if "rewards" in info.keys():
                            dict_rewards = info["rewards"]
                            tmp_dict_rewards = tmp_info["rewards"]
                            info["rewards"] = {rew_name: rew + tmp_dict_rewards[rew_name] for rew_name, rew in dict_rewards.items()}
                        else:
                            info["rewards"] = tmp_info["rewards"]
                elif self._reward_cumul == "last":
                    res_reward = tmp_reward
                    if "rewards" in tmp_info.keys() and not tmp_done:
                        info["rewards"] = tmp_info["rewards"]
                    
                    
                if tmp_done:
                    break
            if done:
                break
        return g2op_obs, res_reward, done, info
    
    def heuristic_actions(self, g2op_obs, reward, done, info) -> List[BaseAction]:
        """To match the description of the environment, this heuristic will:
        
        - return the list of all the powerlines that can be reconnected if any
        - return a list containing an action to get closer to the storage setpoint if the grid is very safe
        - return the list "[do nothing]" if the grid is safe
        - return the empty list (signaling the agent should take control over the heuristics) otherwise

        Parameters
        ----------
        See parameters of :func:`GymEnvWithHeuristics.heuristic_actions`

        Returns
        -------
        See return values of :func:`GymEnvWithHeuristics.heuristic_actions`
        """
        to_reco = (g2op_obs.time_before_cooldown_line == 0) & (~g2op_obs.line_status)
        res = []
        if np.any(to_reco):
            # reconnect something if it can be
            reco_id = np.where(to_reco)[0]
            for line_id in reco_id:
                g2op_act = self.init_env.action_space({"set_line_status": [(line_id, +1)]})
                res.append(g2op_act)
        elif g2op_obs.rho.max() <= self._very_safe_max_rho:
            Emin, Emax = self.init_env.storage_Emin, self.init_env.storage_Emax
            storage_charge = g2op_obs.storage_charge
            storage_setpoint_mwh = self.storage_setpoint[self.init_env.nb_time_step - 1, :] * (Emax - Emin) + Emin
            set_storage = np.clip(- (storage_charge - storage_setpoint_mwh) *12. , - self._ratio_max_storage_power * self.init_env.storage_max_p_prod, self._ratio_max_storage_power * self.init_env.storage_max_p_prod)
            res = [self.init_env.action_space({"set_storage": set_storage})]
        elif g2op_obs.rho.max() <= self._safe_max_rho:
            # play do nothing if there is "no problem" according to the "rule of thumb"
            res = [self.init_env.action_space()]
        return res
    
            
    def step(self, gym_action):
        g2op_act_tmp = self.action_space.from_gym(gym_action)
        g2op_act = self.fix_action(g2op_act_tmp, self._previous_obs)
        if hasattr(type(self), "_gymnasium") and type(self)._gymnasium:
            g2op_obs, reward, done, info = self.init_env.step(g2op_act)
            reward = self._update_reward(g2op_obs, reward)
            if not done:
                g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, done, info)
            self._previous_obs = g2op_obs
            gym_obs = self._observation_space.to_gym(g2op_obs)
            gym_obs = self._update_gym_obs(gym_obs)
            truncated = False
            return gym_obs, float(reward), done, truncated, info
        else:
            g2op_obs, reward, done, info = self.init_env.step(g2op_act)
            reward = self._update_reward(g2op_obs, reward)
            if not done:
                g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, done, info)
            self._previous_obs = g2op_obs
            gym_obs = self._observation_space.to_gym(g2op_obs)
            gym_obs = self._update_gym_obs(gym_obs)
            return gym_obs, float(reward), done, info

    def _get_target_random_walk(self):
        tmp_ = self.init_env.space_prng.uniform(-1, 1, (self.init_env.max_episode_duration() + 1, self.init_env.n_storage))
        try:
            _ = iter(self.init_storage_capa_ratio)
            init_storage_capa_ratio = self.init_env.space_prng.uniform(*self.init_storage_capa_ratio, 1)
        except TypeError as te:
            init_storage_capa_ratio = self.init_storage_capa_ratio

        try:
            _ = iter(self.ratio_setpoint)
            ratio_setpoint = self.init_env.space_prng.uniform(*self.ratio_setpoint, 1)
        except TypeError as te:
            ratio_setpoint = self.ratio_setpoint

        tmp_ *= self.init_env.storage_max_p_prod * 1 / 12.0 * ratio_setpoint / self.init_env.storage_Emax
        if self.smooth is not None:
            for s_id in range(self.init_env.n_storage):
                y = tmp_[:, s_id]
                tmp_[self.smooth - 1:, s_id] = moving_average(y, self.smooth)

        tmp_ = np.cumsum(tmp_, axis=0) + init_storage_capa_ratio
        tmp_ = np.clip(tmp_, 0, 1)
        return tmp_


class GymEnvWithSetPointRemoveCurtail(GymEnvWithSetPoint): 
    def __init__(self, env_init, *args, reward_cumul="sum", safe_max_rho=0.9, curtail_margin=30, weight_penalization_storage=0.25, ind=1, 
                 penality_storage_bound=None,
                 very_safe_max_rho=0.85,
                 n_gen_to_uncurt=1,
                 ratio_to_uncurt=0.05,
                 margin=0.,
                 less_or_more="more",
                 init_storage_capa_ratio=(0.2, 0.8),
                 ratio_setpoint=(0.1, 3.),
                 smooth=12,
                 **kwargs):
        super().__init__(env_init=env_init, *args, reward_cumul=reward_cumul, safe_max_rho=safe_max_rho, curtail_margin=curtail_margin, weight_penalization_storage=weight_penalization_storage, ind=ind,
                         penality_storage_bound=penality_storage_bound,
                         init_storage_capa_ratio=init_storage_capa_ratio,
                         ratio_setpoint=ratio_setpoint,
                         smooth=smooth,
                         **kwargs)
        self._very_safe_max_rho = very_safe_max_rho if (very_safe_max_rho < safe_max_rho) else safe_max_rho
        self._n_gen_to_uncurt = n_gen_to_uncurt
        self._ratio_to_uncurt = ratio_to_uncurt
        self._margin = margin
        self._less_or_more = less_or_more
    
    def heuristic_actions(self, g2op_obs, reward, done, info) -> List[BaseAction]:
        """To match the description of the environment, this heuristic will:
        
        - return the list of all the powerlines that can be reconnected if any
        - return the list "[do nothing]" is the grid is safe
        - return the empty list (signaling the agent should take control over the heuristics) otherwise

        Parameters
        ----------
        See parameters of :func:`GymEnvWithHeuristics.heuristic_actions`

        Returns
        -------
        See return values of :func:`GymEnvWithHeuristics.heuristic_actions`
        """
        to_reco = (g2op_obs.time_before_cooldown_line == 0) & (~g2op_obs.line_status)
        res = []
        if np.any(to_reco):
            # reconnect something if it can be
            reco_id = np.where(to_reco)[0]
            for line_id in reco_id:
                g2op_act = self.init_env.action_space({"set_line_status": [(line_id, +1)]})
                res.append(g2op_act)
        elif g2op_obs.rho.max() <= self._safe_max_rho:
            # play do nothing if there is "no problem" according to the "rule of thumb"
            Emin, Emax = self.init_env.storage_Emin, self.init_env.storage_Emax
            storage_charge = g2op_obs.storage_charge
            storage_setpoint_mwh = self.storage_setpoint[self.init_env.nb_time_step - 1, :] * (Emax - Emin) + Emin
            set_storage = np.clip(- (storage_charge - storage_setpoint_mwh) *12. , -self.init_env.storage_max_p_prod, self.init_env.storage_max_p_prod)
            act_dict = {"set_storage": set_storage}
            # remove useless curtailment      
            curtail_vect = np.zeros(self.init_env.n_gen) - 1.
            useless_limits = (g2op_obs.curtailment_limit - g2op_obs.gen_p_before_curtail / g2op_obs.gen_pmax >= self._margin) & (g2op_obs.curtailment_limit < 1)
            if useless_limits.sum() > 0:
                gen_ids_useless_curtail = np.where(useless_limits)
                curtail_vect = np.zeros(self.init_env.n_gen) - 1.
                curtail_vect[gen_ids_useless_curtail] = 1.
            # increase the limit for the most curtailed generators
            limits = (g2op_obs.curtailment > 0.) & (g2op_obs.curtailment_limit < 1)
            if limits.sum() > 0 and g2op_obs.rho.max() <= self._very_safe_max_rho:
                if self._less_or_more == "more":
                    gen_ids_to_uncurtail = np.argsort(g2op_obs.curtailment_mw)[::-1][:min(self._n_gen_to_uncurt, g2op_obs.curtailment_mw.shape[0])]
                elif self._less_or_more == "less":
                    gen_ids_to_uncurtail = np.argsort(g2op_obs.curtailment_mw)[:min(self._n_gen_to_uncurt, g2op_obs.curtailment_mw.shape[0])]
                curtail_vect[gen_ids_to_uncurtail] = np.clip(g2op_obs.curtailment_limit[gen_ids_to_uncurtail] + self._ratio_to_uncurt, 0., 1)
            act_dict["curtail"]  = curtail_vect
            res = [self.init_env.action_space(act_dict)]
        return res


def load_agent(env, load_path, name,
               gymenv_class,
               gymenv_kwargs=None,
               obs_space_kwargs=None,
               act_space_kwargs=None,
               iter_num=None,
               return_gymenv=False,
             ):


    if obs_space_kwargs is None:
        obs_space_kwargs = {}
    if act_space_kwargs is None:
        act_space_kwargs = {}

    # load the attributes kept
    my_path = os.path.join(load_path, name)        
    with open(os.path.join(my_path, "obs_attr_to_keep.json"), encoding="utf-8", mode="r") as f:
        obs_attr_to_keep = json.load(fp=f)
    with open(os.path.join(my_path, "act_attr_to_keep.json"), encoding="utf-8", mode="r") as f:
        act_attr_to_keep = json.load(fp=f)

    if gymenv_kwargs is None:
        gymenv_kwargs = {}
    gymenv = gymenv_class(env, **gymenv_kwargs)
    gymenv.observation_space.close()
    if obs_space_kwargs is None:
        obs_space_kwargs = {}
    gymenv._observation_space = BoxGymObsSpace(env.observation_space,
                                                attr_to_keep=obs_attr_to_keep,
                                                **obs_space_kwargs)
    gymenv.action_space.close()
    if act_space_kwargs is None:
        act_space_kwargs = {}
    gymenv.action_space = BoxGymActSpace(env.action_space,
                                          attr_to_keep=act_attr_to_keep,
                                          **act_space_kwargs)
    # create the action and observation space
    # gym_observation_space =  BoxGymObsSpace(env.observation_space,
    #                                         attr_to_keep=obs_attr_to_keep,
    #                                         **obs_space_kwargs)
    # gym_action_space = BoxGymActSpace(env.action_space,
    #                                   attr_to_keep=act_attr_to_keep,
    #                                   **act_space_kwargs)
    
    if os.path.exists(os.path.join(load_path, ".normalize_act")):
        for attr_nm in act_attr_to_keep:
            if (("multiply" in act_space_kwargs and attr_nm in act_space_kwargs["multiply"]) or 
                ("add" in act_space_kwargs and attr_nm in act_space_kwargs["add"]) 
               ):
                continue
            gymenv.action_space.normalize_attr(attr_nm)

    if os.path.exists(os.path.join(load_path, ".normalize_obs")):
        for attr_nm in obs_attr_to_keep:
            if (("divide" in obs_space_kwargs and attr_nm in obs_space_kwargs["divide"]) or 
                ("subtract" in obs_space_kwargs and attr_nm in obs_space_kwargs["subtract"]) 
               ):
                continue
            gymenv._observation_space.normalize_attr(attr_nm)
    
    # gymenv = None
    # if gymenv_class is not None and issubclass(gymenv_class, GymEnvWithHeuristics):
    #     if gymenv_kwargs is None:
    #         gymenv_kwargs = {}
    #     gymenv = gymenv_class(env, **gymenv_kwargs)
        
    #     gymenv.action_space.close()
    #     gymenv.action_space = gym_action_space
        
    #     gymenv.observation_space.close()
    #     gymenv._observation_space = gym_observation_space
        
    # define the gym environment from the grid2op env
        
    gymenv.observation_space = gymnasium.spaces.Box(shape=gymenv._observation_space.shape,
                                                     low=gymenv._observation_space.low,
                                                     high=gymenv._observation_space.high,
                                                     dtype=gymenv._observation_space.dtype,
                                                    )
    def to_gym2(self, *args, **kwargs):
        return gymenv._observation_space.to_gym(*args, **kwargs)
    setattr(type(gymenv.observation_space), "to_gym", to_gym2)
    
    def close2(self):
        return gymenv._observation_space.close()
    setattr(type(gymenv.observation_space), "close", close2)
        
    # create a grid2gop agent based on that (this will reload the save weights)
    full_path = os.path.join(load_path, name)
    
    # def get_act2(self, gym_obs, reward, done):
    #     if hasattr(type(self.gymenv), "_update_gym_obs"):
    #         gym_obs = self.gymenv._update_gym_obs(gym_obs)
    #     action, _ = self.nn_model.predict(gym_obs, deterministic=True) 
    #     return action
    
    # setattr(SB3Agent, "get_act", get_act2)
    
    grid2op_agent = SB3Agent(env.action_space,
                             gymenv.action_space,
                             gymenv.observation_space,
                             nn_path=os.path.join(full_path, name),
                             gymenv=gymenv,
                             iter_num=iter_num,
                             )
    if return_gymenv:
        return grid2op_agent, gymenv
    else:
        return grid2op_agent
    
    
    
class GymEnvWithSetPointRecoDN(GymEnvWithSetPoint): 
    def __init__(self, env_init, *args, reward_cumul="sum", safe_max_rho=0.9, curtail_margin=30, weight_penalization_storage=0.25, ind=1, **kwargs):
        super().__init__(env_init=env_init, *args, reward_cumul=reward_cumul, safe_max_rho=safe_max_rho, curtail_margin=curtail_margin, weight_penalization_storage=weight_penalization_storage, ind=ind,**kwargs)
    
    def heuristic_actions(self, g2op_obs, reward, done, info) -> List[BaseAction]:
        """To match the description of the environment, this heuristic will:
        
        - return the list of all the powerlines that can be reconnected if any
        - return the list "[do nothing]" is the grid is safe
        - return the empty list (signaling the agent should take control over the heuristics) otherwise

        Parameters
        ----------
        See parameters of :func:`GymEnvWithHeuristics.heuristic_actions`

        Returns
        -------
        See return values of :func:`GymEnvWithHeuristics.heuristic_actions`
        """
        to_reco = (g2op_obs.time_before_cooldown_line == 0) & (~g2op_obs.line_status)
        res = []
        if np.any(to_reco):
            # reconnect something if it can be
            reco_id = np.where(to_reco)[0]
            for line_id in reco_id:
                g2op_act = self.init_env.action_space({"set_line_status": [(line_id, +1)]})
                res.append(g2op_act)
        elif g2op_obs.rho.max() <= self._safe_max_rho:
            # play do nothing if there is "no problem" according to the "rule of thumb"
            res = [self.init_env.action_space()]
            
        return res