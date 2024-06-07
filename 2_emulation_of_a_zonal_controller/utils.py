import numpy as np
import json
import os
import copy
import warnings
from grid2op.Reward.baseReward import BaseReward
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction
from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace, GymEnv
from grid2op.dtypes import dt_float
from l2rpn_baselines.utils import GymEnvWithHeuristics, GymEnvWithRecoWithDN

from l2rpn_baselines.PPO_SB3.utils import (remove_non_usable_attr,
                                           save_used_attribute,
                                           SB3Agent)

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo import MlpPolicy

from typing import List, Dict, Tuple
import gymnasium

# attributes taken into account in the observation by default
obs_attr_to_keep_default = ["month", "day_of_week", "hour_of_day", "minute_of_hour",
                    "gen_p", "load_p", 
                    "p_or", "rho", "timestep_overflow", "line_status",
                    # dispatch part of the observation
                    "actual_dispatch", "target_dispatch",
                    # storage part of the observation
                    "storage_charge", "storage_power",
                    # curtailment part of the observation
                    "curtailment", "curtailment_limit",  "gen_p_before_curtail",
                    ]
# attributes of the possible actions by default
act_attr_to_keep_default = ["curtail", "set_storage"]

with open("preprocess_obs_old.json", "r", encoding="utf-8") as f:
    obs_space_kwargs_default = json.load(f)
with open("preprocess_act_old.json", "r", encoding="utf-8") as f:
    act_space_kwargs_default = json.load(f) 

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

    def reset(self, seed=None, return_info=True, options=None, return_all=False, reset_until_not_done=True):
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
            if not done or not reset_until_not_done:
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
    
    
def create_gymenv(env,
               gymenv_class,
               gymenv_kwargs=None,
               obs_space_kwargs=None,
               obs_attr_to_keep=None,
               act_space_kwargs=None,
               act_attr_to_keep=None,
               normalize_act=True,
               normalize_obs=True,
             ):

    if obs_space_kwargs is None: obs_space_kwargs = copy.deepcopy(obs_space_kwargs_default)
    if act_space_kwargs is None: act_space_kwargs = copy.deepcopy(act_space_kwargs_default)
    if obs_attr_to_keep is None: obs_attr_to_keep = obs_attr_to_keep_default.copy()
    if act_attr_to_keep is None: act_attr_to_keep = act_attr_to_keep_default.copy()
    
    if issubclass(gymenv_class, GymEnvWithSetPoint):
        if "functs" not in obs_space_kwargs.keys():
            obs_space_kwargs["functs"] = {}
        if "storage_setpoint" not in obs_space_kwargs["functs"]:
            n_storage = env.n_storage # without this line, the PPO model is much heavier (600 Mo)
            obs_space_kwargs["functs"].update({"storage_setpoint": (lambda grid2opobs: np.zeros(n_storage), 0., 1.0, None, None)})
        if obs_attr_to_keep[-1] != "storage_setpoint":
            obs_attr_to_keep.append("storage_setpoint")


    if gymenv_kwargs is None:
        gymenv_kwargs = {}
    gymenv = gymenv_class(env, **gymenv_kwargs)
    gymenv.observation_space.close()
    gymenv._observation_space = BoxGymObsSpace(env.observation_space,
                                                attr_to_keep=obs_attr_to_keep,
                                                **obs_space_kwargs)
    gymenv.action_space.close()
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
    
    if normalize_act:
        for attr_nm in act_attr_to_keep_default:
            if (("multiply" in act_space_kwargs and attr_nm in act_space_kwargs["multiply"]) or 
                ("add" in act_space_kwargs and attr_nm in act_space_kwargs["add"]) 
               ):
                continue
            gymenv.action_space.normalize_attr(attr_nm)

    if normalize_obs:
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
        
    return gymenv
    
def load_agent(env, load_path, name,
               gymenv_class,
               gymenv_kwargs={},
               iter_num=None,
               return_gymenv=False,
               obs_space_kwargs = None,
               act_space_kwargs = None,
               obs_attr_to_keep = None,
               act_attr_to_keep = None,
             ):      
    
    if obs_space_kwargs is None: obs_space_kwargs = copy.deepcopy(obs_space_kwargs_default)
    if act_space_kwargs is None: act_space_kwargs = copy.deepcopy(act_space_kwargs_default)
    if obs_attr_to_keep is None: obs_attr_to_keep = obs_attr_to_keep_default.copy()
    if act_attr_to_keep is None: act_attr_to_keep = act_attr_to_keep_default.copy()
    
    full_path = os.path.join(load_path, name) 
    # whether or not observations and actions are normalized
    normalize_obs = os.path.exists(os.path.join(full_path,".normalize_obs"))
    normalize_act = os.path.exists(os.path.join(full_path,".normalize_act"))
    
    # create the appropriated gym environment
    gymenv = create_gymenv(env,
               gymenv_class,
               gymenv_kwargs=gymenv_kwargs,
               normalize_obs=normalize_obs,
               normalize_act=normalize_act,
               obs_space_kwargs=obs_space_kwargs,
               act_space_kwargs=act_space_kwargs,
               obs_attr_to_keep=obs_attr_to_keep,
               act_attr_to_keep=act_attr_to_keep,
               )
    
    # create a grid2gop agent based on that (this will reload the save weights) 
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
    
    
    
    
def train(env,
          name="PPO_SB3",
          iterations=1,
          save_path=None,
          load_path=None,
          net_arch=None,
          logs_dir=None,
          learning_rate=3e-4,
          checkpoint_callback=None,
          save_every_xxx_steps=None,
          model_policy=MlpPolicy,
          obs_attr_to_keep=copy.deepcopy(obs_attr_to_keep_default),
          obs_space_kwargs=None,
          act_attr_to_keep=copy.deepcopy(act_attr_to_keep_default),
          act_space_kwargs=None,
          policy_kwargs=None,
          normalize_obs=False,
          normalize_act=False,
          gymenv_class=GymEnv,
          gymenv_kwargs=None,
          verbose=True,
          seed=None,  # TODO
          eval_env=None,  # TODO
          **kwargs):
    """
    This function will use stable baselines 3 to train a PPO agent on
    a grid2op environment "env".

    It will use the grid2op "gym_compat" module to convert the action space
    to a BoxActionSpace and the observation to a BoxObservationSpace.

    It is suited for the studying the impact of continuous actions:

    - on storage units
    - on dispatchable generators
    - on generators with renewable energy sources

    Parameters
    ----------
    env: :class:`grid2op.Environment`
        The environment on which you need to train your agent.

    name: ``str```
        The name of your agent.

    iterations: ``int``
        For how many iterations (steps) do you want to train your agent. NB these are not episode, these are steps.

    save_path: ``str``
        Where do you want to save your baseline.

    load_path: ``str``
        If you want to reload your baseline, specify the path where it is located. **NB** if a baseline is reloaded
        some of the argument provided to this function will not be used.

    net_arch:
        The neural network architecture, used to create the neural network
        of the PPO (see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

    logs_dir: ``str``
        Where to store the tensorboard generated logs during the training. ``None`` if you don't want to log them.

    learning_rate: ``float``
        The learning rate, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

    save_every_xxx_steps: ``int``
        If set (by default it's None) the stable baselines3 model will be saved
        to the hard drive each `save_every_xxx_steps` steps performed in the
        environment.

    model_policy: 
        Type of neural network model trained in stable baseline. By default
        it's `MlpPolicy`

    obs_attr_to_keep: list of string
        Grid2op attribute to use to build the BoxObservationSpace. It is passed
        as the "attr_to_keep" value of the
        BoxObservation space (see
        https://grid2op.readthedocs.io/en/latest/gym.html#grid2op.gym_compat.BoxGymObsSpace)
        
    obs_space_kwargs:
        Extra kwargs to build the BoxGymObsSpace (**NOT** saved then NOT restored)

    act_attr_to_keep: list of string
        Grid2op attribute to use to build the BoxGymActSpace. It is passed
        as the "attr_to_keep" value of the
        BoxAction space (see
        https://grid2op.readthedocs.io/en/latest/gym.html#grid2op.gym_compat.BoxGymActSpace)
        
    act_space_kwargs:
        Extra kwargs to build the BoxGymActSpace (**NOT** saved then NOT restored)

    verbose: ``bool``
        If you want something to be printed on the terminal (a better logging strategy will be put at some point)

    normalize_obs: ``bool``
        Attempt to normalize the observation space (so that gym-based stuff will only
        see numbers between 0 and 1)
    
    normalize_act: ``bool``
        Attempt to normalize the action space (so that gym-based stuff will only
        manipulate numbers between 0 and 1)
    
    gymenv_class: 
        The class to use as a gym environment. By default `GymEnv` (from module grid2op.gym_compat)
    
    gymenv_kwargs: ``dict``
        Extra key words arguments to build the gym environment., **NOT** saved / restored by this class
        
    policy_kwargs: ``dict``
        extra parameters passed to the PPO "policy_kwargs" key word arguments
        (defaults to ``None``)
    
    kwargs:
        extra parameters passed to the PPO from stable baselines 3

    Returns
    -------

    baseline: 
        The trained baseline as a stable baselines PPO element.


    .. _Example-ppo_stable_baseline:

    Examples
    ---------

    Here is an example on how to train a ppo_stablebaseline .

    First define a python script, for example

    .. code-block:: python

        import re
        import grid2op
        from grid2op.Reward import LinesCapacityReward  # or any other rewards
        from grid2op.Chronics import MultifolderWithCache  # highly recommended
        from lightsim2grid import LightSimBackend  # highly recommended for training !
        from l2rpn_baselines.PPO_SB3 import train

        env_name = "l2rpn_case14_sandbox"
        env = grid2op.make(env_name,
                           reward_class=LinesCapacityReward,
                           backend=LightSimBackend(),
                           chronics_class=MultifolderWithCache)

        env.chronics_handler.real_data.set_filter(lambda x: re.match(".*00$", x) is not None)
        env.chronics_handler.real_data.reset()
        # see https://grid2op.readthedocs.io/en/latest/environment.html#optimize-the-data-pipeline
        # for more information !

        try:
            trained_agent = train(
                  env,
                  iterations=10_000,  # any number of iterations you want
                  logs_dir="./logs",  # where the tensorboard logs will be put
                  save_path="./saved_model",  # where the NN weights will be saved
                  name="test",  # name of the baseline
                  net_arch=[100, 100, 100],  # architecture of the NN
                  save_every_xxx_steps=2000,  # save the NN every 2k steps
                  )
        finally:
            env.close()

    """
    
    # keep only usable attributes (if default is used)
    act_attr_to_keep = remove_non_usable_attr(env, act_attr_to_keep)
    
    # save the attributes kept
    if save_path is not None:
        my_path = os.path.join(save_path, name)
        dict_to_save = {
            "name": name,
            "load_path": load_path,
            "reward_class": str(type(env.get_reward_instance())),
            "nomalize_obs": normalize_obs,
            "normalize_act": normalize_act,
            "gymenv_class": gymenv_class.__name__,
            "gymenv_kwargs": gymenv_kwargs,
            "n_available_chronics": len(env.chronics_handler.real_data.available_chronics()),
            "iterations": iterations,     
        }
    save_used_attribute(save_path, name, obs_attr_to_keep, act_attr_to_keep)
    

    # define the gym environment from the grid2op env
    env_gym = create_gymenv(env,
               gymenv_class,
               gymenv_kwargs,
               obs_space_kwargs=obs_space_kwargs,
               obs_attr_to_keep=obs_attr_to_keep,
               act_space_kwargs=act_space_kwargs,
               act_attr_to_keep=act_attr_to_keep,
               normalize_act=normalize_act,
               normalize_obs=normalize_obs,
             )


    if normalize_act:
        if save_path is not None:
            with open(os.path.join(my_path, ".normalize_act"), encoding="utf-8", 
                      mode="w") as f:
                f.write("I have encoded the action space !\n DO NOT MODIFY !")

    if normalize_obs:
        if save_path is not None:
            with open(os.path.join(my_path, ".normalize_obs"), encoding="utf-8", 
                      mode="w") as f:
                f.write("I have encoded the observation space !\n DO NOT MODIFY !")
    
    # define the policy
    if load_path is None:
        if policy_kwargs is None:
            policy_kwargs = {}
        if net_arch is not None:
            policy_kwargs["net_arch"] = net_arch
        if logs_dir is not None:
            if not os.path.exists(logs_dir):
                os.mkdir(logs_dir)
            this_logs_dir = os.path.join(logs_dir, name)
        else:
            this_logs_dir = None
                
        nn_kwargs = {
            "policy": model_policy,
            "env": env_gym,
            "verbose": verbose,
            "learning_rate": learning_rate,
            "tensorboard_log": this_logs_dir,
            "policy_kwargs": policy_kwargs,
            **kwargs
        }
        
        agent = SB3Agent(env.action_space,
                         env_gym.action_space,
                         env_gym.observation_space,
                         nn_kwargs=nn_kwargs,
        )
        
    else:        
        agent = SB3Agent(env.action_space,
                         env_gym.action_space,
                         env_gym.observation_space,
                         nn_path=os.path.join(load_path, name)
        )


    # Save a checkpoint every "save_every_xxx_steps" steps
    if checkpoint_callback is None:
        if save_every_xxx_steps is not None:
            if save_path is None:
                warnings.warn("save_every_xxx_steps is set, but no path are "
                            "set to save the model (save_path is None). No model "
                            "will be saved.")
            else:
                checkpoint_callback = CheckpointCallback(save_freq=save_every_xxx_steps,
                                                        save_path=my_path,
                                                        name_prefix=name)
    # save hyperparameters
    if save_path is not None:
        dict_to_save.update({
            "obs_attr_to_keep": env_gym._observation_space._attr_to_keep,
            "act_attr_to_keep": env_gym.action_space._attr_to_keep,
            "net_arch": agent.nn_model.policy.net_arch,
            "activation_fn": str(agent.nn_model.policy.activation_fn),
            "learning_rate": agent.nn_model.learning_rate,
            "batch_size": agent.nn_model.batch_size,
            "n_steps": agent.nn_model.n_steps,
            **kwargs
    })
        with open(os.path.join(my_path, 'dict_hyperparameters.json'), 'x') as fp:
            json.dump(dict_to_save, fp, indent=4)
                
    # train it
    agent.nn_model.learn(total_timesteps=iterations,
                        callback=checkpoint_callback,
                        # eval_env=eval_env  # TODO
                        )
    
    # save it    
    if save_path is not None:
        agent.nn_model.save(os.path.join(my_path, name))

    env_gym.close()
    return agent




class CustomGymEnvZone1(GymEnvWithHeuristics):
    """This environment is slightly more complex that the other one.
    
    It consists in 2 things:
    
    #. reconnecting the powerlines if possible
    #. doing nothing is the state of the grid is "safe" (for this class, the notion of "safety" is pretty simple: if all
       flows are bellow 90% (by default) of the thermal limit, then it is safe)
    
    If for a given step, non of these things is applicable, the underlying trained agent is asked to perform an action
    
    .. warning::
        When using this environment, we highly recommend to adapt the parameter `safe_max_rho` to suit your need.
        
        Sometimes, 90% of the thermal limit is too high, sometimes it is too low.
        
    """
    def __init__(self, env_init, *args, reward_cumul="init", safe_max_rho=0.9, curtail_margin=30, **kwargs):
        super().__init__(env_init, reward_cumul=reward_cumul, *args, **kwargs)
        self._safe_max_rho = safe_max_rho
        self._curtail_margin = curtail_margin
        self.lines_I_care_about = [135, 136, 137, 138, 139, 141, 142, 143, 144, 146, 147, 148, 149,
            150, 152, 153, 154, 155, 156, 157, 158, 159, 160, 163, 164, 165,
            166, 167, 168, 180, 181, 182]
        
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
        elif g2op_obs.rho[self.lines_I_care_about].max() <= self._safe_max_rho:
            # play do nothing if there is "no problem" according to the "rule of thumb"
            res = [self.init_env.action_space()]
        return res
    
    def fix_action(self, grid2op_action, g2op_obs):
        # We try to limit to end up with a "game over" because actions on curtailment or storage units.
        # this is "required" because we use curtailment and action on storage units
        # but the main goal is to 
        grid2op_action.limit_curtail_storage(g2op_obs, margin=self._curtail_margin)
        return grid2op_action
    
    
class GymEnvWithSetPointRecoDNZone1(GymEnvWithSetPoint): 
    def __init__(self, env_init, *args, reward_cumul="sum", safe_max_rho=0.9, curtail_margin=30, weight_penalization_storage=0.25, ind=1, **kwargs):
        super().__init__(env_init=env_init, *args, reward_cumul=reward_cumul, safe_max_rho=safe_max_rho, curtail_margin=curtail_margin, weight_penalization_storage=weight_penalization_storage, ind=ind,**kwargs)
        self.lines_I_care_about = [135, 136, 137, 138, 139, 141, 142, 143, 144, 146, 147, 148, 149,
            150, 152, 153, 154, 155, 156, 157, 158, 159, 160, 163, 164, 165,
            166, 167, 168, 180, 181, 182]
        
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
        elif g2op_obs.rho[self.lines_I_care_about].max() <= self._safe_max_rho:
            # play do nothing if there is "no problem" according to the "rule of thumb"
            res = [self.init_env.action_space()]
        return res