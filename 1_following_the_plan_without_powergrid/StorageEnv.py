# # Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of storage_env, storage_env a over simplified gym environment to test control storage units


from gymnasium.spaces import Box
import gymnasium as gym
import numpy as np
from numpy.random import default_rng
from typing import Any, Union, Tuple, Optional, Literal

# import pdb
# from scipy.interpolate import splrep, BSpline

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class StorageEnv(gym.Env):
    """
    The "gym environment" that emulates a few storage units.
    
    Each call to "reset" will:
    1) generate setpoint for the new episode
    
    The goal of the agent is to control the storage unit to match the setpoint as close as possible. It cannot die.
    
    Attributes
    -----------
    episode_duration:
        Duration of each episode
        
    nb_storage:
        Number of storage unit to control
        
    init_storage_capa_ratio:
        At which ratio of Emax the storage units will be initialized at the beginning of an episode
        
        If a ``float`` is provided, then this is used. If a tuple of 2 floats is provided
        then initially the initial charge will be sampled from a uniform distribution bounded by this 2 floats.
        
        **NB** this needs to be provided as a ratio of Emax.
        
    ratio_setpoint:
        How "noisy" / "rough"  will be the storage unit setpoints at the beginning of the episode.
        
        The higher this number, the more "rough" the setpoint will be (harder to follow)
        
        If a ``float`` is provided, then this is used. If a tuple of 2 floats is provided
        then initially the ratio will be sampled from a uniform distribution bounded by this 2 floats.
    
    nb_iter:
        Current number of iterations performed until now in the environment
        
    smooth:
        If ``None`` no smoothing is performed for the setpoint. If an ``int`` is given then the 
        storage unit setpoint is smoothed by applying a rolling average with the window of size "smooth"
    
    action_space:
        A gym Box containing as many elements as the number of storage. Each dimension controls a storage unit.
        
    observation_space:
        A gym Box with 2 elements for each storage units.
        
        The first ``nb_storage`` elements are the target for each storage unit (target for the next step)
        
        The last ``nb_storage`` elements are the actual charge for each storage unit (current charge of each storage units)
        
        It is already "reduced" in [0, 1]
        
    reward_shape:
        Which type of function to use for the reward:
        
        - `abs` : reward proportional to: 1. - avg |setpoint - current_charge| / emax
        - `square` : reward proportional to: 1. - avg  (setpoint - current_charge) ** 2 / emax**2
        - `sqrt` : reward proportional to: avg  1. - sqrt( |setpoint - current_charge| / emax)
    """
    def __init__(self,
                 episode_duration : int,
                 nb_storage : int,
                 storage_Emin : np.ndarray ,
                 storage_Emax : np.ndarray ,
                 storage_MaxP : np.ndarray ,
                 init_storage_capa_ratio : Union[float, Tuple[float, float]]=0.5,
                 ratio_setpoint : Union[float, Tuple[float, float]]=0.5,
                 smooth : Optional[int]=None,
                 reward_shape: Literal["abs", "square", "sqrt"]="abs",
                 target_shape: Literal["random_walk", "step_function"]="random_walk",
                 nb_intervals: Union[float, Tuple[float, float]]=10,
                 ) -> None:
        super().__init__()
        self.episode_duration : int = episode_duration
        self.nb_storage : int = nb_storage
        self._emin : np.ndarray = storage_Emin
        self._emax : np.ndarray  = storage_Emax
        self._maxP : np.ndarray  = storage_MaxP
        self.prng = default_rng()
        self.init_storage_capa_ratio : Union[float, Tuple[float, float]] = init_storage_capa_ratio
        self.ratio_setpoint : Union[float, Tuple[float, float]] = ratio_setpoint
        self._storage_charge : np.ndarray  = np.zeros(self.nb_storage, dtype=np.float32)
        self.smooth: bool = smooth
        self.reward_shape : Literal["abs", "square", "sqrt"] = reward_shape
        self.target_shape : Literal["random_walk", "step_function"] = target_shape
        self.nb_intervals : Union[float, Tuple[float, float]] = nb_intervals
        
        # power to add / substract
        self.action_space : Box = Box(low=-np.ones(self.nb_storage, dtype=np.float32),
                                      high=np.ones(self.nb_storage, dtype=np.float32),
                                      dtype=np.float32)
        
        # current charge, then current target
        self.observation_space : Box = Box(low=np.zeros(2 * self.nb_storage, dtype=np.float32),
                                           high=np.ones(2 * self.nb_storage, dtype=np.float32),
                                           dtype=np.float32)
        
        # initialize the target
        self._target : np.ndarray = np.zeros((self.episode_duration + 1, self.nb_storage), dtype=np.float32)
        self._target_norm : np.ndarray = np.zeros((self.episode_duration + 1, self.nb_storage), dtype=np.float32)
        self.nb_iter : int = 0
    
    def seed(self, seed) -> None:
        """seed the env for reproducible experiments"""
        self.prng = default_rng(seed)
    
    def _normalize_storage_charge(self, storage_charge):
        return (storage_charge - self._emin) / (self._emax - self._emin)
    
    def reset(self, seed = None) -> Box:
        # sample the initial storage capacity
        if seed is not None:
            self.seed(seed)
        
        try:
            _ = iter(self.init_storage_capa_ratio)
            init_storage_capa_ratio = self.prng.uniform(*self.init_storage_capa_ratio, 1)
        except TypeError as te:
            init_storage_capa_ratio = self.init_storage_capa_ratio 

        # Sample the target setpoint
        if self.target_shape == "random_walk":
            tmp_ = self._get_target_random_walk(init_storage_capa_ratio)
        elif self.target_shape == "step_function":
            tmp_ = self._get_target_step_function()
        else:
            raise RuntimeError(f"Unknown target type {self.target_shape}")
        self._target_norm[:] = tmp_
        self._target = self._target_norm * self._emax
        
        # update the env
        self.nb_iter = 0
        self.init_storage_capa_mwh = self._emax * init_storage_capa_ratio
        self._storage_charge[:] = self.init_storage_capa_mwh
        
        # update the obs
        obs = np.zeros(2 * self.nb_storage, dtype=np.float32)
        obs[self.nb_storage:] = self._target_norm[0,:]
        obs[:self.nb_storage] = self._normalize_storage_charge(np.full(shape=(self.nb_storage,), fill_value=self.init_storage_capa_mwh))
        return obs, {}
    
    def _get_target_random_walk(self, init_storage_capa_ratio):
        tmp_ = self.prng.uniform(-1, 1, (self.episode_duration + 1, self.nb_storage)) # -1,1
        # sample "how noisy" is the setpoint
        try:
            _ = iter(self.ratio_setpoint)
            ratio_setpoint = self.prng.uniform(*self.ratio_setpoint, 1)
        except TypeError as te:
            ratio_setpoint = self.ratio_setpoint    
        tmp_ *= self._maxP * 1/12. * ratio_setpoint / self._emax  # (-maxp, maxp)/emax
        
        if self.smooth is not None:
            for s_id in range(self.nb_storage):
                y = tmp_[:, s_id]
                tmp_[(self.smooth- 1):, s_id] = moving_average(y, self.smooth)            
        
        # integrate intial storage cap               
        tmp_= np.cumsum(tmp_, axis=0) + init_storage_capa_ratio
        tmp_ = np.clip(tmp_, 0, 1)
        
        return tmp_
    
    def _get_target_step_function(self):
        # sample the number of intervals
        try:
            _ = iter(self.nb_intervals)
            self.nb_intervals = self.prng.randint(*self.nb_intervals, 1)[0]
        except TypeError as te:
            pass
        
        # sample the intervals' delimitations
        ts_intervals = np.sort(self.prng.choice(range(1, self.episode_duration +1), (self.nb_intervals - 1, self.nb_storage)), axis=0)
        ts_intervals = np.concatenate([np.full((1, self.nb_storage), fill_value=0), 
                                    ts_intervals, 
                                    np.full((1, self.nb_storage), fill_value=self.episode_duration+1)])
        
        #sample the value of each interval
        values =  self.prng.uniform(size = (self.nb_intervals, self.nb_storage))
        
        # creation of the final target trajectory
        tmp_ = np.concatenate([np.concatenate([np.full((ts_intervals[i+1, i_stor] - ts_intervals[i, i_stor], 1), fill_value=values[i, i_stor]) 
                                                for i in range(self.nb_intervals)])
                                for i_stor in range(self.nb_storage)], 
                                axis=1)
        return tmp_
        
    
    def compute_reward(self):
        # if the distance to the normalized target is 1/2, we want a zero reward
        # target = self._target_norm[self.nb_iter, :] * self._emax
        target = self._target[self.nb_iter, :]
        if self.reward_shape == "abs":
            return 1. - 2. * np.mean(np.abs((self._storage_charge - target) / self._emax))
        if self.reward_shape == "sqrt":
            return 1. - 0.5 * np.mean(np.sqrt(np.abs((self._storage_charge - target) / self._emax)))
        if self.reward_shape == "square":
            return 1. - 4. * np.mean((self._storage_charge - target)**2 / self._emax**2)
        raise RuntimeError(f"Unknown reward type {self.reward_shape}")
    
    def step(self, action: Box):
        action_unnorm = action * self._maxP
        action_mwh = action_unnorm * 1/12.
        
        # update the env
        self._storage_charge[:] += action_mwh
        self._storage_charge[:] = np.clip(self._storage_charge, self._emin, self._emax)
        
        # update the reward
        reward = self.compute_reward()
        
        # update the obs
        self.nb_iter += 1
        obs = np.zeros(2 * self.nb_storage, dtype=np.float32)
        obs[:self.nb_storage] = self._normalize_storage_charge(self._storage_charge)
        obs[self.nb_storage:] = self._target_norm[self.nb_iter, :]
        return obs, reward, self.nb_iter == self.episode_duration, False, {}
    
    
if __name__ == "__main__":
    import grid2op
    env_g2op = grid2op.make("educ_case14_storage", test=True)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        env = StorageEnv(env_g2op.max_episode_duration(),
                         env_g2op.n_storage,
                         env_g2op.storage_Emin,
                         env_g2op.storage_Emax,
                         env_g2op.storage_max_p_prod)
        
    obs, _ = env.reset()
    assert (obs[:2] == env.init_storage_capa_mwh / env._emax).all()
    assert (env._storage_charge == env.init_storage_capa_mwh).all()
    obs, *_ = env.step((-1, -1))
    assert np.allclose(env._storage_charge, env.init_storage_capa_mwh - 1. / 12. * env._maxP)
    assert np.allclose(obs[:2], env.init_storage_capa_mwh / env._emax - 1. / 12. * env._maxP / env._emax )
    
    obs, _ = env.reset()
    assert (obs[:2] == env.init_storage_capa_mwh / env._emax).all()
    assert (env._storage_charge == env.init_storage_capa_mwh).all()
    obs, *_ = env.step((1, -1))
    assert np.allclose(env._storage_charge, env.init_storage_capa_mwh + 1. / 12. * np.array((1, -1)) * env._maxP)
    assert np.allclose(obs[:2], env.init_storage_capa_mwh / env._emax + 1. / 12. * np.array((1, -1)) * env._maxP / env._emax )
    
    obs, _ = env.reset()
    assert (obs[:2] == env.init_storage_capa_mwh / env._emax).all()
    assert (env._storage_charge == env.init_storage_capa_mwh).all()
    obs, *_ = env.step((-1, 1))
    assert np.allclose(env._storage_charge, env.init_storage_capa_mwh + 1. / 12. * np.array((-1, 1)) * env._maxP)
    assert np.allclose(obs[:2], env.init_storage_capa_mwh / env._emax + 1. / 12. * np.array((-1, 1)) * env._maxP / env._emax )
    
    obs, _ = env.reset()
    assert (obs[:2] == env.init_storage_capa_mwh / env._emax).all()
    assert (env._storage_charge == env.init_storage_capa_mwh).all()
    obs, *_ = env.step((1, 1))
    assert np.allclose(env._storage_charge, env.init_storage_capa_mwh + 1. / 12. * np.array((1, 1)) * env._maxP)
    assert np.allclose(obs[:2], env.init_storage_capa_mwh / env._emax + 1. / 12. * np.array((1, 1)) * env._maxP / env._emax )
    
    env = StorageEnv(env_g2op.max_episode_duration(),
                     env_g2op.n_storage,
                     env_g2op.storage_Emin,
                     env_g2op.storage_Emax,
                     env_g2op.storage_max_p_prod,
                     smooth=True)
    obs, _ = env.reset()