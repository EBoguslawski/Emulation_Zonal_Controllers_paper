import pdb

import grid2op
from grid2op.Reward import BaseReward

import json
import numpy as np
from gymnasium.spaces import Box
import sys
import os
import copy
from tqdm import tqdm

from utils import *

class ZeroReward(BaseReward):

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):        
        reward = 0
        return reward
    
class OneReward(BaseReward):

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):        
        reward = 1.
        return reward
    
class CurtailmentMWReward(BaseReward):

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):        
        obs = env.get_obs()
        curtailment_mw = obs.curtailment_mw.sum()        
        
        return curtailment_mw
    
class CurtailmentLimitReward(BaseReward):

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):        
        obs = env.get_obs()
        curtailment_limit = obs.curtailment_limit[env.gen_renewable].sum()
        return curtailment_limit
    
def get_obs_without_setpoint(gym_obs, gym_env):
    idx_begin, idx_end = gym_env._observation_space.get_indexes("storage_setpoint")
    bool_to_keep = np.full_like(gym_obs, True, dtype=bool)
    bool_to_keep[idx_begin:idx_end] = False
    gym_obs_no_setpoint = gym_obs[bool_to_keep]
    return gym_obs_no_setpoint

env_name = "l2rpn_idf_2023_val"

reward_cumul = "sum"
# reward_cumul = "init"

# show_tqdm = True
show_tqdm = False

nb_scenario = 34
# nb_scenario = 2

# load_path = "./saved_models/"
# load_path = "/home/boguslawskieva/PSCC_idf_2023/saved_model_2023/"
iter_num = None  # put None for the latest version
verbose = True

safe_max_rho = 0.9

curtail_margin = 30

def evaluate_agent(agent_name, 
            gymenv_class_training,
            gymenv_class_evaluation,
            my_agent=None,
            DoNothing_agent=False,
            load_path = "./saved_models/",
            env_name = "l2rpn_idf_2023_val",
            nb_scenario = 2,
            safe_max_rho = 0.9,
            curtail_margin = 30,
            show_tqdm=True,
            iter_num = None,  
            save_dict = False,  
):
    env_val = grid2op.make(env_name, reward_class=ZeroReward, other_rewards={"timesteps":OneReward, "curtailment_mw":CurtailmentMWReward, "curtailment_limit":CurtailmentLimitReward})        

    if my_agent is None:
        my_agent = load_agent(env_val, load_path,
                                    agent_name,
                                    gymenv_class_training,
                                    # {"safe_max_rho": safe_max_rho, "curtail_margin": curtail_margin, "reward_cumul":reward_cumul},
                                    iter_num=iter_num,
                        )
        
        print(f"Agent {agent_name} loaded with gymenv_class {gymenv_class_training.__name__}!")

    gymenv_kwargs={"safe_max_rho": safe_max_rho, "curtail_margin": curtail_margin, "reward_cumul":"sum"}
    if issubclass(gymenv_class_evaluation, GymEnvWithSetPoint):
        gymenv_kwargs["weight_penalization_storage"] = - 0.5
        
    if gymenv_class_evaluation == GymEnvWithSetPointRemoveCurtail:
        gymenv_kwargs.update({'very_safe_max_rho': 0.85, 'n_gen_to_uncurt': 1, 'ratio_to_uncurt': 0.05, 'margin': 0.0, 'less_or_more': 'more'})


    print("gymenv_kwargs:", gymenv_kwargs, "reward_cum:", reward_cumul, "iter_num:", iter_num)

    gym_env = create_gymenv(env_val,
               gymenv_class_evaluation,
               gymenv_kwargs=gymenv_kwargs
                )


    # env_seeds=[0 for _ in range(nb_scenario)]
    env_seeds = [1558339257, 1953351278, 368231021, 1039865328, 1262305624, 612239120, 2099139284, 2119137781, 
                 1988556056, 431758685, 1485800608, 348197074, 1329383771, 1520600050, 1661332927, 1553317924, 
                 1952114582, 1307414814, 1828284229, 594142571, 1454445973, 2014219556, 2032911575, 1932799384, 
                 1251080764, 896704283, 121724528, 628201122, 1064146026, 597708608, 1978904151, 1896826479, 723392956, 789224689
    ][:nb_scenario]
    agent_seeds=[0 for _ in range(nb_scenario)]

    ts_survived_array = np.full(nb_scenario, np.nan)
    reward_array = np.full(nb_scenario, np.nan)
    storage_diff_smr_array = np.full(nb_scenario, np.nan)
    above_smr_array = np.full(nb_scenario, np.nan)
    curtailment_mw_array = np.full(nb_scenario, np.nan)
    curtailment_limit_array = np.full(nb_scenario, np.nan)
    curtailment_limit_smr_array = np.full(nb_scenario, np.nan)
    one_reward_array = np.full(nb_scenario, np.nan)

    idx0_setpoint, idx1_setpoint = gym_env._observation_space.get_indexes("storage_setpoint")

    for i in range(nb_scenario):
        gym_env.init_env.set_id(i)
        gym_env.init_env.seed(env_seeds[i])
        
        gym_obs, reward, done, info = gym_env.reset(return_all=True, seed=env_seeds[i])
        my_agent.seed(agent_seeds[i])

        reward_tot = reward
        storage_diff_smr = 0
        curtailment_limit_smr = 0
        above_smr = 1
        if "rewards" in info.keys(): # happends when reset applies the heuristics
            curtailment_mw_tot = info["rewards"]["curtailment_mw"]
            curtailment_limit_tot = info["rewards"]["curtailment_limit"]
            one_reward_tot = info["rewards"]["timesteps"]
            # print(i, gym_env.init_env.nb_time_step, reward_tot / gym_env.init_env.nb_time_step, 
                # curtailment_mw_tot/ (gym_env.init_env.nb_time_step * gym_env.init_env.gen_renewable.sum()), 
                # curtailment_limit_tot / (gym_env.init_env.nb_time_step * gym_env.init_env.gen_renewable.sum()), 
                # info["rewards"])
        else: # happends when no heuristics has been apply in reset
            curtailment_mw_tot = 0
            curtailment_limit_tot = gym_env.init_env.get_obs().curtailment_limit[gym_env.init_env.gen_renewable].sum()
            one_reward_tot = 1
            # print(i, gym_env.init_env.nb_time_step, reward_tot, curtailment_mw_tot, curtailment_limit_tot)
            
        
        
        idx_begin, idx_end = gym_env.action_space.get_indexes("curtail")
        dn_act = np.zeros(gym_env.action_space.shape[0])
        dn_act[idx_begin:idx_end] = -1.
        if show_tqdm:
            pbar = tqdm(total=gym_env.init_env.max_episode_duration())
            pbar.update(gym_env.init_env.nb_time_step)
        ts_survived = gym_env.init_env.nb_time_step
        while not done:
            gym_obs_for_agent = gym_obs
            if not issubclass(gymenv_class_training, GymEnvWithSetPoint):
                gym_obs_for_agent = get_obs_without_setpoint(gym_obs_for_agent, gym_env)
            gym_act = my_agent.get_act(gym_obs_for_agent, reward, done)
            if DoNothing_agent:
                gym_act = dn_act
            # print(gym_env.init_env.nb_time_step, "gym_obs:", gym_obs)
            # print(gym_env.init_env.nb_time_step, "gym_act:", gym_act)
            
            gym_obs, reward, done, _, info = gym_env.step(gym_act)
            if not done:
                g2op_obs = gym_env.init_env.get_obs()
                storage_diff_smr += np.abs(gym_obs[idx0_setpoint:idx1_setpoint] - g2op_obs.storage_charge / gym_env.init_env.storage_Emax).mean()
                curtailment_limit_smr += g2op_obs.curtailment_limit[g2op_obs.gen_renewable].sum()
            # print(gym_env.init_env.nb_time_step, info["rewards"]["curtailment_mw"])
            reward_tot += reward
            if not np.isnan(info["rewards"]["curtailment_mw"]):
                curtailment_mw_tot += info["rewards"]["curtailment_mw"]
            curtailment_limit_tot += info["rewards"]["curtailment_limit"]
            one_reward_tot += info["rewards"]["timesteps"]
            above_smr+=1
            # print(gym_env.init_env.nb_time_step, reward, info["rewards"])
            # print("curtailment_limit:", g2op_obs.curtailment_limit[g2op_obs.gen_renewable].sum())
            # print("curtailment_mw:", g2op_obs.curtailment_mw.sum())
            # print(gym_env.init_env.nb_time_step, "storage_setpoint:", np.abs(gym_obs[idx0_setpoint:idx1_setpoint] - g2op_obs.storage_charge / gym_env.init_env.storage_Emax).mean())
            if show_tqdm:
                pbar.update(gym_env.init_env.nb_time_step - ts_survived)
            ts_survived = gym_env.init_env.nb_time_step
        if show_tqdm:
            pbar.close()
        ts_survived_array[i] = gym_env.init_env.nb_time_step
        reward_array[i] = reward_tot
        curtailment_mw_array[i] = curtailment_mw_tot
        curtailment_limit_array[i] = curtailment_limit_tot
        one_reward_array[i] = one_reward_tot
        storage_diff_smr_array[i] = storage_diff_smr
        above_smr_array[i] = above_smr
        curtailment_limit_smr_array[i] = curtailment_limit_smr
        # print(i, gym_env.init_env.nb_time_step, 
        #     reward_tot / one_reward_tot, 
        #     curtailment_mw_tot / (one_reward_tot * gym_env.init_env.gen_renewable.sum()), 
        #     curtailment_limit_tot / (one_reward_tot * gym_env.init_env.gen_renewable.sum()), 
        #     info["rewards"])
        
    print("Average timesteps survived:", np.mean(ts_survived_array))
    print("Median of timesteps survived", np.median(ts_survived_array))
    print("Average reward", np.sum(reward_array) / (one_reward_array.sum()))
    print("Average curtaiment (MW)", np.sum(curtailment_mw_array) / (one_reward_array.sum() * gym_env.init_env.gen_renewable.sum()))
    print("Average limits for curtailment", curtailment_limit_array.sum()/(one_reward_array.sum() * gym_env.init_env.gen_renewable.sum()))
    print("Average storage setpoint diff above smr", storage_diff_smr_array.sum()/(above_smr_array.sum())) 
    print("Average curtailment limits above smr", curtailment_limit_smr_array.sum()/(above_smr_array.sum() * gym_env.init_env.gen_renewable.sum()))   
    
    dict_to_save = {"Average timesteps survived:": np.mean(ts_survived_array),
                    "ts_survived_array": ts_survived_array, 
                    "reward_array": reward_array,
                    "Average reward": np.sum(reward_array) / (one_reward_array.sum()),
                    "curtailment_mw_array": curtailment_mw_array,
                    "curtaiment_limit_array": curtailment_limit_array,
                    "Average limits for curtailment": curtailment_limit_array.sum()/(one_reward_array.sum() * gym_env.init_env.gen_renewable.sum()),
                    "one_reward_array": one_reward_array,
                    "storage_diff_smr_array": storage_diff_smr_array,
                    "curtailment_limit_smr_array": curtailment_limit_smr_array,
                    "above_smr_array": above_smr_array
                    }
    
    if save_dict:
        os.makedirs(os.path.join("./final_results"), exist_ok=True)
        np.save(os.path.join("./final_results", f"dict_{agent_name}_{gymenv_class_evaluation.__name__}.npy"), dict_to_save)
        
    return dict_to_save
    

# agents_names = [f"agent_default_no_redisp_LinesCapacityReward_CustomGymEnv_0.85_0.9_sum_3x300_bis_lr2_10M_{j}" for j in range(3,13)]
# agents_names = [f"agent_default_no_redisp_PenalizeSetpointPosReward_CustomGymEnv_0.85_0.9_sum_3x300_bis_lr2_10M_{j}" for j in range(3,12)]
# agents_names = ["agent_default_no_redisp_PenalizeSetpointPosReward_GymEnvWithSetPointRemoveCurtail_0.85_0.9_0.25_sum_3x300_bis_lr2_10M_3_gpu1",
#                 "agent_default_no_redisp_PenalizeSetpointPosReward_GymEnvWithSetPointRemoveCurtail_0.85_0.9_0.25_sum_3x300_bis_lr2_10M_3_gpu2"] + \
#                 [f"agent_default_no_redisp_PenalizeSetpointPosReward_GymEnvWithSetPointRemoveCurtail_0.85_0.9_0.25_sum_3x300_bis_lr2_10M_{j}" for j in range(4,8)]

# load_path = "/home/boguslawskieva/PSCC_idf_2023/saved_model_2023/"

# for agent_name in agents_names:
#     evaluate_agent(agent_name, 
#                 # CustomGymEnv, 
#                 # GymEnvWithSetPointRecoDN, 
#                 GymEnvWithSetPointRemoveCurtail,
#                 GymEnvWithSetPointRemoveCurtail,
#                 load_path=load_path,
#                 nb_scenario=nb_scenario,
#                 show_tqdm=False,
#                 )
    
# agents_names = [f"agent_default_no_redisp_LinCapRew_sum_3x300_lr2_10M_{j}" for j in range(3)]
# agents_names = ["agent_default_no_redisp_PenSetpointPosRew_sum_3x300_bis_lr2_10M_0"]
# agents_names = ["agent_default_no_redisp_LinCapRew_sum_3x300_lr2_10M_0"]

# load_path = "/data/boguslawskieva_data/ppo_stable_baselines_idf_2023/saved_model_2023/"


# print("END OF SCRIPT")

