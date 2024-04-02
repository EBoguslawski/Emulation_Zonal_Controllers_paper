# %%

from StorageEnv import StorageEnv
from stable_baselines3 import PPO
import grid2op
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from stable_baselines3.common.base_class import BaseAlgorithm
import os
import pandas as pd
from evaluation_functions import *

from A_train_ppo import MAIN_FOLDER

NB_RUN = 34


# %%

expe_names = ["expe_118_substations_power_grid"]
agents_dict = {}
agents_dict["expert"] = act_expert
for expe_name in expe_names:
    for dir_nm in sorted(os.listdir(os.path.join(MAIN_FOLDER, expe_name))):
        try:
            # if "rew_abs_lr_3e-06_n_steps_32" in dir_nm:
            tmp = PPO.load(os.path.join(MAIN_FOLDER, expe_name, dir_nm, f"{dir_nm}.zip"))
            agents_dict[dir_nm] = tmp
        except:
            pass

agents_dict.keys()



# %%
# agent = PPO.load("/home/boguslawskieva/storage_env_expe/saved_experiments/case_118_lr_n_steps_ratio_setpoint/rew_abs_lr_3e-06_n_steps_32/rew_abs_lr_3e-06_n_steps_32.zip")
# agents_dict.update({"best_agent":agent})
seed_coef=2
for seed, smooth, target_shape, title in zip(
    [1*seed_coef, 2*seed_coef, 3*seed_coef],
    [12, None, 12],
    ["random_walk", "random_walk", "step_function"],
    ["Smooth shape", "Noisy shape", "Step function shape"]
    
):
    env = StorageEnv(env_g2op.max_episode_duration(),
                    env_g2op.n_storage,
                    env_g2op.storage_Emin,
                    env_g2op.storage_Emax,
                    env_g2op.storage_max_p_prod,
                    init_storage_capa_ratio=(0.1, 0.9),
                    ratio_setpoint=(0.2, 3),
                    smooth=smooth,
                    #  target_shape="step_function",
                    target_shape=target_shape,
                    nb_intervals=6
                    )
    
    print(title)
    res_eval = eval_all(agents_dict, env, seed=0, nb_run=NB_RUN)
    for id_, (agent_nm, (states, actions, target, rewards)) in enumerate(res_eval.items()):
        print(f"agent \"{agent_nm}\": avg distance: {fun_eval(env, target, states, ind=1):.3f}")
        

