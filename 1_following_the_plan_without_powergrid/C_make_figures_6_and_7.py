# %%

from StorageEnv import StorageEnv
from stable_baselines3 import PPO
import matplotlib.pylab as plt
from stable_baselines3.common.base_class import BaseAlgorithm
import os
import pandas as pd

from evaluation_functions import *

from A_train_ppo import MAIN_FOLDER

NB_RUN = 34

#%%
# LOADING AGENTS IN THE DIRECTORIES in expe_names INTO THE DICT agents_dict

# expe_names = ["case_118_lr_n_steps_ratio_setpoint"]
expe_names = ["expe_118_substations_power_grid"]

agents_dict = {}
agents_dict["expert"] = act_expert
for expe_name in expe_names:
    for dir_nm in sorted(os.listdir(os.path.join(MAIN_FOLDER, expe_name))):
        try:
            tmp = PPO.load(os.path.join(MAIN_FOLDER, expe_name, dir_nm, f"{dir_nm}.zip"))
            agents_dict[dir_nm] = tmp
        except:
            pass

agents_dict.keys()

# %%

# MAKE FIGURE 8

ratio_emax_pmax_list = [0.05, 0.1, 0.5, 1, 3, 5, 10, 15, 20]

# agent = PPO.load("/home/boguslawskieva/storage_env_expe/saved_experiments/case_118_lr_n_steps_ratio_setpoint/rew_abs_lr_3e-06_n_steps_32/rew_abs_lr_3e-06_n_steps_32.zip")
agent = agents_dict["without_power_grid_agent"]
scores = compute_perf_with_ratio(ratio_emax_pmax_list, agent)


plt.figure(figsize=(4, 3))
plt.plot(ratio_emax_pmax_list, scores[:, 0], marker=".", label="RL agent")
plt.plot(ratio_emax_pmax_list, scores[:, 1], marker="*", label="expert agent")
plt.axvline(x=1., color="grey", ls="--", label="Normal storage powers")
plt.ylabel("Average distance to setpoint", fontsize=14)
plt.xlabel("Ratio by which we multiplied \n the storage powers for test phase", fontsize=14)
plt.xscale("log")
plt.legend(fontsize=12)
# plt.savefig("storage_characteristics_perf_until20.png", format='png', dpi=400, bbox_inches = 'tight')



# %%

# MAKE FIGURE 7

# agent = PPO.load("/home/boguslawskieva/storage_env_expe/saved_experiments/case_118_lr_n_steps_ratio_setpoint/rew_abs_lr_3e-06_n_steps_32/rew_abs_lr_3e-06_n_steps_32.zip")
agent = agents_dict["without_power_grid_agent"]
fig, axs = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(9,3))
seed_coef=2
for ax, seed, smooth, target_shape, title in zip(
    [axs[0], axs[1], axs[2]],
    [1*seed_coef, 2*seed_coef, 3*seed_coef],
    [12, None, 12],
    ["random_walk", "random_walk", "step_function"],
    ["Smooth shape", "Noisy shape", "Step function shape"]
    
):
    env = StorageEnv(12 * 24, 
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
    
    states, actions, target, rewards = eval_one(env, agent, seed=seed, nb_run=1)
    
    # env.reset()
    # ax.plot(env._target_norm[:,0])
    ax.plot(target[0,:-1,0] / env_g2op.storage_Emax[0], color="blue", label="target charge")
    ax.plot(states[0,1:,0] / env_g2op.storage_Emax[0], color="orange", label="observed charge")
    ax.set_title(title)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Normalized storage setpoint")
    ax.legend()

