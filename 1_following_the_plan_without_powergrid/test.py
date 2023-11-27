# %%

from storage_env import StorageEnv
from stable_baselines3 import PPO
import grid2op
import numpy as np
import plotly.express as px
import matplotlib.pylab as plt
from stable_baselines3.common.base_class import BaseAlgorithm
import os
import pandas as pd

from A_train_ppo import MAIN_FOLDER

NB_RUN = 34

#%%

env_kwargs={"init_storage_capa_ratio":(0.1, 0.9), "ratio_setpoint":(0.2, 3), "smooth":12}

env_g2op = grid2op.make("l2rpn_idf_2023")

# env = StorageEnv(env_g2op.max_episode_duration(),
#                  env_g2op.n_storage,
#                  env_g2op.storage_Emin,
#                  env_g2op.storage_Emax,
#                 # np.array([24, 48, 48, 24, 24, 48, 48]),
#                  env_g2op.storage_max_p_prod,
#                  init_storage_capa_ratio=(0.1, 0.9),
#                  ratio_setpoint=(0.2, 12),
#                  smooth=6,
#                 #  target_shape="step_function",
#                  target_shape="random_walk",
#                  )

def act_expert(env, obs):
    obs_unnorm = np.concatenate((obs[:env.nb_storage] * env._emax, obs[env.nb_storage:] * env._emax))
    return np.clip((obs_unnorm[env.nb_storage:] - obs_unnorm[:env.nb_storage]) *12. / env._maxP, -1, 1)

def eval_one(env, agent, seed=0, nb_run=NB_RUN):
    env.seed(0)
    states = np.zeros((nb_run, env.episode_duration, env.nb_storage))
    actions = np.full((nb_run, env.episode_duration, env.nb_storage), np.nan)
    target = np.zeros((nb_run, env.episode_duration, env.nb_storage))
    rewards = np.zeros((nb_run, env.episode_duration))
    env.seed(0)
    for run_id in range(nb_run):
        obs, _ = env.reset()
        # states[run_id, env.nb_iter, :] =  env._storage_charge
        while True:
            if isinstance(agent, BaseAlgorithm):
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = agent(env, obs)
            target[run_id, env.nb_iter, :] =  env._target_norm[env.nb_iter, :] * env._emax
            obs, reward, dones, _, _ = env.step(action)
            if not dones:
                actions[run_id, env.nb_iter, :] = action
                states[run_id, env.nb_iter, :] =  env._storage_charge
                rewards[run_id, env.nb_iter] = reward
            if dones:
                break
    return states, actions, target, rewards


def fun_eval(env, target, state, ind=2):
    return np.mean((np.abs(target - state)**ind / env._emax**ind))**(1/ind)

expe_names = ["case_118_lr_n_steps_ratio_setpoint"]

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

#%%

def compute_perf(ratio_emax_pmax_list, seed=0, nb_run=NB_RUN, env_kwargs=env_kwargs):
    ratio_emax_pmax_list = np.array(ratio_emax_pmax_list)
    scores = np.zeros((ratio_emax_pmax_list.shape[0], 2))
    for i, r in enumerate(ratio_emax_pmax_list):
        env = StorageEnv(env_g2op.max_episode_duration(),
                    env_g2op.n_storage,
                    env_g2op.storage_Emin,
                    env_g2op.storage_Emax,
                    env_g2op.storage_max_p_prod * r,
                    **env_kwargs
                    )
        env.seed(seed)
        agent = agents_dict['rew_abs_lr_3e-06_n_steps_32']
        states, actions, target, rewards = eval_one(env, agent, seed, nb_run)
        scores[i, 0] = fun_eval(env, target, states, ind=1)
        states, actions, target, rewards = eval_one(env, act_expert, seed, nb_run)
        scores[i, 1] = fun_eval(env, target, states, ind=1)
    return scores

# %%

ratio_emax_pmax_list = [0.05, 0.1, 0.5, 1, 3, 5, 10, 15, 20, 40, 50, 100, 1000, 10000]

scores = compute_perf(ratio_emax_pmax_list)

# %%

plt.plot(ratio_emax_pmax_list, scores[:, 0], marker=".", label="RL agent")
plt.plot(ratio_emax_pmax_list, scores[:, 1], marker="*", label="expert agent")
plt.axvline(x=1., color="grey", ls="--", label="Normal storage powers")
plt.ylabel("Average distance to setpoint")
plt.xlabel("Ratio by which we multiplied the storage powers for test phase")
plt.xscale("log")
plt.legend()


# %%
