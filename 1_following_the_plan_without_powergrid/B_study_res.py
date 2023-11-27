# %%

from storage_env import StorageEnv
from stable_baselines3 import PPO
import grid2op
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from stable_baselines3.common.base_class import BaseAlgorithm
import os
import pandas as pd

from A_train_ppo import MAIN_FOLDER

NB_RUN = 34

# %%

# env_g2op = grid2op.make("educ_case14_storage", test=True)
env_g2op = grid2op.make("l2rpn_idf_2023")
# env_g2op = grid2op.make("l2rpn_wcci_2022", test=True)

env = StorageEnv(env_g2op.max_episode_duration(),
                 env_g2op.n_storage,
                 env_g2op.storage_Emin,
                 env_g2op.storage_Emax,
                 env_g2op.storage_max_p_prod,
                 init_storage_capa_ratio=(0.1, 0.9),
                 ratio_setpoint=(0.2, 3),
                 smooth=None,
                 target_shape="step_function",
                #  target_shape="random_walk",
                 )

# %%

def act_expert(env, obs):
    obs_unnorm = np.concatenate((obs[:env.nb_storage] * env._emax, obs[env.nb_storage:] * env._emax))
    return np.clip((obs_unnorm[env.nb_storage:] - obs_unnorm[:env.nb_storage]) *12. / env._maxP, -1, 1)

def eval_one(env, agent, seed=0, nb_run=NB_RUN):
    env.seed(seed)
    states = np.zeros((nb_run, env.episode_duration + 1, env.nb_storage))
    actions = np.full((nb_run, env.episode_duration + 1, env.nb_storage), np.nan)
    target = np.zeros((nb_run, env.episode_duration + 1, env.nb_storage))
    rewards = np.zeros((nb_run, env.episode_duration + 1))
    for run_id in range(nb_run):
        obs, _ = env.reset()
        target[run_id, env.nb_iter, :] =  env._target_norm[env.nb_iter, :] * env._emax
        states[run_id, env.nb_iter, :] =  env._storage_charge
        while True:
            if isinstance(agent, BaseAlgorithm):
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = agent(env, obs)
            obs, reward, dones, _, _ = env.step(action)
            actions[run_id, env.nb_iter - 1, :] = action
            target[run_id, env.nb_iter, :] =  env._target_norm[env.nb_iter, :] * env._emax
            states[run_id, env.nb_iter, :] =  env._storage_charge
            rewards[run_id, env.nb_iter] = reward
            if dones:
                break
    return states, actions, target, rewards

def eval_all(dict_agent, env, seed, nb_run=NB_RUN):
    res = {}
    for agent_nm, agent in dict_agent.items():
        res[agent_nm] = eval_one(env, agent, seed, nb_run)
    return res

def fun_eval(env, target, state, ind=2):
    return np.mean((np.abs(target - state)**ind / env._emax**ind))**(1/ind)

# %%

# expe_names = ["reward_shape", "n_steps_impact"]
# expe_names = ["n_steps_impact"]
# expe_names = ["storage_number"]
# expe_names = ["case_118_lr_n_steps"]
# expe_names = ["case_118_lr_n_steps_ratio_setpoint"]
# expe_names = ["case_118_n_steps_ratio_setpoint_max"]
expe_names = ["case_118_final_agent_instances"]
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

res_eval = eval_all(agents_dict, env, seed=0, nb_run=NB_RUN)

#%%
for id_, (agent_nm, (states, actions, target, rewards)) in enumerate(res_eval.items()):
    print(f"agent \"{agent_nm}\": avg distance: {fun_eval(env, target, states, ind=1):.3f}")
    
# %%

data = pd.DataFrame([[run_id, stor_id, ts, states[run_id,ts,stor_id], states[run_id,ts,stor_id]/env._emax[stor_id], actions[run_id,ts,stor_id], actions[run_id,ts,stor_id]/env._emax[stor_id], agent_nm] 
                    for run_id in range(NB_RUN) 
                    for stor_id in range(env.nb_storage) 
                    for ts in range(env.episode_duration + 1)
                     for agent_nm, (states, actions, _, rewards) in res_eval.items()
                    ] + 
                    
                    [[run_id, stor_id, ts+1, target[run_id,ts,stor_id], target[run_id,ts,stor_id]/env._emax[stor_id], np.nan, np.nan, "target"] 
                    for run_id in range(NB_RUN) 
                    for stor_id in range(env.nb_storage) 
                    for ts in range(env.episode_duration)
                     for target in [res_eval["expert"][2]]
                    ]
                    
                    ,
             columns = ["run_id", "stor_id", "ts", "storage_charge", "storage_charge_norm", "set_storage", "set_storage_norm", "agent_nm"]
             )

# %%

g = px.line(data, x="ts", y="storage_charge_norm", facet_col="stor_id", facet_row="run_id", color="agent_nm", height=2000)
g.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
g.show()

# %%
agent = PPO.load("/home/boguslawskieva/storage_env_expe/saved_experiments/case_118_lr_n_steps_ratio_setpoint/rew_abs_lr_3e-06_n_steps_32/rew_abs_lr_3e-06_n_steps_32.zip")
fig, axs = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(8,8))
seed_coef=2
# fig.suptitle('Horizontally stacked subplots')
for ax, seed, ratio_setpoint, smooth, target_shape, title in zip(
    [axs[0,0], axs[0,1], axs[1,0]],
    [1*seed_coef, 2*seed_coef, 3*seed_coef],
    [(0.2, 3), (0.2, 3), (0.2,3)], 
    [12, None, 12],
    ["random_walk", "random_walk", "step_function"],
    ["Smooth shape", "Noisy shape", "Step function shape"]
    
):
    env = StorageEnv(12 * 24, #env_g2op.max_episode_duration(),
                    env_g2op.n_storage,
                    env_g2op.storage_Emin,
                    env_g2op.storage_Emax,
                    env_g2op.storage_max_p_prod,
                    init_storage_capa_ratio=(0.1, 0.9),
                    ratio_setpoint=ratio_setpoint,
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


# %%

plt.plot(actions[0,:,0])