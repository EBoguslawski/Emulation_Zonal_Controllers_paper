from StorageEnv import StorageEnv
import numpy as np
import grid2op
from stable_baselines3.common.base_class import BaseAlgorithm

env_kwargs={"init_storage_capa_ratio":(0.1, 0.9), "ratio_setpoint":(0.2, 3), "smooth":12}

env_g2op = grid2op.make("l2rpn_idf_2023")


def act_expert(env, obs):
    obs_unnorm = np.concatenate((obs[:env.nb_storage] * env._emax, obs[env.nb_storage:] * env._emax))
    return np.clip((obs_unnorm[env.nb_storage:] - obs_unnorm[:env.nb_storage]) *12. / env._maxP, -1, 1)

def eval_one(env, agent, seed=0, nb_run=34):
    env.seed(seed)
    states = np.zeros((nb_run, env.episode_duration, env.nb_storage))
    actions = np.full((nb_run, env.episode_duration, env.nb_storage), np.nan)
    target = np.zeros((nb_run, env.episode_duration, env.nb_storage))
    rewards = np.zeros((nb_run, env.episode_duration))
    for run_id in range(nb_run):
        obs, _ = env.reset()
        while True:
            if isinstance(agent, BaseAlgorithm): # case of a trained agent
                action, _ = agent.predict(obs, deterministic=True)
            else: # case of the expert policy
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

def eval_all(dict_agent, env, seed, nb_run=34):
    res = {}
    for agent_nm, agent in dict_agent.items():
        res[agent_nm] = eval_one(env, agent, seed, nb_run)
    return res


def fun_eval(env, target, state, ind=2):
    return np.mean((np.abs(target - state)**ind / env._emax**ind))**(1/ind)


def compute_perf_with_ratio(ratio_emax_pmax_list, agent, seed=0, nb_run=34, episode_duration=None, env_kwargs=env_kwargs):
    ratio_emax_pmax_list = np.array(ratio_emax_pmax_list)
    scores = np.zeros((ratio_emax_pmax_list.shape[0], 2))
    for i, r in enumerate(ratio_emax_pmax_list):
        episode_duration = episode_duration if episode_duration is not None else env_g2op.max_episode_duration()
        env = StorageEnv(episode_duration,
                    env_g2op.n_storage,
                    env_g2op.storage_Emin,
                    env_g2op.storage_Emax,
                    env_g2op.storage_max_p_prod * r,
                    **env_kwargs
                    )
        env.seed(seed)
        states, actions, target, rewards = eval_one(env, agent, seed, nb_run)
        scores[i, 0] = fun_eval(env, target, states, ind=1)
        states, actions, target, rewards = eval_one(env, act_expert, seed, nb_run)
        scores[i, 1] = fun_eval(env, target, states, ind=1)
    return scores