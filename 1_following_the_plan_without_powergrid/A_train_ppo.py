# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of storage_env, storage_env a over simplified gym environment to test control storage units


from storage_env import StorageEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import grid2op
import torch
import numpy as np
import os
import warnings
import copy
import json
import pdb

MAIN_FOLDER = "./saved_experiments/"

def check_cuda(use_cuda, cuda_device):
    """
    Check if cuda is available and send a warning if necessary.
    """
    if use_cuda:
        assert torch.cuda.is_available(), "cuda is not available on your machine with pytorch"
        torch.cuda.set_device(int(cuda_device))
    else:
        warnings.warn("You won't use cuda")
        if int(cuda_device) != 0:
            warnings.warn("You specified to use a cuda_device (\"cuda_device = XXX\") yet you tell the program not to use cuda (\"use_cuda = False\"). "
                          "This program will ignore the \"cuda_device = XXX\" directive.")
    return use_cuda

def build_dict_to_save(expe_name, name, g2op_env_name, n_storage, env_kwargs, PPO_kwargs, total_timesteps):
    """
    Create the dictionary with the hyperparameters used to train the model.
    This disctionary will then be saved in a json file.
    """
    dict_to_save = {}
    dict_to_save["expe_name"] = expe_name
    dict_to_save["name"] = name
    dict_to_save["n_storage"] = n_storage
    dict_to_save["g2op_env_name"] = g2op_env_name
    dict_to_save["total_timesteps"] = total_timesteps
    dict_to_save["env_kwargs"] = copy.deepcopy(env_kwargs)
    dict_to_save["PPO_kwargs"] = copy.deepcopy(PPO_kwargs)
    if "policy_kwargs" in dict_to_save["PPO_kwargs"].keys():
        if "activation_fn" in dict_to_save["PPO_kwargs"]["policy_kwargs"].keys():
            dict_to_save["PPO_kwargs"]["policy_kwargs"]["activation_fn"] = str(dict_to_save["PPO_kwargs"]["policy_kwargs"]["activation_fn"])
    return dict_to_save

def train_PPO(expe_name, name, env_kwargs, PPO_kwargs, 
              g2op_env_name="educ_case14_storage", 
              total_timesteps=5_000_000, 
              use_cuda=True, 
              cuda_device=0,
              save_every_xxx_steps = None
              ):
    """
    Train the model. Save the model, logs and used hyperparameters.
    """
    use_cuda = check_cuda(use_cuda, cuda_device)
    device = torch.device("cuda" if use_cuda else "cpu")

    # We will save the model here
    path_expe = os.path.join(MAIN_FOLDER, expe_name, name)
    
    # Define the training environment
    g2op_env = grid2op.make(g2op_env_name, test=True) # l2rpn_wcci_2022
    env = StorageEnv(g2op_env.max_episode_duration(),
                     g2op_env.n_storage,
                     g2op_env.storage_Emin,
                     g2op_env.storage_Emax,
                     g2op_env.storage_max_p_prod,
                     **env_kwargs)
    # env = StorageEnv(g2op_env.max_episode_duration(),
    #                  g2op_env.n_storage * 3,
    #                  np.tile(g2op_env.storage_Emin, 3),
    #                  np.tile(g2op_env.storage_Emax, 3),
    #                  np.tile(g2op_env.storage_max_p_prod, 3),
    #                  **env_kwargs)
    env.reset()
    
    # Initialize the model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=path_expe, gamma=0.999,
                **PPO_kwargs
                )
    
    # Define callbacks
    if save_every_xxx_steps is None:
        callbacks_list = None
    else:
        checkpoint_callback = CheckpointCallback(save_freq=save_every_xxx_steps,
                                                     save_path=path_expe,
                                                     name_prefix=name)
        callbacks_list = [checkpoint_callback]
    
    # Train and save the model
    model.learn(total_timesteps=total_timesteps, callback=callbacks_list)
    os.makedirs(path_expe, exist_ok=True)
    model.save(os.path.join(path_expe, name))
    
    # Save main hyperparameters
    dict_to_save = build_dict_to_save(expe_name, name, g2op_env_name, env.nb_storage, env_kwargs, PPO_kwargs, total_timesteps)
    with open(os.path.join(path_expe, "dict_hyperparameters.json"), 'x') as fp:
        json.dump(dict_to_save, fp, indent=4)

    
if __name__ == "__main__":
    # Choose to use or not cuda
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Initialize some default hyperparameters for the training environment and the training
    # total_timesteps = 20_000_000
    total_timesteps = 4096
    save_every_xxx_steps = min(total_timesteps // 10, 5000000)
    env_kwargs_default = dict(init_storage_capa_ratio=(0.1, 0.9),
                     ratio_setpoint=(0.2, 3),
                     smooth=12,
                     reward_shape="abs")
    PPO_kwargs_default = dict(learning_rate = 3e-6,
                policy_kwargs={"activation_fn": torch.nn.ReLU, 
                               "net_arch": {'pi': [300, 300, 300], 'vf': [300, 300, 300]}
                               },
                n_steps=32,
                )
    
    #### Experiment
    
    for i in range(2):
        expe_name = "case_118_final_agent_instances"
        name = f"rew_abs_lr_1e-6_n_steps_32_{i}"
        path_expe = os.path.join(expe_name, name)
        env_kwargs = copy.deepcopy(env_kwargs_default)
        PPO_kwargs = copy.deepcopy(PPO_kwargs_default)
        PPO_kwargs["n_steps"] = 32
        PPO_kwargs["learning_rate"] = 3e-6
        train_PPO(expe_name, name, env_kwargs, PPO_kwargs, 
                  g2op_env_name="l2rpn_idf_2023", # "l2rpn_wcci_2022"
                  total_timesteps=total_timesteps, 
                  use_cuda=True, cuda_device=i, 
                  save_every_xxx_steps=save_every_xxx_steps)

