import os
import re
import json
import numpy as np
import copy
from grid2op.Action import PlayableAction
from utils import *

from A_prep_env import NB_TRAINING


env_name = "l2rpn_idf_2023_train"
save_path = "./saved_models" 
max_iter = 7 * 24 * 12  # This parameter is set so that each scenario lasts a week      

# the heuristic and the PenalizationStorage term of the reward are implemented in the gym environment
gymenv_class = GymEnvWithSetPointRemoveCurtail

safe_max_rho = 0.9  # the grid is said "safe" if the rho is lower than this value, it is a really important parameter to tune !
curtail_margin = 30  # it is a really important parameter to tune !

####### /!\
# We set nb_iter = 4096 to check our code quickly, but to obtain a good agent you need 
# to set nb_iter = 10_000_000 and. The training will last several hours.
#######

nb_iter = 4096
# nb_iter = 10_000_000 # the NoPlan agent, LightCurtailment agent and WithPlan agent were trained with 10 000 000 iterations

reward_class = PenalizeSetpointPosReward
    
if __name__ == "__main__":
    
    import grid2op
    from lightsim2grid import LightSimBackend  # highly recommended !
    from grid2op.Chronics import MultifolderWithCache  # highly recommended for training
    import torch
    
    torch.cuda.set_device(3)
    
    # attributes taken into account in the observation
    obs_attr_to_keep = ["month", "day_of_week", "hour_of_day", "minute_of_hour",
                        "gen_p", "load_p", 
                        "p_or", "rho", "timestep_overflow", "line_status",
                        # dispatch part of the observation
                        "actual_dispatch", "target_dispatch",
                        # storage part of the observation
                        "storage_charge", "storage_power",
                        # curtailment part of the observation
                        "curtailment", "curtailment_limit",  "gen_p_before_curtail",
                        # the setpoint we have to follow
                        "storage_setpoint"
                        ]
    # attributes of the possible actions
    act_attr_to_keep = ["curtail", "set_storage"]
    
    # parameters for the training
    learning_rate = 3e-5
    net_arch = {'pi': [300, 300, 300], 'vf': [300, 300, 300]}
    
    env = grid2op.make(env_name,
                       action_class=PlayableAction,
                       reward_class=reward_class,
                       backend=LightSimBackend(),
                       chronics_class=MultifolderWithCache,
                       )
    
    # loading coefficients used to normalize observations and actions
    with open("preprocess_obs.json", "r", encoding="utf-8") as f:
        obs_space_kwargs = json.load(f)
    # we need this line because "storage_setpoint" is not in the attributes by default
    n_storage = env.n_storage # without this line, the PPO model is much heavier (600 Mo)
    obs_space_kwargs["functs"] ={"storage_setpoint": (lambda grid2opobs: np.zeros(n_storage),
                                                            0., 1.0, None, None)}
    with open("preprocess_act.json", "r", encoding="utf-8") as f:
        act_space_kwargs = json.load(f)
        
    # hyper-parameters specific to the heuristic
    gymenv_kwargs = {"safe_max_rho": safe_max_rho, "curtail_margin": curtail_margin, "reward_cumul": "sum"}
    # hyper-parameters specific to GymEnvWithSetpoint[something] classes
    gymenv_kwargs["weight_penalization_storage"] = 0.25 # Coefficient of the penalty
    gymenv_kwargs["very_safe_max_rho"] = 0.85 
    gymenv_kwargs["penality_storage_bound"] = 0.5 # Maximum value of the penalty including the coefficient
    
    
    # we train only on the february month (a very cold month)
    env.chronics_handler.real_data.set_filter(lambda x: re.match(r".*2035-02-.*$", x) is not None)
    env.chronics_handler.real_data.reset()
    # see https://grid2op.readthedocs.io/en/latest/environment.html#optimize-the-data-pipeline
    # for more information !
    
    env.set_max_iter(max_iter)  # set the duration of a scenario to one week
    env.reset()      
    
    print("environment loaded !")
    
    for i in range(NB_TRAINING):
            trained_agent = train(
                    env,
                    iterations=nb_iter,
                    logs_dir="./logs",
                    save_path=save_path, 
                    obs_attr_to_keep=obs_attr_to_keep,
                    obs_space_kwargs=copy.deepcopy(obs_space_kwargs),
                    act_attr_to_keep=act_attr_to_keep,
                    act_space_kwargs=copy.deepcopy(act_space_kwargs),
                    normalize_act=True,
                    normalize_obs=True,
                    name=f"WithPlan_agent_{i}",
                    learning_rate=learning_rate,
                    net_arch=net_arch,
                    save_every_xxx_steps=min(nb_iter // 10, 200_000),
                    verbose=1,
                    gamma=0.999,
                    gymenv_class=gymenv_class,
                    gymenv_kwargs=gymenv_kwargs,
                    # batch_size = batch_size,
                    # n_steps = n_steps
                    )