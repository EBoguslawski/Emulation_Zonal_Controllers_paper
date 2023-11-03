import os
import re
import json
from grid2op.Action import PlayableAction
from utils import *


env_name = "l2rpn_idf_2023_train"
save_path = "./saved_models"
max_iter = 7 * 24 * 12  # None to deactivate it    

# parameters for the training
gymenv_class = CustomGymEnv

safe_max_rho = 0.9  # the grid is said "safe" if the rho is lower than this value, it is a really important parameter to tune !
curtail_margin = 30  # it is a really important parameter to tune !

nb_iter = 4096
# nb_iter = 10_000_000

reward_class = PenalizeSetpointPosReward

if __name__ == "__main__":
    
    import grid2op
    from l2rpn_baselines.PPO_SB3 import train
    from lightsim2grid import LightSimBackend  # highly recommended !
    from grid2op.Chronics import MultifolderWithCache  # highly recommended for training
    import torch
    
    torch.cuda.set_device(3)
    
    # you can change below (full list at https://grid2op.readthedocs.io/en/latest/observation.html#main-observation-attributes)
    obs_attr_to_keep = ["month", "day_of_week", "hour_of_day", "minute_of_hour",
                        "gen_p", "load_p", 
                        "p_or", "rho", "timestep_overflow", "line_status",
                        # dispatch part of the observation
                        "actual_dispatch", "target_dispatch",
                        # storage part of the observation
                        "storage_charge", "storage_power",
                        # curtailment part of the observation
                        "curtailment", "curtailment_limit",  "gen_p_before_curtail",
                        ]
    # same here you can change it as you please
    act_attr_to_keep = ["curtail", "set_storage"]
    
    # parameters for the learning
    learning_rate = 3e-5
    net_arch = {'pi': [300, 300, 300], 'vf': [300, 300, 300]}
    
    env = grid2op.make(env_name,
                       action_class=PlayableAction,
                       reward_class=reward_class,
                       backend=LightSimBackend(),
                       chronics_class=MultifolderWithCache,
                       )
    
    with open("preprocess_obs.json", "r", encoding="utf-8") as f:
        obs_space_kwargs = json.load(f)
    with open("preprocess_act.json", "r", encoding="utf-8") as f:
        act_space_kwargs = json.load(f)
        
    gymenv_kwargs = {"safe_max_rho": safe_max_rho, "curtail_margin": curtail_margin, "reward_cumul": "sum"}    
    
    # train on all february month, why not ?
    env.chronics_handler.real_data.set_filter(lambda x: re.match(r".*2035-02-.*$", x) is not None)
    env.chronics_handler.real_data.reset()
    # see https://grid2op.readthedocs.io/en/latest/environment.html#optimize-the-data-pipeline
    # for more information !
    
    if max_iter is not None:
        env.set_max_iter(max_iter)  # one week
    obs = env.reset()    
    
    
    print("environment loaded !")
    for i in range(5):
            trained_agent = train(
                    env,
                    iterations=nb_iter,
                    logs_dir="./logs",
                    save_path=save_path, 
                    obs_attr_to_keep=obs_attr_to_keep,
                    obs_space_kwargs=obs_space_kwargs,
                    act_attr_to_keep=act_attr_to_keep,
                    act_space_kwargs=act_space_kwargs,
                    normalize_act=True,
                    normalize_obs=True,
                    name=f"LightCurtailment_agent_{i}",
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