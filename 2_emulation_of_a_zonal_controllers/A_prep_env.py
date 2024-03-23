import os
import json
import sys
import numpy as np
import grid2op
from grid2op.dtypes import dt_int
from grid2op.Agent import RecoPowerlineAgent
from grid2op.utils import EpisodeStatistics, ScoreICAPS2021, ScoreL2RPN2020
from lightsim2grid import LightSimBackend

is_windows = sys.platform.startswith("win32")

env_name = "l2rpn_idf_2023"

# We need coefficients to normalize our observations. 
# To get them, we lauch the Recopowerline agent on the validation set and compute 
# average and standard deviation for each coordinate of the observation space.
# The normalization coefficients are then saved in the file named preprocess_act.json.
# If you want to recompute these coefficients, change this parameter to True.
recompute_normalization_coefficients = False

deep_copy = is_windows  # force the deep copy on windows (due to permission issue in symlink in windows)




if __name__ == "__main__":
    # create the environment 
    
    env = grid2op.make(env_name)

    # split into train / val / test
    # it is such that there are 34 chronics for val and 34 for test
    env.seed(1)
    env.reset()
    try:
        nm_train, nm_val, nm_test = env.train_val_split_random(add_for_test="test",
                                                           pct_val=4.2,
                                                           pct_test=4.2,
                                                           deep_copy=deep_copy)
    except Exception:
        pass
    
    nm_val = env_name + "_val"
    
    if recompute_normalization_coefficients:
        max_int = max_int = np.iinfo(dt_int).max
        env_val = grid2op.make(nm_val, backend=LightSimBackend())
        nb_scenario = len(env_val.chronics_handler.subpaths)
        print(f"{nm_val}: {nb_scenario} scenarios")

        # compute statistics for reco powerline from the validation set
        env_seeds = env.space_prng.randint(low=0,
                                high=max_int,
                                size=nb_scenario,
                                dtype=dt_int)
        reco_powerline_agent = RecoPowerlineAgent(env_val.action_space)
        stats_reco = EpisodeStatistics(env_val, name_stats="reco_powerline")
        stats_reco.compute(nb_scenario=nb_scenario,
                        agent=reco_powerline_agent,
                        env_seeds=env_seeds,
                        pbar=True,
                        )
        
        # compute the normalization coefficients from the statistics and save
        dict_ = {"subtract": {}, 'divide': {}}
        for attr_nm in ["gen_p", "load_p", "p_or", "rho",
                        "timestep_overflow", "line_status",
                        "actual_dispatch", "target_dispatch",
                        "storage_charge", "storage_power",
                        "curtailment", "curtailment_limit",  "gen_p_before_curtail",
                        ]:
            avg_ = stats_reco.get(attr_nm)[0].mean(axis=0)
            std_ = stats_reco.get(attr_nm)[0].std(axis=0)
            dict_["subtract"][attr_nm] = [float(el) for el in avg_]
            dict_["divide"][attr_nm] = [max(float(el), 1.0) for el in std_]
        
        with open("preprocess_obs.json", "w", encoding="utf-8") as f:
            json.dump(obj=dict_, fp=f)
            
        act_space_kwargs = {"add": {"redispatch": [0. for gen_id in range(env.n_gen) if env.gen_redispatchable[gen_id]],
                                    "set_storage": [0. for _ in range(env.n_storage)]},
                            'multiply': {"redispatch": [max(float(el), 1.0) for gen_id, el in enumerate(env.gen_max_ramp_up) if env.gen_redispatchable[gen_id]],
                                        "set_storage": [max(float(el), 1.0) for el in env.storage_max_p_prod]}
                        }
        with open("preprocess_act.json", "w", encoding="utf-8") as f:
            json.dump(obj=act_space_kwargs, fp=f)
