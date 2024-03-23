# %%

from evaluation_function import *
import pandas as pd

#%% 

load_path = "./saved_models/"
nb_scenario = 2
res_dict = {}

# %%

res_dict["NoPlan_agent_0"] = evaluate_agent("NoPlan_agent_0", 
                CustomGymEnv, 
                GymEnvWithSetPointRecoDN, 
                load_path=load_path,
                nb_scenario=nb_scenario,
                show_tqdm=False,
                )

# %%

res_dict["LightCurtailment_agent_0"] = evaluate_agent("LightCurtailment_agent_0", 
                CustomGymEnv, 
                GymEnvWithSetPointRecoDN, 
                load_path=load_path,
                nb_scenario=nb_scenario,
                show_tqdm=False,
                )

# %%

res_dict["WithPlan_agent_0"] = evaluate_agent("WithPlan_agent_0", 
                GymEnvWithSetPointRemoveCurtail, 
                GymEnvWithSetPointRemoveCurtail,
                load_path=load_path,
                nb_scenario=nb_scenario,
                show_tqdm=False,
                )

# %%
# COMPUTE RESULTS IN TABLE 3

pd.DataFrame({"name":res_dict.keys(), 
 "survival_time T_end":[v['Average timesteps survived:'] for v in res_dict.values()],
'Curtailment Limits CL':[v['Average limits for curtailment'] for v in res_dict.values()],
"D_StorageCharge":[v['Average reward'] for v in res_dict.values()],
             
             
             })