# %%

from evaluation_function import *
import pandas as pd

from A_prep_env import NB_TRAINING

#%% 

load_path = "./saved_models/"
nb_scenario = 2
res_dict = {}

# %%

res_dict["DoNothing_agent"] = evaluate_agent("NoPlan_agent_0", 
                CustomGymEnv, 
                GymEnvWithSetPointRecoDN, 
                load_path=load_path,
                nb_scenario=nb_scenario,
                show_tqdm=False,
                DoNothing_agent=True,
                )
res_dict["DoNothing_agent"].update({"agent_type":"DoNothing"})
# %%

for i in range(NB_TRAINING):
    res_dict[f"NoPlan_agent_{i}"] = evaluate_agent(f"NoPlan_agent_{i}", 
                    CustomGymEnv, 
                    GymEnvWithSetPointRecoDN, 
                    load_path=load_path,
                    nb_scenario=nb_scenario,
                    show_tqdm=False,
                    )
    res_dict[f"NoPlan_agent_{i}"].update({"agent_type":"NoPlan_agent"})

# %%

for i in range(NB_TRAINING):
    res_dict[f"LightCurtailment_agent_{i}"] = evaluate_agent(f"LightCurtailment_agent_{i}", 
                    CustomGymEnv, 
                    GymEnvWithSetPointRecoDN, 
                    load_path=load_path,
                    nb_scenario=nb_scenario,
                    show_tqdm=False,
                    )
    res_dict[f"LightCurtailment_agent_{i}"].update({"agent_type":"LightCurtailment_agent"})

# %%
for i in range(NB_TRAINING):
    res_dict[f"WithPlan_agent_{i}"] = evaluate_agent(f"WithPlan_agent_{i}", 
                    GymEnvWithSetPointRemoveCurtail, 
                    GymEnvWithSetPointRemoveCurtail,
                    load_path=load_path,
                    nb_scenario=nb_scenario,
                    show_tqdm=False,
                    )
    res_dict[f"WithPlan_agent_{i}"].update({"agent_type":"WithPlan_agent"})

# %%

table_res = pd.DataFrame({"name":res_dict.keys(), 
              "agent_type":[v["agent_type"] for v in res_dict.values()],
                "survival_time T_end":[v['Average timesteps survived:'] for v in res_dict.values()],
                'Curtailment_Limits_CL':[v['Average limits for curtailment'] for v in res_dict.values()],
                "D_StorageCharge":[v['Average reward'] for v in res_dict.values()],             
            })
table_res = table_res.groupby(["agent_type"]).mean().reset_index()
# %%

print(table_res)