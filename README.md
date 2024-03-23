# Emulation_Zonal_Controllers_paper
This repository aims to reproduce the results presented in the "Emulation of Zonal Controllers for the Power System Transport Problem" paper.

# Python libraries
We recommend using a dedicated Python virtual environment. You can install the needed libraries with the command:

`pip install -r requirements.txt`

# Organization of the code

The code is organized in two independent parts:
1. The folder `1_following_the_plan_without_powergrid` concerns the agent that aims at following a storage charge plan without operating the grid (section 5.3)
2. The folder `2_emulation_of_a_zonal_controller` concerns the agents that aim at operating the grid with or without a plan (sections 5.2 and 5.4): the Noplan agent, the LightCurtailment agent and the WithPlan agent.

In both cases, the training or evaluation code has to be launched in the concerned folder. For example, don't launch `A_train_PPO.py` if you are in the root folder. Do :

```
cd [your_directory]/Emulation_Zonal_Controllers_paper/1_following_the_plan_without_powergrid/
python3 A_train_PPO.py
```

Files start with a letter, we suggest following this order to go through the training and evaluation steps. 


# Number of iterations / Number of instances / Number of test scenarios.

We set the number of training iterations at 4096 to obtain very short training times. To get efficient agents, you need to set this hyper-parameters (`nb_iter`) to 1e7.

In the paper, we trained 6 instances per type of agent (NoPlan / LightCurtailment / WithPlan), which led to 18 trainings. You can choose the number of instances with the `NB_TRAINING` predefined in `A_prep_env.py`. Its default value here is 1.

The `C_evaluate_agents.py` script evaluates only on 2 scenarios but you can set `nb_scenario = 34` (like in our experiments).

# Used hyper-parameters


| First Header  | Second Header |
| ------------- | ------------- |
| RL algorithm  | PPO (Actor Critic)  |
| Model type  | Multi-Layer-Perceptron  |
| Non shared hidden layers  | 3 of 300 neurons each  |
| Activation function  | tanh  |
| Batch size | 2048  |
| Gamma  | 0.999  |
| Epochs loss optimization  | 10  |
| Rollout buffer size (n_steps) | 32 |
| Optimizer  | ADAM  |
| Normalization of the observation space  | Yes  |
| Normalization of the action space  | Yes  |
