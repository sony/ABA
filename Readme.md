# Adaptive Budget Optimization for Multichannel Advertising Using Combinatorial Bandits (AAMAS (Extended Abstract) 2025)

This is the code contribution for simulation of long running ad campaigns and TUCBMAE algorithm

# System Requirements

The code has been tested in systems with the following OS

- Ubuntu 20.04.2 LTS

# Dependencies:

Run the following command to create and activate the environment:

```
$ conda create --name tucbmae
$ conda activate tucbmae
$ python -m pip install -r requirements.txt
$ pip install -e .

```

# Hyperparameter configurations

The hyperparameters can be set in configpolicy. The hyperparameters are described as follows:

num_month : The number of months the campaigns are to be simulated. For example 16 for AI Prediction analysis tool and 1 for Criterio dataset. 

data_google : Path to data from Platform source A ex : data/aianalysistool_campaign_data.csv

data_smn :Path to data from Platform source B

data_criterio: Path to criteriob dataset ex: data/criterio_data_filtered.csv

use_wandb : Enables login results to wandb

data_channel : Depends on data source can be can be google (Platform A), smn(Platform B), criterio depending on platform source

baseline_name : GPUCBBaseline_Policy (For GP based baseline) or Human (for comparison with human allocation)

exploration_strategy : ucb (for UCB exploration) or ts (for thompson sampling exploration)

adaptation_strategy : Can be any of the following options - discounted_reward sliding_window no_change_detection mae_test

noise : Controls the noise level in the reward function default 0.1

use_psudo_conversion : If set to True uses pseudo_conversion as reward otherwise uses clicks

non_stationarity_detection_threshold : Threshold for mae test

beta : exploration factor default 2

learning_data_points : Number of data points to learn the current model default 6


# Simulation Environment

Considerations for the environment:

    1. Each episode is of 1 month the simulation moves to next month once the episode ends
    2. Actions : Budget allocation
    3. Reward : Clicks, Pseudo conversion
    4. Observations : Performance data cost/conversion/history etc (extend as per requirement)
    5. Data column format : date, campaign_name, cost, click, ctv, vtv (view through conversion) if available
    6. Tp = 20 (future window size and stationary days) and change threshold = 1 bedget granualarity=500

    Specifications for the environment:

        Monthly budget setting : Sum of the total costs consumed by the campaigns in the month
        Reward : Modeled as a function of cost and clicks or cost and pseudo conversions
        Cost : Cost control is based on two simple rules as used by google ads

# Reproducing results of TUCBMAE algorithm and baselines

For TUCBMAE algorithm run

```
$ python main.py
```

For baselines run

```
$ python eval_baseline.py
```
