from simulator import CampaignSimulatorEnv
from baselines import HumanPolicy, GPUCBPolicyWithSlidingWindow
import pandas as pd
import wandb
import configparser


if __name__ == '__main__':

    # Read the config file
    config = configparser.ConfigParser()

    try:
        config.read('./configpolicy.ini')
    except Exception as e:
        print('Error reading the config file', e)

    # Num months to simulate the campaign
    num_month = config.getint('DEFAULT', 'num_month')
    use_wandb = config.getboolean('DEFAULT', 'use_wandb')
    data_channel = config['DEFAULT']['data_channel']
    policy_name = config['BASELINE']['baseline_name']
    exploration_strategy = config['BASELINE']['exploration_strategy']
    adaptation_strategy = config['BASELINE']['adaptation_strategy']
    


    if data_channel == "google":
        advertiser_id = config['DEFAULT']['campaign_name']
        # Load from the config file to initialize the simulator
        data_simulator = pd.read_csv(config['DEFAULT']['data_google'])
        data_policy = data_simulator

    if data_channel == "smn":
        data = pd.read_csv(config['DEFAULT']['data_smn'], sep='\t')
        # Filter data as per specified advertiser ID
        advertiser_id = config['DEFAULT']['advertiser_id']
        data_simulator = data[data['advertiser_id'] == int(advertiser_id)]
        data_policy = data_simulator
        
    if data_channel == "criterio":
        advertiser_id = config['DEFAULT']['campaign_name']
        data_simulator = pd.read_csv(config['DEFAULT']['data_criterio'])
        data_policy = data_simulator
    else:
        RuntimeError("Invalid data channel")

    start_date_simulator = data_simulator['date'].min()
    end_date_simulator = data_simulator['date'].max()

    # Initialize the environment
    env = CampaignSimulatorEnv(data_simulator, config, False)
    policy = None

    if policy_name == 'Human_Policy':
        # Initialize the baseline policy
        policy = HumanPolicy(data_simulator)

    if policy_name == 'GPUCBBaseline_Policy':
        policy = GPUCBPolicyWithSlidingWindow(data_policy, config, exploration_strategy, adaptation_strategy)

    if use_wandb:
        wandb.init(project='Campaign Simulator New', name=policy_name+'-'+exploration_strategy+'-'+adaptation_strategy+'-'+advertiser_id+'-'+data_channel,
                   group=adaptation_strategy)
        
         # Save the configuration parameters to wandb
        wandb.config.update({section: dict(config[section]) for section in config.sections()})

    cummilative_reward = 0
    cummilative_regret = 0
    cummilative_cost = 0
    cummilative_cpc = 0

    for i in range(num_month):
        print(f'Simulating month {i+1}')
        observation = env.reset()
        # set policy parameters as per env data
        policy.reset(env)

        days = env.current_days

        for day in range(days):

            if policy_name == 'Human_Policy':
                action = policy.get_action(env.current_date, env.campaigns)
                # print(f'action ======== {action} ======== {optimal_expected}')
            else:
                action, _ = policy.get_action(observation, day)

            observation, reward, regret, done, _ , info = env.step(action)

            for campaign in observation.keys():
                # Add the latest cost incurred to cummulative cost
                cummilative_cost += observation[campaign][0][-1]

            cpc = cummilative_cost/cummilative_reward if cummilative_reward > 0 else 0

            cummilative_reward += reward
            cummilative_regret += regret
            cummilative_cpc += cpc


            if use_wandb:
                wandb.log({
                    'reward': reward,
                    'regret': regret,
                    'cumulative_reward': cummilative_reward,
                    'cumulative_regret': cummilative_regret,
                    'cumulative_cpc': cummilative_cpc
                })

            if done:
                break
