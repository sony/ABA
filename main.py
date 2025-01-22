from simulator import CampaignSimulatorEnv
from policy import GPPolicyOptimization
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


    if data_channel == "google":
        advertiser_id = config['DEFAULT']['campaign_name']
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

    # Calculate the number of months to simulate
    num_month_simulate = (pd.to_datetime(end_date_simulator) - pd.to_datetime(start_date_simulator)).days // 30
    print(f'Number of months to simulate: {num_month_simulate}')

    # Initialize the environment
    env = CampaignSimulatorEnv(data_simulator, config, False)
    env_window = 'random_changes_to_function_10_TargetedExplorationWithSaturatingMeanUCB'+data_channel

    # Initialize the policy
    policy = GPPolicyOptimization(data_policy, config)

    reward_over_entire_duration = []

    if use_wandb:
        wandb.init(project='Campaign Simulator New', name='TUCBMAE - '+advertiser_id+'-'+data_channel)

        # Save the configuration parameters to wandb
        wandb.config.update({section: dict(config[section]) for section in config.sections()})

    # Some metrics to compare the campaign performance

    cummilative_reward = 0
    cummilative_regret = 0
    cummilative_cost = 0
    cummilative_cpc = 0

    for i in range(num_month):
        print(f'Simulating month {i+1}')
        observation = env.reset()

        # set policy parameters as per env data
        policy.reset(env)

        # llm_agent = LLM_Agent(env.current_month)

        days = env.current_days

        for day in range(days):

            action, optimal_expected = policy.get_action(observation, day)
            # print(f'action ======== {action} ======== {optimal_expected}')

            observation, reward, regret, done, _ , info = env.step(action)

            for campaign in observation.keys():
                cummilative_cost += observation[campaign][0][-1]

            cpc = cummilative_cost/cummilative_reward if cummilative_reward > 0 else 0

            cummilative_reward += reward
            cummilative_regret += regret
            cummilative_cpc += cpc

            reward_over_entire_duration.append(reward)

            if use_wandb:
                wandb.log({
                    'reward': reward,
                    'regret': regret,
                    'cumulative_reward': cummilative_reward,
                    'cumulative_regret': cummilative_regret,
                    'cumulative_cpc': cummilative_cpc
                })


            # log_metrics_daily(reward)
            print(f'actual environment reward ========= {reward}')
            if done:
                break
