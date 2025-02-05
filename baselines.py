# Implementation of a set of baseline for the Combinatorial Bandit budget allocation problem
# 1. The first baseline is the actions taken by the human operator from the dataset
# 2. The second baseline is the GP-UCB policy with sliding window
# 3. The third baseline is the GP-TS policy with sliding window

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, ConstantKernel as C, WhiteKernel, ConvergenceWarning
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pulp
from pulp import PULP_CBC_CMD
from scipy.stats import norm
from utility import get_log_likelihood, simple_prediction_test
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

''' --------------- Class for Bayesian Change Point Detection --------------- '''
class BOCD:
    def __init__(self, hazard, prior_mean, prior_var):
        self.hazard = hazard # probability of change occuring at each timestep
        self.prior_mean = prior_mean  # The initial mean of the reward distribution (before seeing any data).
        self.prior_var = prior_var # The initial variance of the reward distribution.
        self.run_length_probs = np.array([1.0]) # Start with run length = 0 with probability 1.

    def update(self, observation):
        T = len(self.run_length_probs) # Time step
        pred_means = np.zeros(T)
        pred_vars = np.zeros(T)

        # Predictive step: Calculate the predictive mean and variance for all run lengths
        for t in range(T):
            pred_means[t] = self.prior_mean
            pred_vars[t] = self.prior_var + 1 # Gaussian likelihood with unit variance

        # Observation likelihood for the new data point
        observation_likelihoods = norm.pdf(observation, loc=pred_means, scale=np.sqrt(pred_vars))

        # Growth step: Propagate the probabilities for the current run length
        growth_probs = self.run_length_probs * (1 - self.hazard) * observation_likelihoods

        # Change step: Calculate the probabilities of a new change point
        new_run_prob = np.sum(self.run_length_probs *self.hazard * observation_likelihoods)

        # Update run-length probabilities
        self.run_length_probs = np.append(growth_probs, new_run_prob)
        self.run_length_probs /=np.sum(self.run_length_probs) # Normalize

        # Update the mean and variance (assuming Gaussian update for simplicity)
        self.prior_mean = (self.prior_mean * T + observation) / (T + 1)
        self.prior_var = 1/(1/self.prior_var + 1)

    def detect_change_point(self):
        # If a change point id likely reset
        return self.run_length_probs[-1] > 0.5


''' ------- Code for simulating the actions taken by a human operator form the dataset ------- '''
class HumanPolicy():
    def __init__(self, data):
        self.data = data
        self.current_action = {}

    def reset(self, env):
        pass

    def get_action(self, current_date, campaigns):
        human_actions = {}
        for campaign in campaigns:
            human_data = self.data[(self.data['date'] == current_date) & (self.data['campaign_name'] == campaign)]

            if human_data.empty:
                if campaign not in human_actions.keys():
                    self.current_action[campaign] = 1 # adding a dummy value if data does not exist
                human_actions[campaign] = self.current_action[campaign]
            else:
                human_actions[campaign] = human_data['campaign_budget_total_amount'].values[0]

        # print(f'Human actions ======= {human_actions}')
        self.current_action = human_actions

        return human_actions

''' -------------- Code for GP policy with sliding window and different exploration strategy and adaptive strategy -------------- 

GP Kernel is vanilla GP kernel without saturating kernel and guided ucb exploration

1. No exploration : ne
2. Thompson sampling : ts
3. Upper Confidence Bound : ucb

Different adaption strategies

1. Sliding window : sliding_window
2. Discounted reward : discount the reward based on the time
3. Log likelihood test : log_likelihood_test
4. mae test : Mean average error test
5.
'''
class GPUCBPolicyWithSlidingWindow():
    def __init__(self, data, config, exploration_type='ucb', adaptation_type='sliding_window'):
                
        self.offline_data = data # pd.concat([data_brand, data_nonbrand, data_display])
        self.offline_data['date'] = pd.to_datetime(self.offline_data['date'], errors='coerce')
        self.current_date = self.offline_data['date'].min()
        self.current_month = self.current_date.month
        self.current_year = self.current_date.year
        self.adaptation_type = adaptation_type # Type of reward function update policy
        self.exploration_type = exploration_type
        self.beta = config.getint('POLICY', 'beta') # High beta for exploration
        self.use_prior = False
        self.seed = config.getint('POLICY', 'seed')
        self.channel = config.get('SIMULATOR', 'data_channel')

        # Dictionaries for the campaign data
        self.reward_models = {}
        self.cost_data = {}
        self.click_data = {}
        self.budget_data = {}
        self.avg_cpc = {}
        self.gamma = 0.9

        self.budget_granualarity = 500

        # To be set every month based on the camapign data

        self.monthly_budget = 0
        self.daily_budget_max = 0

        # Store the previous action
        self.prev_action = {}

        # Intialize click and cost data for the different channels
        self.campaigns = self.offline_data['campaign_name'].unique()
        self.bocd_models = {campaign: BOCD(hazard=0.01, prior_mean=0, prior_var=1) for campaign in self.campaigns}

        for campaign in self.campaigns:

            if self.use_prior:
                self.cost_data[campaign] = self.offline_data[self.offline_data['campaign_name'] == campaign]['cost'].values
                self.click_data[campaign] = self.offline_data[self.offline_data['campaign_name'] == campaign]['click'].values
                self.budget_data[campaign] = self.offline_data[self.offline_data['campaign_name'] == campaign]['campaign_budget_total_amount'].values
                self.reward_models[campaign] = self.learn_reward_model(self.cost_data[campaign], self.click_data[campaign], campaign, True)
            else:
                self.cost_data[campaign] = [0]
                self.click_data[campaign] = [0]
                self.budget_data[campaign] = []

            self.avg_cpc[campaign] = 0

            # print(f'campaign ========== {len(self.cost_data[campaign])} ======== {len(self.click_data[campaign])} ======== {len(self.budget_data[campaign])}')

    def reset(self, env):

        self.env = env # For regret calculation store the current env the policy is working on
        self.current_month = env.current_month
        self.current_year = env.current_year
        self.daily_budget_max = env.daily_budget_max
        np.random.seed(self.seed)

        # Maintain a list of current cost and reward of 7 days
        self.cost_data_current = {}
        self.click_data_current = {}
        for campaign in self.campaigns:
            self.cost_data_current[campaign] = np.array([])
            self.click_data_current[campaign] = np.array([])

    def learn_reward_model(self, cost_data, reward_data, campaign_name, plot_function):
        # Learn the reward function using Gaussian Process Regression

        X = np.array(cost_data).reshape(-1, 1)
        y = np.array(reward_data)

        kernel = (Matern(length_scale=1.0, nu=1.5) + RBF(length_scale=1.0))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        gp.fit(X, y)

        return gp

    def predict_reward_budgetlevel(self, budget_levels):
        reward_dict = {}
        budget_levels = np.array(budget_levels).reshape(-1, 1)
        for campaign in self.campaigns:
            gp = self.reward_models[campaign]
            y_pred, sigma = gp.predict(budget_levels, return_std=True)
            explore_reward = y_pred

            if self.exploration_type == 'ucb':
                explore_reward = y_pred + self.beta * sigma

            if self.exploration_type == 'ts':
                sampled_rewards = gp.sample_y(budget_levels, n_samples=1).flatten()
                explore_reward = sampled_rewards

            reward_dict[campaign] = explore_reward # ucb_reward # y_pred

        return reward_dict
    

    '''
            Function to calculate the optimal budget allocation using the predicted rewards
            While maintaining the budget constraint of not exceeding the daily budget limit
            args : Dictionary of Gaussian Process models
            Output: Dictionary of optimal budget allocation
    '''
    def optimization(self):
        
        budget_levels = np.arange(0, self.daily_budget_max + 1, self.budget_granualarity)
        reward_dict = self.predict_reward_budgetlevel(budget_levels)

        num_budgets = len(budget_levels)

        # Campaign IDs
        campaign_ids = list(reward_dict.keys())

        # Create the problem
        prob = pulp.LpProblem("BudgetAllocation", pulp.LpMaximize)

        # Decision variables (allocation percentage)
        allocation_vars = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat='Binary') for j in range(num_budgets)] for i in range(len(campaign_ids))]

        # Objective function
        prob += pulp.lpSum(allocation_vars[i][j] * reward_dict[campaign_ids[i]][j]
                   for i in range(len(campaign_ids)) for j in range(num_budgets))

        # Constraints : Each campaign's total budget should not exceed budget_max
        for i in range(len(campaign_ids)):
            prob += pulp.lpSum(allocation_vars[i][j] * budget_levels[j] for j in range(num_budgets)) <= self.daily_budget_max

        # Each campaign must allocate its budget to exactly one level
        for i in range(len(campaign_ids)):
            prob += pulp.lpSum(allocation_vars[i][j] for j in range(num_budgets)) == 1

        # Total budget allocation must not exceed the daily budget limit
        prob += pulp.lpSum(allocation_vars[i][j] * budget_levels[j] for i in range(len(campaign_ids)) for j in range(num_budgets)) <= self.daily_budget_max

        # Solve the problem and suprress the output message
        prob.solve(PULP_CBC_CMD(msg=False))

        # Extract the optimal budget allocation
        budget_allocation = {}

        for i in range(len(campaign_ids)):
            for j in range(num_budgets):
                if pulp.value(allocation_vars[i][j]) > 0:
                    budget_allocation[campaign_ids[i]] = budget_levels[j]
                    break

        # Extract optimal results
        optimal_reward = pulp.value(prob.objective)

        return budget_allocation, optimal_reward


    def get_action(self, observation, day):
        if day > 0:
            for campaign in self.campaigns:
                self.cost_data[campaign] = np.append(self.cost_data[campaign], observation[campaign][0][-1])
                self.click_data[campaign] = np.append(self.click_data[campaign], observation[campaign][1][-1])
                self.budget_data[campaign] = np.append(self.budget_data[campaign], self.prev_action[campaign])

                # Maintain a list of current cost and reward of 7 days
                self.cost_data_current[campaign] = np.append(self.cost_data_current[campaign], observation[campaign][0][-1])
                self.click_data_current[campaign] = np.append(self.click_data_current[campaign], observation[campaign][1][-1])

                if len(self.cost_data_current[campaign]) > 7:
                    # take only the last 7 days data
                    self.cost_data_current[campaign] = self.cost_data_current[campaign][-10:]
                    self.click_data_current[campaign] = self.click_data_current[campaign][-10:]

        if self.adaptation_type == 'no_change_detection':
            for campaign in self.campaigns:
                self.reward_models[campaign] = self.learn_reward_model(self.cost_data[campaign], self.click_data[campaign], campaign, True)

        # Passive Approch: Sliding window with a 
        if self.adaptation_type == 'sliding_window':
            for campaign in self.campaigns:
                self.reward_models[campaign] = self.learn_reward_model(self.cost_data[campaign], self.click_data[campaign], campaign, True)
                # Taking a sliding window of 10 days
                if len(self.cost_data[campaign]) > 30:
                    # print(f'cost_data ========= {len(self.cost_data[campaign])}')
                    self.cost_data[campaign] = self.cost_data[campaign][10:]
                    self.click_data[campaign] = self.click_data[campaign][10:]
                    # print(f'cost_data ========= {len(self.cost_data[campaign])}')

        # Code for discounting reward as per time
        if self.adaptation_type == 'discounted_reward':
            for campaign in self.campaigns:
                # Adjust the importance of cost and click based on the time
                time_steps = len(self.cost_data[campaign])

                # Generate discount weights for the past observations
                discount_weights = np.array([self.gamma**(time_steps - i -1) for i in range(time_steps)])

                # Reweight cost and click data with the discount factor
                discounted_costs = self.cost_data[campaign] * discount_weights
                discounted_clicks = self.click_data[campaign] * discount_weights

                self.reward_models[campaign] = self.learn_reward_model(discounted_costs, discounted_clicks, campaign, True)

        # Code for Bayesian Change Point Detection
        if self.adaptation_type == 'bayesian_change_point':
            for campaign in self.campaigns:
                if len(observation[campaign][0]) > 1:
                    new_observation = observation[campaign][0][-1]

                    # Update the BOCD model with the new observation
                    self.bocd_models[campaign].update(new_observation)

                    # Detect if there is a change point
                    if self.bocd_models[campaign].detect_change_point():
                        print(f'Change point detected for campaign {campaign} at day {day}')
                        self.cost_data[campaign] = np.array([])
                        self.click_data[campaign] = np.array([])
                self.reward_models[campaign] = self.learn_reward_model(self.cost_data[campaign], self.click_data[campaign], campaign, False)

        if self.adaptation_type == 'mae_test':
            # Use the current data to learn a new model
            new_gp_models = {}
            for campaign in self.campaigns:
                self.reward_models[campaign] = self.learn_reward_model(self.cost_data[campaign], self.click_data[campaign], campaign, False)

                # Learn only when there are atleast some available data points
                if len(self.cost_data_current[campaign]) > 6:
                    new_gp_models[campaign] = self.learn_reward_model(self.cost_data_current[campaign], self.click_data_current[campaign], campaign, False)
                    # Calculate the log likelihood of the new model and the old model
                    # likelihood_ratio = get_log_likelihood(self.reward_models[campaign], new_gp_models[campaign], self.cost_data_current[campaign], self.click_data_current[campaign])
                    avg_diff, p_value = simple_prediction_test(self.reward_models[campaign], new_gp_models[campaign], self.cost_data_current[campaign], self.daily_budget_max, self.current_month)

                    # compare the reward learnt by the policy and the reward learnt by the environment
                    current_date = self.env.current_date + pd.Timedelta(days=self.env._day)

                    # use new data as cost
                    if avg_diff > 10:
                        print(f'Change detectesd in campaign {campaign} MAE_DIFF ========= {avg_diff}')
                        self.cost_data[campaign] = self.cost_data_current[campaign]
                        self.click_data[campaign] = self.click_data_current[campaign]


        # Code for log likehood test for change detection
        if self.adaptation_type == 'log_likelihood_test':
            new_gp_models = {}
            for campaign in self.campaigns:
                self.reward_models[campaign] = self.learn_reward_model(self.cost_data[campaign], self.click_data[campaign], campaign, False)

                # Learn only when there are atleast some available data points
                if len(self.cost_data_current[campaign]) > 9:
                    new_gp_models[campaign] = self.learn_reward_model(self.cost_data_current[campaign], self.click_data_current[campaign], campaign, False)
                    # Calculate the log likelihood of the new model and the old model
                    likelihood_ratio = get_log_likelihood(self.reward_models[campaign], new_gp_models[campaign], self.cost_data_current[campaign], self.click_data_current[campaign])

                    print(f'likelihood_ratio ===={campaign}===== {likelihood_ratio}')
                    # use new data as cost
                    if likelihood_ratio > 10:
                        print(f'Change detectesd in campaign {campaign} likelihood_ratio ========= {likelihood_ratio}')
                        self.cost_data[campaign] = self.cost_data_current[campaign]
                        self.click_data[campaign] = self.click_data_current[campaign]



        budget_allocation, optimal_reward = self.optimization()

        # Normalize the budget allocation
        total_budget = sum(budget_allocation.values())
        for campaign in self.campaigns:
            budget_allocation[campaign] = budget_allocation[campaign] * self.daily_budget_max / total_budget

        # Make sure each campaign gets atleast 500 (for google channel) 50 (for smn channel) budget adjust from campaign with highest budget
        if self.channel == 'smn':
            for campaign in self.campaigns:
                if budget_allocation[campaign] < 100:
                    budget_allocation[campaign] = 100
                    budget_allocation[max(budget_allocation, key=budget_allocation.get)] -= 100
        else:
            for campaign in self.campaigns:
                if budget_allocation[campaign] < 500:
                    budget_allocation[campaign] = 500
                    budget_allocation[max(budget_allocation, key=budget_allocation.get)] -= 500


        self.prev_action = budget_allocation
        print(f' budget_allocation ========= {budget_allocation}')


        return budget_allocation, optimal_reward



''' -------------- Simple Heuristic Baseline that samples the highest point from budget allocation -------------- '''
class SimpleBaselinePolicy():
    def get_action(self, day):
        pass