'''
    This is the developed policy based on the simulation environment using real logged data
    The policy uses Gaussian Process Regression to learn the reward from offline data and MCK to allocate budget
'''

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel, ConvergenceWarning
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pulp
from pulp import PULP_CBC_CMD
from utility import get_log_likelihood, simple_prediction_test
from matplotlib import cm
from baselines import BOCD

# Suppress ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Designing a saturating mean function for the Gaussian Process
# Modeling saturating mean function
class SaturatingMean:
    def __init__(self, y_max, x_max):
        self.y_max = y_max
        self.x_max = x_max

    # Saturate the function over the last observed highest budget level as per y_max
    def __call__(self, x):
        return np.where( x > self.x_max, self.y_max, 0)

class CustomGaussianProcess(GaussianProcessRegressor):
    def __init__(self, kernel=None, x_max=0, **kwargs):
        super().__init__(kernel=kernel, **kwargs)
        self.y_max = 0
        self.x_max = x_max

    def predict(self, X, return_std=False, return_cov=False):
        y_pred, std = super().predict(X, return_std=True)
        self.y_max = np.max(y_pred)
        # Use the saturating mean function to predict the y_value
        self.mean_func = SaturatingMean(self.y_max, self.x_max)
        
        # Modify y_pred using the SaturatingMean function
        y_pred = np.maximum(y_pred, self.mean_func(X.flatten()))

        if return_std:
            return y_pred, std
        else:
            return y_pred


class GPPolicyOptimization():

    def __init__(self, data, config):
                
        self.offline_data = data # pd.concat([data_brand, data_nonbrand, data_display])
        self.offline_data['date'] = pd.to_datetime(self.offline_data['date'], errors='coerce')
        self.current_date = self.offline_data['date'].min()
        self.current_month = self.current_date.month
        self.current_year = self.current_date.year
        self.beta = config.getint('POLICY', 'beta') # High beta for exploration (keep this adaptive later)
        self.policy_type = config['POLICY']['policy_type'] # 'cpt' for change point detection
        self.use_prior_data = False
        self.seed = config.getint('POLICY', 'seed')
        self.channel = config.get('SIMULATOR', 'data_channel')

        # Dictionaries for the campaign data
        self.reward_models = {}
        self.cost_data = {}
        self.click_data = {}
        self.budget_data = {}
        self.avg_cpc = {}
        self.non_stationarity_detection_threshold = config.getint('POLICY', 'non_stationarity_detection_threshold')
        self.learning_points = config.getint('POLICY', 'learning_data_points')

        self.budget_granualarity = 500

        # To be set every month based on the camapign data

        self.monthly_budget = 0
        self.daily_budget_max = 0

        # Store the previous action
        self.prev_action = {}

        # Intialize click and cost data for the different channels
        self.campaigns = self.offline_data['campaign_name'].unique()

        for campaign in self.campaigns:

            # Let us start the campaign from begining without the use of any prior data

            if self.use_prior_data:
                self.cost_data[campaign] = self.offline_data[self.offline_data['campaign_name'] == campaign]['cost'].values
                self.click_data[campaign] = self.offline_data[self.offline_data['campaign_name'] == campaign]['click'].values
                self.budget_data[campaign] = self.offline_data[self.offline_data['campaign_name'] == campaign]['campaign_budget_total_amount'].values
                self.reward_models[campaign] = self.learn_reward_model(self.cost_data[campaign], self.click_data[campaign], campaign, True)
            else:
                self.cost_data[campaign] = []
                self.click_data[campaign] = []
                self.budget_data[campaign] = []
            self.avg_cpc[campaign] = 0

    def _set_seed(self, seed):
        np.random.seed(seed)


    def reset(self, env):
        self._set_seed(self.seed)
        self.env = env # For regret calculation store the current env the policy is working on
        self.current_month = env.current_month
        self.current_year = env.current_year
        self.daily_budget_max = env.daily_budget_max
        self.monthly_budget = env.monthly_budget

        # Maintain a list of current cost and reward of 7 days
        self.cost_data_current = {}
        self.click_data_current = {}
        for campaign in self.campaigns:
            self.cost_data_current[campaign] = np.array([])
            self.click_data_current[campaign] = np.array([])


    """
        Plot the results of GP learning
    """
    def plot_data(self, X, y, gp, campaign_name, data_type):

        # Create a test vector from 0 to X.max
        X_test = np.linspace(0, self.daily_budget_max, 100).reshape(-1, 1)
        y_pred, sigma = gp.predict(X_test, return_std=True)

        plt.figure()
        # Normalize the indices of the data points to create a gradient
        norm = plt.Normalize(0, len(X) - 1)
        colors = cm.viridis(norm(range(len(X))))  # You can use any colormap like 'viridis', 'plasma', etc.

        # Scatter the original data points
        plt.scatter(X, y, c=colors, s=100, edgecolor='k', label='Observations')

        # Plot the prediction
        plt.plot(X_test, y_pred, 'b-', label='Prediction')
        plt.fill_between(X_test.ravel(), y_pred - 1.96*sigma, y_pred + 1.96*sigma,
                        alpha=0.2, color='k', label='90% confidence interval')

        # Add a colorbar to represent the time gradient
        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='Data Point Order')
        
        plt.xlabel('Cost')

        plt.ylabel('Click')
        plt.title('Gaussian Process Regression '+ str(campaign_name) + ' month ' + str(self.current_month) + ' year ' + str(self.current_year))
        plt.legend()
        plt.savefig(f'PolicyPlots/GaussianProcessPolicy_{str(campaign_name)}_{self.current_month}_{self.current_year}_{data_type}.png')
        plt.close()


    def learn_reward_model(self, cost_data, reward_data, campaign_name, plot_function, data_type):
        # Learn the reward function using Gaussian Process Regression

        X = np.array(cost_data).reshape(-1, 1)
        y = np.array(reward_data)

        # Always add the point with (0,0) to the data
        X = np.append(X, [[0], [0], [0], [0]], axis=0)
        y = np.append(y, [0, 0, 0, 0])
        x_max = np.max(X)

        kernel = C(1.0, (1e-4, 1e1)) * (Matern(length_scale=1.0, nu=1.5) + RBF(length_scale=1.0)) # C(1.0, (1e-4, 1e1)) * (Matern(length_scale=1.0, nu=1.5) + RBF(length_scale=1.0))
        gp = CustomGaussianProcess(kernel=kernel, x_max=x_max, n_restarts_optimizer=10, random_state=42)
        gp.fit(X, y)

        # Plot the data
        if plot_function:
            self.plot_data(X, y, gp, campaign_name, data_type)

        return gp
    
    def predict_reward_budgetlevel(self, budget_levels):
        reward_dict = {}
        budget_levels = np.array(budget_levels).reshape(-1, 1)
        for campaign in self.campaigns:
            reward_max, cost_max = 0, 0
            gp = self.reward_models[campaign]
            y_pred, sigma = gp.predict(budget_levels, return_std=True)

            # Find the current maximum predicted reward
            max_y_pred_index = np.argmax(y_pred)
            max_y_pred_budget = budget_levels.flatten()[max_y_pred_index]


            # Apply UCB only for budget levels higher than the budget level of max_y_pred with avg_cpc as the exploration parameter
            # The idea is to explore more for highly efficient algorithms
            
            # Normalize the avg_cpc to be between 0 and 1 with respect to other campaigns and avoid division by zero
            normalized_avg_cpc = self.avg_cpc[campaign] / max(max(self.avg_cpc.values()), 1)
            # print(f'normalized_avg_cpc ====={campaign}==== {normalized_avg_cpc}')

            # ucb_normal = y_pred + self.beta * (1 - normalized_avg_cpc) * sigma

            ucb_reward = y_pred + np.where(budget_levels.flatten() > max_y_pred_budget, self.beta * (1 - normalized_avg_cpc) * sigma, 0)

            reward_dict[campaign] = ucb_reward # y_pred # ucb_normal # ucb_reward # y_pred # ucb_reward

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

    '''
        Function to calculate avg CPC of each campaign 
    ''' 
    def get_avg_cpc(self):
        for campaign in self.campaigns:
            total_cost = sum(self.cost_data[campaign])
            total_clicks = sum(self.click_data[campaign])
            # handle cases where there are no clicks
            total_clicks = max(total_clicks, 1)
            self.avg_cpc[campaign] = total_cost / total_clicks

    '''
        Since cost consumed by the platform 
    '''
    def calculate_remaining_budget(self, observation, day):
        total_consumed_cost = 0
        # Find the total cost consumed by all the campaigns
        for campaign in self.campaigns:
            total_consumed_cost += np.array(observation[campaign][0]).sum()

        # Calculate the remaining budget
        remaining_budget = self.monthly_budget - total_consumed_cost
        self.daily_budget_max = remaining_budget / (self.num_days - (day - 1))

        # Cap the daily budget to (monthly budget / num of days of the month) * 2
        self.daily_budget_max = min(self.daily_budget_max, self.monthly_budget / self.num_days * 2)


    '''
        Choose the budget allocation based on the current reward model and LP optimization
    '''
    def get_action(self, observation, day, suggested_action=[]):

        # update the cost and click for the agent based on the latest observation
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
                    self.cost_data_current[campaign] = self.cost_data_current[campaign][-7:]
                    self.click_data_current[campaign] = self.click_data_current[campaign][-7:]

        
        # log likelihood check 
        if self.policy_type == 'cpt':
            # Use the current data to learn a new model
            new_gp_models = {}
            for campaign in self.campaigns:
                self.reward_models[campaign] = self.learn_reward_model(self.cost_data[campaign], self.click_data[campaign], campaign, False, 'old')

                # Learn only when there are atleast some available data points
                if len(self.cost_data_current[campaign]) > self.learning_points:
                    new_gp_models[campaign] = self.learn_reward_model(self.cost_data_current[campaign], self.click_data_current[campaign], campaign, False, 'new')
                    # Calculate the log likelihood of the new model and the old model
                    # likelihood_ratio = get_log_likelihood(self.reward_models[campaign], new_gp_models[campaign], self.cost_data_current[campaign], self.click_data_current[campaign], self.env, campaign)
                    
                    avg_diff, p_value = simple_prediction_test(self.reward_models[campaign], new_gp_models[campaign], self.cost_data_current[campaign], self.daily_budget_max, self.current_month)

                    # compare the reward learnt by the policy and the reward learnt by the environment
                    current_date = self.env.current_date + pd.Timedelta(days=self.env._day)

                    # use new data as cost
                    if avg_diff > self.non_stationarity_detection_threshold:

                        print(f'Change detected in campaign {campaign} mae ========= {avg_diff}')
                        self.cost_data[campaign] = self.cost_data_current[campaign]
                        self.click_data[campaign] = self.click_data_current[campaign]
                    
            self.get_avg_cpc()

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