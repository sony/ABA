'''
    This environment simulates the real data from logged Ad campaigns

    Considerations for the environment:

    1. Each episode is of 1 month the simulation moves to next month once the episode ends
    2. Actions : Budget allocation, Target CPA (in the next future step)
    3. Reward : Clicks, Pseudo conversion
    4. Observations : Performance data cost/conversion/history etc (undefined extend as per requirement)
    5. Data column format : date, campaign_name, cost, click, ctv, vtv (view through conversion) if available

    Specifications for the environment:

        Monthly budget setting : Sum of the total costs consumed by the campaigns in the month
        Reward : Modeled as a function of cost and clicks or cost and pseudo conversions
        Cost : Cost control is based on two simple rules as used by google ads (extend based on target cpa in the future)

'''

import numpy as np
import warnings
import pandas as pd 
import gym
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from simutil import fit_power_law, func, solve_optimal_allocation, fit_logistic, logistic_function, _bisection_method
from scipy.stats import ttest_ind
from matplotlib import cm
import os
import configparser

# Set the random seed for reproducibility of the gaussian learning process

class CampaignSimulatorEnv(gym.Env):
    """
    Description:
        The environment simulates the real data from Real logged Ad campaigns
    """
    def __init__(self, data, config, plot = False):
        # Create seperate dataframes for the campaigns in the data
        self.data = data
        self.plot = plot

        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')

        # Extract the start and the end date from the data
        self.start_date = self.data['date'].min()
        self.end_date = self.data['date'].max()

        print(f'Start date == {self.start_date} === End date == {self.end_date}')
        
        # Seperate the data by campaign name
        self.campaigns = self.data['campaign_name'].unique()


        # Data from current month
        self.campaign_cost_cmdata = {}
        self.campaign_click_cmdata = {}
        self.campaign_conversion_cmdata = {}
        self.campaign_vtv_cmdata = {}
        self.campaign_psconv_cmdata = {} # To maintain pseudo conversions
        self.models_click_current = {}
        self.models_psconv_current = {} # To maintain reward models for pseudo conversions

        # We want to maintain two data sources for the reward model in order to check non stationarity
        
        self.cost_from_data_reward = {} # Cost data used for learning the reward model
        self.click_from_data_reward = {} # Click data used for learning the reward model
        self.psconv_from_data_reward = {} # Pseudo conversion data used for learning the reward model

        # Store the cost values for the current campaign this will help in controlling the cost for the month
        self.current_cost = {}
        self.cumulative_results = {}

        # Initialize the past cost and click data for each campaign this is used to effectively learn the reward model by discarding some amount of the past data
        for campaign in self.campaigns:
            self.cost_from_data_reward[campaign] = []
            self.click_from_data_reward[campaign] = []
            self.psconv_from_data_reward[campaign] = []
            self.cumulative_results[campaign] = []

        self.monthly_budget = 0
        self.daily_budget_max = 0
        self.current_date = self.start_date
        self.current_days = 0
        self._day = 0 # This parameter is used for keeping track of the simulated time
        self.future_window_size = 20
        self.a_change_threshold = 1 # We will consider a change in the function if the a parameter changes by 100%
        self.learning_period = 20 # First 20 days of change no change point detection is performed
        self.changes = 0 # Number of changes detected in the function
        self.counter = 0 # Once the change is detected only do the detection after 20 days
        self.seed = config.getint('SIMULATOR', 'seed')
        self.noise = config.getfloat('SIMULATOR', 'noise') # 0.1 # Assuming small noise in the observed function (right now not using this)
        self.channel = config.get('SIMULATOR', 'data_channel')
        self.use_psudo_coversion = config.getboolean('SIMULATOR', 'use_psudo_conversion')
        self.weight = 0.3 # Weight to give to the view through conversion data

    def _set_seed(self, seed):
        np.random.seed(seed)


    '''
        The simulation proceeds as per month
        Each month the environment will provide the data for each campaign
        The monthly budget is adjusted as the sum of the monthly cost consumed by each campaign
    '''
    def reset(self):

        # set the seed for reproducibility
        self._set_seed(self.seed)

        # Calculate the date from start_date to end of current month
        self._day = 0
        self.monthly_budget = 0
        self.current_month = self.current_date.month
        self.current_year = self.current_date.year
        self.current_month_end = self.current_date + pd.offsets.MonthEnd(0)
        print(f'Current month == {self.current_date} === {self.current_month_end}')

        # Take the data from the current_month to the end of the month plus 20 days to understand abrupt change
        self.current_data = self.data[(self.data['date'] >= self.current_date) & (self.data['date'] <= self.current_month_end + pd.Timedelta(days=20))]

        # Set the current days of the month
        self.current_days = (self.current_month_end - self.current_date).days + 1

        self.observation = {}

        # Initialize the dictionaries for each campaign campaign_id : [] # cost data, clicks data, conversion data, view through convesion data
        for campaign in self.campaigns:
            self.campaign_cost_cmdata[campaign] = self.current_data[self.current_data['campaign_name']==campaign]['cost'].values
            self.campaign_click_cmdata[campaign] = self.current_data[self.current_data['campaign_name']==campaign]['click'].values
            self.campaign_conversion_cmdata[campaign] = self.current_data[self.current_data['campaign_name']==campaign]['ctv'].values

            self.observation[campaign] = [[], []] # list of costs and rewards for each campaign
            self.current_cost[campaign] = []
            if self.channel == 'smn':
                # for smn data since the spent amount is restricted we will use 1.5 times the maximum cost of the campaign as the monthly budget
                cost_max = self.current_data['cost'].max()
                self.monthly_budget = 1.5 * cost_max * 30.4
            else:
                self.monthly_budget += self.campaign_cost_cmdata[campaign].sum()
                self.campaign_conversion_cmdata[campaign] = self.campaign_conversion_cmdata[campaign].astype(float)
                # Add vtv only if campaign contains vtv values
                if 'vtv' in self.current_data.columns:
                    self.campaign_vtv_cmdata[campaign] = self.current_data[self.current_data['campaign_name']==campaign]['vtv'].values
                    # Add view through conversion data to the conversion data
                    self.campaign_conversion_cmdata[campaign] += self.weight * self.campaign_vtv_cmdata[campaign]

        
        # Calculate pseudo conversion for the data
        if self.use_psudo_coversion:
            self.calculate_pseudo_coversion()

        # Set the daily budget max as monthly budget / number of days in the month
        self.daily_budget_max = self.monthly_budget / abs(((self.current_month_end - self.current_date).days + 1))

        self.current_reward_function()

        print(f'Monthly budget of {self.current_month}-{self.current_year} == {self.monthly_budget} === Daily budget max == {self.daily_budget_max}')

        # Set the current_date to the next month and also handle the case when the current month is 12 year is incremented
        self.current_date = self.current_month_end + pd.Timedelta(days=1)
        self.current_date = self.current_date.replace(day=1)

        return self.observation

    '''
        Function to calculate pseudo conversions
    '''
    def calculate_pseudo_coversion(self):
        for campaign in self.campaigns:
            clicks = self.campaign_click_cmdata[campaign]
            conversions = self.campaign_conversion_cmdata[campaign]
            ps_conv = [] * len(clicks)
            for i in range(len(clicks)):
                if i < 6:
                    avg_clicks = np.mean(clicks[:i+1])
                    avg_clicks = max(avg_clicks, 1)
                    clicks[i] = max(clicks[i], 1)
                    ps_conv.append(clicks[i]/avg_clicks)
                else:
                    avg_clicks = np.mean(clicks[i-6:i+1]) # Average of last 7 days for clicks and conversions
                    avg_conversions = np.mean(conversions[i-6:i+1])
                    # Set avg clicks and coversion to minimum 1
                    avg_clicks = max(avg_clicks, 1)
                    avg_conversions = max(avg_conversions, 1)
                    clicks[i] = max(clicks[i], 1)
                    ps_conv.append(clicks[i] *  avg_conversions/ avg_clicks)
            # scale the pseudo conversion by 10 and convert to integer
            ps_conv = [x * 10 for x in ps_conv]
            self.campaign_psconv_cmdata[campaign] = ps_conv


    """
        Plot the results of GP learning
    """
    def plot_data(self, X, y, gp, campaign_name):

        # if EnvPlots directory does not exist create it
        if not os.path.exists('EnvPlots'):
            os.makedirs('EnvPlots')

        # Create a test vector from 0 to X.max (daily_budget_max)
        X_test = np.linspace(0, self.daily_budget_max , 100).reshape(-1, 1)
        y_pred = func(X_test, *gp)

        # y_pred, sigma = gp.predict(X_test, return_std=True)
        plt.figure()
        # Normalize the indices of the data points to create a gradient
        norm = plt.Normalize(0, len(X) - 1)
        colors = cm.viridis(norm(range(len(X))))  # You can use any colormap like 'viridis', 'plasma', etc.

        # Scatter the original data points
        plt.scatter(X, y, c=colors, s=100, edgecolor='k', label='Observations')

        # Plot the prediction
        plt.plot(X_test, y_pred, 'b-', label='Prediction')

        # Add a colorbar to represent the time gradient
        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='Data Point Order')
        
        plt.xlabel('Cost')

        plt.ylabel('Click')
        plt.title('Power Law Function '+str(campaign_name)+ ' month ' + str(self.current_month) + ' year ' + str(self.current_year))
        plt.legend()

        plt.savefig(f'EnvPlots/PowerLawFunction_{campaign_name}_{self.current_month}_{self.current_year}.png')

        # Only visualize the end of the month prediction
        if self._day == self.current_days:
            print(f'storing cummulative results ------------- {self._day} -------- {self.current_month} ------- {self.current_year}')
            self.cumulative_results[campaign_name].append((X_test, y_pred, self.current_month, self.current_year))
        
        plt.close()


    def plot_comparative_result(self, X_old, y_old, popt_old, X_new, y_new, popt_new, campaign_name, month, year):
        X_test = np.linspace(0, self.daily_budget_max , 100).reshape(-1, 1)
        y_pred_old = func(X_test, *popt_old)
        y_pred_new = func(X_test, *popt_new)
        # plot new and old data
        plt.figure()
        plt.plot(X_test, y_pred_old, 'r-', label='Old Prediction')
        plt.plot(X_test, y_pred_new, 'b-', label='New Prediction')
        plt.scatter(X_old, y_old, c='g', label='Old Observations') 
        plt.scatter(X_new, y_new, c='y', label='New Observations')
        plt.xlabel('Cost')
        plt.ylabel('Click')
        plt.title('Comparative Power Law Function '+str(campaign_name)+ ' month ' + str(month) + ' year ' + str(year))
        plt.legend()
        plt.savefig(f'EnvPlots/ComparativePowerLawFunction_{campaign_name}_{month}_{year}_{self.changes}.png')
        plt.close()



    """
        Function for plotting every months prediction to observe the change in the reward model
    """
    def plot_cumulative_results(self):
        for campaign in self.campaigns:
            # Generate different colors for each month
            colors = plt.cm.jet(np.linspace(0, 1, len(self.cumulative_results[campaign])))
            plt.figure()
            for data, color in zip(self.cumulative_results[campaign], colors):
                X_test, y_pred, month, year = data
                # Plot the results for each month

                # Find out the slope change in the data
                # reshape the data to a 1D array
                X_test = X_test.reshape(-1)
                y_pred = y_pred.reshape(-1)
                popt = fit_power_law(X_test, y_pred)
                plt.plot(X_test, y_pred, '-', label=f' {month}/{year}', color=color)

            plt.xlabel('Cost', fontsize=14)
            plt.ylabel('Click', fontsize=14)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=10)  # Move legend to upper left to avoid overlap with the plot title
            plt.savefig(f'EnvPlots/CumulativeGaussianProcess_{campaign}.png')
            plt.close()

    """
        Function to update for pseudo conversions using a power law model
        The logic follows similar to updating click models
    """
    def update_pseudo_conversion(self, campaign):

        # For the first month take the data as it is 
        if len(self.cost_from_data_reward[campaign]) == 0:
            self.cost_from_data_reward[campaign] = self.campaign_cost_cmdata[campaign].tolist()
            self.psconv_from_data_reward[campaign] = self.campaign_psconv_cmdata[campaign]
        else:
            # Create a non overlapping window of the future data
            data_future_cost = self.campaign_cost_cmdata[campaign][self._day:self._day + self.future_window_size]
            data_future_psconv = self.campaign_psconv_cmdata[campaign][self._day:self._day + self.future_window_size]

            # Check if the parameters for the old data and future data are different
            popt_old = fit_power_law(self.cost_from_data_reward[campaign], self.psconv_from_data_reward[campaign])
            if len(data_future_cost) != len(data_future_psconv):
                popt_new = fit_power_law(data_future_cost, data_future_psconv[:len(data_future_cost)])
            else:
                popt_new = fit_power_law(data_future_cost, data_future_psconv)

            a_old = popt_old[0]
            a_new = popt_new[0]

            change_percentage_parameter = abs(a_new - a_old) / a_old

            if change_percentage_parameter > self.a_change_threshold and self.counter >= 20:
                self.changes += 1
                print(f'Change detected for pseudo conversion in the function for {campaign} at date {self.current_date + pd.Timedelta(days=self._day)} with a_old = {a_old} b_old = {popt_old[1]} a_future = {a_new} b_future = {popt_new[1]}')

                # Use the data from the future to learn the new model
                self.cost_from_data_reward[campaign] = self.campaign_cost_cmdata[campaign][self._day:self._day + self.future_window_size].tolist()
                cost_data_length = len(self.cost_from_data_reward[campaign])
                self.psconv_from_data_reward[campaign] = self.campaign_psconv_cmdata[campaign][self._day:self._day + cost_data_length]
                self.counter = 0
            else:
                # keep adding the data to the reward model
                if(len(self.campaign_cost_cmdata[campaign]) > self._day):
                    self.cost_from_data_reward[campaign].append(self.campaign_cost_cmdata[campaign][self._day])
                    self.psconv_from_data_reward[campaign].append(self.campaign_psconv_cmdata[campaign][self._day])      



    """
        Function to update the GP policy as per a and b parameters
    """
    def update_campaign_data(self, campaign):

        # For the first month take the data as it is
        if len(self.cost_from_data_reward[campaign]) == 0:
            self.cost_from_data_reward[campaign] = self.campaign_cost_cmdata[campaign].tolist()
            self.click_from_data_reward[campaign] = self.campaign_click_cmdata[campaign].tolist()
        else:
            # Create a non overlapping window of the future data
            data_future_cost = self.campaign_cost_cmdata[campaign][self._day:self._day + self.future_window_size]
            data_future_click = self.campaign_click_cmdata[campaign][self._day:self._day + self.future_window_size]

            # Check if the parameters for the old data and future data are different
            popt_old = fit_power_law(self.cost_from_data_reward[campaign], self.click_from_data_reward[campaign])
            popt_new = fit_power_law(data_future_cost, data_future_click)

            a_old = popt_old[0]
            a_new = popt_new[0]

            change_percentage_parameter = abs(a_new - a_old) / a_old

            # print the percentage difference between a_old and a_new
            # print(f'Percentage difference in a_old === {a_old} and a_new === {a_new} for {campaign} ==== parameter change ===== {change_percentage_parameter}')


            # Check if the parameters differ by a certain threshold discard the data and relearn the model also print the date on which the change is detected and lets restrict this change 
            # Assuming a change in 
            if change_percentage_parameter > self.a_change_threshold and self.counter >= 20:
                self.changes += 1

                print(f'Change detected in the function for {campaign} at date {self.current_date + pd.Timedelta(days=self._day)} with a_old = {a_old} b_old = {popt_old[1]} a_future = {a_new} b_future = {popt_new[1]}')
                if self.plot:
                    self.plot_comparative_result(self.cost_from_data_reward[campaign], self.click_from_data_reward[campaign], popt_old, data_future_cost, data_future_click, popt_new, campaign, self.current_month, self.current_year)

                # Use the data from the future to learn the new model
                self.cost_from_data_reward[campaign] = self.campaign_cost_cmdata[campaign][self._day:self._day + self.future_window_size].tolist()
                self.click_from_data_reward[campaign] = self.campaign_click_cmdata[campaign][self._day:self._day + self.future_window_size].tolist()
                self.counter = 0
            else:
                # keep adding the data to the reward model
                if(len(self.campaign_cost_cmdata[campaign]) > self._day):
                    self.cost_from_data_reward[campaign].append(self.campaign_cost_cmdata[campaign][self._day])
                    self.click_from_data_reward[campaign].append(self.campaign_click_cmdata[campaign][self._day])




    """
        Function to learn the gaussian model for the reward function using cost as a feature
    """
    def _learn_click_model(self, cost_data, reward_data, campaign_name):

        popt = fit_power_law(cost_data, reward_data) # fit_power_law(cost_data, reward_data)
        if self.plot:
            self.plot_data(cost_data, reward_data, popt, campaign_name)
        return popt


    '''
        Function to learn the pseudo conversion model if use_psudo_conversion is set to True
    '''
    def _learn_pseudo_conversion_model(self, cost_data, ps_conversion_data, campaign_name):
        popt = fit_power_law(cost_data, ps_conversion_data) # fit_power_law(cost_data, reward_data)
        if self.plot:
            self.plot_data(cost_data, ps_conversion_data, popt, campaign_name)
        return popt
        

    '''
        The reward function is estimated from the cost and the click of the actual data 
    '''
    def current_reward_function(self):  
        # Build a reward model for each campaign using the current cost and reward data
        for campaign in self.campaigns:
            if self.use_psudo_coversion:
                self.update_pseudo_conversion(campaign)
                self.models_psconv_current[campaign] = self._learn_pseudo_conversion_model(self.cost_from_data_reward[campaign], self.psconv_from_data_reward[campaign], campaign)
            else:
                self.update_campaign_data(campaign)
                self.models_click_current[campaign] = self._learn_click_model(self.cost_from_data_reward[campaign], self.click_from_data_reward[campaign], campaign)
        self.counter += 1

    '''
        Function to simulate the daily cost for each campaign 
        This is based on the following rules:
        1. Your daily spending limit (two times your average daily budget for most campaigns) on any particular day. (but cost is rarely this high)
        2. Your monthly spending limit (30.4 times your average daily budget for most campaigns) in any particular month.
        A truncated normal distribution is used to simulate the daily cost
    '''
    def simulate_daily_cost(self, campaign_budget : float, campaign : str):

        # Define the daily spending limit

        daily_spending_limit = 1.2 * campaign_budget

        # Parameters for truncated normal distribution
        lower, upper = 0, daily_spending_limit
        mu, sigma = campaign_budget, campaign_budget * 0.3 # mean and standard deviation

        # Generate a guess of daily costs based on the current budget and limit it to the daily spending limit
        daily_costs = truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(self.current_days)
        
        # Ensure the monthly spending limit is not exceeded take current cost upto self._day and add predicted costs for the remaining days
        daily_costs[:self._day - 1] = self.current_cost[campaign][:self._day - 1]

        total_cost = np.sum(daily_costs)
        if total_cost > self.monthly_budget:
            scaling_factor = self.monthly_budget / total_cost
            daily_costs = daily_costs * scaling_factor

        # return one cost at random from the daily costs
        return daily_costs[self._day - 1]
        

    '''
        Function to assign the observation (cost) and reward (clicks) for the action chosen by the policy
        action = {campaign_id : budget} (extend this to target cpa later)
        return : observation, reward, done, info
        rewrard and observation is as per individual campaign
        Observation : capaign id: list of cost and list of clicks per month
    '''
    def step(self, action):

        self._day += 1

        total_reward = 0

        for campaign_id, budget in action.items():
            estimated_reward = 0

            # Simulate the daily cost for the campaign

            daily_cost = self.simulate_daily_cost(budget, campaign_id)
            
            # Get the reward from the reward model

            if self.use_psudo_coversion:
                estimated_reward = func(daily_cost, *self.models_psconv_current[campaign_id]) + np.random.normal(0, self.noise)
            else:
                estimated_reward = func(daily_cost, *self.models_click_current[campaign_id]) + np.random.normal(0, self.noise) # Add a small random noise to the reward
            self.current_cost[campaign_id].append(daily_cost)

            self.observation[campaign_id][0].append(daily_cost)
            self.observation[campaign_id][1].append(estimated_reward)

            total_reward += estimated_reward

        # find the optimal clicks that can be obtained from the budget
        if self.use_psudo_coversion:
            optimal_allocation, optimal_reward = solve_optimal_allocation(self.models_psconv_current, self.daily_budget_max, len(self.campaigns))
        else:
            optimal_allocation, optimal_reward = solve_optimal_allocation(self.models_click_current, self.daily_budget_max, len(self.campaigns)) # _bisection_method(self.models_click_current, self.daily_budget_max, len(self.campaigns))

        regret = optimal_reward - total_reward
        # print(f'Optimal allocation ===== {optimal_allocation} Optimal reward == {optimal_reward} === Total reward == {total_reward} === Regret == {regret}')

        regret = max(0, regret)

        # Update the reward model every day
        self.current_reward_function()

        # End episode if day is greater than the current days (ToDo also add the constraint of budget being exhausted)
        done = (self._day >= self.current_days)
        
        return self.observation, total_reward, regret, done, False, {}
        

def plot_budget_cost(campaign_budget, campaign_cost, iteration):

    # Plot the budget and cost for each day for each campaign

    for campaign, budget in campaign_budget.items():
        plt.figure()
        plt.plot(budget, label='Budget')
        plt.plot(campaign_cost[campaign], label='Cost')
        plt.xlabel('Day')
        plt.ylabel('Amount')
        plt.title(f'Budget and Cost for {campaign}')
        plt.legend()
        plt.savefig(f'EnvPlots/BudgetCost_{campaign}_{iteration}.png')
        plt.close()



def simple_policy(observation, env):
    """
    A simple policy that allocates the budget equally among the campaigns for testing.
    """
    campaigns = observation.keys()
    num_campaigns = len(campaigns)
    action = {campaign: env.daily_budget_max / num_campaigns for campaign in campaigns}
    return action

if __name__ == '__main__':
    # Load the data from different campaigns 
    # Currently campaigns of AKASHI are used which have brand, non-brand and general campaigns
     # Read the config file
    config = configparser.ConfigParser()

    try:
        config.read('./configpolicy.ini')
    except Exception as e:
        print('Error reading the config file', e)

    # Analysing SMN data
    data_smn = pd.read_csv('data/smn_data_cleaned_updated.tsv', sep='\t')
    print(data_smn.columns)
    data_smn = data_smn[data_smn['advertiser_id'] == 17276]
    data_google = pd.read_csv('data/aianalysistool_campaign_data.csv')

    data = data_google # data_smn # data_google

    start_date_simulator = str(data['date'].min())
    end_date_simulator = str(data['date'].max())
    env = CampaignSimulatorEnv(data, config, True)

    for i in range(16):
        # Simulate for 
        observation = env.reset()
        campaign_budget = {campaign : [] for campaign in env.campaigns}
        campaign_cost = {campaign : [] for campaign in env.campaigns}

        days = env.current_days

        for day in range(days):
            # print(f'day ======== {day}')
            action = simple_policy(observation, env)
            print(f'Action == {action}')
            observation, reward, regret ,  done, _ , info = env.step(action)
            # print(f'Observation == {observation} === Reward == {reward} === Done == {done} === Info == {info}')
            for campaign, data in observation.items():
                campaign_budget[campaign].append(action[campaign])
                campaign_cost[campaign].append(data[0])
        
        
    env.plot_cumulative_results()