# Utility functions for additional calculations

import matplotlib.pyplot as plt
# import wandb
from scipy.stats import norm, ttest_rel
import numpy as np
from simutil import fit_power_law, func
    

def plot_rewards(reward):
    plt.plot(reward)
    plt.xlabel('Days')
    plt.ylabel('Reward')
    plt.title('Reward vs Days')
    plt.savefig('SimPlots/reward.png')
    plt.close()

# Utility functions for policy with the log likelihood function

def plot_gp(gp_model_old, gp_model_new, budget_level, current_month):
    X_test = np.linspace(0, budget_level, 1000).reshape(-1, 1)
    y_pred, _ = gp_model_old.predict(X_test, return_std=True)
    y_pred_new, _ = gp_model_new.predict(X_test, return_std=True)
    plt.plot(X_test, y_pred, label='Prediction')
    plt.plot(X_test, y_pred_new, label='New Prediction')
    plt.xlabel('Budget')
    plt.ylabel('Reward')
    plt.title(f'GP Model Comparison Prediction date_{current_month}')
    plt.legend()
    plt.savefig(f'SimPlots/GPModelComparison_{current_month}.png')
    plt.close()

def get_log_likelihood(gp_model_old, gp_model_new, new_cost, new_reward, env, campaign):
    '''
        Calculate the log likelihood ratio for the old and new models
        parameters:
        - gp_model_old: the old Gaussian Process model
        - gp_model_new: the new Gaussian Process model
        - new_cost: the new cost
        - new_reward: the new reward
        returns:
        - log_likelihood_ratio: the log likelihood ratio
    '''
    new_cost = new_cost.reshape(-1, 1)
    
    # get the log likelihood of the old model for new data
    y_pred_old, sigma_old = gp_model_old.predict(new_cost, return_std=True)
    log_likelihood_old = np.sum(norm.logpdf(new_reward, loc = y_pred_old, scale=sigma_old * 1000))

    # get the log likelihood of the new model for new data
    y_pred_new, sigma_new = gp_model_new.predict(new_cost, return_std=True)
    log_likelihood_new = np.sum(norm.logpdf(new_reward, loc=y_pred_new, scale=sigma_new * 1000))
    compare_reward_plots_gp(gp_model_old, gp_model_new, env.daily_budget_max, env.current_date, env.current_month, campaign)

    # Calculate the log likelihood ratio
    log_likelihood_ratio = -2 * (log_likelihood_old - log_likelihood_new)
    print(f'log_likelihood_ratio: {log_likelihood_ratio}')

    return log_likelihood_ratio

'''
    A simple test to check if the prediction between the old and new models are consistent
    parameters: old_gp_model, new_gp_model, new_cost, new_reward
    returns: True if the prediction is consistent, False otherwise
'''
def simple_prediction_test(gp_model_old, gp_model_new, new_cost, budget_level, current_month):
    # print(f'New Cost: {new_cost[0]}')
    X_test = np.linspace(0, budget_level, 1000).reshape(-1, 1)
    # Take the first value of the new cost
    # pred_value = new_cost[0].reshape(-1, 1)
    # new_cost = np.array([5000]).reshape(-1, 1) # new_cost.reshape(-1, 1)
    y_pred, _ = gp_model_old.predict(X_test, return_std=True)
    y_pred_new, _ = gp_model_new.predict(X_test, return_std=True)
    # y_pred_old = gp_model_old.predict(pred_value)
    # y_pred_new = gp_model_new.predict(pred_value)
    # plot the predictions
    # plot_gp(gp_model_old, gp_model_new, budget_level, current_month)

    # 1. Mean avg value to detect if the prediction is consistent
    avg_diff = np.mean(np.abs(y_pred - y_pred_new))
    print(f'MEAN DIFF: {avg_diff}')
    # 2. Paired t-test to check if the prediction is consistent
    # t_stat, p_value = ttest_rel(y_pred_old, y_pred_new)
    p_value = 0
    return avg_diff, p_value


# Compare the reward estimated by the policy and the true reward function
def compare_reward_plots(policy_gp, env_gp, budget_level, current_date, current_month):
    '''
        Compare the reward estimated by the policy and the true reward function
        parameters:
        - policy_gp: the policy Gaussian Process model
        - env_gp: the environment Gaussian Process model
        - budget_level: the budget level
    '''
    for campaign in policy_gp.keys():
        # generate x values upto budget level with 1000 points
        x = np.linspace(0, budget_level, 1000).reshape(-1, 1)
        y_policy, _ = policy_gp[campaign].predict(x, return_std = True)
        y_env = func(x , *env_gp[campaign]) # env_gp[campaign].predict(x, return_std = True)

        # plot the reward functions
        plt.plot(x, y_policy, label='Policy')
        plt.plot(x, y_env, label='Environment')
        plt.xlabel('Budget')
        plt.ylabel('Reward')
        plt.title(f'{campaign} Policy and environment Reward Function date_{current_date}')
        plt.legend()
        plt.savefig(f'SimPlots/ComparisonPlots_{campaign}_{current_month}.png')
        plt.close()

    # Compare the reward estimated by the policy and the true reward function
def compare_reward_plots_gp(old_gp, new_gp, budget_level, current_date, current_month, campaign):
    '''
        Compare the reward estimated by the policy and the true reward function
        parameters:
        - policy_gp: the policy Gaussian Process model
        - env_gp: the environment Gaussian Process model
        - budget_level: the budget level
    '''
    # generate x values upto budget level with 1000 points
    x = np.linspace(0, budget_level, 1000).reshape(-1, 1)
    y_policy, _ = old_gp.predict(x, return_std = True)
    y_new, _ = new_gp.predict(x, return_std = True)

    # plot the reward functions
    plt.plot(x, y_policy, label='Policy')
    plt.plot(x, y_new, label='New')
    plt.xlabel('Budget')
    plt.ylabel('Reward')
    plt.title(f'{campaign} Old Policy and New Reward Function date_{current_date}')
    plt.legend()
    plt.savefig(f'SimPlots/ComparisonPlots_{campaign}_{current_month}.png')
    plt.close()