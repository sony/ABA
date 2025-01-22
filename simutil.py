from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


np.random.seed(42)


def func(x, a, b):
    epsilon = 1e-10
    return a * (x + epsilon) ** b

def fit_power_law(x, y):
    x = np.array(x)
    y = np.array(y)

    # Append points 0 and 0 to the x and y arrays
    x = np.append(x, [0,0])
    y = np.append(y, [0,0])
    
    popt, pcov = curve_fit(func, x, y, maxfev=10000)
    return popt[0], popt[1]

def logistic_function(X, L, X0, k, b):
    return L / (1 + np.exp(-k * (X - X0))) + b

def fit_logistic(x, y):
    x = np.array(x)
    y = np.array(y)

    # Append points 0 and 0 to the x and y arrays
    x = np.append(x, [0,0])
    y = np.append(y, [0,0])

    # print(f'x ======= {x} ======= {y}')
    L_initial = max(y)
    X0_initial = x[np.abs(y - L_initial / 2).argmin()]
    k_initial = 1 / (x.max() - x.min())
    b_initial = 0
    initial_guess = [L_initial, X0_initial, k_initial, b_initial]
    popt, pcov = curve_fit(logistic_function, x, y, p0 = initial_guess, maxfev=10000)
    return popt

def preprocess_dataset(data_file):
    data = pd.read_csv(data_file)

'''
    Function to solve the optimal allocation problem using the SLSQP method.
    Parameters:
    - modles : Reward models for each campaign
    budget : Daily budget for allocation

    Returns:
    tuple : Optimal allocation and total reward
'''
def solve_optimal_allocation(models, budget, num_campaigns):
    a = []
    b = []

    # Extract the coefficients from the models
    for campaign in models.keys():
        a.append(models[campaign][0])
        b.append(models[campaign][1])

    # Define the objectve function
    def objective(x):
        # Create the optimization ob
        return -sum([a[i] * x[i] ** b[i] for i in range(num_campaigns)])
    
    # Define the constraint
    def constraint(x):
        return budget - sum(x)
    
    # Start with equal allocation as a guess
    x0 = [budget / num_campaigns] * num_campaigns

    # Define the bounds for eache variable as per num campaigns
    bnds = [(0, budget)] * num_campaigns

    # Define the constraints as a dictionry
    cons = {'type': 'eq', 'fun': constraint}

    options = {'maxiter': 1000, 'ftol': 1e-6}  # Increase iterations and set tolerance
    # Use SLSQP method to minmize the objective function
    result = minimize(objective, x0, method = 'SLSQP', bounds = bnds, constraints = cons, options=options)
    # Define the constraint object
    # result = differential_evolution(objective, bnds, constraints=(cons,),)

    if not result.success:
        print(f"Optimization failed: {result.message}")

    # Return the optimal allocation and the total reward
    return result.x, -result.fun


# Trying a different optimization method for regret calculation as SQLSP is not being able to find the optimal solution sometimes
def _total_cost(params: dict, max_id: int, cost_of_max_arm: float):
        a_of_max_b, max_b = params[max_id]
        ret = 0
        for arm_id, (a, b) in params.items():
            if arm_id != max_id:
                ret += (((a_of_max_b*max_b)/(a*b)) * (cost_of_max_arm**(max_b-1)))**(1/(b-1))
            else:
                ret += cost_of_max_arm
        
        return ret

def _bisection_method(models: dict, budget: float, num_campaigns: int)-> float:
        """find budget allocation that meets the following conditions
        - the sum of each arm's budget equals budget
        - the first derivative of each arm is the same

        Args:
            params (dict): key: arm_id, value: two parameters of an exponential function
            budget (float): budget

        Returns:
            value: allocated budget
        """
        a = []
        b = []

        max_b = 0.0
        a_of_max_b = 1.0
        max_arm_id = ""
        for arm_id, (a, b) in models.items():
            if b > max_b:
                max_b = b
                a_of_max_b = a
                max_arm_id = arm_id
        
        max_cost = budget
        min_cost = 1
        epsilon = 1
        mid = 0
        #binary search
        while abs(max_cost - min_cost) > epsilon:
            mid = (max_cost + min_cost) / 2
            total = _total_cost(models, max_arm_id, mid)
            if total > budget - epsilon and total < budget + epsilon:
                break
            elif total < budget:
                min_cost = mid
            else:
                max_cost = mid

        ret = {}
        for arm_id, (a, b) in models.items():
            ret[arm_id] = (((a_of_max_b*max_b)/(a*b)) * (mid**(max_b-1)))**(1/(b-1))

        # find the optimal reward from the allocation
        reward = 0
        for key in ret:
            reward += models[key][0] * ret[key] ** models[key][1]

        return ret, reward