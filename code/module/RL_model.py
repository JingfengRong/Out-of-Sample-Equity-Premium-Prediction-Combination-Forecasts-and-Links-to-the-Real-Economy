import gym
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import Type, Tuple
import matplotlib.pyplot as plt
import sys
sys.path.append('../module')

from data_handler import get_econ_predictors, \
                         get_quarterly_date_format, \
                        get_monthly_date_format, \
                        get_equities_returns_volatility,\
                        get_equities_returns,\
                        get_env_data

def get_policy_in_sample_performance(env: gym.Env, model:Type[BaseAlgorithm]) -> pd.DataFrame:
    """
    Simulates the performance of a PPO policy on an OpenAI Gym environment.

    Parameters
    ----------
    model_path : str
        Path to the saved PPO model.
    env : gym.Env
        The OpenAI Gym environment to run the simulation on.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the actions taken by the policy, the resulting portfolio returns, and the dates of the actions.
    """
    state = env.reset()
    done = False 
    total_reward = 0  
    action_list = []
    port_ret_list = []
    date_list = []

    while not done:
        action, _state = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
        total_reward += reward 
        action_list.append(action[0])
        port_ret_list.append(info['portfolio return'])
        date_list.append(info['date'])

    performance = pd.DataFrame({'action': action_list, 'portfolio return': port_ret_list}, index=date_list)
    performance.index = performance.index.asfreq('Q')

    print(f'Total reward: {total_reward}')

    return performance

def evaluate_agent_action(df: pd.DataFrame) -> Tuple[float, plt.Figure, pd.DataFrame]:
    """
    Calculates the Sharpe ratio for the portfolio returns in the given data frame.

    Parameters:
        df: A pandas DataFrame containing 'action' and 'portfolio return' columns.

    Returns:
        A tuple containing the Sharpe ratio, a figure object, and the performance DataFrame.
        The performance DataFrame contains the 'portfolio return', 'benchmark 60/40', 'Market Index',
        and 'Rfree' columns, which are used to calculate the Sharpe ratio and plot the cumulative
        returns.
        
    Raises:
        ValueError: If the input portfolio return does not match the calculated portfolio return.
    """
    action = df['action'].values
    portfolio_return_input = df['portfolio return'].values
    equities_returns = get_equities_returns(data_freq='quarterly')
    equities_returns.index = equities_returns.index.asfreq('Q')
    index = df.index.asfreq('Q')
    equities_returns = equities_returns.loc[index].values

    w = np.column_stack([action, 1 - action])
    portfolio_return = np.sum(equities_returns * w, axis=1)
    if not np.array_equal(portfolio_return_input, portfolio_return):
        raise ValueError("Input portfolio return does not match the calculated portfolio return.")

    benchmark_60_40 = equities_returns @ np.array([0.6, 0.4])
    performance = pd.DataFrame(columns=['portfolio return', 'benchmark 60/40', 'Market Index', 'Rfree'],
                               data=np.column_stack([portfolio_return, benchmark_60_40, equities_returns]),
                               index=index)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    performance.cumsum().plot(ax=ax1)
    plt.xticks(rotation=45)
    df['action'].plot(ax=ax2)
    ax1.set(title='Cumulative Summation of Returns')
    ax2.set(title='Action')
    plt.xticks(rotation=45)
    
    sharpe_ratio = np.mean(performance[['portfolio return', 'benchmark 60/40']]) \
                    / np.std(performance[['portfolio return', 'benchmark 60/40']])
    
    return sharpe_ratio, fig, performance

