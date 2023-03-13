#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
import gym
env = gym.make('CliffWalking-v0')
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    py
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    a = np.random.uniform(0, 1)
    if a < epsilon:
        action = np.random.choice(nA)
    else:
        action = np.argmax(Q[state])





    ############################
    return action

def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):


    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    eps_min = 0.01
    decay = 0.99
    ############################
    # YOUR IMPLEMENTATION HERE #
    nA = 4


    for i in range(n_episodes):
        epsilon = epsilon*decay
        state = env.reset()

        action = epsilon_greedy(Q, state, 4, epsilon)

        while True:

            next_state, reward, done, info = env.step(action)
            next_action = epsilon_greedy(Q, next_state, 4, epsilon)

            td_target = reward + gamma * (Q[next_state][next_action])
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alpha * td_error
            state = next_state
            action = next_action

            if done:

                break
    return Q

    # loop n_episodes

        # define decaying epsilon


        # initialize the environment 

        
        # get an action from policy

        # loop for each step of episode

            # return a new state, reward and done

            # get next action

            
            # TD update
            # td_target

            # td_error

            # new Q

            
            # update state

            # update action

    ############################

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    for _ in range(n_episodes):

        state = env.reset()
        done = False

        while not done:
            action = epsilon_greedy(Q_values, state, 4, epsilon = 0.1)

            next_state, reward, done, info = env.step(action)

            td_targets = reward + gamma * np.max(Q_values[next_state])
            td_error = td_targets - Q_values[state][action]
            Q_values[state][action] = Q_values[state][action] + alpha * td_error

            state = next_state
    return Q_values
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # loop n_episodes

        # initialize the environment 

        
        # loop for each step of episode

            # get an action from policy
            
            # return a new state, reward and done
            
            # TD update
            # td_target with best Q

            # td_error

            # new Q
            
            # update state

    ############################
sarsa(env, n_episodes=50000, gamma = 1.0, alpha = 0.01, epsilon= 0.1)
