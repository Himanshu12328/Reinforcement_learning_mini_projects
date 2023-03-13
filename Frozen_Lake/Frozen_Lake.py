### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
import numpy as np
import math
import gym

from gym.envs.registration import register
np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
        tuple of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""


register(
    id='Deterministic-4x4-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False})


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    
    value_function = np.zeros(nS)
    while True:
        difference = 0
        for a in range(nS):
            sum = 0
            for b, action_probability in enumerate(policy[a]):
                for prob, nextstage, reward, terminal in P[a][b]:
                    sum = sum + action_probability * prob * (reward + gamma * value_function[nextstage])

            difference = max(difference, np.abs(value_function[a] - sum))
            value_function[a] = sum

        if difference < tol:
            break
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    new_policy = np.ones([nS, nA]) / nA

    for j in range(nS):
        k = np.zeros(nA)

        for i in range (nA):
            for prob, nextstage, reward, terminal in (P[j][i]):
                k[i] = k[i] + prob * (reward + gamma * value_from_policy[nextstage])
        bestaction = np.argmax(k)
        k = np.zeros(nA)
        k[bestaction] = 1
        new_policy[j] = k

    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()

    while(True):
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, value_function, gamma)
        value_function_new = policy_evaluation(P, nS, nA, new_policy, gamma, tol)

        if np.max(abs(value_function - value_function_new)) < tol:
            break

        policy = new_policy.copy();

    return new_policy, value_function_new


def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """

    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        difference = 0

        for s in range(nS):
            v = V[s]
            q = np.zeros(nA)

            for a in range(nA):

                for probaility, next_step, reward, terminal in (P[s][a]):
                    q[a] += probaility * (reward + gamma * V[next_step])

            V[s] = max(q)

            difference = max(difference, abs(V[s] - v))

        if difference < tol:
            break

    policy = policy_improvement(P, nS, nA, V, gamma)

    ############################
    # YOUR IMPLEMENTATION HERE #

    while (True):

        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, value_function, gamma)
        value_function_new = policy_evaluation(P, nS, nA, new_policy, gamma, tol)
        #    element wise comparison of value function
        if np.max(abs(value_function - value_function_new)) < tol:
            break

        policy = new_policy.copy();
    return policy, value_function_new
    ############################

def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob = env.reset() # initialize the episode
        done = False
        while not done:
            if render:
                env.render() # render the game
            ############################
            # YOUR IMPLEMENTATION HERE #

            value = 0
            interersted = 0
            for i in range(4):
                sample = policy[ob][i]
                if(sample > value):
                    value = sample
                    interersted = i
            #print(interersted)
            ob, reward, done, info = env.step(interersted)
            if done:
                if reward == 1:
                    total_rewards = total_rewards + 1

    return total_rewards
env = gym.make("FrozenLake-v1")
env = env.unwrapped
random_policy1 = np.ones([env.nS, env.nA])/env.nA
V1 = policy_evaluation(env.P, env.nS, env.nA, random_policy1, 1, tol = 1e-8)

np.random.seed(595)
V1 = np.random.rand(env.nS)
policy = policy_improvement(env.P, env.nS, env.nA, V1)
#print(policy)
V = np.zeros(env.nS)


