"""
Solving FrozenLake8x8 environment using Value iteration.
"""
import argparse

import gym
import numpy as np


def run_episode(env, policy, gamma=1.0, render=False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.

    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.

    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma=1.0, n=100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
        run_episode(env, policy, gamma=gamma, render=False)
        for _ in range(n)]
    return np.mean(scores)


def extract_policy(env, v, gamma=1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma=1.0):
    """ Value-iteration algorithm """
    v = np.zeros(env.observation_space.n)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            # TODO previously gamma missing? p * (r +  prev_v[s_]) --> p * (r + gamma * prev_v[s_]) ???
            q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in
                    range(env.action_space.n)]
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break
    return v


def start(args):
    gamma = args.gamma
    num_score_episodes = args.num_score_episodes

    env_name = 'FrozenLake8x8-v0'
    # env_name = 'Taxi-v2'
    env = gym.make(env_name)
    optimal_v = value_iteration(env, gamma);
    policy = extract_policy(env, optimal_v, gamma)

    print('Running with best policy')
    run_episode(env, policy, gamma, render=True)

    print('Best policy:', policy)

    policy_score = evaluate_policy(env, policy, gamma, n=num_score_episodes)
    print('Policy average score = ', policy_score)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor')

    parser.add_argument('--num_score_episodes', type=int, default=1000,
                        help='Number of episodes to compute the best value-iteration policy score')

    args = parser.parse_args()
    start(args)


if __name__ == '__main__':
    main()
