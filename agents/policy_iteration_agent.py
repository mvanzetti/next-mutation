"""
Solving FrozenLake8x8 environment using Policy iteration.
"""
import argparse

import gym
import numpy as np


def run_episode(env, policy, gamma=1.0, render=False):
    """ Runs an episode and return the total reward """
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
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)


def extract_policy(env, v, gamma=1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s]
    and solve them to find the value function.
    """
    v = np.zeros(env.observation_space.n)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][policy_a]])
        if np.sum((np.fabs(prev_v - v))) <= eps:
            # value converged
            break
    return v


def policy_iteration(env, gamma=1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.action_space.n, size=env.observation_space.n)  # initialize a random policy
    max_iterations = 200000
    # gamma = 1.0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        if np.all(policy == new_policy):
            print('Policy-Iteration converged at step %d.' % (i + 1))
            break
        policy = new_policy
    return policy


def start(args):
    gamma = args.gamma
    num_score_episodes = args.num_score_episodes

    env_name = 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    optimal_policy = policy_iteration(env, gamma=gamma)

    print('Running with best policy')
    run_episode(env, optimal_policy, gamma, render=True)

    print('Best policy:', optimal_policy)

    scores = evaluate_policy(env, optimal_policy, gamma=gamma, n=num_score_episodes)
    print('Average scores = ', np.mean(scores))


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
