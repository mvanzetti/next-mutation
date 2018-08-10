"""
Q-Learning example using OpenAI gym MountainCar enviornment

observation -> [position, velocity]
action -> [0, 1, 2] = [push left, none, push right]
reward -> -1 for each timestep

"""
import gym
import numpy as np
import argparse


def run_episode(env, n_timesteps, n_states, gamma, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(n_timesteps):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a, b = obs_to_state(env, n_states, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward


def obs_to_state(env, n_states, obs):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0]) / env_dx[0])
    b = int((obs[1] - env_low[1]) / env_dx[1])
    return a, b


def pick_greedy_action(env, q_table, a, b, eps):
    if np.random.uniform(0, 1) < eps:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[a][b])


def pick_softmax_action(env, q_table, a, b, temperature):
    logits_exp = np.exp(q_table[a][b] / temperature)
    probs = logits_exp / np.sum(logits_exp)

    return np.random.choice(env.action_space.n, p=probs)


def print_info(args):
    print("Info -----")
    print(args)


def train(env, n_states, n_episodes, n_timesteps, min_lr, initial_lr, gamma, strategy, min_eps, max_eps, eps_decay_rate,
          temperature):
    q_table = np.zeros((n_states, n_states, env.action_space.n))

    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0

        # eta: learning rate is decreased at each step
        eta = max(min_lr, initial_lr * (0.85 ** (episode // 100)))

        # decaying epsilon
        if strategy == 'greedy':
            eps = min_eps + (max_eps - min_eps) * np.exp(-eps_decay_rate * episode)
        # eps = 0.02

        for timestep in range(n_timesteps):
            a, b = obs_to_state(env, n_states, obs)

            if strategy == 'greedy':
                action = pick_greedy_action(env, q_table, a, b, eps)

            elif strategy == 'softmax':
                action = pick_softmax_action(env, q_table, a, b, temperature=temperature)

            else:
                raise Exception("Unavailable strategy")

            obs, reward, done, _ = env.step(action)
            total_reward += reward

            # update q table
            a_, b_ = obs_to_state(env, n_states, obs)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            q_table[a][b][action] = q_table[a][b][action] + eta * (
                    reward + gamma * np.max(q_table[a_][b_]) - q_table[a][b][action])
            if done:
                break
        if episode % 100 == 0:
            print('Episode #%d -- Total reward = %d.' % (episode + 1, total_reward))
            print('Eta', eta)
            # print('Epsilon', eps)

    return q_table


def start(args):
    n_states = args.n_states
    n_episodes = args.n_episodes
    n_timesteps = args.n_timesteps

    # Learning rate
    initial_lr = args.lr
    min_lr = initial_lr

    if args.lr_decay:
        min_lr = 0.003

    gamma = args.gamma

    strategy = args.strategy

    # greedy: epsilon decreasing in time to optimize exploration vs exploitation
    max_eps = args.epsilon
    min_eps = 0.01
    eps_decay_rate = args.epsilon_decay

    # softmax: temperature parameter (temp -> inf turns the agent to act randomly)
    temperature = args.temperature

    num_score_episodes = args.n_score_episodes

    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(0)
    np.random.seed(0)

    print_info(args)

    q_table = train(env, n_states, n_episodes, n_timesteps, min_lr, initial_lr, gamma, strategy, min_eps, max_eps,
                    eps_decay_rate,
                    temperature)

    solution_policy = np.argmax(q_table, axis=2)
    solution_policy_scores = [run_episode(env, n_timesteps, n_states, gamma, solution_policy, False) for _ in
                              range(num_score_episodes)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    # Animate it
    run_episode(env, n_timesteps, n_states, gamma, solution_policy, True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_states', type=int, default=40,
                        help='Number of states to discretize a continuous state space')

    parser.add_argument('--n_episodes', type=int, default=10000,
                        help='Number of training episodes')

    parser.add_argument('--n_timesteps', type=int, default=1000,
                        help='Number of max timesteps for each episode')

    parser.add_argument('--lr', type=float, default=1.0,
                        help='Learning rate')

    parser.add_argument('--lr_decay', type=bool, default=True,
                        help='True if learning rate must decay through episodes, default is True')

    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor')

    parser.add_argument("--strategy",
                        help='Exploration strategy, epsilon greedy or softmax selection',
                        choices=['greedy', 'softmax'], default='greedy')

    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Epsilon parameter for greedy strategy')

    parser.add_argument('--epsilon_decay', type=float, default=0.01,
                        help='Decay rate for epsilon parameter in the greedy strategy')

    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature parameter for softmax strategy')

    parser.add_argument('--n_score_episodes', type=int, default=100,
                        help='Number of episodes to compute the best value-iteration policy score')

    args = parser.parse_args()
    start(args)


if __name__ == '__main__':
    main()
