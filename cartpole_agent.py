import gym
import numpy as np
import argparse
from joblib import Parallel, delayed

'''
The Cartpole Agent
observation: vector of 4 numbers
action space: 0 or 1
reward: +1 for every time step
----
Each policy is represented by 5 params such that
if w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4  + b > 0:
   cart moves to right
else
   cart moves to left
'''

# Solved Requirements (average score >= 195 over 100 consecutive trials
SOLVED_REQ = 195. / 100


def gen_random_policy():
    return np.random.uniform(-1, 1, size=4), np.random.uniform(-1, 1)


def policy_to_action(env, policy, obs):
    if np.dot(policy[0], obs) + policy[1] > 0:
        return 1
    else:
        return 0


def run_episode(env, policy, t_max=1000, render=False):
    print("Policy", policy)

    obs = env.reset()
    total_reward = 0
    for i in range(t_max):
        if render:
            env.render()
            # print("Obs", i, ":", obs)
        selected_action = policy_to_action(env, policy, obs)
        obs, reward, done, info = env.step(selected_action)
        total_reward += reward
        if done:
            break
    print("Total reward", total_reward)
    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes')

    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of timesteps for each episode')

    args = parser.parse_args()
    start(args)


def start(args):
    num_episodes = args.num_episodes
    timesteps = args.timesteps

    env = gym.make('CartPole-v0')

    print("Episodes:", num_episodes)
    print("Timesteps per episode:", timesteps)

    print("Random policy example", gen_random_policy())

    ## Generate a pool or random policies
    n_policy = num_episodes
    policy_list = [gen_random_policy() for _ in range(n_policy)]

    # Evaluate the score of each policy.
    # scores_list = Parallel(n_jobs=10)(delayed(run_episode)(env, p, render=False) for p in policy_list)
    scores_list = [run_episode(env, p, t_max=timesteps, render=False) for p in policy_list]

    print("Scores:", scores_list)

    # Select the best policy.
    print('Best policy score = %f' % max(scores_list))

    best_policy = policy_list[np.argmax(scores_list)]
    print('Running with best policy:\n')
    run_episode(env, best_policy, render=True)

    print("Current average score is", np.mean(scores_list), "over", num_episodes, "episodes")
    score_rate = np.mean(scores_list) / num_episodes
    print("Solved Requirements (average score >= 195 over 100 consecutive trials):")
    print(True if score_rate >= SOLVED_REQ else False)


if __name__ == '__main__':
    main()
