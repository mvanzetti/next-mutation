import gym
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=20,
                        help='Number of episodes')

    parser.add_argument('--timesteps', type=int, default=100,
                        help='Number of timesteps for each episode')

    args = parser.parse_args()
    start(args)


# Cartpole-v1
# Actions: 0 (push left), 1 (push right)

# Breakout-v0
# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
env_to_actions = {
    'Cartpole-v1': [0, 1],
    'Breakout-v0': [0, 1, 2, 3]
}


def stupid_action():
    return 2


def start(args):
    num_episodes = args.num_episodes
    timesteps = args.timesteps

    # env = gym.make('Breakout-v0')
    env = gym.make('Cartpole-v0')
    for i_episode in range(num_episodes):
        observation = env.reset()
        for t in range(timesteps):
            env.render()
            print(observation)
            # action = env.action_space.sample()
            action = stupid_action()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


if __name__ == '__main__':
    main()
