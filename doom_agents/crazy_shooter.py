#!/usr/bin/env python

from vizdoom import *
import vizdoom as vzd
import random
import time

game = DoomGame()
game.load_config("scenarios/basic.cfg")
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
game.set_sound_enabled(True)
game.init()

sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]

episodes = 10
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():

        state = game.get_state()
        n = state.number
        vars = state.game_variables
        img = state.screen_buffer
        misc = state.game_variables
        reward = game.make_action(random.choice(actions))

        print("State #" + str(n))
        print("Game variables:", vars)
        print("Reward:", reward)
        print("=====================")

        time.sleep(sleep_time)

    print("Episode finished.")
    print("Total reward:", game.get_total_reward())
    print("************************")

    time.sleep(2)
