import os
import random
import numpy as np
import time
import random

'''
maps state->action_list, where the list has the qvalue for each action
'''
class Qtable:
    def __init__(self, action_size):
        self.qtable = dict()
        self.action_size = action_size

    def __getitem__(self, state):
        if state not in self.qtable:
            self.qtable[state] = [0 for _ in range(self.action_size)]
        return self.qtable[state]

    def __setitem__(self, key, value):
        self.qtable[key] = value

def print_state(s):
    for r in range(7):
        for c in range(7):
            print(s[r*16+8][c*16+8], end=" ")
        print("\n")

def state_hash(s):
    return hash(s.tostring())

def create_dir(name):
    try:
        os.makedirs(name)
    except:
        pass

'''
returns random action epsilon times, and 'action' (1-epsilon) times
'''
def epsilon_random_action(env, action, epsilon):
    choice = random.uniform(0,1)
    return action if choice > epsilon else env.action_space.sample()


def write_config_to_file(config, filename):
    with open(filename, 'w') as handle:
        for key in config:
            handle.write(key + ": " + config[key] + "\n")