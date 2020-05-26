import os
import random
import numpy as np
import time
import random
from datetime import datetime

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
Dictionary that allows for dot notation
Extracted from: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    - epool
'''
class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Config, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Config, self).__delitem__(key)
        del self.__dict__[key]

'''
returns random action epsilon times, and 'action' (1-epsilon) times
'''
def epsilon_random_action(env, action, epsilon):
    choice = random.uniform(0,1)
    return action if choice > epsilon else env.action_space.sample()


def write_config_to_file(config, filename):
    with open(filename, 'w') as handle:
        handle.write(str(len(config.keys())) + "\n")
        for key in config:
            handle.write(str(key) + ": " + str(config[key]) + "\n")

def write_csv_results(config, episode, reward, elapsed):
    with open(config.logfile, "a+") as handle:
        handle.write(str(episode) + "," + str(reward) + "," + str(elapsed) + "\n")

def current_time():
    now = datetime.now()
    return now.strftime("%H:%M:%S")