import gym
import gym_sokoban
from algorithms.sarsa import run_sarsa
from algorithms.montecarlo import run_montecarlo
from algorithms.qlearning import run_qlearning
from sokoban_utils.policy import run_policy
from sokoban_utils.utils import *
from sokoban_utils.global_configs import GlobalConfigs
import numpy as np

env = gym.make('Boxoban-Train-v1')
 
config = Config()
config.total_episodes = 100

print("[!] Max Epsilon Test")
#Test montecarlo, sarsa and qlearning max epsilon (exploration probability at start) 
sarsa_max_epsilon = open("logs/sarsa_max_epsilon.txt", "a+")
montecarlo_max_epsilon = open("logs/montecarlo_max_epsilon.txt", "a+")
qlearning_max_epsilon = open("logs/qlearning_max_epsilon.txt", "a+")

max_epsilon_config = copy_config(config)

# Defining list of values for the max epsilon
max_epsilon_list =  [1.0]

for max_epsilon in max_epsilon_list:
    max_epsilon_config.epsilon = max_epsilon
    max_epsilon_config.max_epsilon = max_epsilon
    _, logfile = run_sarsa(env, log=True, initial_config=max_epsilon_config)
    sarsa_max_epsilon.write(logfile + "\n")
    
    _, logfile = run_montecarlo(env, log=True, initial_config=max_epsilon_config)
    montecarlo_max_epsilon.write(logfile + "\n")
    
    _, logfile = run_qlearning(env, log=True, initial_config=max_epsilon_config)
    qlearning_max_epsilon.write(logfile + "\n")

sarsa_max_epsilon.close()
montecarlo_max_epsilon.close()
qlearning_max_epsilon.close()

env.close()
