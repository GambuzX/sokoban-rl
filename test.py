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

create_dir(GlobalConfigs.logs_dir)        

print("[!] Alpha Test")

# Test sarsa and qlearning alpha (learning rate)
sarsa_alpha = open("logs/sarsa_alpha.txt", "w")
qlearning_alpha = open("logs/qlearning_alpha.txt", "w")

alpha_config = copy_config(config)

# Defining list of values for the learning rate (alpha)
learning_rate_list = np.arange(0.5, 1.0, 0.1).tolist()

for learning_rate in learning_rate_list:
    alpha_config.alpha = learning_rate
    _, logfile = run_sarsa(env, log=True, initial_config=alpha_config)
    sarsa_alpha.write(logfile + "\n")
    
    _, logfile = run_qlearning(env, log=True, initial_config=alpha_config)
    qlearning_alpha.write(logfile + "\n")
    
sarsa_alpha.close()
qlearning_alpha.close()

print("[!] Gamma Test")
#Test montecarlo, sarsa and qlearning gamma (discounting rate) 
sarsa_gamma = open("logs/sarsa_gamma.txt", "w")
montecarlo_gamma = open("logs/montecarlo_gamma.txt", "w")
qlearning_gamma = open("logs/qlearning_gamma.txt", "w")

gamma_config = copy_config(config)

# Defining list of values for the discounting rate (gamma)
discounting_rate_list = [0.8, 0.9, 0.95, 1.0]

for discounting_rate in discounting_rate_list:
    gamma_config.gamma = discounting_rate
    _, logfile = run_sarsa(env, log=True, initial_config=gamma_config)
    sarsa_gamma.write(logfile + "\n")
    
    _, logfile = run_montecarlo(env, log=True, initial_config=gamma_config)
    montecarlo_gamma.write(logfile + "\n")
    
    _, logfile = run_qlearning(env, log=True, initial_config=gamma_config)
    qlearning_gamma.write(logfile + "\n")

sarsa_gamma.close()
montecarlo_gamma.close()
qlearning_gamma.close()

print("[!] Max Epsilon Test")
#Test montecarlo, sarsa and qlearning max epsilon (exploration probability at start) 
sarsa_max_epsilon = open("logs/sarsa_max_epsilon.txt", "w")
montecarlo_max_epsilon = open("logs/montecarlo_max_epsilon.txt", "w")
qlearning_max_epsilon = open("logs/qlearning_max_epsilon.txt", "w")

max_epsilon_config = copy_config(config)

# Defining list of values for the max epsilon
max_epsilon_list = np.arange(0, 1.0, 0.2).tolist()

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