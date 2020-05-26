import gym
import gym_sokoban
from algorithms.sarsa import run_sarsa
from algorithms.montecarlo import run_montecarlo
from sokoban_utils.policy import run_policy
from sokoban_utils.utils import *
from sokoban_utils.global_configs import GlobalConfigs

env = gym.make('Boxoban-Train-v1')
 
config = Config()
config.total_episodes = 5

create_dir(GlobalConfigs.logs_dir)        

# Test sarsa alpha (learning rate)
sarsa_alpha = open("logs/sarsa_alpha.txt", "w")

sarsa_alpha_config = copy_config(config)
learning_rate = 0.1
while learning_rate <= 1.0:
    sarsa_alpha_config.alpha = learning_rate
    _, logfile = run_sarsa(env, log=True, initial_config=sarsa_alpha_config)
    sarsa_alpha.write(logfile + "\n")
    learning_rate += 0.1
    
sarsa_alpha.close()

#Test montecarlo and sarsa gamma (discounting rate) 
sarsa_gamma = open("logs/sarsa_gamma.txt", "w")
montecarlo_gamma = open("logs/montecarlo_gamma.txt", "w")

gamma_config = copy_config(config)
discounting_rate = 0.1
while discounting_rate <= 1.0:
    gamma_config.gamma = discounting_rate
    _, logfile = run_sarsa(env, log=True, initial_config=gamma_config)
    sarsa_gamma.write(logfile + "\n")
    
    _, logfile = run_montecarlo(env, log=True, initial_config=gamma_config)
    montecarlo_gamma.write(logfile + "\n")
    
    discounting_rate += 0.1

sarsa_gamma.close()
montecarlo_gamma.close()


#Test montecarlo and sarsa max epsilon (exploration probability at start) 
sarsa_max_epsilon = open("logs/sarsa_max_epsilon.txt", "w")
montecarlo_max_epsilon = open("logs/montecarlo_max_epsilon.txt", "w")

max_epsilon_config = copy_config(config)
max_epsilon = 0.1
while max_epsilon <= 1.0:
    max_epsilon_config.max_epsilon = max_epsilon
    _, logfile = run_sarsa(env, log=True, initial_config=max_epsilon_config)
    sarsa_max_epsilon.write(logfile + "\n")
    
    _, logfile = run_montecarlo(env, log=True, initial_config=max_epsilon_config)
    montecarlo_max_epsilon.write(logfile + "\n")
    
    max_epsilon += 0.1

sarsa_max_epsilon.close()
montecarlo_max_epsilon.close()

env.close()