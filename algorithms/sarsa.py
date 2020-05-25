import random
import numpy as np
from sokoban_utils.global_configs import GlobalConfigs
from sokoban_utils.policy import Policy
from sokoban_utils.utils import *

# Set hyperparameters

# @hyperparameters
total_episodes = 200       # Total episodes
alpha = 0.8           # Learning rate
max_steps = 100                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 0.2            # Exploration rate
max_epsilon = 0.2             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.001             # Exponential decay rate for exploration

def run_sarsa(env, config, log=False):
    if 'total_episodes' in config: total_episodes = config['total_episodes']
    if 'alpha' in config: alpha = config['alpha']
    if 'max_steps' in config: max_steps = config['max_steps']
    if 'gamma' in config: gamma = config['gamma']
    if 'epsilon' in config: epsilon = config['epsilon']
    if 'max_epsilon' in config: max_epsilon = config['max_epsilon']
    if 'min_epsilon' in config: min_epsilon = config['min_epsilon']
    if 'decay_rate' in config: decay_rate = config['decay_rate']

    if log:
        create_dir(GlobalConfigs.logs_dir)
        logfile = "sarsa_" + str(random.randint(1, 9999999)) + ".txt"
        write_config_to_file(config, logfile)

    return sarsa(env, log)  


def sarsa(env, log=False):
    global epsilon
    
    policy = Policy(env)
    qtable = Qtable(env.action_space.n)

    for ep in range(total_episodes):
        print("[+] Episode %d\r" % (ep+1), end="")
        
        s_hash = state_hash(env.reset()) # reset state
        a = epsilon_random_action(env, policy[s_hash], epsilon) # choose epsilon greedy action
        
        done = False        
        while not done:
            # step
            env.render()
            new_state, r, done, info = env.step(a) # handle done differently?
            new_s_hash = state_hash(new_state)            
            
            # choose next action
            next_a = epsilon_random_action(env, policy[new_s_hash], epsilon)     
                        
            # calculate new Q(s,a)
            curr_q = qtable[s_hash][a]
            next_q = qtable[new_s_hash][next_a]
            qtable[s_hash][a] = curr_q + alpha*(r + gamma*next_q - curr_q)

            # improve policy
            s_actions = qtable[s_hash]
            policy[s_hash] = s_actions.index(max(s_actions))
            
            # Next iteration values
            s_hash = new_s_hash
            a = next_a # use same action in next step as determined above

            # update epsilon
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * ep)
            
    print('')
    return policy