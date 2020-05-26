import numpy as np
import time
from sokoban_utils.global_configs import GlobalConfigs
from sokoban_utils.policy import Policy
from sokoban_utils.utils import *

def run_qlearning(env, initial_config, log=False):
    if not initial_config:
        config = Config()
    else:
        config = initial_config

    # default paramaters values
    if 'total_episodes' not in config: config.total_episodes = 200 # Total episodes
    if 'alpha' not in config: config.alpha = 0.8 # Learning rate
    if 'max_steps' not in config: config.max_steps = 100 # Max steps per episode
    if 'gamma' not in config: config.gamma = 0.95 # Discounting rate
    if 'epsilon' not in config: config.epsilon = 0.2 # Exploration rate
    if 'max_epsilon' not in config: config.max_epsilon = 0.2 # Exploration probability at start
    if 'min_epsilon' not in config: config.min_epsilon = 0.01 # Minimum exploration probability 
    if 'decay_rate' not in config: config.decay_rate = 0.001 # Exponential decay rate for exploration

    if log:
        create_dir(GlobalConfigs.logs_dir)
        logfile = GlobalConfigs.logs_dir + "qlearning_" + current_time() + ".txt"
        write_config_to_file(config, logfile)
        config.logfile = logfile

    return q_learning(env, config, log),logfile  

def q_learning(env, config, log):
    policy = Policy(env)
    qtable = Qtable(env.action_space.n)

    for ep in range(config.total_episodes):
        print("[+] Episode %d\r" % (ep+1), end="")
        initial_state = env.reset()
        done = False
        prev_hash = state_hash(initial_state)

        for _ in range(config.max_steps):
            # choose next action
            a = epsilon_random_action(env, policy[prev_hash], config.epsilon) 

            env.render()
            new_state, reward, done, info = env.step(a)

            s_hash = state_hash(new_state)

            q_max = np.max(qtable[s_hash])

            prev_q = qtable[prev_hash][a]
            qtable[prev_hash][a] = prev_q + config.alpha * (reward + config.gamma * q_max - prev_q)
                
            if done:
                break

            # update epsilon
            config.epsilon = config.min_epsilon + (config.max_epsilon - config.min_epsilon) * np.exp(-config.decay_rate * ep)
            prev_hash = s_hash

            # improve policy
            s_actions = qtable[prev_hash]
            policy[prev_hash] = s_actions.index(max(s_actions))
        
    return policy