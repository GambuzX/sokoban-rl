import numpy as np
import time
from sokoban_utils.global_configs import GlobalConfigs
from sokoban_utils.policy import Policy
from sokoban_utils.utils import *

def run_qlearning(env, config, log=False):
    if not config:
        config = Config()

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

    return q_learning(env, config, log)  

def q_learning(env, config, log):
    policy = Policy(env)
    qtable = Qtable(env.action_space.n)

    for ep in range(config.total_episodes):
        print("[+] Episode %d\r" % (ep+1), end="")
        initial_state = env.reset()
        done = False
        a = env.action_space.sample() # start with random action
        prev_hash = state_hash(initial_state)

        greedy_start = qtable.get_greedy(prev_hash)
        if greedy_start is -1:
            policy[prev_hash] = a
        else:
            policy[prev_hash] = greedy_start

        for _ in range(config.max_steps):
            env.render()
            new_state, reward, done, info = env.step(a)

            s_hash = state_hash(new_state)

            if not qtable[s_hash] is None:
                qtable[prev_hash][a] = qtable[prev_hash][a] + config.alpha * (reward + config.gamma * np.max(qtable[s_hash]) - qtable[prev_hash][a])
            else:
                qtable[prev_hash][a] = qtable[prev_hash][a] + config.alpha * (reward + config.gamma * reward - qtable[prev_hash][a])
                
            if done:
                break

            greedy_action = qtable.get_greedy(s_hash)
            policy[s_hash] = greedy_action
            if greedy_action is -1:
                a = epsilon_random_action(env, env.action_space.sample(), config.epsilon)
            else:
                a = epsilon_random_action(env, greedy_action, config.epsilon)

            # update epsilon
            config.epsilon = config.min_epsilon + (config.max_epsilon - config.min_epsilon) * np.exp(-config.decay_rate * ep)
        
    return policy