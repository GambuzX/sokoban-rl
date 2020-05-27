import numpy as np
import time
from sokoban_utils.global_configs import GlobalConfigs
from sokoban_utils.policy import Policy
from sokoban_utils.utils import *

def run_sarsa(env, initial_config, log=False, render=False):
    config = copy_config(initial_config)

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
        logfile = GlobalConfigs.logs_dir + "sarsa_" + current_time() + ".txt"
        write_config_to_file(config, logfile)
        config.logfile = logfile

    policy = sarsa(env, config, log, render)
    return (policy, logfile) if log else policy


def sarsa(env, c, log=False, render=False):

    print("[!] Starting Sarsa")    
    policy = Policy(env)
    qtable = Qtable(env.action_space.n)

    for ep in range(c.total_episodes):
        print("[+] Episode %d\r" % (ep+1), end="")
        ep_start_t = time.time()
        
        s_hash = state_hash(env.reset()) # reset state
        a = epsilon_random_action(env, policy[s_hash], c.epsilon) # choose epsilon greedy action
        
        max_reward = 0
        total_reward = 0
        done = False        
        while not done:
            # step
            if render: env.render()
            new_state, r, done, info = env.step(a)
            new_s_hash = state_hash(new_state)
            total_reward += r
            max_reward = max(max_reward, total_reward)    
            
            # choose next action
            next_a = epsilon_random_action(env, policy[new_s_hash], c.epsilon)     
                        
            # calculate new Q(s,a)
            curr_q = qtable[s_hash][a]
            next_q = qtable[new_s_hash][next_a]
            qtable[s_hash][a] = curr_q + c.alpha*(r + c.gamma*next_q - curr_q)

            # improve policy
            s_actions = qtable[s_hash]
            policy[s_hash] = s_actions.index(max(s_actions))
            
            # Next iteration values
            s_hash = new_s_hash
            a = next_a # use same action in next step as determined above

            # update epsilon
            c.epsilon = c.min_epsilon + (c.max_epsilon - c.min_epsilon) * np.exp(-c.decay_rate * ep)
        
        if log: 
            ep_end_t = time.time()
            elapsed = ep_end_t - ep_start_t # time in seconds
            write_csv_results(c, ep+1, total_reward, max_reward, elapsed)
            
    print('')
    return policy