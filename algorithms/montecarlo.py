import numpy as np
import random
from sokoban_utils.global_configs import GlobalConfigs
from sokoban_utils.policy import Policy
from sokoban_utils.utils import *

# @hyperparameters
total_episodes = 200       # Total episodes
max_steps = 100                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 0.2            # Exploration rate
max_epsilon = 0.2             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.001             # Exponential decay rate for exploration


def run_montecarlo(env, config, log=False):
    if 'total_episodes' in config: total_episodes = config['total_episodes']
    if 'max_steps' in config: max_steps = config['max_steps']
    if 'gamma' in config: gamma = config['gamma']
    if 'epsilon' in config: epsilon = config['epsilon']
    if 'max_epsilon' in config: max_epsilon = config['max_epsilon']
    if 'min_epsilon' in config: min_epsilon = config['min_epsilon']
    if 'decay_rate' in config: decay_rate = config['decay_rate']

    if log:
        create_dir(GlobalConfigs.logs_dir)
        logfile = "montecarlo_" + str(random.randint(1, 9999999)) + ".txt"
        write_config_to_file(config, logfile)
    
    return montecarlo(env)

'''
attempt to find an optimal policy over a number of episodes
'''
def montecarlo(env, log=False):
    print("[!] Starting Montecarlo")
    policy = Policy(env)
    qtable = Qtable(env.action_space.n)
    r = dict() # rewards for each tuple (state,action)->[list of rewards over many steps]

    for ep in range(total_episodes):
        print("[+] Episode %d\r" % (ep+1), end="")
        sag_list = evaluate_policy(env, policy) # state, action, reward
        visited_states_actions = []

        for s,a,G in sag_list: # s = str(state)
            if (s,a) not in visited_states_actions:
                # update rewards for (s,a)
                if (s,a) not in r:
                    r[(s,a)] = []
                r[(s,a)].append(G)

                # update qtable
                qtable[s][a] = sum(r[(s,a)]) / len(r[(s,a)]) # avg

                # mark visited
                visited_states_actions.append((s,a))
        
        # improve policy
        for s in policy:
            actions = qtable[s]
            policy[s] = actions.index(max(actions))

        # update epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * ep)

    print('')
    return policy

'''
play episode until the end, recording every state, action and reward
'''
def evaluate_policy(env, policy):
    env.reset()
    sar_list = [] # state, action, reward
    done = False
    a = env.action_space.sample() # start with random action
    for _ in range(max_steps):
        env.render()
        new_state, reward, done, info = env.step(a)
        s_hash = state_hash(new_state)

        if done:
            sar_list.append((s_hash, 0, reward))
            break
        else:
            a = epsilon_random_action(env, policy[s_hash], epsilon)
            sar_list.append((s_hash, a, reward))
    
    # compute the return
    G = 0
    sag_list = []
    for s,a,r in sar_list[::-1]: # start loop in end
        G = r + gamma*G
        sag_list.append((s,a,G))

    # return the computed list
    return sag_list[::-1]