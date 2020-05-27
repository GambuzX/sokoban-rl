import numpy as np
from sokoban_utils.global_configs import GlobalConfigs
from sokoban_utils.policy import Policy
from sokoban_utils.utils import *

def run_montecarlo(env, initial_config, log=False, render=False):
    config = copy_config(initial_config)

    # default paramaters values
    if 'total_episodes' not in config: config.total_episodes = 200 # Total episodes
    if 'max_steps' not in config: config.max_steps = 100 # Max steps per episode
    if 'gamma' not in config: config.gamma = 0.95 # Discounting rate
    if 'epsilon' not in config: config.epsilon = 0.2 # Exploration rate
    if 'max_epsilon' not in config: config.max_epsilon = 0.2 # Exploration probability at start
    if 'min_epsilon' not in config: config.min_epsilon = 0.01 # Minimum exploration probability 
    if 'decay_rate' not in config: config.decay_rate = 0.001 # Exponential decay rate for exploration

    if log:
        create_dir(GlobalConfigs.logs_dir)
        logfile = GlobalConfigs.logs_dir + "montecarlo_" + current_time() + ".txt"          
        write_config_to_file(config, logfile)
        config.logfile = logfile

    policy = montecarlo(env, config, log, render)
    return (policy, logfile) if log else policy

'''
attempt to find an optimal policy over a number of episodes
'''
def montecarlo(env, c, log=False, render=False):

    print("[!] Starting Montecarlo")
    policy = Policy(env)
    qtable = Qtable(env.action_space.n)
    r = dict() # rewards for each tuple (state,action)->[list of rewards over many steps]

    for ep in range(c.total_episodes):
        print("[+] Episode %d\r" % (ep+1), end="")
        ep_start_t = time.time()

        sag_list, total_reward, max_reward = evaluate_policy(env, c, policy, render) # state, action, reward
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
        c.epsilon = c.min_epsilon + (c.max_epsilon - c.min_epsilon) * np.exp(-c.decay_rate * ep)
        
        if log:
            ep_end_t = time.time()
            elapsed = ep_end_t - ep_start_t # time in seconds
            write_csv_results(c, ep+1, total_reward, max_reward, elapsed)

    print('')
    return policy

'''
play episode until the end, recording every state, action and reward
'''
def evaluate_policy(env, c, policy, render=False):
    env.reset()
    sar_list = [] # state, action, reward
    done = False
    a = env.action_space.sample() # start with random action

    max_reward = 0
    total_reward = 0
    for _ in range(c.max_steps):
        # step
        if render: env.render()
        new_state, reward, done, info = env.step(a)
        s_hash = state_hash(new_state)
        total_reward += reward
        max_reward = max(max_reward, total_reward)   

        if done:
            sar_list.append((s_hash, 0, reward))
            break
        else:
            a = epsilon_random_action(env, policy[s_hash], c.epsilon)
            sar_list.append((s_hash, a, reward))
    
    # compute the return
    G = 0
    sag_list = []
    for s,a,r in sar_list[::-1]: # start loop in end
        G = r + c.gamma*G
        sag_list.append((s,a,G))

    # return the computed list
    return (sag_list[::-1], total_reward, max_reward)