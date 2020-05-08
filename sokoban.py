import gym
import gym_sokoban
import random
import numpy as np

env = gym.make('Sokoban-v0')

'''
* action is a number from 0 to 8 specifying the action, as in the gym-sokoban repo
* step(action) return:
    - observation (state): board's pixels (width x height). each element represents the rgb color of the pixel in human mode
    - reward: double value representing the reward, as explained in the gym-sokoban repo
    - done: boolean, has episode terminated
    - info: {'action.name': 'push down (example)', 'action.moved_player': BOOLEAN, 'action.moved_box': BOOLEAN}
'''

# Set hyperparameters

# @hyperparameters
total_episodes = 200        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.001             # Exponential decay rate for exploration prob

'''
returns random action epsilon times, and 'action' (1-epsilon) times
'''
def epsilon_random_action(action):
    return action if random.uniform(0,1) > epsilon else env.action_space.sample()

'''
play episode until the end, recording every state, action and reward
'''
def evaluate_policy(policy):
    env.reset()
    sar_list = [] # state, action, reward
    done = False
    next_action = env.action_space.sample()
    
    while not done:
        new_state, reward, done, info = env.step(next_action)

        if done:
            sar_list.append((new_state, None, reward))
        else:
            a = epsilon_random_action(policy[new_state])
            sar_list.append((new_state, a, reward))

'''
generate random policy for each state
'''
def random_policy():
    return []

'''
attempt to find an optimal policy over a number of episodes
'''
def find_optimal_policy():
    policy = random_policy()
    Q = [[]] # TODO use numpy arrays
    r = [] # rewards for each tuple (state,action)

    for _ in range(max_steps):
        sag_list = evaluate_policy(policy) # state, action, reward
        visited_states_actions = []

        for (s,a,G) in sag_list:
            if (s,a,G) not in visited_states_actions:
                r[(s,a)].append(G)
                Q[s][a] = sum(r[(s,a)]) / len(r[(s,a)])
                visited_states_actions.add(s,a)
        
        for s in policy:
            policy[s] = []#action_with_max(Q[s])


#state = env.reset()
#for _ in range(1000):
#    env.render()
#    action = env.action_space.sample()
#    observation, reward, done, info = env.step(action)

env.close()

# TODO create wrapper around env.step to change returned state