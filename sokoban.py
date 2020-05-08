import gym
import gym_sokoban
import random
import numpy as np

env = gym.make('Sokoban-v0')
action_size = env.action_space.n

'''
maps state->action_list, where the list has the qvalue for each action
'''
class Qtable:
    def __init__(self):
        self.qtable = dict()

    def __getitem__(self, state):
        if state not in self.qtable:
            self.qtable[state] = [0 for _ in range(action_size)] # TODO should this start at 0 ???
        return self.qtable[state]

    def __setitem__(self, key, value):
        self.qtable[key] = value

'''
maps state->action
'''
class Policy:
    def __init__(self):
        self.policy = dict()
    
    def __getitem__(self, state):
        if state not in self.policy:
            self.policy[state] = env.action_space.sample()
        return self.policy[state]

    def __setitem__(self, state, action):
        self.policy[state] = action

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
    
    for _ in range(max_steps):
        new_state, reward, done, info = env.step(next_action)
        new_state = str(new_state)

        if done:
            sar_list.append((new_state, 0, reward))
        else:
            a = epsilon_random_action(policy[new_state])
            sar_list.append((new_state, a, reward))
    
    # compute the return
    G = 0
    sag_list = []
    sar_list.reverse() # start loop in end
    for s,a,r in sar_list:
        G = r + gamma*G
        sag_list.append((s,a,G))

    # return the computed list
    sag_list.reverse()
    return sag_list

'''
attempt to find an optimal policy over a number of episodes
'''
def find_optimal_policy():
    policy = Policy()
    qtable = Qtable()
    r = dict() # rewards for each tuple (state,action)->[list of rewards over many steps]

    for _ in range(total_episodes):
        print("Episode " + str(_))
        sag_list = evaluate_policy(policy) # state, action, reward
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

def montecarlo():
    policy = find_optimal_policy()
    done = False
    state = env.reset()
    while not done:
        env.render()
        action = policy[str(state)]
        state, r, done, info = env.step(action)

montecarlo()
env.close()