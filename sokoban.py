import random
import numpy as np
import time

'''
* action is a number from 0 to 8 specifying the action, as in the gym-sokoban repo
* step(action) return:
    - observation (state): board's pixels (width x height). each element represents the rgb color of the pixel in human mode
    - reward: double value representing the reward, as explained in the gym-sokoban repo
    - done: boolean, has episode terminated
    - info: {'action.name': 'push down (example)', 'action.moved_player': BOOLEAN, 'action.moved_box': BOOLEAN}
'''

#env = gym.make('Boxoban-Train-v1')
#action_size = 0


'''
maps state->action_list, where the list has the qvalue for each action
'''
class Qtable:
    def __init__(self, action_size):
        self.qtable = dict()
        self.action_size = action_size

    def __getitem__(self, state):
        if state not in self.qtable:
            self.qtable[state] = [0 for _ in range(self.action_size)] # TODO should this start at 0 ???
        return self.qtable[state]

    def get_action(self, state, action, done):
        if state not in self.qtable:
            self.qtable[state] = [0 if done else 0 for _ in range(self.action_size)] # TODO random value if not done??
        return self.qtable[state][action]

    def __setitem__(self, key, value):
        self.qtable[key] = value

'''
maps state->action
'''
class Policy:
    def __init__(self, env):
        self.policy = dict()
        self.env = env
    
    def __getitem__(self, state):
        if state not in self.policy:
            self.policy[state] = self.env.action_space.sample()
        return self.policy[state]

    def __setitem__(self, state, action):
        self.policy[state] = action

    def __len__(self):
        return len(self.policy)

    def __iter__(self):
        return iter(self.policy)

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
decay_rate = 0.001             # Exponential decay rate for exploration prob

def print_state(s):
    for r in range(7):
        for c in range(7):
            print(s[r*16+8][c*16+8], end=" ")
        print("\n")

def state_hash(s):
    return hash(s.tostring())

'''
returns random action epsilon times, and 'action' (1-epsilon) times
'''
def epsilon_random_action(env, action):
    choice = random.uniform(0,1)
    return action if choice > epsilon else env.action_space.sample()

'''
runs the environment using given policy
'''
def run_policy(env, policy):
    done = False
    s_hash = state_hash(env.reset()) # reset state
    for i in range(1000):
        time.sleep(0.5)
        env.render()

        action = policy[s_hash]
        s, r, done, info = env.step(action)

        new_s_hash = state_hash(s)
        if new_s_hash == s_hash: # did not change state
            break
        s_hash = new_s_hash

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
            a = epsilon_random_action(env, policy[s_hash])
            sar_list.append((s_hash, a, reward))
    
    # compute the return
    G = 0
    sag_list = []
    for s,a,r in sar_list[::-1]: # start loop in end
        G = r + gamma*G
        sag_list.append((s,a,G))

    # return the computed list
    return sag_list[::-1]

def run_montecarlo(env, dic):
    total_episodes = 200 if not 'total_episodes' in dic else dic['total_episodes']  
    alpha = 0.8 if not 'alpha' in dic else dic['alpha']
    max_steps = 100 if not 'max_steps' in dic else dic['max_steps']               
    gamma = 0.95 if not 'gamma' in dic else dic['gamma']                  
    epsilon = 0.2 if not 'epsilon' in dic else dic['epsilon']
    max_epsilon = 0.2 if not 'max_epsilon' in dic else dic['max_epsilon']
    min_epsilon = 0.01 if not 'min_epsilon' in dic else dic['min_epsilon']
    decay_rate = 0.001 if not 'decay_rate' in dic else dic['decay_rate']
    
    policy = montecarlo(env)
    
    return policy

'''
attempt to find an optimal policy over a number of episodes
'''
def montecarlo(env):
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

def run_sarsa(env, dic):
    total_episodes = 200 if not 'total_episodes' in dic else dic['total_episodes']  
    alpha = 0.8 if not 'alpha' in dic else dic['alpha']
    max_steps = 100 if not 'max_steps' in dic else dic['max_steps']               
    gamma = 0.95 if not 'gamma' in dic else dic['gamma']                  
    epsilon = 0.2 if not 'epsilon' in dic else dic['epsilon']
    max_epsilon = 0.2 if not 'max_epsilon' in dic else dic['max_epsilon']
    min_epsilon = 0.01 if not 'min_epsilon' in dic else dic['min_epsilon']
    decay_rate = 0.001 if not 'decay_rate' in dic else dic['decay_rate']
    
    action_size = env.action_space.n
    
    policy = montecarlo(env)
    
    return policy


def sarsa(env):
    # set exploration parameters
    epsilon = 0.2
    max_epsilon = 0.2
    min_epsilon = 0.01

    policy = Policy(env)
    qtable = Qtable(env.action_space.n)

    for ep in range(total_episodes):
        print("[+] Episode %d\r" % (ep+1), end="")
        
        s_hash = state_hash(env.reset()) # reset state
        a = epsilon_random_action(env, policy[s_hash]) # choose epsilon greedy action
        
        done = False        
        while not done:
            # step
            env.render()
            new_state, r, done, info = env.step(a) # handle done differently?
            new_s_hash = state_hash(new_state)            
            
            # choose next action
            next_a = epsilon_random_action(env, policy[new_s_hash])     
                        
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
       
 
#policy = montecarlo()
#policy = sarsa()

#input("Press any key to continue...")
#run_policy(env, policy)
#env.close()