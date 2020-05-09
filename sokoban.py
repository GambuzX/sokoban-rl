import gym
import gym_sokoban
import random
import numpy as np
import time

env = gym.make('Boxoban-Train-v1')
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

    def actions(self, state, done):
        if state not in self.qtable:
            if done:
                self.qtable[state] = [0 for _ in range(action_size)]
            else:
                self.qtable[state] = [env.action_space.sample() for _ in range(action_size)]
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

    def __len__(self):
        return len(self.policy)

    def __iter__(self):
        return iter(self.policy)

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
total_episodes = 1000       # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 200                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1            # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
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
def epsilon_random_action(action):
    choice = random.uniform(0,1)
    return action if choice > epsilon else env.action_space.sample()

'''
play episode until the end, recording every state, action and reward
'''
def evaluate_policy(policy):
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
            a = epsilon_random_action(policy[s_hash])
            sar_list.append((s_hash, a, reward))
    
    # compute the return
    G = 0
    sag_list = []
    for s,a,r in sar_list[::-1]: # start loop in end
        G = r + gamma*G
        sag_list.append((s,a,G))

    # return the computed list
    return sag_list[::-1]

'''
attempt to find an optimal policy over a number of episodes
'''
def find_optimal_policy():
    print("[!] Finding optimal policy")
    policy = Policy()
    qtable = Qtable()
    r = dict() # rewards for each tuple (state,action)->[list of rewards over many steps]

    for ep in range(total_episodes):
        print("[+] Episode %d\r" % (ep+1), end="")
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

        # update epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * ep)

    print('')
    return policy

def montecarlo():
    print("[!] Starting Montecarlo")
    policy = find_optimal_policy()
    print("[!] Found policy")
    input("Press any key to continue...")
    done = False
    state = env.reset()
    s_hash = state_hash(state)
    for i in range(1000):
        time.sleep(0.5)
        env.render()

        action = policy[s_hash]
        state, r, done, info = env.step(action)

        new_s_hash = state_hash(state)
        if new_s_hash == s_hash:
            break
        s_hash = new_s_hash

#montecarlo()

env.close()

# TODO player is doing nothing for some reason
# think its because states are different every time

# TODO analisar frequencia de estados
# gerar mais tabuleiros a começar em sitios diferentes

def sarsa():
    policy = Policy()
    qtable = Qtable()
    r = dict() # rewards for each tuple (state,action)->[list of rewards over many steps]

    epsilon = 0.1 # random action 10% of the time
    alpha = 0.1

    for ep in range(total_episodes):
        print("[+] Episode %d\r" % (ep+1), end="")
        
        s = env.reset() # start state
        done = False
        
        s_hash = state_hash(s)
        
        actions = qtable.actions(s_hash, done)
        
        a = epsilon_random_action(policy[s_hash])
        
        for _ in range(max_steps):
            env.render()
            new_state, reward, new_done, info = env.step(a)
            
            new_s_hash = state_hash(new_state)
            
            new_a = epsilon_random_action(policy[new_s_hash])
            new_actions = qtable.actions(new_s_hash, new_done)
            
                        
            for index in range(len(actions)):
                actions[index] = actions[index] + alpha*(reward+gamma*new_actions[index]-actions[index])
            
            qtable[s_hash] = actions
            
            # Next iteration values
            done = new_done
            a = new_a
            s = new_state
            actions = new_actions
            s_hash = new_s_hash
            
            if done:
                break
            
    
    done = False
    state = env.reset()
    s_hash = state_hash(state)
    for i in range(1000):
        time.sleep(0.5)
        env.render()

        action = qtable[s_hash]
        state, r, done, info = env.step(action.index(max(action)))

        if done:
            print("Finish")
            break

        new_s_hash = state_hash(state)
        if new_s_hash == s_hash:
            break
        s_hash = new_s_hash
        

sarsa()