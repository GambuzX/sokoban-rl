import time
from sokoban_utils.utils import state_hash

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

'''
runs the environment using given policy
'''
def run_policy(env, policy):
    print("[+] Running policy")
    done = False
    s_hash = state_hash(env.reset()) # reset state
    total_reward = 0
    for i in range(1000):        
        env.render()
        time.sleep(0.5)

        if done:
            break

        action = policy[s_hash]
        s, r, done, info = env.step(action)
        total_reward += r

        new_s_hash = state_hash(s)
        if new_s_hash == s_hash: # did not change state
            break
        s_hash = new_s_hash
    print("[!] Achieved reward: " + str(total_reward))