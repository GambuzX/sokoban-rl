import gym
import gym_sokoban
from algorithms.sarsa import run_sarsa
from algorithms.montecarlo import run_montecarlo
from algorithms.qlearning import run_qlearning
from sokoban_utils.policy import run_policy
from sokoban_utils.utils import Config

'''
* action is a number from 0 to 8 specifying the action, as in the gym-sokoban repo
* step(action) return:
    - observation (state): board's pixels (width x height). each element represents the rgb color of the pixel in human mode
    - reward: double value representing the reward, as explained in the gym-sokoban repo
    - done: boolean, has episode terminated
    - info: {'action.name': 'push down (example)', 'action.moved_player': BOOLEAN, 'action.moved_box': BOOLEAN}
'''

env = gym.make('Boxoban-Train-v1')
 
config = Config()
config.total_episodes = 200

#policy,_ = run_montecarlo(env, log=True, initial_config=config)
policy,_ = run_sarsa(env, log=True, initial_config=config, render=True)
#policy,_ = run_qlearning(env, log=True, initial_config=config, render =True)

input("Press any key to continue...")
run_policy(env, policy)
env.close()