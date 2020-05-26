import gym
import gym_sokoban

from sokoban import run_montecarlo

dic = dict() 

env = gym.make('Boxoban-Train-v1')
run_montecarlo(env, dic)
env.close()