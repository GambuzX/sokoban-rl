import gym
import gym_sokoban

env = gym.make('Sokoban-v0')
env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

env.close()