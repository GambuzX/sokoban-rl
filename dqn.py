import gym
import gym_sokoban

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

# hyperparameters
gamma = 0.99 #discount factor
learning_rate = 0.00025 #learning rate for adam optimizer
buffer_size = 50000 #size of the replay buffer
exploration_fraction=0.1 #fraction of entire training period over which the exploration rate is annealed
exploration_final_eps=0.02 #final value of random action probability
exploration_initial_eps=1.0 #initial value of random action probability
train_freq=1 #update the model every train_freq steps. set to None to disable printing
batch_size=32 #size of a batched sampled from replay buffer for training
double_q=True #whether to enable Double-Q learning or not.
learning_starts=1000 #how many steps of the model to collect transitions for before learning starts
timesteps = 10#2000
verbose = 1

env = gym.make('Boxoban-Train-v1')

model = DQN(MlpPolicy, env, 
    gamma=gamma,
    learning_rate=learning_rate,
    buffer_size=buffer_size,
    exploration_fraction=exploration_fraction,
    exploration_final_eps=exploration_final_eps,
    exploration_initial_eps=exploration_initial_eps,
    train_freq=train_freq,
    batch_size=batch_size,
    double_q=double_q,
    learning_starts=learning_starts,
    verbose=1)
model.learn(total_timesteps=timesteps)
model.save("trained_models/dqn_sokoban_model")

# Enjoy trained agent
obs = env.reset()
print(model.action_probability(obs))
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()