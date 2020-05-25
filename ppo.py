import gym
import gym_sokoban

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

# hyperparameters
gamma = 0.99 #Discount factor
ent_coef = 0.01 #Entropy coefficient for the loss calculation
learning_rate = 0.00025 #The learning rate, it can be a function
vf_coef = 0.5 #Value function coefficient for the loss calculation
max_grad_norm = 0.5 #The maximum value for the gradient clipping
lam = 0.95 #Factor for trade-off of bias vs variance for Generalized Advantage Estimator
timesteps = 10#2000
verbose = 1

# multiprocess environment
env = make_vec_env('Boxoban-Train-v1', n_envs=4)

model = PPO2(MlpPolicy, env,
    gamma=gamma,
    ent_coef=ent_coef,
    learning_rate=learning_rate,
    vf_coef=vf_coef,
    max_grad_norm=max_grad_norm,
    lam=lam,
    verbose=1)
model.learn(total_timesteps=timesteps)
model.save("trained_models/ppo2_sokoban_model") # save model to disk

# Enjoy trained agent
obs = env.reset()
print(model.action_probability(obs))
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")