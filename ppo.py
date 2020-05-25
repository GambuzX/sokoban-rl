import gym
import gym_sokoban
import time 
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines import results_plotter

# folders
log_dir = "./logs/"
models_path = "./trained_models/"
best_model_save_path = models_path + "ppo_sokoban_model"

# hyperparameters
gamma = 0.99 #Discount factor
ent_coef = 0.01 #Entropy coefficient for the loss calculation
n_envs = 4 # number of environments
n_steps = 20 # The number of steps to run for each environment per update (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
learning_rate = 0.00025 #The learning rate, it can be a function
vf_coef = 0.5 #Value function coefficient for the loss calculation
max_grad_norm = 0.5 #The maximum value for the gradient clipping
lam = 0.95 #Factor for trade-off of bias vs variance for Generalized Advantage Estimator
timesteps = 100
verbose = 1

n_measurements = 10 # number of measurements for the graph
eval_callback_freq = 20#timesteps / n_measurements # interval between callbacks to achieve desired n_measurements

# multiprocess environment
#env = make_vec_env('Boxoban-Train-v1', n_envs=n_envs)
env = gym.make('Boxoban-Train-v1')
#first_env = env.envs[0]
first_env = env
first_env = Monitor(first_env, log_dir)
eval_callback = EvalCallback(first_env, best_model_save_path=best_model_save_path,
                                log_path=log_dir, eval_freq=eval_callback_freq, 
                                deterministic=True, render=False)

model = PPO2(MlpPolicy, env,
    gamma=gamma,
    ent_coef=ent_coef,
    n_steps=n_steps,
    learning_rate=learning_rate,
    vf_coef=vf_coef,
    max_grad_norm=max_grad_norm,
    lam=lam,
    verbose=1)
model.learn(total_timesteps=timesteps, callback=eval_callback)
#model.save("trained_models/ppo2_sokoban_model") # save model to disk

# Enjoy trained agent
'''
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if rewards[0] > 10:
        print("Completed the puzzle")
    time.sleep(0.1)
    env.render("human")
'''
results_plotter.plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Sokoban PPO")
plt.savefig("mygraph.png")