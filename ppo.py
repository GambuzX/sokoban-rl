import gym
import gym_sokoban
import time 

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback

# folders
log_dir = "./logs/"
models_path = "./trained_models/"
best_model_save_path = models_path + "ppo_sokoban_model"

# hyperparameters
gamma = 0.99 #Discount factor
ent_coef = 0.01 #Entropy coefficient for the loss calculation
learning_rate = 0.00025 #The learning rate, it can be a function
vf_coef = 0.5 #Value function coefficient for the loss calculation
max_grad_norm = 0.5 #The maximum value for the gradient clipping
lam = 0.95 #Factor for trade-off of bias vs variance for Generalized Advantage Estimator
timesteps = 2000
verbose = 1

# multiprocess environment
env = make_vec_env('Boxoban-Train-v1', n_envs=4)
eval_callback = EvalCallback(env.envs[0], best_model_save_path=best_model_save_path,
                                log_path=log_dir, eval_freq=500, 
                                deterministic=True, render=False)

model = PPO2(MlpPolicy, env,
    gamma=gamma,
    ent_coef=ent_coef,
    learning_rate=learning_rate,
    vf_coef=vf_coef,
    max_grad_norm=max_grad_norm,
    lam=lam,
    verbose=1)
model.learn(total_timesteps=timesteps, callback=eval_callback)
#model.save("trained_models/ppo2_sokoban_model") # save model to disk

# Enjoy trained agent
env = make_vec_env('Boxoban-Train-v1', n_envs=1)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if rewards[0] > 10:
        print("Completed the puzzle")
    time.sleep(0.1)
    env.render("human")