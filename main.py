'''rom environment import BEMSEnv
import time

env = BEMSEnv(initial_desired_temp=25)
while (True):
    obs = env.step(True)

    print(obs)

    time.sleep(0.01)'''

import gymnasium as gym

from stable_baselines3 import PPO

from environment import BEMSEnv

# Parallel environments
env = BEMSEnv(initial_desired_temp=25)

model = PPO("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=100000)
model.save("ppo_bems")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_bems")

"""obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")"""