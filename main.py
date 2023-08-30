import gymnasium as gym
from stable_baselines3 import PPO
from environment import BEMSEnv
import matplotlib as plt

# Parallel environments
env = BEMSEnv(initial_desired_temp=25)

model = PPO("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=10000)
model.save("ppo_bems")

del model # remove to demonstrate saving and loading

#######################################
#######################################

model = PPO.load("ppo_bems")
obs, _ = env.reset()

print(obs)

obs_list = []

for i in range(1440):
    action, _states = model.predict(obs)
    obs, rewards, dones, info, _ = env.step(action)
    obs_list.append(obs)
    print(obs)

