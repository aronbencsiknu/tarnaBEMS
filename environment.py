
import math
import random
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces


class BEMSEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, initial_desired_temp, render_mode: Optional[str] = None):
        

        self.desired_temp = initial_desired_temp
        self.heating = False

        self.heating_temp = 40
        self.alpha_heating = 0.005
        self.alpha_heat_conductance= 0.003
        #self.beta_heating = 2
        #self.beta_heat_conductance = 2

        high = np.array(
            [
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.counter = 0 # midnight

        self.min_external_temp = -10
        self.max_external_temp = 15
        self.external_temp = self._update_external_temp(self.counter, self.min_external_temp, self.max_external_temp)
        self.temp = self.external_temp

    def step(self, action):

        if action:
            self.heating = True
        
        else:
            self.heating = False

        self.external_temp = self._update_external_temp(self.counter, self.min_external_temp, self.max_external_temp)

        if self.heating:
            self.temp += self._compute_heat_transfer(self.temp, self.heating_temp, self.alpha_heating)

        self.temp += self._compute_heat_transfer(self.temp, self.external_temp, self.alpha_heat_conductance)

        if self.render_mode == "human":
            self.render()

        reward = self._compute_reward(self.heating, self.temp, self.desired_temp)
        observation = [self.temp, self.external_temp, self.heating]
        terminated = False

        self.counter += 1

        return observation, reward, terminated, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.desired_temp = 20
        self.heating = False

        self.heating_temp = 40
        self.alpha_heating = 0.005
        self.alpha_heat_conductance= 0.003
        #self.beta_heating = 2
        #self.beta_heat_conductance = 2

        self.counter = 0 # midnight

        self.min_external_temp = -10
        self.max_external_temp = 15
        self.external_temp = self._update_external_temp(self.counter, self.min_external_temp, self.max_external_temp)
        self.temp = self.external_temp

        observation = [self.temp, self.external_temp, self.heating]

        return observation, {}

    def render(self):
        pass

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def _update_external_temp(self, index, min_temp, max_temp, scatter=0.0):
        # compute current outside temperature
        # and forecast temperature

        cosine_temp = (math.cos(index/229)*((max_temp-min_temp)/2))+((max_temp-min_temp)/2)+min_temp
        current_temp = cosine_temp + random.uniform(-scatter/2, scatter/2)

        return current_temp
    
    def _compute_heat_transfer(self, mobile_temp, immobile_temp, alpha=1, beta=1):

        """
        mobile_temp: internal temp of building, can be modified by heating or outside temperature.
        immobile_temp: the heating and outside temperature are abstracted as immobile, non-modifiable, temperatures
        alpha: represents the effect of the immobile temperature on the mobile temperature
        beta: scaling factor
        """
        return math.tanh((immobile_temp - mobile_temp) * alpha) * beta
    
    def _compute_reward(self, heating, internal_temp, desired_temp):
        reward = 0.0
        if heating:
            reward -= 1.0

        if internal_temp < desired_temp:
            reward -= (desired_temp - internal_temp)

        return reward