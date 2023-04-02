# -*- coding: utf-8 -*-
"""
Script for random process impolementation
"""

import numpy as np

#class for noise from Ornstein Uhlenbeck Process
class OrnsteinUhlenbeckProcess:
    
    # object initialization
    def __init__(self, size, std, theta=0.15, dt = 1e-2, x0 = None):
        
        # initialization of parameters
        self.size = size
        self.theta = theta
        self.mu = 0.0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.reset_states()
    
    # function for reset states
    def reset_states(self):
        self.x_prev  = self.x0 if self.x0 is not None else np.zeros(self.size)
    
    # generate sample of noise    
    def sample(self):
        x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + self.std*np.random.normal(loc=0.0,scale=1.0,size=(self.size))*np.sqrt(self.dt)
        self.x_prev = x
        return x
                                                                       