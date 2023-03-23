# -*- coding: utf-8 -*-
"""
Base agent with the basic functions for save and load model
"""

import torch
import numpy as np
import pickle

class BaseAgent:
        
    def save_model(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
    
    def load_model(self, filename):
        state_dict = torch.load('%s.model' % filename)
        self.network.load_state_dict(state_dict)
 
            
    