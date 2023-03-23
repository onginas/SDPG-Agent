# -*- coding: utf-8 -*-
"""
Utils for network
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math

"""
Fills the input Tensor with a (semi) orthogonal matrix. It's described in the
publication 'Exact solutions to the nonlinear dynamics
of learning in deep linear neural networks - Saxe, A. et al. (2013)'.
"""   
def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)  
    return layer


"Convert tensor to numpy"
def to_np(t):
    return t.cpu().detach().numpy()