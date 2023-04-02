# -*- coding: utf-8 -*-
"""
Utils for creating bodies neral netowrks
 
"""
from utils.network_utils import *

class FCBody(nn.Module):
    
    def __init__(self, state_dim, hidden_units = (128, 128, 64), function_unit = F.relu):
        
        # initializes the parent class object into the child class
        super(FCBody, self).__init__()
        self.dims = (state_dim,) + hidden_units
        
        # create module list with layers
        self.layers = nn.ModuleList(
             [layer_init(nn.Linear(dim_in, dim_out))  for dim_in, dim_out in zip(self.dims[:-1], self.dims[1:])])
        self.batch_layers = [nn.LayerNorm([hidden_unit]) for hidden_unit in hidden_units]
        self.dropout_layers = [torch.nn.Dropout(p=0.2) for _ in range(len(self.dims))]
        # define function unit
        self.function_unit = function_unit
        self.feature_dim = self.dims[-1]
    
    # forward function for prediction
    def forward(self, x):
        for layer, dropout, batch_layer in zip(self.layers, self.dropout_layers, self.batch_layers):
            x = layer(x)
            #x = batch_layer(x)
            x = dropout(x)
            x = self.function_unit(x)
        return x
        
class DummyBody(nn.Module):
    
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim
        
    def forward(self, x):
        return x

class CriticBody(nn.Module):
            
    def __init__(self, state_dim, action_dim, num_atoms, hidden_units = (400, 300), function_unit = F.relu):
                
        # initializes the parent class object into the child class
        super(CriticBody, self).__init__()
        self.function_unit = function_unit
        self.state_dim = state_dim        
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.feature_dim = hidden_units[-1]
        self.batch_norm = nn.LayerNorm([hidden_units[0]])
            # forward function for prediction
        self.linear0 = nn.Linear(state_dim, hidden_units[0])
        #self.linear1 = nn.Linear(hidden_units[0], hidden_units[0])
        self.linear2a = nn.Linear(hidden_units[0], hidden_units[1])
        self.linear2b = nn.Linear(action_dim, hidden_units[1])
        self.linear2c = nn.Linear(num_atoms, hidden_units[1])
        self.linear2 = nn.Linear(hidden_units[1], num_atoms)

    def forward(self, state, action, noise):
        #x1 =  self.function_unit(self.batch_norm(self.linear1(state)))
        x0 =  self.function_unit(self.linear0(state))
        #x1 =  self.function_unit(self.linear1(x0))
        #print(action.shape)
        x2a = self.function_unit(self.linear2a(x0))
        x2b = self.function_unit(self.linear2b(action))
        x2c = self.function_unit(self.linear2c(noise))
        x =   self.function_unit(x2a + x2b + x2c)
        return x