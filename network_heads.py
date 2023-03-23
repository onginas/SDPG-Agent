# -*- coding: utf-8 -*-
"""
Function for Deterministic Critic Network
"""
from network_utils import *
from network_body import *
from torch.nn import functional as F
import torch


class DeterministicActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, obs_state_range,  action_range, num_atoms,
                 actor_optim_fn, critic_optim_fn,
                 phi_body=None, actor_body=None, critic_body=None, device = 'cpu'):
        
        # initializes the parent class object into the child class
        super(DeterministicActorCriticNet, self).__init__()
        
        # create dummy bodies in case that some body is not defined 
        if phi_body is None:
            phi_body = DummyBody(state_dim)
        
        if actor_body is None:
            actor_body = DummyBody(state_dim)
        
        if critic_body is None:
            critic_body = DummyBody(state_dim)
        
        #initialize input bodies
        self.device = device
        self.phi_body = phi_body.to(device)
        self.actor_body = actor_body.to(device)
        self.critic_body = critic_body.to(device)
        self.v_min_critic = -20
        self.v_max_critic =  0
        self.v_min_actor = action_range[0]
        self.v_max_actor = action_range[1]
        self.num_atoms = num_atoms

        # init output layers
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim).to(device), 1e-4).to(self.device)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, num_atoms).to(device), 1e-4).to(self.device)
        
        # concatenate params for actor and critic
        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        # set optimization for actor and critic
        self.actor_opt = actor_optim_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_optim_fn(self.critic_params + self.phi_params)
        
    # select feature from data
    def feature(self, obs):
        obs = torch.tensor(obs).to(self.device)
        return(self.phi_body(obs).to(self.device))
    
    # prediction for actor's neural network
    def action(self, phi):
        return torch.tanh(self.fc_action(self.actor_body.forward(phi)).to(self.device)).to(self.device)
        #return torch.tensor(0.5).to(self.device) * (torch.tensor(self.v_max_actor + self.v_min_actor)+torch.tensor((self.v_max_actor - self.v_min_actor))*torch.tanh(self.fc_action(self.actor_body.forward(phi)))).to(self.device)
   
    # prediction for critic's neural network
    def critic(self, data_phi, data_action, data_q):
        
        #probs = [torch.tensor(0.5) * (torch.tensor(self.v_max_critic + self.v_min_critic)+torch.tensor((self.v_max_critic - self.v_min_critic))*torch.tanh(self.fc_critic(self.critic_body.forward(phi, action, q)))).to(self.device) for phi, action, q in zip(data_phi, data_action, data_q)]
        probs = [torch.nn.functional.log_softmax(self.fc_critic(self.critic_body.forward(phi, action, q)), dim = -1).to(self.device) for phi, action, q in zip(data_phi, data_action, data_q)]
        #probs = [torch.tanh(self.fc_critic(self.critic_body.forward(phi, action, q))).to(self.device) for phi, action, q in zip(data_phi, data_action, data_q)]
        #probs = [torch.nn.functional.relu(self.fc_critic(self.critic_body.forward(phi, action, q))).to(self.device) for phi, action, q in zip(data_phi, data_action, data_q)]
        return torch.stack((probs), dim = 0)

    
    def calculate_action_grad(self, data_phi, data_action, data_q): 
        grads = []
        for phi, action, q in zip(data_phi, data_action, data_q):
            action.requires_grad_(True)
            #z_j =  torch.tensor(0.5) * (torch.tensor(self.v_max_critic + self.v_min_critic)+torch.tensor((self.v_max_critic - self.v_min_critic))*torch.tanh(self.fc_critic(self.critic_body.forward(phi, action, q)))).to(self.device)
            z_j =  torch.nn.functional.log_softmax(self.fc_critic(self.critic_body.forward(phi, action, q)), dim = -1).to(self.device)
            #z_j =  torch.tanh(self.fc_critic(self.critic_body.forward(phi, action, q))).to(self.device)
            #z_j =  torch.nn.functional.relu(self.fc_critic(self.critic_body.forward(phi, action, q))).to(self.device)
            grad = torch.autograd.grad(z_j.mean(), action, retain_graph=True)
            grads.append(grad)
        grads=[torch.mul(x[0], -1) for x in grads]
        return torch.stack((grads), dim = 0 )