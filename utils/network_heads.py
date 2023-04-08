# -*- coding: utf-8 -*-
"""
Function for Deterministic Critic Network
"""
from utils.network_utils import *
from utils.network_body import *
from torch.nn import functional as F
import torch


class DeterministicActorCriticNet(nn.Module):
    def __init__(self, batch_size, state_dim, action_dim, obs_state_range,  action_range, num_atoms,
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
        self.v_min_critic = 0
        self.v_max_critic = 10
        self.v_min_actor = action_range[0]
        self.v_max_actor = action_range[1]
        self.num_atoms = num_atoms
        self.batch_size = batch_size

        # init output layers
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim).to(device), 1e-4).to(self.device)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, num_atoms).to(device), 1e-4).to(self.device)
        
        # concatenate params for actor and critic
        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        # set optimization for actor and critic
        self.actor_opt = actor_optim_fn(self.actor_params)
        self.critic_opt = critic_optim_fn(self.critic_params)
        
    # select feature from data
    def feature(self, obs):
        obs = torch.tensor(obs).to(self.device)
        return(self.phi_body(obs).to(self.device))
    
    # prediction for actor's neural network
    def action(self, phi):
        #return self.fc_action(self.actor_body.forward(phi)).to(self.device)
        return F.relu(self.fc_action(self.actor_body.forward(phi))).to(self.device)
        #return torch.tensor(0.5).float().to(self.device) * (torch.tensor((self.v_max_actor + self.v_min_actor)).float()+torch.tensor((self.v_max_actor - self.v_min_actor)).float()*torch.tanh(self.fc_action(self.actor_body.forward(phi)))).to(self.device)
    
    # prediction for critic's neural network
    def critic(self, data_phi, data_action, data_q):
        
        probs = torch.tensor(0.5) * (torch.tensor(self.v_max_critic + self.v_min_critic)+torch.tensor((self.v_max_critic - self.v_min_critic))*torch.tanh(self.fc_critic(self.critic_body.forward(data_phi, data_action, data_q)))).to(self.device)
        #probs = torch.tanh(self.fc_critic(self.critic_body.forward(data_phi, data_action, data_q))).to(self.device)
        #probs = F.relu(self.fc_critic(self.critic_body.forward(data_phi, data_action, data_q))).to(self.device)
        #probs = torch.nn.functional.log_softmax(self.fc_critic(self.critic_body.forward(data_phi, data_action, data_q)), dim = 1)
        #probs = torch.nn.functional.sigmoid(self.fc_critic(self.critic_body.forward(data_phi, data_action, data_q)))
        #probs = self.fc_critic(self.critic_body.forward(data_phi, data_action, data_q))
        #probs = [torch.tanh(self.fc_critic(self.critic_body.forward(phi, action, q))).to(self.device) for phi, action, q in zip(data_phi, data_action, data_q)]
        #probs = [torch.nn.functional.relu(self.fc_critic(self.critic_body.forward(phi, action, q))).to(self.device) for phi, action, q in zip(data_phi, data_action, data_q)]
        #return torch.stack((probs), dim = 0)
        return probs
    
    def calculate_action_grad(self, z_j_data, data_action): 
        
        
        # calculate gradients
        grads = torch.autograd.grad(z_j_data, data_action, grad_outputs=torch.ones_like(z_j_data))
        
        # calculate means of gradients
        grads = [torch.mul(x, torch.tensor(1/self.num_atoms)) for x in grads[0]]
        
        # convert gradients from tuple to list
        grads = torch.stack((grads), dim = 0 )
        
        return grads
    
    def train_critic(self, z_j, z_target):
        
        self.zero_grad()
        
        # calculate differences between z_target_distribution and z_distribution_sorted
        z_j, idx = torch.sort(z_j, dim = 2, descending = False)
        z_target, _ = torch.sort(z_target, dim = 2, descending = False)
        z_j =  torch.transpose(z_j, 1, 2)
        #z_target =  torch.transpose(z_target, 1, 2)
        diff_distribution = z_target - z_j

        #calculate mean of Huber quantile loss
        tau = torch.from_numpy(np.array([(2*(i+1) - 1)/(2*self.num_atoms) for i in range(self.num_atoms)])).reshape(1, self.num_atoms).to(self.device)
        inv_tau = 1.0 - tau
        huber_loss =    torch.nn.functional.huber_loss(z_j, z_target, reduction = 'none', delta = 1.0)
        
        # calculate loss function
        critic_losses = torch.where(torch.less(diff_distribution, 0.), inv_tau * huber_loss, tau * huber_loss).to(self.device)
        critic_losses = torch.sum(torch.mean(huber_loss,dim = 2, keepdim = True), dim = 1, keepdim = True).to(self.device)
        
        # calculate gradient
        self.critic_opt.zero_grad()
        critic_losses.mean(dim = 0).backward()
        
        # clip gradient
        torch.nn.utils.clip_grad_norm_(self.critic_params, 10)
        
        # train critic network
        self.critic_opt.step()
        
    def train_actor(self, action_preds, action_grad, z_j_act):
        self.zero_grad()
        
        # calculate gradients of parameters
        grads = torch.autograd.grad(action_preds, self.actor_params, grad_outputs= -action_grad)
        
        # calculate mean of gradient of actor's neural network parameters
        grads = [torch.div(x, self.batch_size) for x in grads]
        
        #clip norm gradient
        torch.nn.utils.clip_grad_norm_(self.actor_params, 10)
        
        # apply gradients
        self.actor_opt.zero_grad()
        for grad, param in zip(grads, self.actor_params):
            param.grad =  torch.mul(grad, 1)   
     
        # optimize actor neural networks
        self.actor_opt.step()