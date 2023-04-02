# -*- coding: utf-8 -*-
"""
Script for DDPG Agent and Replay buffer
"""

from collections import deque, namedtuple
import random
import numpy as np
from utils.randomProcess import *
from agent_based import *
from utils.network_body import *
from utils.network_heads import *
from utils.network_utils import *
import torch.nn.functional as F
import torch

"""
DDPG Agent algorithm
"""

class SDPGAgent(BaseAgent):
    
    # initialize agent object
    def __init__(self, state_dim, action_dim, obs_state_range, action_range, device='cpu'):
        
        # initialize parent object class into children object
        super(BaseAgent, self).__init__()
        
        # set basic parameters of agent
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = 0.98
        self.target_network_mix = 1e-3
        self.device = device
        self.n_step = 0
        self.train_step = 0
        self.warm_up = 4
        self.buffer_size = int(0.2*1e6)
        self.batch_size = 256
        self.seed = 0
        self.NUM_ATOMS = 80
        self.obs_state_range = obs_state_range
        self.action_range = action_range
        self.gamma = torch.from_numpy(np.vstack([self.discount**i for i in range(self.batch_size)])).float().to(self.device)
        self.gamma = self.gamma[:,None,:]
        
        # create local network
        self.network =  DeterministicActorCriticNet(self.state_dim, self.action_dim, self.obs_state_range,  self.action_range, self.NUM_ATOMS,
                                                                   actor_optim_fn  = lambda params: torch.optim.Adam(params, lr=1e-4),
                                                                   critic_optim_fn = lambda params: torch.optim.Adam(params, lr=1e-4), 
                                                                   actor_body = FCBody(self.state_dim, hidden_units = (400, 256), function_unit = F.relu),
                                                                   critic_body = CriticBody(self.state_dim,  self.action_dim,  self.NUM_ATOMS, hidden_units = (400, 256), function_unit = F.relu), device=device)
        # create target network
        self.target_network = DeterministicActorCriticNet(self.state_dim, self.action_dim, self.obs_state_range,  self.action_range, self.NUM_ATOMS,
                                                                   actor_optim_fn  = lambda params: torch.optim.Adam(params, lr=1e-4),
                                                                   critic_optim_fn = lambda params: torch.optim.Adam(params, lr=1e-4), 
                                                                   actor_body = FCBody(self.state_dim, hidden_units = (400, 256), function_unit = F.relu),
                                                                   critic_body = CriticBody(self.state_dim,  self.action_dim,  self.NUM_ATOMS, hidden_units = (400, 256), function_unit = F.relu), device=device)
        
        # copy parameters to target network from local network
        self.target_network.load_state_dict(self.network.state_dict())
        
        # add noise from Ornstein-Uhlenbeck process
        self.random_process_fn = OrnsteinUhlenbeckProcess(
                                            size=(self.action_dim, ),
                                            std=0.2)
        
        # create replay buffer
        self.replay_buffer = ReplayBuffer(action_size=self.action_dim, buffer_size=self.buffer_size, batch_size=self.batch_size, seed=self.seed, device = self.device, num_atoms=self.NUM_ATOMS)
    # function for copy all parameters between models    
    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1 - self.target_network_mix) + param*self.target_network_mix).to(self.device)
    
    # function for selection of action
    def select_action(self, state, noise = True):
        # convert input from numpy to tensor
        state = torch.from_numpy(state).to(self.device)
        #state = torch.FloatTensor(state).to(self.device)
        # predict action from local neural network
        self.network.eval()
        with torch.no_grad():
            action = self.network.action(state).to(self.device)
        self.network.train()
        
        # convert action into numpy
        action = to_np(action)
        
        # add noise to input data
        if noise:
            action = action + 0.3*self.random_process_fn.sample()
        action = np.clip(action, -1, 1)
        return action
    
    # make step in learning process
    def step(self, state, action, reward, next_state, done):
        # increment number step
        self.n_step += 1
        
        # add the experience into buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        if self.replay_buffer.__len__() == self.replay_buffer.batch_size - 1:
            self.learn_start = True
        else:
            self.learn_start = False
        
        # run learning process if the length of buffer is same or higher than batch size
        if self.replay_buffer.__len__() >= self.replay_buffer.batch_size:
            self.learn()
    
    # function for learning process of agent
    def learn(self):
        self.train_step += 1
        # choose random sample from object
        b_states, b_actions, b_rewards, b_next_states, b_dones, q_j, q_j_target = self.replay_buffer.sample()
        
        # add dimension
        b_states = b_states[:,None,:]
        b_actions = b_actions[:,None,:]
        b_rewards = b_rewards[:,None,:]
        b_next_states = b_next_states[:,None,:]
        b_dones = b_dones[:,None,:]
        q_j = q_j[:,None,:]
        q_j_target = q_j_target[:,None,:]
        
        
        # predict next action from next states
        with torch.no_grad():
            b_next_actions = self.target_network.action(b_next_states)
       
        # calculate z target distribution
        with torch.no_grad():
            z_target =  torch.div(b_rewards, 1) + self.gamma * self.target_network.critic(b_next_states, b_next_actions, q_j_target) #* (1 - b_dones)
            #z_target = torch.transpose(z_target, 1, 2)
            #z_target, _ = torch.sort(z_target, dim = 1, descending = False)
        z_target.requires_grad_(False)
        
        
        #calculate z_j
        z_j = self.network.critic(b_states, b_actions, q_j).requires_grad_(True)
        
        
        # calculate differences between z_target_distribution and z_distribution_sorted
        z_j, _ = torch.sort(z_j, dim = 2, descending = False)
        #z_target, _ = torch.sort(z_target, dim = 2, descending = False)
        z_j =  torch.transpose(z_j, 1, 2)
        #z_target =  torch.transpose(z_target, 1, 2)
        diff_distribution = z_target - z_j

        #calculate mean of Huber quantile loss
        tau = torch.from_numpy(np.array([(2*(i+1) - 1)/(2*self.NUM_ATOMS) for i in range(self.NUM_ATOMS)])).reshape(1, self.NUM_ATOMS).to(self.device)
        inv_tau = 1.0 - tau
        huber_loss =    torch.nn.functional.huber_loss(z_j, z_target, reduction = 'none', delta = 1.0)
        critic_losses = torch.where(torch.less(diff_distribution, 0.), inv_tau * huber_loss, tau * huber_loss).to(self.device)
        critic_losses = torch.mean(torch.mean(torch.mean(huber_loss,dim = 2, keepdim = True), dim = 1, keepdim = True), dim = 0).to(self.device).requires_grad_(True)
        
        
        # Compute actor loss
        b_states.requires_grad_(True)
        action_preds = self.network.action(b_states).to(self.device).requires_grad_(True)
        z_j_act = self.network.critic(b_states, action_preds, q_j).requires_grad_(True)
        #z_j_act, _ = torch.sort(z_j_act, dim = 1, descending = False)
        action_losses_mean = self.network.calculate_action_grad(z_j_act, action_preds)
        action_losses_mean = action_losses_mean[0].requires_grad_(True)
        action_losses_mean = torch.div(action_losses_mean, self.batch_size)
                
        # minimize the critic losses
        self.network.critic_opt.zero_grad()
        critic_losses.backward(retain_graph = True)
        torch.nn.utils.clip_grad_norm_(self.network.critic_params, 1)
        self.network.critic_opt.step()
        
   
        # calculate gradients
        grads = torch.autograd.grad(action_preds, self.network.actor_params, grad_outputs= action_losses_mean, retain_graph = True)

        grads = [x/self.batch_size for x in grads]
        self.network.actor_opt.zero_grad()

        # apply gradients
        for grad, param in zip(grads, self.network.actor_params):
            param.grad = grad
            param._grad = grad
        
        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.actor_params, 1)
     
        # optimize actor neural networks
        self.network.actor_opt.step()
        
        
        self.network.zero_grad()

        
        
        # update params of target neural network
        if self.n_step % self.warm_up  == 0:
            self.soft_update(self.target_network, self.network)
 
    
     
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    # Initialize a ReplayBuffer object.
    def __init__(self, action_size, buffer_size, batch_size, seed, device, num_atoms):

        # parameters initialization
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
        self.NUM_ATOMS = num_atoms
        
    # function for adding experience into buffer
    def add(self, state, action, reward, next_state, done):
        ex = self.experience(state, action, reward, next_state, done)
        self.memory.append(ex)
        
    # function to generate random sample
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        # convert outputs into tensor
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        q_j =   torch.from_numpy(np.vstack([np.random.normal(size=[self.NUM_ATOMS]) for e in experiences])).float().to(self.device)
        q_j_target = torch.from_numpy(np.vstack([np.random.normal(size=[self.NUM_ATOMS]) for e in experiences])).float().to(self.device)
  
       # print(states)
        return (states, actions, rewards, next_states, dones, q_j, q_j_target)

    # function to get length of the memory
    def __len__(self):
        return len(self.memory)

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()        