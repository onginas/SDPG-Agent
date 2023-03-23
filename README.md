#SDPG Algorithm for PyTorch
Udacity Deep Reinforcement Learning Nanodegree Program - implementation of Sample-Based Distributional Policy Gradient (SDPG) in PyTorch (https://arxiv.org/pdf/2001.02652.pdf).


### Observations:
- There's <b>Rerpot.ipynb</b> file for jupyter notebook execution where is described and showed the implementation of SDPG Agent
- The necessary python files are below. There's necessary to keep all these files in current workdir
	* network_utils.py
		*utilities for neural network
	
	* network_body.py
		*files with classes for Fully Connected Neural Network or Dummy Body
	
	* network_heads.py
		*file with critic and actor neural network with function for prediction for Q_values or action
	
	* agent_based.py
		* file with base function for each agent
	
	* SDPG_agent.py
		*file with DDPG Agent with functions
	
	* randomProcess.py
		*file with Orstein-Uhlenbeck process for adding noise

### Requeriments:
- numpy
- jupyter
- gym
- pandas
- sklearn
- ipykernel
- torch
- seaborn
- matplotlib

### Tested model
LunarLander-v2 fro OpenAI Gym

### The hyperparameters:
- the hyperpameters are in the file <b>SDPG_agent.py</b>.
- The actual configuration of the hyperparameters is: 
  - Learning Rate: 1e-4 (in both DNN)
  - Batch Size: 180
  - Replay Buffer: 1e6
  - Gamma: 0.99
  - Tau: 1e-3
  - Ornstein-Uhlenbeck noise parameters (0.15 theta and 0.2 sigma.)
  - warm-up: 4 (number of episodes before the target network is updated)
  - num_atoms: 100
  
- For the neural models:    
  - Actor    
    - Hidden: (input, 128)  - function unit: ReLU
    - Hidden: (128, 400)    - function unit: ReLU
    - Output: (400, 2)      - function unit: TanH

  - Critic
    - Hidden: (input, 128)	                        - function unit: ReLU
    - Hidden_2a: (128, 128)  				        - function unit: ReLU
	- Hidden_2b: (action_size, 128)			        - function unit: ReLU
	- Hidden_2c: (num_atoms, 128)			        - function unit: ReLU
	- Hidden_2: (Hidden_2a + Hidden_2b + Hidden_2c) - function unit: ReLU
    - Output: (128, num_atoms)                      - function unit: ReLU
