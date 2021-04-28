# import necessary pacakges
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingNetwork(nn.Module):
	"""
		Incoming layer is a batch of states. Each row represents 
		[velocity, and ray based perception] of the environment.
		We use two Fc layers and then a value function and advantage
		function layer
	"""
	def __init__(self, state_size, action_size, seed, fc_units=[64,64]):
		"""
			Initialize parameters and build model
				state_size (int): We observe 37 states - [velocity, and ray based perception]
				action_size(int): Four actions - [foward, backward, left, right]
				seed       (int): Random seed
				fc_units  (array int): array of number of nodes in fc layers
		"""
		super(DuelingNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)
		# store layers before value function
		# first layer : state and first connected layer
		self.op_1  = nn.Linear(state_size, fc_units[0])
		nn.init.orthogonal_(self.op_1.weight)
		# second layer : first connected layer and second connected layer
		self.op_2  = nn.Linear(fc_units[0], fc_units[-1])
		nn.init.orthogonal_(self.op_2.weight)

		self.value_function = nn.Linear(fc_units[-1], 1)
		nn.init.orthogonal_(self.value_function.weight)
		# advantage function layer
		self.advantage_function = nn.Linear(fc_units[-1], action_size)
		nn.init.orthogonal_(self.advantage_function.weight)
		

	def forward(self, states):
		"""
			Building the forward pass
			Args:
				Batch of states
			Returns:
				Q values for each possible action state in the batch
		"""
		model = F.relu(self.op_1(states))
		model = F.relu(self.op_2(model))
		

		# break up into advantage and value function layer
		# value function learns important states
		value_function = self.value_function(model)
		# benerfits of taking indivudial actions in each presented state
		advantage_function = self.advantage_function(model)
		# Output is the the aggregate of both - we subtract average advantage over all actions. This forces
		# the advantage function to be zero biased and then we avoid the identifiabbility issue
		output = value_function + torch.sub(advantage_function, advantage_function.mean(dim=1, keepdim=True))
		# return the output
		return output