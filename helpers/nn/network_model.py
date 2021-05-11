# import necessary pacakges
import torch
import torch.nn as nn
import torch.nn.functional as F

# initializes the weight and bias data
def layer_init(layer, w_scale=1.0, gain_func="relu"):
	nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain(gain_func))
	layer.weight.data.mul_(w_scale)
	nn.init.constant_(layer.bias.data, 0)
	return layer

# flatten class
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# Actual network
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
		self.op_1  = layer_init(nn.Linear(state_size, fc_units[0]))
		# second layer : first connected layer and second connected layer
		self.op_2  = layer_init(nn.Linear(fc_units[0], fc_units[-1]))			
		# value function
		self.value_function = layer_init(nn.Linear(fc_units[-1], 1))	
		# advantage function layer
		self.advantage_function = layer_init(nn.Linear(fc_units[-1], action_size))		
		# headless - network
		self.network = nn.Sequential(self.op_1,
									nn.ReLU(),
									nn.BatchNorm1d(fc_units[0]),
									self.op_2,
									nn.ReLU(),
									nn.BatchNorm1d(fc_units[-1]),
									Flatten())	

	def forward(self, state):
		"""
			Building the forward pass
			Args:
				Batch of states
			Returns:
				Q values for each possible action state in the batch
		"""
		model = self.network(state)
		# break up into advantage and value function layer
		# tanh activation to center around 0
		advantage_function = torch.tanh(self.advantage_function(model))
		# value function learns important states
		value_function = torch.tanh(self.value_function(model))
		# expand value_function to size of advantage function
		value_function = value_function.expand_as(advantage_function)
		
		# Output is the the aggregate of both - we subtract average advantage over all actions. This forces
		# the advantage function to be zero biased and then we avoid the identifiabbility issue
		output = value_function + advantage_function - advantage_function.mean(dim=1, 
														keepdim=True).expand_as(advantage_function)
		# return the output
		return output