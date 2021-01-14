"""
Define encoders and decoders.

"""
__date__ = "January 2021"


import torch


class FcNet(torch.nn.Module):

	def __init__(self, dims, activation=torch.nn.ReLU):
		"""
		Simple fully-connected network with activations between layers.

		Parameters
		----------
		dims : list of int
		activation : torch.nn.Module?
		"""
		super(FcNet, self).__init__()
		layers = []
		assert len(dims) > 1
		if len(dims) == 2:
		for i in range(len(dims)-2):
			layers.append(torch.nn.Linear(dims[i],dims[i+1]))
			layers.append(activation())
		layers.append(torch.nn.Linear(dims[-2], dims[-1]))
		self.net = torch.nn.Sequential(*layers)


	def forward(self, x):
		"""
		Send `x` through the network.

		Parameters
		----------
		x : torch.Tensor
		"""
		return self.net(x)




if __name__ == '__main__':
	pass



###
