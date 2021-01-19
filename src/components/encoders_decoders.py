"""
Define encoders and decoders.

"""
__date__ = "January 2021"


import torch


class NetworkList(torch.nn.Module):

	def __init__(self, nets):
		"""
		Collection of networks for separate modalities.

		Parameters
		----------
		nets : torch.nn.ModuleList
			Networks, one for each modality.
		"""
		super(NetworkList, self).__init__()
		self.nets = nets


	def forward(self, xs):
		"""
		Send `xs` through the networks.

		NOTE: TO DO explain the if/else block

		Parameters
		----------
		xs : list of torch.Tensor
		"""
		if type(xs) == type([]):
			assert len(xs) == len(self.nets)
			outputs = [net(x) for net, x in zip(self.nets, xs)]
		else:
			outputs = [net(xs) for net in self.nets]
		return outputs



class MLP(torch.nn.Module):

	def __init__(self, dims, activation=torch.nn.ReLU):
		"""
		Simple fully-connected network with activations between layers.

		Parameters
		----------
		dims : list of int
		activation : torch.nn.Module?
		"""
		super(MLP, self).__init__()
		layers = []
		assert len(dims) > 1
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



class SplitLinearLayer(torch.nn.Module):

	def __init__(self, in_dim, out_dims):
		"""
		Wraps single input, multi-output matrix multiplies.

		Parameters
		----------
		in_dim : int
		out_dims : list of int
		"""
		super(SplitLinearLayer, self).__init__()
		self.layers = [torch.nn.Linear(in_dim, out_dim) for out_dim in out_dims]
		self.layers = torch.nn.ModuleList(self.layers)

	def forward(self, x):
		"""
		Send `x` through the network.

		Parameters
		----------
		x : torch.Tensor
		"""
		return [layer(x) for layer in self.layers]



class ModalityEmbeddingCatLayer(torch.nn.Module):

	def __init__(self, n_modalities, feature_dim=8):
		"""


		"""
		super(ModalityEmbeddingCatLayer, self).__init__()
		self.embed = \
				torch.nn.Parameter(torch.randn(1, n_modalities, feature_dim))


	def forward(self, x):
		"""
		Send `x` through the network.

		Parameters
		----------
		x : torch.Tensor
		"""
		print("forward x", x.shape)
		assert len(x.shape) == 3
		temp_embed = torch.tanh(self.embed)
		temp_embed = temp_embed.expand(x.shape[0], -1, -1)
		x = x.expand(-1, temp_embed.shape[1], -1)
		return torch.cat([x,temp_embed], dim=2)



if __name__ == '__main__':
	pass



###
