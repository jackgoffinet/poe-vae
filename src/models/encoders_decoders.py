"""
Define encoders and decoders.

"""
__date__ = "January 2021"


import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector



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
		if isinstance(xs, (tuple,list)):
			assert len(xs) == len(self.nets)
			outputs = tuple(net(x) for net, x in zip(self.nets, xs))
		else:
			outputs = tuple(net(xs) for net in self.nets)
		return outputs



class MLP(torch.nn.Module):

	def __init__(self, dims, activation=torch.nn.ReLU, last_activation=False):
		"""
		Simple fully-connected network with activations between layers.

		Parameters
		----------
		dims : list of int
		activation : torch.nn.Module?
		"""
		super(MLP, self).__init__()
		layers = torch.nn.ModuleList()
		assert len(dims) > 1
		for i in range(len(dims)-2):
			layers.append(torch.nn.Linear(dims[i],dims[i+1]))
			layers.append(activation())
		layers.append(torch.nn.Linear(dims[-2], dims[-1]))
		if last_activation:
			layers.append(activation())
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
		return tuple(layer(x) for layer in self.layers)



class EncoderModalityEmbedding(torch.nn.Module):

	def __init__(self, n_modalities, embed_dim=8):
		"""


		"""
		super(EncoderModalityEmbedding, self).__init__()
		self.embed = \
				torch.nn.Parameter(torch.randn(1, n_modalities, embed_dim))


	def forward(self, x):
		"""
		Send `x` through the network.

		Parameters
		----------
		x : torch.Tensor
			Shape: [batch,modalities,m_dim]

		Returns
		-------
		x_out : torch.Tensor
			Shape: [batch,modalities,m_dim+embed_dim]
		"""
		assert len(x.shape) == 3
		temp_embed = torch.tanh(self.embed)
		temp_embed = temp_embed.expand(x.shape[0], -1, -1)
		x = x.expand(-1, temp_embed.shape[1], -1)
		return torch.cat([x,temp_embed], dim=2)


class DecoderModalityEmbedding(torch.nn.Module):

	def __init__(self, n_modalities, embed_dim=8):
		"""


		"""
		super(DecoderModalityEmbedding, self).__init__()
		self.embed = \
				torch.nn.Parameter(torch.randn(1, 1, n_modalities, embed_dim))


	def forward(self, x):
		"""
		Send `x` through the network.

		Parameters
		----------
		x : torch.Tensor
			Shape: [batch,samples,_] or [batch,samples,modalities,_]

		Returns
		-------
		x_out : torch.Tensor
			Shape: [batch,samples,modalities,_+embed_dim]
		"""
		if len(x.shape) == 3:
			x = x.unsqueeze(2).expand(-1,-1,self.embed.shape[2],-1)
		temp_embed = torch.tanh(self.embed)
		temp_embed = temp_embed.expand(x.shape[0], x.shape[1], -1, -1)
		return torch.cat([x,temp_embed], dim=3)


if __name__ == '__main__':
	pass



###
