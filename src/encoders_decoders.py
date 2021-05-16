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
		xs : tuple of torch.Tensor
		"""
		# if isinstance(xs, (tuple,list)):
		assert len(xs) == len(self.nets)
		outputs = tuple(net(x) for net, x in zip(self.nets, xs))
		# else:
		# outputs = tuple(net(xs) for net in self.nets)
		return outputs



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



class ConcatLayer(torch.nn.Module):

	def __init__(self, dim=-1):
		super(ConcatLayer, self).__init__()
		self.dim = dim

	def forward(self, args):
		return torch.cat(args, dim=self.dim)



class GatherLayer(torch.nn.Module):

	def __init__(self, transpose=False):
		"""
		Take inputs and wrap them in a tuple.
		"""
		super(GatherLayer, self).__init__()
		self.transpose = transpose

	def forward(self, args):
		temp = (args,)
		if self.transpose:
			temp = tuple(tuple(t) for t in zip(*temp))
		return temp



class SqueezeLayer(torch.nn.Module):

	def __init__(self, dim=-1):
		"""
		Squeeze the given dimension.
		"""
		super(SqueezeLayer, self).__init__()
		self.dim = dim

	def forward(self, x):
		return x.squeeze(self.dim)



class UnsqueezeLayer(torch.nn.Module):

	def __init__(self, dim=-1):
		"""
		Unsqueeze the given dimension.
		"""
		super(UnsqueezeLayer, self).__init__()
		self.dim = dim

	def forward(self, x):
		return x.unsqueeze(self.dim)



class EncoderModalityEmbedding(torch.nn.Module):

	def __init__(self, n_modalities, embed_dim=8):
		"""
		Create a trainable modality representation, concatenate to inputs.

		Parameters
		----------
		n_modalities : int
		embed_dim : int, optional
		"""
		super(EncoderModalityEmbedding, self).__init__()
		self.embed = torch.nn.Parameter(
				torch.randn(1, n_modalities, embed_dim),
		)


	def forward(self, x):
		"""
		Concatenate the modality representation to x.

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
		assert x.shape[1] == self.embed.shape[1]
		temp_embed = torch.tanh(self.embed) # [1,m,e]
		temp_embed = temp_embed.expand(x.shape[0], -1, -1) # [b,m,e]
		return torch.cat([x,temp_embed], dim=2) # [b,m,m_dim+e]


class DecoderModalityEmbedding(torch.nn.Module):

	def __init__(self, n_modalities, embed_dim=8):
		"""
		Create a trainable modality representation, concatenate to inputs.

		Parameters
		----------
		n_modalities : int
		embed_dim : int, optional
		"""
		super(DecoderModalityEmbedding, self).__init__()
		self.m = n_modalities
		self.embed = torch.nn.Parameter(
				torch.randn(1, 1, self.m, embed_dim),
		)


	def forward(self, x):
		"""
		Concatenate the modality representation to x.

		Parameters
		----------
		x : torch.Tensor
			Shape: [b,s,z]

		Returns
		-------
		x_out : torch.Tensor
			Shape: [batch,samples,modalities,_+embed_dim]
		"""
		assert len(x.shape) == 3
		x = x.unsqueeze(2).expand(-1,-1,self.m,-1) # [b,s,m,z]
		temp_embed = torch.tanh(self.embed) # [1,1,m,e]
		temp_embed = temp_embed.expand(len(x), x.shape[1], -1, -1) # [b,s,m,e]
		return torch.cat([x,temp_embed], dim=3) # [b,s,m,z+e]


if __name__ == '__main__':
	pass



###
