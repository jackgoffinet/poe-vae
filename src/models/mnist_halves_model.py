"""
Binary MNIST halves networks.

"""
__date__ = "May 2021"


import torch
import torch.nn as nn

from .encoders_decoders import SplitLinearLayer



def get_vae(**kwargs):
	"""
	Main method: return VAE parts for this dataset.

	Returns
	-------
	vae : dict
		Maps keys `'encoder'` and `'decoder'` to `torch.nn.Module`s.
	"""
	return {
		'encoder': encoder_helper(**kwargs),
		'decoder': decoder_helper(**kwargs),
	}


def encoder_helper(variational_strategy='gaussian_poe', latent_dim=20,
	vmf_dim=4, n_vmfs=5, **kwargs):
	if variational_strategy == 'gaussian_poe':
		z_dim = latent_dim
	elif variational_strategy == 'vmf_poe':
		z_dim = (vmf_dim + 1) * n_vmfs
	else:
		raise NotImplementedError(f"{variational_strategy} not implemented!")
	return nn.Sequential(
		nn.Sequential(
			nn.Linear(784//2,500),
			nn.ReLU(),
			nn.Linear(500,500),
			nn.ReLU(),
			nn.Linear(500,200),
			nn.ReLU(),
		),
		SplitLinearLayer(200,(z_dim,z_dim)),
	)


def decoder_helper(variational_strategy='gaussian_poe', latent_dim=20, \
	vmf_dim=4, n_vmfs=5, **kwargs):
	if variational_strategy == 'gaussian_poe':
		z_dim = latent_dim
	elif variational_strategy == 'vmf_poe':
		z_dim = (vmf_dim + 1) * n_vmfs
	else:
		raise NotImplementedError(f"{variational_strategy} not implemented!")
	return nn.Sequential(
		nn.Linear(z_dim,200),
		nn.Linear(200,500),
		nn.Linear(500,500),
		nn.Linear(500,784),
	)



class Encoder(nn.Module):

	def __init__(self):
		super(Encoder, self).__init__()
		self.encoder_1 = encoder_helper()
		self.encoder_2 = encoder_helper()

	def forward(self, x_1, x_2):
		out_1 = self.encoder_1(x_1)
		out_2 = self.encoder_2(x_2)
		return tuple([i,j] for i,j in zip(out_1,out_2))


class Decoder(nn.Module):

	def __init__(self):
		super(Decoder, self).__init__()
		self.decoder = decoder_helper()

	def forward(self, z):
		return self.decoder(z)



if __name__ == '__main__':
	pass



###
