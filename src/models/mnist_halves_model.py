"""
Binary MNIST halves networks.

"""
__date__ = "May 2021"


import torch
import torch.nn as nn

from .encoders_decoders import SplitLinearLayer, NetworkList



def get_vae(**kwargs):
	"""
	Main method: return VAE parts for this dataset.

	Returns
	-------
	vae : dict
		Maps keys `'encoder'` and `'decoder'` to `torch.nn.Module`s.
	"""
	return {
		'encoder': NetworkList(
			nn.ModuleList([
				encoder_helper(**kwargs),
				encoder_helper(**kwargs),
			]),
		),
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



if __name__ == '__main__':
	pass



###
