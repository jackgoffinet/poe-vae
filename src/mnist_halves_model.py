"""
Binarized MNIST halves networks.

"""
__date__ = "May 2021"


import torch
import torch.nn as nn

from .encoders_decoders import SplitLinearLayer, NetworkList, \
		GatherLayer
from .likelihoods import GroupedLikelihood
from .param_maps import LIKELIHOOD_MAP



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
				make_single_encoder(**kwargs),
				make_single_encoder(**kwargs),
			]),
		),
		'decoder': make_decoder(**kwargs),
		'likelihood': likelihood_helper(**kwargs),
	}


def make_single_encoder(variational_strategy='gaussian_poe', latent_dim=20,
	vmf_dim=4, n_vmfs=5, **kwargs):
	""" """
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


def make_decoder(variational_strategy='gaussian_poe', latent_dim=20, \
	vmf_dim=4, n_vmfs=5, **kwargs):
	""" """
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
		SplitLinearLayer(500,(784//2,784//2)),
		GatherLayer(),
	)


def likelihood_helper(likelihood='spherical_gaussian', **kwargs):
	"""Make a different likelihood for each modality."""
	return GroupedLikelihood([
		LIKELIHOOD_MAP[likelihood](**kwargs) for _ in range(2)
	])



if __name__ == '__main__':
	pass



###
