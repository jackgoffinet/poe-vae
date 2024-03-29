"""
Binarized MNIST halves networks.

"""
__date__ = "May 2021"


import torch
import torch.nn as nn

from .encoders_decoders import SplitLinearLayer, NetworkList, GatherLayer, \
		UnsqueezeLayer, ConcatLayer
from .likelihoods import GroupedLikelihood
from .param_maps import LIKELIHOOD_MAP



def get_vae(
		variational_strategy='gaussian_poe',
		likelihood='bernoulli',
		unstructured_encoder=False,
		**kwargs,
):
	"""
	Main method: return VAE parts for this dataset.

	Returns
	-------
	vae : dict
		Maps keys 'encoder', 'decoder', and 'likelihood' to torch.nn.Modules.
	"""
	vae = {}
	if unstructured_encoder:
		vae['encoder'] = make_unstructured_encoder(
				variational_strategy=variational_strategy,
				**kwargs,
		)
	else:
		vae['encoder'] = NetworkList(
			nn.ModuleList([
				make_single_encoder(
						variational_strategy=variational_strategy,
						**kwargs,
				),
				make_single_encoder(
						variational_strategy=variational_strategy,
						**kwargs,
				),
			]),
		)
	vae['decoder'] = make_decoder(
			variational_strategy=variational_strategy,
			**kwargs,
	)
	vae['likelihood'] = likelihood_helper(
			likelihood=likelihood,
			**kwargs,
	)
	return vae


def make_single_encoder(variational_strategy='gaussian_poe', latent_dim=20,
	vmf_dim=4, n_vmfs=5, theta_dim=4, **kwargs):
	if variational_strategy in ['gaussian_poe', 'gaussian_moe']:
		out_dims = (latent_dim, latent_dim)
	elif variational_strategy == 'vmf_poe':
		out_dims = ((vmf_dim + 1) * n_vmfs,)
	elif variational_strategy == 'loc_scale_ebm':
		out_dims = (theta_dim,latent_dim,latent_dim)
	else:
		err_str = f"{variational_strategy} not implemented for " + \
				f"mnist_halves_model!"
		raise NotImplementedError(err_str)
	return nn.Sequential(
		nn.Linear(784//2,500),
		nn.ReLU(),
		nn.Linear(500,500),
		nn.ReLU(),
		nn.Linear(500,200),
		nn.ReLU(),
		SplitLinearLayer(200, out_dims),
	)


def make_unstructured_encoder(variational_strategy='gaussian_poe', \
	latent_dim=20, vmf_dim=4, n_vmfs=5, theta_dim=4, **kwargs):
	""" """
	if variational_strategy in ['gaussian_poe', 'gaussian_moe']:
		out_dims = (latent_dim, latent_dim)
	elif variational_strategy == 'vmf_poe':
		out_dims = ((vmf_dim + 1) * n_vmfs,)
	elif variational_strategy == 'loc_scale_ebm':
		out_dims = (theta_dim,latent_dim,latent_dim)
	else:
		err_str = f"{variational_strategy} not implemented for " + \
				f"mnist_halves_model!"
		raise NotImplementedError(err_str)
	return nn.Sequential(
		ConcatLayer(),
		nn.Linear(784,500),
		nn.ReLU(),
		nn.Linear(500,500),
		nn.ReLU(),
		nn.Linear(500,200),
		nn.ReLU(),
		SplitLinearLayer(200, out_dims),
		GatherLayer(transpose=False),
	)


def make_decoder(variational_strategy='gaussian_poe', latent_dim=20, \
	vmf_dim=4, n_vmfs=5, **kwargs):
	if variational_strategy in ['gaussian_poe', 'gaussian_moe','loc_scale_ebm']:
		z_dim = latent_dim
	elif variational_strategy == 'vmf_poe':
		z_dim = (vmf_dim + 1) * n_vmfs
	else:
		err_str = f"{variational_strategy} not implemented for " + \
				f"mnist_halves_model!"
		raise NotImplementedError(err_str)
	return nn.Sequential(
		nn.Linear(z_dim,200),
		nn.ReLU(),
		nn.Linear(200,500),
		nn.ReLU(),
		nn.Linear(500,500),
		nn.ReLU(),
		UnsqueezeLayer(dim=-2),
		SplitLinearLayer(500,(784//2,784//2)),
		GatherLayer(),
	)


def likelihood_helper(likelihood='spherical_gaussian', **kwargs):
	return GroupedLikelihood([
		LIKELIHOOD_MAP[likelihood](**kwargs) for _ in range(2)
	])



if __name__ == '__main__':
	pass



###
