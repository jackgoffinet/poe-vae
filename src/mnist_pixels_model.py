"""
Binarized MNIST pixels networks.

"""
__date__ = "May 2021"


import torch
import torch.nn as nn

from .encoders_decoders import SplitLinearLayer, NetworkList, GatherLayer, \
		EncoderModalityEmbedding, DecoderModalityEmbedding, SqueezeLayer
from .likelihoods import GroupedLikelihood
from .param_maps import LIKELIHOOD_MAP



def get_vae(
		variational_strategy='gaussian_poe',
		variational_posterior='diag_gaussian',
		prior='standard_gaussian',
		latent_dim=20,
		embed_dim=8,
		**kwargs,
):
	"""
	Main method: return VAE parts for this dataset.

	Returns
	-------
	vae : dict
		Maps keys 'encoder', 'decoder', and 'likelihood' to torch.nn.Modules.
	"""
	return {
		'encoder': make_encoder(
				variational_strategy=variational_strategy,
				embed_dim=embed_dim,
				**kwargs,
		),
		'decoder': make_decoder(
				variational_strategy=variational_strategy,
				**kwargs,
		),
	}


def make_encoder(variational_strategy='gaussian_poe', latent_dim=20,
	vmf_dim=4, n_vmfs=5, theta_dim=4, embed_dim=8, **kwargs):
	if variational_strategy == 'gaussian_poe':
		out_dims = (latent_dim,latent_dim)
	elif variational_strategy == 'vmf_poe':
		out_dims = (latent_dim,)
	elif variational_strategy == 'loc_scale_ebm':
		out_dims = (theta_dim,latent_dim,latent_dim)
	else:
		err_str = f"{variational_strategy} not implemented for " + \
				f"mnist_pixels_model!"
		raise NotImplementedError(err_str)
	return nn.Sequential(
		EncoderModalityEmbedding(784, embed_dim=embed_dim),
		nn.Linear(1+embed_dim,128),
		nn.ReLU(),
		nn.Linear(128,128),
		nn.ReLU(),
		SplitLinearLayer(128, out_dims),
	)


def make_decoder(variational_strategy='gaussian_poe', latent_dim=20, \
	vmf_dim=4, n_vmfs=5, embed_dim=8, **kwargs):
	if variational_strategy in ['gaussian_poe', 'loc_scale_ebm']:
		z_dim = latent_dim
	elif variational_strategy == 'vmf_poe':
		z_dim = (vmf_dim + 1) * n_vmfs
	else:
		err_str = f"{variational_strategy} not implemented for " + \
				f"mnist_pixels_model!"
		raise NotImplementedError(err_str)
	return nn.Sequential(
		DecoderModalityEmbedding(784,embed_dim=embed_dim),
		nn.Linear(z_dim+embed_dim,128),
		nn.ReLU(),
		nn.Linear(128,128),
		nn.ReLU(),
		nn.Linear(128,1),
		GatherLayer(),
	)



if __name__ == '__main__':
	pass



###
