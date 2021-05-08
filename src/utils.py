"""
Useful functions and classes.

Contains
--------
* Logger: class
* make_datasets: function
* make_dataloaders: function
* make_vae: function
* make_objective: function
* make_encoder: function
* make_decoder: function
* make_likelihood: function
* check_args: function
* hash_json_str: function
* generate: function
* reconstruct: function
* get_nan_mask: function

"""
__date__ = "January 2021"


import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
from zlib import adler32

from .param_maps import DATASET_MAP, VAR_STRATEGY_MAP, VAR_POSTERIOR_MAP, \
		PRIOR_MAP, LIKELIHOOD_MAP, OBJECTIVE_MAP, MODEL_MAP
# from .encoders_decoders import SplitLinearLayer, NetworkList, \
# 		EncoderModalityEmbedding, DecoderModalityEmbedding


DIR_LEN = 8 # for naming the logging directory
# Keys to ignore when hashing JSON strings.
IGNORED_KEYS = [
		'pre_trained',
		'epochs',
		'data_dir',
		'mll_freq',
		'test_freq',
		'no_cuda',
]



class Logger(object):
	"""
	Logger object for copying stdout to a file.

	Copied from: https://stackoverflow.com/questions/14906764/
	"""

	def __init__(self, filename, mode='a'):
		self.terminal = sys.stdout
		self.log = open(filename, mode)

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		pass



def make_dataloaders(device, dataset='mnist_halves', train_m=0.5, test_m=0.0, \
	batch_size=256, data_dir='/data', **kwargs):
	"""
	Make train and test DataLoaders.

	Parameters
	----------
	dataset : str, optional
	train_m : float, optional
	test_m : float, optional
	batch_size : str, optional
	data_dir : str, optional

	Returns
	-------
	dataloaders : dict
		Maps the keys 'train' and 'test' to respective DataLoaders.
	"""
	datasets = {
		'train': DATASET_MAP[dataset](
				device,
				missingness=train_m,
				data_dir=data_dir,
				train=True,
		),
		'test': DATASET_MAP[dataset](
				device,
				missingness=test_m,
				data_dir=data_dir,
				train=False,
		),
	}
	dataloaders = {}
	for key in datasets:
		dset = datasets[key]
		dataloaders[key] = DataLoader(dset, batch_size=batch_size, \
				shuffle=(key == 'train'))
	return dataloaders


def make_objective(dataset='mnist_halves', variational_strategy='gaussian_poe',\
	variational_posterior='diag_gaussian', prior='standard_gaussian', \
	likelihood='spherical_gaussian', objective='elbo', **kwargs):
	"""
	NOTE: HERE!

	Parameters
	----------

	"""
	# Assemble the VAE.
	vae = MODEL_MAP[dataset](**kwargs)
	assert 'encoder' in vae, f"{MODEL_MAP[dataset]} must have encoder!"
	assert 'decoder' in vae, f"{MODEL_MAP[dataset]} must have decoder!"
	# Fill in the missing pieces.
	keys = [
		'variational_strategy',
		'variational_posterior',
		'prior',
		'likelihood',
	]
	values = [
		variational_strategy,
		variational_posterior,
		prior,
		likelihood,
	]
	maps = [
		VAR_STRATEGY_MAP,
		VAR_POSTERIOR_MAP,
		PRIOR_MAP,
		LIKELIHOOD_MAP,
	]
	for key, value, map in zip(keys, values, maps):
		if key not in vae:
			vae[key] = map[value](**kwargs)
	# Transform to ModuleDict.
	vae = torch.nn.ModuleDict(vae)
	# Then pass that to the objective.
	objective = OBJECTIVE_MAP[objective](vae, **kwargs)
	return objective


def check_args(
		variational_strategy='gaussian_poe',
		variational_posterior='diag_gaussian',
		prior='standard_gaussian',
		likelihood='spherical_gaussian',
		objective='elbo',
		latent_dim=20,
		vmf_dim=4,
		n_vmfs=5,
		**kwargs,
):
	"""
	Check the arguments.

	Parameters
	----------
	TO DO: finish this
	"""
	# Check von Mises-Fisher-related issues.
	if variational_strategy == 'vmf_poe':
		z_dim = (vmf_dim + 1) * n_vmfs
		assert z_dim == latent_dim, "When using vMF distributions, " + \
				"`latent_dim` should be `(vmf_dim + 1) * n_vmfs`."
		assert variational_posterior == 'vmf_product', \
				"Only the variational posterior 'vmf_product' can be used with"\
				+ " the variational strategy 'vmf_poe'."
	elif variational_strategy == 'gaussian_poe':
		assert variational_posterior == 'diag_gaussian', "When using a " + \
				"Gaussian product of experts, the variational posterior " + \
				"should be diagonal Gaussian."
	elif variational_strategy == 'gaussian_moe':
		raise NotImplementedError
	elif variational_strategy == 'loc_scale_ebm':
		raise NotImplementedError
	else:
		raise NotImplementedError


def hash_json_str(json_str):
	"""Hash the JSON string to get a logging directory name."""
	json_str = [line for line in json_str.split('\n') if \
			not any(ignore_str in line for ignore_str in IGNORED_KEYS)]
	json_str = '\n'.join(json_str)
	exp_dir = str(adler32(str.encode(json_str))).zfill(DIR_LEN)[-DIR_LEN:]
	return exp_dir


def generate(vae, n_samples=9, decoder_noise=False):
	"""
	Generate data with a VAE.

	Parameters
	----------
	vae :
	n_samples : int
	decoder_noise : bool

	Returns
	-------
	generated : numpy.ndarray
	"""
	vae.eval()
	with torch.no_grad():
		z_samples = vae.prior.rsample(n_samples=n_samples) # [1,n,z]
		like_params = vae.decoder(z_samples) # [m][param_num][1,n,z]
		if decoder_noise:
			assert hasattr(vae.likelihood, 'rsample')
			generated = vae.likelihood.rsample(like_params, n_samples=n_samples)
		else:
			assert hasattr(vae.likelihood, 'mean')
			generated = vae.likelihood.mean(like_params, n_samples=n_samples)
	return np.array([g.detach().cpu().numpy() for g in generated])


def reconstruct(vae, data, decoder_noise=False):
	"""
	Reconstruct data with a VAE.

	Parameters
	----------

	Returns
	-------

	"""
	vae.eval()
	with torch.no_grad():
		nan_mask = get_nan_mask(data)
		for i in range(len(data)):
			data[i][nan_mask[i]] = 0.0
		var_dist_params = vae.encoder(data)
		var_post_params = vae.variational_strategy(*var_dist_params, \
				nan_mask=nan_mask)
		z_samples, _ = vae.variational_posterior(*var_post_params, \
				transpose=False)
		like_params = vae.decoder(z_samples)
		if decoder_noise:
			assert hasattr(vae.likelihood, 'rsample')
			generated = vae.likelihood.rsample(like_params, n_samples=1)
		else:
			assert hasattr(vae.likelihood, 'mean')
			generated = vae.likelihood.mean(like_params, n_samples=1)
	return np.array([g.detach().cpu().numpy() for g in generated])


def get_nan_mask(xs):
	"""Return a mask indicating which minibatch items are NaNs."""
	if type(xs) == type([]):
		return [torch.isnan(x[:,0]) for x in xs]
	else:
		return torch.isnan(xs[:,:,0]) # [b,m]



if __name__ == '__main__':
	pass



###
