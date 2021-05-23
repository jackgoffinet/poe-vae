"""
Useful functions and classes.

TO DO
-----
* detect hash collisions!
* update check_args

Contains
--------
* Logger: class
* make_dataloaders: function
* make_objective: function
* check_args: function
* hash_json_str: function

"""
__date__ = "January - May 2021"


import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
from zlib import adler32

from .param_maps import DATASET_MAP, VAR_STRATEGY_MAP, VAR_POSTERIOR_MAP, \
		PRIOR_MAP, LIKELIHOOD_MAP, OBJECTIVE_MAP, MODEL_MAP


DIR_LEN = 8 # for naming the logging directory
# Keys to ignore when hashing JSON strings.
IGNORED_KEYS = [
		'pre_trained',
		'epochs',
		'data_dir',
		'mll_freq',
		'test_freq',
		'no_cuda',
		'save_model',
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



def make_dataloaders(device, dataset='mnist_halves', train_m=0.5, valid_m=0.0, \
	test_m=0.0, batch_size=256, data_dir='/data', **kwargs):
	"""
	Make train, valid, and test DataLoaders.

	Parameters
	----------
	dataset : str, optional
	train_m : float, optional
	valid_m : float, optional
	test_m : float, optional
	batch_size : str, optional
	data_dir : str, optional

	Returns
	-------
	dataloaders : dict
		Maps the keys 'train', 'valid', and 'test' to respective DataLoaders.
	"""
	datasets = {
		'train': DATASET_MAP[dataset](
				device,
				missingness=train_m,
				data_dir=data_dir,
				mode='train',
		),
		'valid': DATASET_MAP[dataset](
				device,
				missingness=valid_m,
				data_dir=data_dir,
				mode='valid',
		),
		'test': DATASET_MAP[dataset](
				device,
				missingness=test_m,
				data_dir=data_dir,
				mode='test',
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
	Make an objective.

	Parameters
	----------
	dataset :
	variational_strategy :
	variational_posterior :
	prior :
	likelihood :
	objective :
	"""
	# Assemble the VAE.
	vae = MODEL_MAP[dataset](
		variational_strategy=variational_strategy,
		variational_posterior=variational_posterior,
		prior=prior,
		likelihood=likelihood,
		**kwargs,
	)
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
	objective = OBJECTIVE_MAP[objective](vae, dataset=dataset, **kwargs)
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

	TO DO: some strategies assume standard normal priors.

	Parameters
	----------
	variational_strategy : str, optional
	variational_posterior : str, optional
	prior : str, optional
	likelihood : str, optional
	objective : str, optional
	latent_dim : int, optional
	vmf_dim : int, optional
	n_vmfs : int, optional
	"""
	# Make sure we recognize the inputs.
	assert variational_strategy in VAR_STRATEGY_MAP.keys(), \
			f"Invalid variational strategy {variational_strategy}! " + \
			f"Choose one of: {list(VAR_STRATEGY_MAP.keys())}"
	assert variational_posterior in VAR_POSTERIOR_MAP.keys(), \
			f"Invalid variational posterior {variational_posterior}! " + \
			f"Choose one of: {list(VAR_POSTERIOR_MAP.keys())}"
	assert prior in PRIOR_MAP.keys(), \
			f"Invalid prior {prior}! Choose one of: {list(PRIOR_MAP.keys())}"
	assert likelihood in LIKELIHOOD_MAP.keys(), \
			"Invalid likelihood {likelihood}! " + \
			f"Choose one of: {list(LIKELIHOOD_MAP.keys())}"
	assert objective in OBJECTIVE_MAP.keys(), \
			f"Invalid objective {objective}! " + \
			f"Choose one of: {list(OBJECTIVE_MAP.keys())}"
	# Check variational strategy-related issues.
	if variational_strategy == 'vmf_poe':
		z_dim = vmf_dim * n_vmfs
		assert z_dim == latent_dim, \
				"When using vMF distributions, latent_dim should be " \
				+ "vmf_dim * n_vmfs."
		assert variational_posterior == 'vmf_product', \
				"Only the variational posterior 'vmf_product' can be used with"\
				+ " the variational strategy 'vmf_poe'."
		assert prior == 'uniform_hyperspherical', \
				"Only the prior 'uniform_hyperspherical' can be used with the"\
				+ " variational strategy 'vmf_poe'."
	elif variational_strategy == 'gaussian_poe':
		assert variational_posterior == 'diag_gaussian', \
				"When using a Gaussian product of experts, the variational " \
				+ "posterior must be diagonal Gaussian."
	elif variational_strategy == 'gaussian_moe':
		assert variational_posterior == 'diag_gaussian_mixture', \
				"When using a Gaussian mixture of experts, the variational " \
				+ "posterior must be a diagonal Gaussian mixture."
		assert objective in ['iwae', 'dreg_iwae', 'mmvae_elbo'], \
				"The Gaussian mixture of experts strategy can only be used " \
				"with objectives that don't require analytic KL-divergence. " \
				"Choose from ['iwae', 'dreg_iwae', 'mmvae_elbo']."
	elif variational_strategy == 'loc_scale_ebm':
		assert variational_posterior == 'loc_scale_ebm', \
				"Only the variational posterior 'loc_scale_ebm' can be used" \
				+ " with the variational_strategy 'loc_scale_ebm'."
		assert objective in ['iwae', 'dreg_iwae', 'ar_elbo', 'mvae_elbo'], \
				"Only the IWAE, DReG IWAE, MVAE, and AR-ELBO objectives can " \
				+ "be used with a Location/Scale EBM variational strategy."
	# Check objective-related issues.
	if objective == 'mmvae_elbo':
		assert variational_posterior == 'diag_gaussian_mixture', \
				"Only posteriors that support stratified sampling can be " \
				+ "used with the MMVAE ELBO. Right now, that is just " \
				+ "'diag_gaussian_mixture'."
	elif objective == 'ar_elbo':
		assert variational_posterior in ['diag_gaussian', 'loc_scale_ebm'], \
				"The AR-ELBO is only compatible with variational posteriors " \
				+ "that factorize and have within-family KL divergences " \
				+ "implemented. Right now, this is just 'diag_gaussian' and " \
				+ "'loc_scale_ebm'."


def hash_json_str(json_str):
	"""Hash the JSON string to get a logging directory name."""
	json_str = [line for line in json_str.split('\n') if \
			not any(ignore_str in line for ignore_str in IGNORED_KEYS)]
	json_str = '\n'.join(json_str)
	exp_dir = str(adler32(str.encode(json_str))).zfill(DIR_LEN)[-DIR_LEN:]
	return exp_dir



if __name__ == '__main__':
	pass



###
