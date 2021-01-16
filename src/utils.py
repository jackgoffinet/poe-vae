"""
Useful functions and classes.

"""
__date__ = "January 2021"


import sys
import torch
from torch.utils.data import DataLoader

from .components import DATASET_MAP, ENCODER_DECODER_MAP, \
		VARIATIONAL_STRATEGY_MAP, VARIATIONAL_POSTERIOR_MAP, PRIOR_MAP, \
		LIKELIHOOD_MAP, OBJECTIVE_MAP
from.components.encoders_decoders import SplitLinearLayer, NetworkList
from .vae import VAE



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
		# this flush method is needed for python 3 compatibility.
		# this handles the flush command by doing nothing.
		# you might want to specify some extra behavior here.
		pass



def make_datasets(args, train_ratio=0.8):
	"""
	Make train and test Datasets.

	Note: generator should already be seeded -- check this

	Parameters
	----------
	args : argparse.Namespace
	train_ratio : float

	Returns
	-------
	datasets : dict
		Maps the keys 'train' and 'test' to respective Datasets.
	"""
	big_dataset = args.dataset(args.data_fn)
	train_len = int(round(train_ratio * len(big_dataset)))
	test_len = len(big_dataset) - train_len
	dset_splits = \
			torch.utils.data.random_split(big_dataset, [train_len,test_len])
	return {'train': dset_splits[0], 'test': dset_splits[1]}


def make_dataloaders(datasets, batch_size):
	"""
	Make train and test DataLoaders.

	Parameters
	----------
	args : argparse.Namespace

	Returns
	-------
	dataloaders : dict
		Maps the keys 'train' and 'test' to respective DataLoaders.
	"""
	dataloaders = {}
	for key in datasets:
		dset = datasets[key]
		dataloaders[key] = DataLoader(dset, batch_size=batch_size, \
				shuffle=(key == 'train'))
	return dataloaders


def make_vae(args):
	"""
	Make a VAE.

	Parameters
	----------
	args : argparse.Namespace

	Returns
	-------
	model : .vae.VAE
	"""
	model = VAE( \
			make_encoder(args),
			args.variational_strategy(),
			args.variational_posterior(),
			args.prior(),
			make_decoder(args),
			args.likelihood(),
			args.objective(),
	)
	return model


def make_encoder(args):
	"""
	Make the encoder.

	Each modality gets its own encoder, all with identical architectures.

	Parameters
	----------
	args : argparse.Namespace

	Returns
	-------
	encoders : .components.encoders_decoders.NetworkList (torch.nn.Module)
	"""
	# Collect parameters.
	n_modalities = args.dataset.n_modalities
	modality_dim = args.dataset.modality_dim
	vectorized_modalities = args.dataset.vectorized_modalities
	z_dim = args.latent_dim
	output_dims = args.variational_posterior.parameter_dim_func(z_dim)
	# Make layers.
	if vectorized_modalities:
		raise NotImplementedError
	else:
		# Make everything up to the last layer.
		dims = [modality_dim] + [args.hidden_layer_dim]*args.num_hidden_layers
		encoders = [args.encoder(dims) for _ in range(n_modalities)]
		for i in range(n_modalities):
			last_layer = SplitLinearLayer(args.hidden_layer_dim, output_dims)
			encoders[i] = torch.nn.Sequential(encoders[i], last_layer)
	return NetworkList(encoders)


def check_args(args):
	"""
	Check the arguments, replacing names with objects.

	TO DO: check compatibility!

	Parameters
	----------
	args : argparse.Namespace
	"""
	# First, map VAE component names to classes.
	args.dataset = DATASET_MAP[args.dataset]
	args.encoder = ENCODER_DECODER_MAP[args.encoder]
	args.variational_strategy = \
				VARIATIONAL_STRATEGY_MAP[args.variational_strategy]
	args.variational_posterior = \
				VARIATIONAL_POSTERIOR_MAP[args.variational_posterior]
	args.prior = PRIOR_MAP[args.prior]
	args.decoder = ENCODER_DECODER_MAP[args.decoder]
	args.likelihood = LIKELIHOOD_MAP[args.likelihood]
	args.objective = OBJECTIVE_MAP[args.objective]
	# Next, make sure the components are compatible.
	# TO DO: finish this!
	pass



if __name__ == '__main__':
	pass



###
