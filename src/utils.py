"""
Useful functions and classes.

"""
__date__ = "January 2021"


import sys
from torch.utils.data import DataLoader

from .components import DATASET_MAP, ENCODER_DECODER_MAP, \
		VARIATIONAL_STRATEGY_MAP, VARIATIONAL_POSTERIOR_MAP, PRIOR_MAP, \
		LIKELIHOOD_MAP, OBJECTIVE_MAP



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



def make_datasets(args):
	"""
	...
	"""



def make_dataloaders(datasets, batch_size):
	"""
	...
	"""
	res = {} # Maps keys ('train', 'test') to DataLoaders.
	for key in datasets:
		dset = dset[key]
		res[key] = DataLoader(dset, batch_size=batch_size, shuffle=dset.shuffle)
	return res


def make_vae(args):
	"""
	...

	"""
	pass


def check_args(args):
	"""
	Check the arguments, replace names with objects.

	Parameters
	----------
	args : ...

	Returns
	-------
	args : ...
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
	# TO DO!
	pass



if __name__ == '__main__':
	pass



###
