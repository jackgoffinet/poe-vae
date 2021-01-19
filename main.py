"""
Main script: Train a model.

Notes
-----
* Loading a model and continuing to train breaks the random seeds. Train in one
  go for reproducability.
"""
__date__ = "January 2021"


import argparse
import datetime
from collections import defaultdict
import json
import numpy as np
import os
import sys
from time import perf_counter
import torch

from src.utils import Logger, make_vae, make_datasets, make_dataloaders, \
		check_args, make_objective, hash_json_str, generate, reconstruct
from src.components import DATASET_KEYS, ENCODER_DECODER_KEYS, \
		VARIATIONAL_STRATEGY_KEYS, VARIATIONAL_POSTERIOR_KEYS, PRIOR_KEYS, \
		LIKELIHOOD_KEYS, OBJECTIVE_KEYS


LOGGING_DIR = 'logs'
ARGS_FN = 'args.json'
LOG_FN = 'run.log'
STATE_FN = 'state.tar'
GENERATE_FN = 'generations.pdf'
TRAIN_RECONSTRUCT_FN = 'train_reconstructions.pdf'
TEST_RECONSTRUCT_FN = 'test_reconstructions.pdf'

INDENT = 4 # for pretty JSON



# Handle the arguments.
parser = argparse.ArgumentParser(description='Multimodal VAEs')

# VAE components.
parser.add_argument('-x', '--dataset', type=str, default='mnist_halves',
					choices=DATASET_KEYS,
					help='experiment dataset name')
parser.add_argument('-e', '--encoder', type=str, default='mlp',
					choices=ENCODER_DECODER_KEYS,
					help='encoder name')
parser.add_argument('-v', '--variational-strategy', type=str,
					default='gaussian_poe',
					choices=VARIATIONAL_STRATEGY_KEYS,
					help='variational strategy name')
parser.add_argument('-q', '--variational-posterior', type=str,
					default='diag_gaussian',
					choices=VARIATIONAL_POSTERIOR_KEYS,
					help='variational posterior name')
parser.add_argument('-p', '--prior', type=str, default='standard_gaussian',
					choices=PRIOR_KEYS,
					help='prior name')
parser.add_argument('-d', '--decoder', type=str, default='mlp',
					choices=ENCODER_DECODER_KEYS,
					help='decoder name')
parser.add_argument('-l', '--likelihood', type=str,
					default='spherical_gaussian',
					choices=LIKELIHOOD_KEYS,
					help='likelihood name')
parser.add_argument('-o', '--objective', type=str, default='elbo',
					choices=OBJECTIVE_KEYS,
					help='objective name')

# Other arguments.
parser.add_argument('--K', type=int, default=10, metavar='K',
					help='number of particles to use for IWAE (default: 10)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
					help='batch size for data (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
					help='number of epochs to train (default: 10)')
parser.add_argument('--latent-dim', type=int, default=20, metavar='L',
					help='latent dimension (default: 20)')
parser.add_argument('--m-dim', type=int, default=8, metavar='K',
					help='modality embedding dimension (default: 8)')
parser.add_argument('--obs-std-dev', type=float, default=0.1, metavar='L',
					help='Observation standard deviation (default: 0.1)')
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H',
					help='number of hidden layers (default: 1)')
parser.add_argument('--hidden-layer-dim', type=int, default=64, metavar='H',
					help='hidden layer dimension (default: 64)')
parser.add_argument('--pre-trained', action='store_true', default=False,
					help='load a saved model')
parser.add_argument('--logp', action='store_true', default=False,
					help='estimate marginal likelihood on completion')
parser.add_argument('--print-freq', type=int, default=0, metavar='f',
					help='frequency with which to print stats (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--data-fn', type=str, default='',
					help='data filename')


# Parse and print args.
args = parser.parse_args()
args_json_str = json.dumps(args.__dict__, sort_keys=True, indent=4)


# Hash the JSON string to make a logging directory.
exp_dir = hash_json_str(args_json_str)


# Make various filenames.
exp_dir = os.path.join(LOGGING_DIR, exp_dir)
log_fn = os.path.join(exp_dir, LOG_FN)
state_fn = os.path.join(exp_dir, STATE_FN)
generate_fn = os.path.join(exp_dir, GENERATE_FN)
train_reconstruct_fn = os.path.join(exp_dir, TRAIN_RECONSTRUCT_FN)
test_reconstruct_fn = os.path.join(exp_dir, TEST_RECONSTRUCT_FN)


# See if we've already started this experiment.
if args.pre_trained:
	assert os.path.exists(exp_dir)
else:
	if os.path.exists(exp_dir):
		_ = input("Experiment path already exists! Continue? ")
		try:
			os.remove(log_fn)
		except FileNotFoundError:
			pass
	else:
		os.makedirs(exp_dir)


# Write the parameters to a JSON file.
args_fn = os.path.join(exp_dir, ARGS_FN)
with open(args_fn, 'w') as fp:
	json.dump(args.__dict__, fp, sort_keys=True, indent=4)


# Check the arguments, convert component names to objects and classes.
check_args(args)


# Set up a Logger object to log stdout.
sys.stdout = Logger(log_fn)
print(args_json_str)
print(datetime.datetime.now().isoformat())


# Set a random seed.
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)


# Set up CUDA.
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device("cuda" if args.cuda else "cpu")



def train_epoch(objective, loader, optimizer, epoch, agg):
	"""
	Train for a single epoch.

	Parameters
	----------
	...
	"""
	objective.train()
	b_loss = 0
	for i, batch in enumerate(loader):
		optimizer.zero_grad()
		loss = objective(batch)
		loss.backward()
		optimizer.step()
		b_loss += loss.item() * get_batch_len(batch)
	agg['train_loss'].append(b_loss / len(loader.dataset))
	print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))


def test_epoch(objective, loader, epoch, agg):
	"""
	Test on the full test set.

	Parameters
	----------
	...
	"""
	objective.eval()
	b_loss = 0
	for i, batch in enumerate(loader):
		optimizer.zero_grad()
		loss = objective(batch)
		loss.backward()
		optimizer.step()
		b_loss += loss.item() * get_batch_len(batch)
	agg['test_loss'].append(b_loss / len(loader.dataset))
	print('====> Epoch: {:03d} Test loss: {:.4f}'.format(epoch, agg['test_loss'][-1]))


def save_state(objective, optimizer, epoch):
	"""Save state."""
	print("Saving state to:", state_fn)
	torch.save({
			'objective_state_dict': objective.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'epoch': epoch,
		}, state_fn)


def load_state(objective, optimizer):
	"""Load state."""
	print("Loading state from:", state_fn)
	checkpoint = torch.load(state_fn, map_location=args.device)
	objective.load_state_dict(checkpoint['objective_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	return checkpoint['epoch']


def get_batch_len(batch):
	if type(batch) == type([]): # non-vectorized modalities
		return len(batch[0])
	return len(batch) # vectorized modalities


def make_plots(model, datasets, dataloaders):
	# Plot generated data.
	generated_data = generate(model, n_samples=5) # [m,b,s,x]
	datasets['train'].dataset.plot(generated_data, generate_fn)
	# Plot train reconstructions.
	for batch in dataloaders['train']:
		data = [b[:5] for b in batch]
		recon_data = reconstruct(model, data) # [m,b,s,x]
		break
	data = np.array([g.detach().cpu().numpy() for g in data])
	data = data.reshape(recon_data.shape) # [m,b,s,x]
	datasets['train'].dataset.plot([data,recon_data], train_reconstruct_fn)
	# Plot test reconstructions.
	for batch in dataloaders['test']:
		data = [b[:5] for b in batch]
		recon_data = reconstruct(model, data) # [m,b,s,x]
		break
	data = np.array([g.detach().cpu().numpy() for g in data])
	data = data.reshape(recon_data.shape) # [m,b,s,x]
	datasets['test'].dataset.plot([data,recon_data], test_reconstruct_fn)


def estimate_log_marginal(objective, loader, k=1000, reduction='mean'):
	"""
	Simple log marginal estimation.

	Take the approximate posterior as a proposal distribution, do an
	importance-weighted estimate.
	"""
	assert reduction in ['sum', 'mean']
	batch_res = []
	with torch.no_grad():
		for i, batch in enumerate(loader):
			log_m = objective.estimate_log_marginal(batch)
			batch_res.append(log_m)
		batch_res = torch.cat(batch_res, dim=0).detach().cpu().numpy()
	assert len(batch_res.shape) == 1 # dataset size
	if reduction == 'sum':
		return np.sum(batch_res)
	return np.mean(batch_res)




if __name__ == '__main__':
	# Make Datasets.
	datasets = make_datasets(args)
	# Make Dataloaders.
	dataloaders = make_dataloaders(datasets, args.batch_size)
	# Make the VAE.
	model = make_vae(args)
	# Make the objective.
	objective = make_objective(model, args)
	# Make the optimizer.
	optimizer = torch.optim.Adam(objective.parameters(), lr=1e-3)
	# Load pretrained models.
	if args.pre_trained:
		prev_epochs = load_state(objective, optimizer)
	else:
		prev_epochs = 0
	# Set up a data aggregrator.
	agg = defaultdict(list)
	# Enter a train loop.
	for epoch in range(prev_epochs + 1, prev_epochs + args.epochs + 1):
		train_epoch(objective, dataloaders['train'], optimizer, epoch, agg)
		test_epoch(objective, dataloaders['test'], epoch, agg)
		# NOTE: HERE!
	# if args.logp:
	tic = perf_counter()
	log_p = estimate_log_marginal(objective, dataloaders['test'])
	toc = perf_counter()
	print("test", log_p, round(toc-tic,5)*1e3)
	tic = perf_counter()
	log_p = estimate_log_marginal(objective, dataloaders['train'])
	toc = perf_counter()
	print("train", log_p, round(toc-tic,5)*1e3)
	make_plots(model, datasets, dataloaders)
	save_state(objective, optimizer, prev_epochs + args.epochs)


###
