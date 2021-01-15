"""
Main script: Train a model.

"""
__date__ = "January 2021"


import argparse
import datetime
from collections import defaultdict
import json
import numpy as np
import os
import sys
import torch
from zlib import adler32 # hash function

from src.utils import Logger, make_vae, make_datasets, make_dataloaders, \
		check_args
from src.components import DATASET_KEYS, ENCODER_DECODER_KEYS, \
		VARIATIONAL_STRATEGY_KEYS, VARIATIONAL_POSTERIOR_KEYS, PRIOR_KEYS, \
		LIKELIHOOD_KEYS, OBJECTIVE_KEYS


LOGGING_DIR = 'logs'
ARGS_FN = 'args.json'
LOG_FN = 'run.log'

INDENT = 4 # for pretty JSON
DIR_LEN = 8 # for naming the logging directory


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
					help='latent dimensionality (default: 20)')
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H',
					help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--pre-trained', type=str, default="",
					help='path to pre-trained model (train from scratch if empty)')
parser.add_argument('--learn-prior', action='store_true', default=False,
					help='learn model prior parameters')
parser.add_argument('--logp', action='store_true', default=False,
					help='estimate marginal likelihood on completion')
parser.add_argument('--print-freq', type=int, default=0, metavar='f',
					help='frequency with which to print stats (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')


# Parse and print args.
args = parser.parse_args()
args_json_str = json.dumps(args.__dict__, sort_keys=True, indent=4)
print(args_json_str)


# TO DO: make sure args are compatible!
print(type(args.prior))
check_args(args)
print(type(args.prior))
quit()


# Hash the JSON string to make a logging directory.
exp_dir = str(adler32(str.encode(args_json_str))).zfill(DIR_LEN)[-DIR_LEN:]
exp_dir = os.path.join(LOGGING_DIR, exp_dir)
log_fn = os.path.join(exp_dir, LOG_FN)
if os.path.exists(exp_dir):
	_ = input("Experiment path already exists! Continue?")
	try:
		os.remove(log_fn)
	except FileNotFoundError:
		pass
else:
	os.makedirs(exp_dir)
args_fn = os.path.join(exp_dir, ARGS_FN)
with open(args_fn, 'w') as fp:
	json.dump(args.__dict__, fp, sort_keys=True, indent=4)


# Set up a Logger object to log stdout and record the time.
sys.stdout = Logger(log_fn)
print(datetime.datetime.now().isoformat())


# Set a random seed.
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Set up CUDA.
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


# TO DO!!!!
# # load args from disk if pretrained model path is given
# pretrained_path = ""
# if args.pre_trained:
# 	pretrained_path = args.pre_trained
# 	args = torch.load(args.pre_trained + '/args.rar')


# # load model
# modelC = getattr(models, 'VAE_{}'.format(args.model))
# model = modelC(args).to(device)


# if pretrained_path:
# 	print('Loading model {} from {}'.format(model.modelName, pretrained_path))
# 	model.load_state_dict(torch.load(pretrained_path + '/model.rar'))
# 	model._pz_params = model._pz_params

# # ???
# if not args.experiment:
# 	args.experiment = model.modelName


# # -- also save object because we want to recover these for other things
# torch.save(args, '{}/args.rar'.format(runPath))


# # preparation for training
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
# 					   lr=1e-3, amsgrad=True)
# train_loader, test_loader = model.getDataLoaders(args.batch_size, device=device)
# objective = getattr(objectives,
# 					('m_' if hasattr(model, 'vaes') else '')
# 					+ args.obj
# 					+ ('_looser' if (args.looser and args.obj != 'elbo') else ''))
# t_objective = getattr(objectives, ('m_' if hasattr(model, 'vaes') else '') + 'iwae')


def train(epoch, agg):
	model.train()
	b_loss = 0
	for i, dataT in enumerate(train_loader):
		data = unpack_data(dataT, device=device)
		optimizer.zero_grad()
		loss = -objective(model, data, K=args.K)
		loss.backward()
		optimizer.step()
		b_loss += loss.item()
		if args.print_freq > 0 and i % args.print_freq == 0:
			print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))
	agg['train_loss'].append(b_loss / len(train_loader.dataset))
	print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))


def test(epoch, agg):
	model.eval()
	b_loss = 0
	with torch.no_grad():
		for i, dataT in enumerate(test_loader):
			data = unpack_data(dataT, device=device)
			loss = -t_objective(model, data, K=args.K)
			b_loss += loss.item()
			if i == 0:
				model.reconstruct(data, runPath, epoch)
				if not args.no_analytics:
					model.analyse(data, runPath, epoch)
	agg['test_loss'].append(b_loss / len(test_loader.dataset))
	print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))


def estimate_log_marginal(K):
	"""Compute an IWAE estimate of the log-marginal likelihood of test data."""
	model.eval()
	marginal_loglik = 0
	with torch.no_grad():
		for dataT in test_loader:
			data = unpack_data(dataT, device=device)
			marginal_loglik += -t_objective(model, data, K).item()

	marginal_loglik /= len(test_loader.dataset)
	print('Marginal Log Likelihood (IWAE, K = {}): {:.4f}'.format(K, marginal_loglik))


if __name__ == '__main__':
	# Make Datasets.
	datasets = make_datasets(args)
	# Make Dataloaders.
	dataloaders = make_dataloaders(datasets, args.batch_size)
	# Make VAE.
	model = make_vae(args)
	# Set up a data aggregrator.
	agg = defaultdict(list)
	# Enter a train loop.
	for epoch in range(1, args.epochs + 1):
		train(epoch, agg)
		test(epoch, agg)
		save_model(model, runPath + '/model.rar')
		save_vars(agg, runPath + '/losses.rar')
		model.generate(runPath, epoch)
	if args.logp:  # compute as tight a marginal likelihood as possible
		estimate_log_marginal(5000)


###
