"""
Main script: Train a model.

Notes
-----
* Loading a model and continuing to train breaks the random seeds. Train in one
  go for reproducability.
"""
__date__ = "January - May 2021"


import datetime
from collections import defaultdict
import fire
import json
import numpy as np
import os
import sys
from time import perf_counter
import torch

from src.misc import LOGGING_DIR, ARGS_FN, LOG_FN, STATE_FN, AGG_FN, INDENT
from src.utils import Logger, make_dataloaders, check_args, make_objective, \
		hash_json_str, generate, reconstruct



def get_grad_norm(obj):
	return sum(p.grad.data.norm(2).item()**2 for p in obj.parameters()) ** 0.5


def train_epoch(objective, loader, optimizer, epoch, agg):
	"""
	Train for a single epoch.

	Parameters
	----------
	objective :
	loader :
	optimizer :
	epoch :
	agg :
	"""
	objective.train()
	b_loss = 0
	for i, batch in enumerate(loader):
		optimizer.zero_grad()
		loss = objective(batch)
		if torch.isnan(loss):
			print("NaN Loss!")
			quit()
		loss.backward()
		# torch.nn.utils.clip_grad_norm_(objective.parameters(), args.clip)
		optimizer.step()
		b_loss += loss.item() * get_batch_len(batch)
	agg['train_loss'].append(b_loss / len(loader.dataset))
	agg['train_epoch'].append(epoch)
	print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))


def test_epoch(objective, loader, epoch, agg):
	"""
	Test on the full test set.

	Parameters
	----------
	objective :
	loader :
	epoch :
	agg :
	"""
	objective.eval()
	b_loss = 0
	for i, batch in enumerate(loader):
		with torch.no_grad():
			loss = objective(batch)
		b_loss += loss.item() * get_batch_len(batch)
	agg['test_loss'].append(b_loss / len(loader.dataset))
	agg['test_epoch'].append(epoch)
	print('====> Epoch: {:03d} Test loss: {:.4f}'.format(epoch, agg['test_loss'][-1]))


def save_state(objective, optimizer, epoch):
	"""Save state."""
	print("Saving state to:", state_fn)
	torch.save({
			'objective_state_dict': objective.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'epoch': epoch,
		}, state_fn)


def load_state(objective, optimizer, device):
	"""Load state."""
	print("Loading state from:", state_fn)
	checkpoint = torch.load(state_fn, map_location=device)
	objective.load_state_dict(checkpoint['objective_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	return checkpoint['epoch']


def get_batch_len(batch):
	"""Return the number of data items in a batch."""
	if type(batch) == type([]): # non-vectorized modalities
		return len(batch[0])
	return len(batch) # vectorized modalities


def estimate_marginal_log_like(objective, loader, k=2000, mini_k=128, \
	reduction='mean'):
	"""
	Simple log marginal estimation.

	Take the approximate posterior as a proposal distribution, do an
	importance-weighted estimate.

	TO DO: try/except blocks with decreasing mini_k

	Parameters
	----------
	objective :
	loader :
	k : int
	mini_k : int
	reduction : {'mean', 'sum'}

	Returns
	-------
	est_mll : float
	"""
	assert reduction in ['sum', 'mean']
	batch_res = []
	with torch.no_grad():
		for i, batch in enumerate(loader):
			inner_batch_res = []
			j = 0
			while j < k:
				temp_mini_k = min(k-j, mini_k)
				try:
					log_l = objective.estimate_marginal_log_like(batch, \
							n_samples=temp_mini_k, keepdim=True)
					j += temp_mini_k
					inner_batch_res.append(log_l)
				except RuntimeError: # CUDA out of memory
					if mini_k == 1:
						print("MLL failed!")
						quit()
					mini_k //= 2
			log_m = torch.cat(inner_batch_res, dim=1) - np.log(k)
			log_m = torch.logsumexp(log_m, dim=1)
			batch_res.append(log_m)
		batch_res = torch.cat(batch_res, dim=0).detach().cpu().numpy()
	assert len(batch_res.shape) == 1 # dataset size
	if reduction == 'sum':
		return np.sum(batch_res)
	return np.mean(batch_res)


def mll_helper(objective, dataloaders, epoch, agg, train_mll=False):
	"""Estimate marginal log likelihoods."""
	tic = perf_counter()
	mll = estimate_marginal_log_like(objective, dataloaders['test'])
	toc = perf_counter()
	agg['test_mll'].append(mll)
	agg['test_mll_epoch'].append(epoch)
	print("Test MLL: ", mll, ", time:", round(toc-tic,2))
	if train_mll:
		tic = perf_counter()
		mll = estimate_marginal_log_like(objective, dataloaders['train'])
		toc = perf_counter()
		agg['train_mll'].append(mll)
		agg['train_mll_epoch'].append(epoch)
		print("Train MLL time:", mll, ", time:", round(toc-tic,2))


def save_aggregator(agg):
	"""Save the data aggregrator."""
	torch.save(agg, agg_fn)



def main(
		dataset='mnist_halves',
		variational_strategy='gaussian_poe',
		variational_posterior='diag_gaussian',
		prior='standard_gaussian',
		likelihood='spherical_gaussian',
		objective='elbo',
		lr=1e-3,
		K=10,
		ebm_samples=10,
		batch_size=256,
		epochs=10,
		latent_dim=20,
		m_dim=4,
		vmf_dim=4,
		n_vmfs=5,
		theta_dim=4,
		obs_std_dev=0.1,
		pre_trained=False,
		mll_freq=1000,
		test_freq=10,
		no_cuda=False,
		seed=42,
		grad_clip=1e3,
		train_m=0.5,
		test_m=0.0,
		data_dir='/media/jackg/Jacks_Animal_Sounds/torchvision/',
):
	"""
	Main function: train a model.

	Parameters
	----------
	dataset : str, optional
	encoder : str, optional
	variational_strategy : str, optional
	...
	"""
	# Check the arguments.
	args = locals()
	check_args(**args)
	# Parse and print args.
	args_json_str = json.dumps(args, sort_keys=True, indent=INDENT)
	# Hash the JSON string to make a logging directory.
	exp_dir = hash_json_str(args_json_str)
	print("Experiment directory:", exp_dir)
	# Make various filenames.
	exp_dir = os.path.join(LOGGING_DIR, exp_dir)
	log_fn = os.path.join(exp_dir, LOG_FN)
	state_fn = os.path.join(exp_dir, STATE_FN)
	agg_fn = os.path.join(exp_dir, AGG_FN)
	agg = None
	# See if we've already started this experiment.
	if pre_trained:
		assert os.path.exists(exp_dir)
		try:
			agg = torch.load(agg_fn)
			print(f"Loaded agg from {agg_fn}")
		except FileNotFoundError:
			print(f"Couldn't find {agg_fn} to load")
			return
	else:
		if os.path.exists(exp_dir):
			_ = input("Experiment path already exists! Continue? ")
			try:
				os.remove(log_fn)
			except FileNotFoundError:
				print(f"Couldn't find {log_fn} to remove")
				return
		else:
			os.makedirs(exp_dir)
	# Write the parameters to a JSON file.
	args_fn = os.path.join(exp_dir, ARGS_FN)
	with open(args_fn, 'w') as fp:
		json.dump(args, fp, sort_keys=True, indent=4)
	# Set up a Logger object to log stdout.
	sys.stdout = Logger(log_fn)
	print(args_json_str)
	print(datetime.datetime.now().isoformat())
	# Set a random seed.
	torch.backends.cudnn.benchmark = True
	torch.manual_seed(seed)
	np.random.seed(seed)
	# Set up CUDA.
	cuda = not no_cuda and torch.cuda.is_available()
	device = torch.device('cuda' if cuda else 'cpu')
	# Make Dataloaders.
	dataloaders = make_dataloaders(device, **args)
	# Make the objective.
	objective = make_objective(**args).to(device)
	# Make the optimizer.
	optimizer = torch.optim.Adam(objective.parameters(), lr=lr)
	# Load pretrained models.
	if pre_trained:
		prev_epochs = load_state(objective, optimizer, device)
	else:
		prev_epochs = 0
	# Set up a data aggregrator.
	if agg is None:
		agg = defaultdict(list)
	# Enter a train loop.
	for epoch in range(prev_epochs+1, prev_epochs+epochs+ 1):
		train_epoch(objective, dataloaders['train'], optimizer, epoch, agg)
		if epoch % test_freq == 0:
			test_epoch(objective, dataloaders['test'], epoch, agg)
		if epoch % mll_freq == 0 or epoch == prev_epochs + epochs:
			mll_helper(objective, dataloaders, epoch, agg)
	epoch = prev_epochs + epochs
	# Save the aggregrator.
	save_aggregator(agg)
	# Plot reconstructions and generations.
	datasets['train'].make_plots(model, datasets, dataloaders, exp_dir)
	# Save the model/objective.
	save_state(objective, optimizer, epoch)



if __name__ == '__main__':
	fire.Fire(main)


###
