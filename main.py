"""
Main script: Train a model.

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
		hash_json_str



def get_grad_norm(obj):
	return sum(p.grad.data.norm(2).item()**2 for p in obj.parameters()) ** 0.5


def train_epoch(objective, loader, optimizer, epoch, kl_factor, agg, grad_clip):
	"""
	Train for a single epoch.

	Parameters
	----------
	objective : src.objectives.VaeObjective
	loader : torch.utils.DataLoader
	optimizer : torch.optim.Optimizer
	epoch : int
	kl_factor : float
	agg : defaultdict
	grad_clip : float
	"""
	objective.train()
	b_loss = 0
	for i, batch in enumerate(loader):
		optimizer.zero_grad()
		loss = objective(batch, kl_factor=kl_factor)
		if torch.isnan(loss):
			quit("NaN Loss!")
		loss.backward()
		# print(get_grad_norm(objective))
		torch.nn.utils.clip_grad_norm_(objective.parameters(), grad_clip)
		optimizer.step()
		b_loss += loss.item() * get_batch_len(batch)
	agg['train_loss'].append(b_loss / len(loader.dataset))
	agg['train_epoch'].append(epoch)
	print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))


def test_epoch(objective, loader, epoch, kl_factor, agg):
	"""
	Test on the full test set.

	Parameters
	----------
	objective : src.objectives.VaeObjective
	loader : torch.utils.DataLoader
	epoch : int
	agg : defaultdict
	"""
	objective.eval()
	with torch.no_grad():
		b_loss = 0
		for i, batch in enumerate(loader):
			loss = objective(batch, kl_factor=kl_factor)
			b_loss += loss.item() * get_batch_len(batch)
	agg['test_loss'].append(b_loss / len(loader.dataset))
	agg['test_epoch'].append(epoch)
	test_str = '====> Epoch: {:03d} Test loss: {:.4f}'
	print(test_str.format(epoch, agg['test_loss'][-1]))


def save_state(objective, optimizer, epoch, state_fn):
	"""Save state."""
	print("Saving state to:", state_fn)
	torch.save({
			'objective_state_dict': objective.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'epoch': epoch,
		}, state_fn)


def load_state(objective, optimizer, device, state_fn):
	"""Load state."""
	print("Loading state from:", state_fn)
	checkpoint = torch.load(state_fn, map_location=device)
	objective.load_state_dict(checkpoint['objective_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	return checkpoint['epoch']


def get_batch_len(batch):
	"""Return the number of data items in a batch."""
	if isinstance(batch, (tuple,list)): # non-vectorized modalities
		return len(batch[0])
	return len(batch) # vectorized modalities


def estimate_marginal_log_like(objective, loader, k=2000, mini_k=128, \
	reduction='mean'):
	"""
	Simple log marginal estimation.

	Take the approximate posterior as a proposal distribution, do an
	importance-weighted estimate.

	Parameters
	----------
	objective : src.objectives.VaeObjective
	loader : torch.utils.DataLoader
	k : int
	mini_k : int
	reduction : {'mean', 'sum'}

	Returns
	-------
	est_mll : float
		Estimated data marginal log likelihood.
	"""
	objective.eval()
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
						quit("MLL failed, probably due to memory issues!")
					mini_k //= 2
			log_m = torch.cat(inner_batch_res, dim=1) - np.log(k)
			log_m = torch.logsumexp(log_m, dim=1)
			batch_res.append(log_m)
		batch_res = torch.cat(batch_res, dim=0).detach().cpu().numpy()
	assert len(batch_res.shape) == 1 # dataset size
	if reduction == 'sum':
		return np.sum(batch_res)
	return np.mean(batch_res)


def mll_helper(objective, dataloaders, epoch, agg):
	"""Estimate marginal log likelihoods."""
	# First estimate MLL on the validation set.
	tic = perf_counter()
	mll = estimate_marginal_log_like(objective, dataloaders['valid'])
	toc = perf_counter()
	agg['valid_mll'].append(mll)
	agg['valid_mll_epoch'].append(epoch)
	print("Valid MLL: ", mll, ", time:", round(toc-tic,2))
	# If it's the best performance we've seen, also evaluate on the test set.
	if mll == max(agg['valid_mll']):
		tic = perf_counter()
		mll = estimate_marginal_log_like(objective, dataloaders['test'])
		toc = perf_counter()
		agg['test_mll'].append(mll)
		agg['test_mll_epoch'].append(epoch)
		print("Test MLL:", mll, ", time:", round(toc-tic,2))
		return True
	return False


def save_aggregator(agg, agg_fn):
	"""Save the data aggregrator."""
	torch.save(agg, agg_fn)



def main(
		dataset='mnist_halves',
		variational_strategy='gaussian_poe',
		variational_posterior='diag_gaussian',
		prior='standard_gaussian',
		likelihood='spherical_gaussian',
		objective='elbo',
		unstructured_encoder=False,
		lr=1e-3,
		K=10,
		ebm_samples=10,
		latent_dim=20,
		m_dim=4,
		vmf_dim=4,
		n_vmfs=5,
		theta_dim=2,
		embed_dim=8,
		batch_size=256,
		epochs=10,
		kl_anneal_epochs=100,
		no_improvement=100,
		ar_step_size=1,
		obs_std_dev=0.1,
		pre_trained=False,
		mll_freq=1000,
		test_freq=10,
		no_cuda=False,
		seed=42,
		grad_clip=1e2,
		train_m=0.5,
		test_m=0.0,
		data_dir='/media/jackg/Jacks_Animal_Sounds/torchvision/',
		save_model=False,
):
	"""
	Main function: train a model.

	Note that not all parameter settings are compatible. For example, we can't
	set `variational_strategy=gaussian_moe` and
	`variational_posterior=vmf_product`. Inconsistencies like this should be
	caught by `src.utils.check_args`.

	Parameters
	----------
	dataset : str, optional
		The dataset to train on. See `src.param_maps.DATASET_MAP` for all
		options.
	variational_strategy : str, optional
		A strategy for combining modality-specific recognition models. See
		`src.param_maps.VAR_STRATEGY_MAP` for all options.
	variational_posterior : str, optional
		The family of approximate posteriors to use. See
		`src.param_maps.VAR_POSTERIOR_MAP` for all options.
	prior : str, optional
		The prior distribution to use. See `src.param_maps.PRIOR_MAP` for all
		options.
	likelihood : str, optional
		The likelihood distribution to use. See `src.param_maps.LIKELIHOOD_MAP`
		for all options.
	objective : str, optional
		The objective to use. See `src.param_maps.OBJECTIVE_MAP` for all
		options.
	unstructured_encoder : bool, optional
		Whether to concatenate all the modalities together to make a single
		unstructured recognition model.
	lr : float, optional
		Learning rate.
	K : int, optional
		Number of samples for the IWAE objective. For the Mvae objective, this
		is the number of random subsets of observed modalities drawn.
		TO DO: change the name of this!
	ebm_samples : int, optional
		Number of samples for the self-normalized importance sampling (SNIS)
		strategy for the energy-based model (EBM) variational posterior.
	latent_dim : int, optional
		Latent dimension.
	m_dim : int, optional
		NOTE: is this used???
	vmf_dim : int, optional
		The sphere dimension to use for von Mises-Fisher (vMF) distributions:
		S^{vmf_dim}. Internally, samples from the vMF are represented in the
		ambient space \mathbb{R}^{vmf_dim+1}.
	n_vmfs : int, optional
		Number of von Mises-Fisher distributions to use to form the latent
		space.
	theta_dim : int, optional
		For energy-based approximate posteriors.
	embed_dim : int, optional
		For learning a modality embedding.
	batch_size : int, optional
		Size of training batches.
	epochs : int, optional
		Maximum number of epochs to train.
	kl_anneal_epochs : int, optional
		Number of epochs taken to anneal the KL term.
	no_improvement : int, optional
		Terminate after there's been no validation set improvement in this many
		epochs.
	ar_step_size : int, optional
		How many modalities to condition on in each step of the AR-ELBO.
	obs_std_dev : float, optional
		Observation standard deviation. For Gaussian likelihoods.
	pre_trained : bool, optional
		Set to `True` if you want to keep training a previously saved model.
		Loading a model and continuing to train breaks the random seeds. Train
		in one go for reproducability.
	mll_freq : int, optional
		The test set marginal log likelihood is estimated every `mll_freq`
		epochs.
	test_freq : int, optional
		The test set objective is evaluated every `test_freq` epochs.
	no_cuda : bool, optional
		Set to `True` if you don't want to use CUDA, even if it is available.
	seed : int, optional
		Random seed.
	grad_clip : float, optional
		The gradient norm at which gradient clipping kicks in.
	train_m : float, optional
		Training set missingness.
	test_m : float, optional
		Test set missingness.
	data_dir : str, optional
		Data directory.
	save_model : bool, optional
		Whether to save the model after the training run.
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
			quit(f"Couldn't find {agg_fn} to load")
	else:
		if os.path.exists(exp_dir):
			pass
			_ = input("Experiment path already exists! Continue? ")
			try:
				os.remove(log_fn)
			except FileNotFoundError:
				quit(f"Couldn't find {log_fn} to remove")
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
	args['device'] = device
	# Make Dataloaders.
	dataloaders = make_dataloaders(**args)
	# Make the objective.
	objective = make_objective(**args).to(device)
	# Make the optimizer.
	optimizer = torch.optim.Adam(objective.parameters(), lr=lr)
	# Load pretrained models.
	if pre_trained:
		prev_epochs = load_state(objective, optimizer, device, state_fn)
	else:
		prev_epochs = 0
	# Set up a data aggregrator.
	if agg is None:
		agg = defaultdict(list)
	# Enter a train loop.
	last_improvement_epoch = prev_epochs+1
	for epoch in range(prev_epochs+1, prev_epochs+epochs+1):
		train_epoch(
			objective,
			dataloaders['train'],
			optimizer,
			epoch,
			min(1.0, epoch/kl_anneal_epochs),
			agg,
			grad_clip,
		)
		if epoch % test_freq == 0:
			test_epoch(
					objective,
					dataloaders['test'],
					epoch,
					1.0,
					agg,
			)
		if epoch % mll_freq == 0:
			improvement = mll_helper(objective, dataloaders, epoch, agg)
			if improvement:
				last_improvement_epoch = epoch
			elif epoch - last_improvement_epoch >= no_improvement:
				print(f"No improvement in {no_improvement} epochs, stopping.")
				break
	# Save the aggregrator.
	save_aggregator(agg, agg_fn)
	# Plot reconstructions and generations.
	dataloaders['train'].dataset.make_plots(objective, dataloaders, exp_dir)
	# Save the model/objective.
	if save_model:
		save_state(objective, optimizer, epoch, state_fn)



if __name__ == '__main__':
	fire.Fire(main)


###
