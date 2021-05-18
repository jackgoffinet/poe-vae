"""
Different objectives for VAEs.

Objectives play a central role in this VAE abstraction. They take a VAE object,
which is just a collection of various modular parts, and, given a batch of data,
determine how to route the data through these parts to calculate a loss.

Having objectives subclass torch.nn.Module leaves the door open for objectives
to have trainable parameters and other state.

TO DO
-----
* Split this up into multiple files
"""
__date__ = "January - May 2021"


import numpy as np
import torch
import torch.nn.functional as F

from .param_maps import DATASET_MAP



class VaeObjective(torch.nn.Module):
	"""Abstract VAE Objective class"""

	def __init__(self, vae):
		super(VaeObjective, self).__init__()
		self.vae = vae
		self.encoder = vae.encoder
		self.variational_strategy = vae.variational_strategy
		self.variational_posterior = vae.variational_posterior
		self.prior = vae.prior
		self.decoder = vae.decoder
		self.likelihood = vae.likelihood
		n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
		print("Trainabale parameters:", n_params)


	def forward(self, x):
		"""
		Evaluate loss on a minibatch `x`.

		Parameters
		----------
		xs : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise

		Returns
		-------
		loss : torch.Tensor
			Shape: []
		"""
		raise NotImplementedError


	def encode(self, xs, nan_mask, n_samples=1, return_params=False, \
		stratified=False):
		"""
		Standard encoding procedure.

		Parameters
		----------
		xs : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise
		nan_mask : torch.Tensor
			Shape: [batch,modalities]
		n_samples : int, optional
		return_params : bool, optional
		stratified : bool, optional
			Whether to perform stratified sampling

		z_samples : torch.Tensor
			Shape:
				[batch,samples,modalities,z_dim] if stratified
				[batch,samples,z_dim] otherwise
		log_qz : torch.Tensor
			Shape:
				[batch,samples,modalities] if stratified
				[batch,samples] otherwise
		log_pz : torch.Tensor
			Shape:
				[batch,samples,modalities] if stratified
				[batch,samples] otherwise
		variational_posterior_params : tuple of torch.Tensor
			Returned if `return_params`.
			Shape: [n_params][b,*]
		"""
		# Encode data.
		# encoding shape:
		# [n_params][b,m,param_dim] if vectorized
		# [m][n_params][b,param_dim] otherwise
		encoding = self.encoder(xs)
		# Transpose first two dimensions: [n_params][m][b,param_dim]
		if isinstance(xs, (tuple,list)):
			encoding = tuple(tuple(e) for e in zip(*encoding))
		return self._encode_helper(
				encoding,
				nan_mask,
				n_samples=n_samples,
				return_params=return_params,
				stratified=stratified,
		)


	def _encode_helper(self, encoding, nan_mask, n_samples=1, \
		return_params=False, stratified=False):
		"""
		This helper is separate so that MvaeElbo can be a bit more efficient.

		Parameters
		----------
		encoding : tuple of torch.Tensor or tuple of tuple of torch.Tensor
			Shape:
				[n_params][b,m,param_dim] if vectorized
				[m][n_params][b,param_dim] otherwise
		nan_mask : torch.Tensor
			Shape: [batch,modalities]
		n_samples : int, optional
		return_params : bool, optional
		stratified : bool, optional
			Whether to perform stratified sampling

		Returns
		-------
		z_samples : torch.Tensor
			Shape:
				[batch,samples,modalities,z_dim] if stratified
				[batch,samples,z_dim] otherwise
		log_qz : torch.Tensor
			Shape:
				[batch,samples,modalities] if stratified
				[batch,samples] otherwise
		log_pz : torch.Tensor
			Shape:
				[batch,samples,modalities] if stratified
				[batch,samples] otherwise
		variational_posterior_params : tuple of torch.Tensor
			Returned if `return_params`.
			Shape: [n_params][b,*]
		"""
		# Combine evidence.
		# var_post_params shape: [n_params][b,*] where * is parameter dim.s.
		var_post_params = self.variational_strategy(
				*encoding,
				nan_mask=nan_mask,
		)
		# Make a variational posterior and sample.
		# z_samples:
		#	[b,s,m,z] if stratified
		#	[b,s,z] otherwise
		# log_qz:
		#	[b,s,m] if stratified
		#	[b,z] otherwise
		if stratified:
			z_samples, log_qz = self.variational_posterior.stratified_forward(
				*var_post_params,
				n_samples=n_samples,
			)
		else:
			z_samples, log_qz = self.variational_posterior(
					*var_post_params,
					n_samples=n_samples,
			)
		# Evaluate prior.
		log_pz = self.prior(z_samples) # [b,s,m] or [b,s]
		if return_params:
			return z_samples, log_qz, log_pz, var_post_params
		return z_samples, log_qz, log_pz


	def decode(self, z_samples, xs, nan_mask, combine_modalities=True):
		"""
		Standard decoding procedure.

		Parameters
		----------
		z_samples : torch.Tensor
			Shape: [batch,n_samples,z_dim]
		xs : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise
		nan_mask : torch.Tensor
			Shape: [batch,modalities]
		combine_modalities : bool, optional
			Sum the log likelihood over the modality dimension.

		Returns
		-------
		log_likes : torch.Tensor
			Shape:
				[batch,samples] if combine_modalities
				[batch,samples,m] otherwise
		"""
		# Decode samples to get likelihood parameters.
		# like_params shape:
		# [n_params][b,s,m,m_dim] if vectorized
		# [n_params][m][b,s,m',m_dim] otherwise
		like_params = self.decoder(z_samples)
		# Evaluate likelihoods, sum over modalities.
		log_likes = self.likelihood(
				xs,
				like_params,
				nan_mask=nan_mask,
		) # [b,s,m]
		if combine_modalities:
			return torch.sum(log_likes, dim=2) # [b,s]
		return log_likes # [b,s,m]


	def estimate_marginal_log_like(self, xs, n_samples=1000, keepdim=False):
		"""
		Estimate the marginal log likelihood of the data `xs`.

		Parameters
		----------
		xs : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise
		n_samples : int
			Number of importance-weighted samples.
		keepdim : bool
			Keep the sample dimension.

		Returns
		-------
		est_log_m : torch.Tensor
			Estimated marginal log likelihood of the data `xs`. If `keepdim`,
			return the log likelihoods of each individual sample. Shape: [b,k]
			Otherwise, log-mean-exp over the sample dimension. Shape: [b]
		"""
		self.eval()
		# Get missingness pattern, replace with zeros to prevent NaN gradients.
		xs, nan_mask = apply_nan_mask(xs)
		# [b,k,z], [b,k], [b,k]
		z_samples, log_qz, log_pz = self.encode(
				xs,
				nan_mask,
				n_samples=n_samples,
		)
		log_likes = self.decode(z_samples, xs, nan_mask) # [b,k]
		if keepdim: # Keep the sample dimension.
			return log_pz - log_qz + log_likes
		est_log_m = log_pz - log_qz + log_likes - np.log(n_samples)
		est_log_m = torch.logsumexp(est_log_m, dim=1)
		return est_log_m


	def generate(self, n_samples=9, likelihood_noise=False):
		"""
		Generate data.

		Parameters
		----------
		n_samples : int, optional
		likelihood_noise : bool, optional

		Returns
		-------
		generated : list of numpy.ndarray
			Shape: [m][1,s,sub_modalities,m_dim]
		"""
		self.eval()
		with torch.no_grad():
			z_samples = self.prior.rsample(n_samples=n_samples) # [1,s,z]
			# like_params shape:
			# [n_params][1,m,m_dim] if vectorized
			# [n_params][m][1,s,x] otherwise
			like_params = self.decoder(z_samples)
			# generated: [m][1,s,sub_modalities,m_dim]
			if likelihood_noise:
				generated = self.likelihood.sample(like_params)
			else:
				generated = self.likelihood.mean(like_params)
		return [g.detach().cpu().numpy() for g in generated]


	def reconstruct(self, xs, likelihood_noise=False):
		"""
		Reconstruct data.

		Parameters
		----------
		xs : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise
		likelihood_noise : bool, optional

		Returns
		-------
		reconstruction : list of numpy.ndarray
			Shape: [m][b,s,sub_modalities,m_dim]
		"""
		self.eval()
		with torch.no_grad():
			xs, nan_mask = apply_nan_mask(xs)
			# Encode data.
			z_samples, _, _ = self.encode(xs, nan_mask) # [b,s,z]
			# like_params shape:
			# [n_params][1,m,m_dim] if vectorized
			# [n_params][m][1,s,x] otherwise
			like_params = self.decoder(z_samples)
			# generated shape: [m][b,s,sub_m,m_dim]
			if likelihood_noise:
				generated = self.likelihood.sample(like_params)
			else:
				generated = self.likelihood.mean(like_params)
		return [g.detach().cpu().numpy() for g in generated]



class StandardElbo(VaeObjective):

	def __init__(self, vae, **kwargs):
		super(StandardElbo, self).__init__(vae)

	def forward(self, xs):
		"""
		Evaluate the standard single-sample ELBO.

		Note
		----
		* Requires an analytic KL-divergence from the variational posterior to
		  the prior. If we don't have that, consider using the IWAE objective
		  with a single sample. Or I could implement a sampling-based KL
		  approximation in AbstractVariationalPosterior.

		Parameters
		----------
		xs : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise

		Returns
		-------
		loss : torch.Tensor
			Shape: []
		"""
		# Get missingness pattern, replace with zeros to prevent NaN gradients.
		xs, nan_mask = apply_nan_mask(xs)
		# Encode.
		# z_samples : [b,s,z]
		z_samples, _, _ = self.encode(xs, nan_mask)
		# Evaluate KL.
		kld = self.variational_posterior.kld(self.prior).sum(dim=1) # [b]
		# Decode.
		log_likes = self.decode(z_samples, xs, nan_mask) # [b,s]
		assert log_likes.shape[1] == 1 # assert single sample
		log_likes = log_likes.squeeze(1) # [b], remove sample dimension
		# Evaluate loss.
		assert len(log_likes.shape) == 1, f"len({log_likes.shape}) != 1"
		assert len(kld.shape) == 1, f"len({kld.shape}) != 1"
		loss = -torch.mean(log_likes - kld)
		return loss



class IwaeElbo(VaeObjective):

	def __init__(self, vae, K=10, **kwargs):
		super(IwaeElbo, self).__init__(vae)
		self.k = K

	def forward(self, xs):
		"""
		Evaluate the multisample IWAE ELBO.

		Parameters
		----------
		xs : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise

		Returns
		-------
		loss : torch.Tensor
			Shape: []
		"""
		# Get missingness pattern, replace with zeros to prevent NaN gradients.
		xs, nan_mask = apply_nan_mask(xs)
		# Encode.
		# z_samples : [b,s,z]
		# log_qz : [b,s]
		# log_pz : [b,s]
		z_samples, log_qz, log_pz = self.encode(xs, nan_mask, n_samples=self.k)
		assert log_pz.shape[1] == self.k # assert k samples
		assert log_qz.shape[1] == self.k # assert k samples
		# Decode.
		log_likes = self.decode(z_samples, xs, nan_mask) # [b,s]
		assert log_likes.shape[1] == self.k # assert k samples
		# Define importance weights.
		log_ws = log_pz + log_likes - log_qz - np.log(self.k) # [b,k]
		# Evaluate loss.
		loss = -torch.mean(torch.logsumexp(log_ws, dim=1))
		return loss



class DregIwaeElbo(VaeObjective):

	def __init__(self, vae, K=10, **kwargs):
		super(DregIwaeElbo, self).__init__(vae)
		self.k = K

	def forward(self, xs):
		"""
		Evaluate the multisample IWAE ELBO with the DReG estimator.

		DReG = Doubly Reparameterized Gradients from Tucker et al. (2018)
		https://arxiv.org/abs/1810.04152

		Parameters
		----------
		xs : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise

		Returns
		-------
		loss : torch.Tensor
			Shape: []
		"""
		# Get missingness pattern, replace with zeros to prevent NaN gradients.
		xs, nan_mask = apply_nan_mask(xs)
		# Encode observed modalities.
		# z_samples: [b,s,z]
		# log_pz: [b,s]
		# var_post_params: [n_params][b,*]
		z_samples, _, log_pz, var_post_params = self.encode(
				xs,
				nan_mask,
				n_samples=self.k,
				return_params=True,
		)
		# Now stop gradients through the encoder when evaluating log q(z|x).
		detached_params = [param.detach() for param in var_post_params]
		log_qz = self.variational_posterior.log_prob(
				z_samples,
				*detached_params,
		) # [b,s]
		assert log_pz.shape[1] == self.k # assert k samples
		assert log_qz.shape[1] == self.k # assert k samples
		# Decode.
		log_likes = self.decode(z_samples, xs, nan_mask) # [b,s]
		assert log_likes.shape[1] == self.k # assert s samples
		# Define importance weights.
		log_ws = log_pz + log_likes - log_qz - np.log(self.k) # [b,s]
		# Doubly reparameterized gradients
		with torch.no_grad():
			weights = log_ws - torch.logsumexp(log_ws, dim=1, keepdim=True)
			weights = torch.exp(weights) # [b,s]
			if z_samples.requires_grad:
				z_samples.register_hook(lambda grad: weights.unsqueeze(-1)*grad)
		# Evaluate loss.
		loss = -torch.mean(torch.sum(weights * log_ws, dim=1))
		return loss



class MmvaeElbo(VaeObjective):

	def __init__(self, vae, K=10, dataset='mnist_halves', **kwargs):
		super(MmvaeElbo, self).__init__(vae)
		self.k = K
		self.M = DATASET_MAP[dataset].n_modalities

	def forward(self, xs):
		"""
		Stratified sampling IW-ELBO from Shi et al. (2019), Eq. 3.

		K samples are drawn from every mixture component. Uses DReG gradients.
		This is the same as `DregIwaeElbo`, but with stratified sampling.

		Parameters
		----------
		xs : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise
		"""
		# Get missingness pattern, replace with zeros to prevent NaN gradients.
		xs, nan_mask = apply_nan_mask(xs)

		# Encode observed modalities.
		# z_samples: [b,s,m,z]
		# log_pz: [b,s,m]
		# var_post_params: [n_params][b,*]
		z_samples, _, log_pz, var_post_params = self.encode(
				xs,
				nan_mask,
				n_samples=self.k,
				return_params=True,
				stratified=True,
		)

		# Now stop gradients through the encoder when evaluating log q(z|x).
		detached_params = [param.detach() for param in var_post_params]
		log_qz = self.variational_posterior.log_prob(
				z_samples,
				*detached_params,
				stratified=True,
		) # [b,s,m]


		assert log_pz.shape[1] == self.k # assert k samples
		assert log_qz.shape[1] == self.k # assert k samples
		# Decode.
		z_samples = z_samples.contiguous() # not ideal!
		# [b,s*m,z]
		z_samples = z_samples.view(z_samples.shape[0],-1,z_samples.shape[3])
		# [b,s,m]
		log_likes = self.decode(z_samples, xs, nan_mask).view(log_qz.shape)
		assert log_likes.shape[1] == self.k # assert k samples
		# Define importance weights.
		# We're going to logsumexp over the sample dimension (K) and average
		# over the modality dimension (M).
		log_ws = log_pz + log_likes - log_qz - np.log(self.k) # [b,s,m]
		with torch.no_grad():
			weights = log_ws - torch.logsumexp(log_ws, dim=1, keepdim=True)
			weights = torch.exp(weights) # [b,s,m]
			# If the outer function isn't in a torch.no_grad() context,
			# register a hook.
			if log_ws.requires_grad:
				def hook_func(grad):
					return weights.view(grad.shape[0],-1).unsqueeze(-1) * grad
				z_samples.register_hook(hook_func)
		# Evaluate loss.
		elbo = torch.mean(torch.sum(weights * log_ws, dim=1), dim=1)
		assert len(elbo.shape) == 1 and elbo.shape[0] == log_ws.shape[0]
		loss = -torch.mean(elbo)
		return loss



class MvaeElbo(VaeObjective):

	def __init__(self, vae, K=10, **kwargs):
		"""
		Parameters
		----------
		K : int, optional
			The number of random subsets of observed modalities to draw.
		"""
		super(MvaeElbo, self).__init__(vae)
		self.k = K

	def forward(self, xs):
		"""
		Evaluate the subsampling ELBO from Wu & Goodman (2018, Eq. 5).

		Note: this isn't a true evidence lower bound of the observed modalities.

		Parameters
		----------
		xs : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise

		Returns
		-------
		loss : torch.Tensor
			Shape: []
		"""
		# Get missingness pattern, replace with zeros to prevent NaN gradients.
		xs, nan_mask = apply_nan_mask(xs)
		# Encode the data once.
		# encoding shape:
		# [n_params][b,m,param_dim] if vectorized
		# [m][n_params][b,param_dim] otherwise
		encoding = self.encoder(xs)
		# Transpose first two dimensions: [n_params][m][b,param_dim]
		if isinstance(xs, (tuple,list)):
			encoding = tuple(tuple(e) for e in zip(*encoding))
		# Create different NaN masks for each ELBO.
		# Append the fully observed mask.
		nan_masks = [nan_mask]
		# Then append the random subsets masks.
		for i in range(self.k):
			mask = torch.randint(0,1,nan_mask.shape,dtype=torch.bool)
			mask = torch.logical_or(mask, nan_mask)
			nan_masks.append(mask)
		# Then append the single modality masks.
		for i in range(nan_mask.shape[1]):
			mask = F.one_hot(torch.tensor([i]), num_classes=nan_mask.shape[1])
			mask = mask.to(torch.bool).expand(nan_mask.shape[0], -1)
			nan_masks.append(mask)
		# For each NaN mask calculate a standard ELBO loss.
		losses = []
		for mask in nan_masks:
			z_samples, log_qz, log_pz = self._encode_helper(encoding, mask)
			assert log_pz.shape[1] == 1 # assert single sample
			log_pz = log_pz.squeeze(1) # [b], remove sample dimension.
			# Evaluate KL.
			kld = self.variational_posterior.kld(self.prior).sum(dim=1) # [b]
			# Decode.
			log_likes = self.decode(z_samples, xs, nan_mask) # [b,s]
			assert log_likes.shape[1] == 1 # assert single sample
			log_likes = log_likes.squeeze(1) # [b], remove sample dimension
			# Evaluate loss.
			assert len(log_pz.shape) == 1
			assert len(log_likes.shape) == 1
			assert len(kld.shape) == 1
			loss = -torch.mean(log_pz + log_likes - kld)
			losses.append(loss)
		return sum(losses)



class ArElbo(VaeObjective):

	def __init__(self, vae, ar_step_size=1, dataset='mnist_halves', **kwargs):
		"""
		Parameters
		----------
		vae : torch.nn.ModuleDict
		ar_step_size : int, optional
			How many additional modalities conditioned on in each step.
		dataset : str, optional
			Used to determine the number of modalities.
		"""
		super(ArElbo, self).__init__(vae)
		self.step = ar_step_size
		self.M = DATASET_MAP[dataset].n_modalities

	def forward(self, xs):
		"""
		Evaluate an autoregressive ELBO.

		Parameters
		----------
		xs : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise
		"""
		# Get missingness pattern, replace with zeros to prevent NaN gradients.
		xs, nan_mask = apply_nan_mask(xs)
		# Encode data.
		# encoding shape:
		# [n_params][b,m,param_dim] if vectorized
		# [m][n_params][b,param_dim] otherwise
		encoding = self.encoder(xs)
		# Transpose first two dimensions: [n_params][m][b,param_dim]
		if isinstance(xs, (tuple,list)):
			encoding = tuple(tuple(e) for e in zip(*encoding))
		# Get evidence.
		# var_post_params shape: [n_params][b,m,*] where * is parameter dim.s.
		var_post_params = self.variational_strategy(
				*encoding,
				nan_mask=nan_mask,
				collapse=False,
		)
		# Define a random permutation of modalities.
		# Option 1: a single permutation for every batch item (much faster)
		perm = torch.randperm(self.M) # [m]
		# # Option 2: independent permutations for each batch item (very slow)
		# # Maybe this could be sped up with a batched index-select method.
		# perm = torch.stack(
		# 		[torch.randperm(self.M) for _ in range(len(nan_mask))],
		# 		dim=0,
		# ) # [b,m]
		# Collect log likelihoods and KL-divergences.
		log_likes, klds = [], []
		i = 0
		while i < self.M:
			# Option 1:
			perm_slice = perm[i:min(self.M, i+self.step)]
			# # Option 2:
			# perm_slice = perm[:,i:min(self.M, i+self.step)]
			# Combine evidence across a few more modalities.
			kld, z_samples = self.variational_posterior.add_evidence(
					*var_post_params,
					perm_slice,
					start_from_prior=(i==0),
					return_sample=True,
			) # [b], [b,1,z]
			klds.append(kld)
			# Decode.
			# This is a bit wasteful, and could be improved, but we're going to
			# decode all the modalities and only take the ones we want.
			log_like = self.decode(
					z_samples,
					xs,
					nan_mask,
					combine_modalities=False,
			).squeeze(1) # [b,s,m] -> [b,m]
			# Option 1:
			log_like = log_like[:,perm_slice].sum(dim=1) # [b,step] -> [b]
			# # Option 2:
			# log_like = torch.stack(
			# 		[a[j] for a,j in zip(log_like, perm_slice)],
			# 		dim=0,
			# ).sum(dim=1) # [b,step] -> [b]
			log_likes.append(log_like)
			i += self.step
		# Return a loss.
		log_likes = torch.stack(log_likes, dim=1).sum(dim=1) # [b]
		klds = torch.stack(klds, dim=1).sum(dim=1) # [b]
		# print(torch.mean(log_likes).item(), torch.mean(klds).item())
		elbo = torch.mean(log_likes - klds) # [b] -> []
		if torch.isnan(elbo).sum() > 0:
			print("log_likes", torch.isnan(log_likes).sum().item())
			print("klds", torch.isnan(klds).sum().item())
			print("elbo NaN")
			quit()
		return -elbo



def apply_nan_mask(xs):
	"""
	Find out where the data is missing, replace these entries with zeros.

	Modalities are marked as missing with NaNs.

	Parameters
	----------
	xs : torch.Tensor or tuple of torch.Tensor
		Shape:
			[batch,modalities,m_dim] if vectorized
			[modalities][batch,m_dim] otherwise

	Returns
	-------
	xs : torch.Tensor or tuple of torch.Tensors
		With NaNs replaced with zeros.
	nan_mask : torch.Tensor or tuple of torch.Tensor
		Shape: [batch,modalities]
	"""
	if isinstance(xs, (tuple,list)): # non-vectorized modalities
		nan_mask = tuple(torch.isnan(x[:,0]) for x in xs)
		for i in range(len(xs)):
			xs[i][nan_mask[i]] = 0
		nan_mask = torch.stack(nan_mask, dim=1)
	else: # vectorized modalities
		nan_mask = torch.isnan(xs[:,:,0]) # [b,m]
		xs[nan_mask.unsqueeze(-1)] = 0
	return xs, nan_mask



if __name__ == '__main__':
	pass



###
