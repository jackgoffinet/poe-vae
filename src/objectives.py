"""
Different objectives for VAEs.

Objectives play a central role in this VAE abstraction. They take a VAE object,
which is just a collection of various modular parts, and, given a batch of data,
determine how to route the data through these parts to calculate a loss.

Having objectives subclass torch.nn.Module leaves the door open for objectives
to have trainable parameters and other state.
"""
__date__ = "January - May 2021"


import numpy as np
import torch



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
		x : torch.Tensor or list of torch.Tensor
			Shape: [m][b,x] or ???

		Returns
		-------
		loss : torch.Tensor
			Shape: []
		"""
		raise NotImplementedError


	def encode(self, xs, nan_mask, n_samples=1):
		"""
		Standard encoding procedure.

		Parameters
		----------
		xs : list of torch.Tensor or torch.Tensor
			Shape:
				[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise
		nan_mask : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities] if vectorized
				[modalities][b] otherwise

		Returns
		-------
		z_samples : torch.Tensor
			Shape: [batch,samples,z_dim]
		log_qz : torch.Tensor
			Shape: [batch,samples]
		log_pz : torch.Tensor
			Shape: [batch,samples]
		"""
		# Encode data.
		# `encoding` shape:
		# [n_params][b,m,param_dim] if vectorized
		# [m][n_params][b,param_dim] otherwise
		encoding = self.encoder(xs)
		# Transpose first two dimensions: [n_params][m][b,param_dim]
		if isinstance(xs, (tuple,list)):
			encoding = tuple(tuple(e) for e in zip(*encoding))
		# Combine evidence.
		# `var_post_params` shape: [n_params][b,*] where * is parameter dim.s.
		# ??? is vectorized
		var_post_params = self.variational_strategy(
				*encoding,
				nan_mask=nan_mask,
		)
		# Make a variational posterior and sample.
		# z_samples : [b,s,z]
		# log_qz : [b,z]
		if hasattr(self.variational_posterior, 'non_stratified_forward'):
			z_samples, log_qz = \
					self.variational_posterior.non_stratified_forward(
						*var_post_params,
						n_samples=n_samples,
					)
		else:
			z_samples, log_qz = self.variational_posterior(
					*var_post_params,
					n_samples=n_samples,
			)
		# Evaluate prior.
		log_pz = self.prior(z_samples)
		return z_samples, log_qz, log_pz


	def decode(self, z_samples, xs, nan_mask):
		"""
		Standard decoding procedure.

		Parameters
		----------
		z_samples : torch.Tensor
			Shape: [batch,n_samples,z_dim]
		xs : list of torch.Tensor or torch.Tensor
			Shape:
				[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise
		nan_mask : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities] if vectorized
				[modalities][b] otherwise

		Returns
		-------
		log_likes : torch.Tensor
			Shape: [batch,samples]
		"""
		# Decode samples to get likelihood parameters.
		# `likelihood_params` shape:
		# [n_params][b,m,m_dim] if vectorized
		# [n_params][m][b,s,x] otherwise
		likelihood_params = self.decoder(z_samples)
		# Evaluate likelihoods, sum over modalities.
		log_likes = self.likelihood(xs, likelihood_params, \
				nan_mask=nan_mask) # [b,s,m]
		# Sum over modality dimension.
		log_likes = torch.sum(log_likes, dim=2) # [b,s]
		return log_likes


	def estimate_marginal_log_like(self, xs, n_samples=1000, keepdim=False):
		"""
		Estimate the MLL of the data `xs`.

		Parameters
		----------
		xs : list of torch.Tensor
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
		generated : numpy.ndarray
		"""
		self.eval()
		with torch.no_grad():
			# z_samples : [b,s,z]
			z_samples = self.prior.rsample(n_samples=n_samples) # [1,s,z]
			# like_params shape:
			# [n_params][1,m,m_dim] if vectorized
			# [n_params][m][1,s,x] otherwise
			like_params = self.decoder(z_samples)
			if likelihood_noise:
				generated = self.likelihood.sample(like_params)
			else:
				generated = self.likelihood.mean(like_params)
		return np.array([g.detach().cpu().numpy() for g in generated])


	def reconstruct(self, xs, likelihood_noise=False):
		"""
		Reconstruct data.

		Parameters
		----------
		xs : ...
		likelihood_noise : bool, optional

		Returns
		-------
		reconstruction : numpy.ndarray
			Shape: ???
		"""
		self.eval()
		with torch.no_grad():
			xs, nan_mask = apply_nan_mask(xs)
			# Encode data.
			var_dist_params = self.encoder(xs) # [n_params][b,m,param_dim]
			# Combine evidence.
			var_post_params = self.variational_strategy(*var_dist_params, \
					nan_mask=nan_mask)
			# Make a variational posterior and sample.
			z_samples, _ = self.variational_posterior(*var_post_params)
			like_params =self.decoder(z_samples)
			if likelihood_noise:
				generated = self.likelihood.sample(like_params)
			else:
				generated = self.likelihood.mean(like_params)
		if isinstance(xs, (tuple, list)):
			return np.array([g.detach().cpu().numpy() for g in generated])
		return generated.detach().cpu().numpy()



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
		  with a single sample. Or I could implement a sampling-based KL in
		  AbstractVariationalPosterior.

		Parameters
		----------
		xs : list of torch.Tensor or torch.Tensor
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
		z_samples, log_qz, log_pz = self.encode(xs, nan_mask)
		assert log_pz.shape[1] == 1 # assert single sample
		log_pz = log_pz.sum(dim=1) # [b], sum over sample dimension.
		# Evaluate KL.
		kld = self.variational_posterior.kld(self.prior).sum(dim=1) # [b]
		# Decode.
		log_likes = self.decode(z_samples, xs, nan_mask) # [b,s]
		assert log_likes.shape[1] == 1 # assert single sample
		log_likes = log_likes.sum(dim=1) # [b], sum over sample dimension
		# Evaluate loss.
		assert len(log_pz.shape) == 1
		assert len(log_likes.shape) == 1
		assert len(kld.shape) == 1
		loss = -torch.mean(log_pz + log_likes - kld)
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
		xs : list of torch.Tensor or torch.Tensor
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

		TO DO: test with non-vectorized data

		Parameters
		----------
		xs : torch.Tensor
		"""
		# Get missingness pattern, replace with zeros to prevent NaN gradients.
		xs, nan_mask = apply_nan_mask(xs)
		# Encode data.
		var_dist_params = self.encoder(xs) # [n_params][b,m,param_dim]
		# Combine evidence.
		var_post_params = self.variational_strategy(*var_dist_params, \
				nan_mask=nan_mask)
		# Make a variational posterior and sample.
		z_samples, _ = self.variational_posterior(*var_post_params, \
				n_samples=self.k) # [b,s,z]
		# Evaluate prior.
		log_pz = self.prior(z_samples) # [b,s]
		# Now stop gradients through the encoder when evaluating log q(z|x).
		detached_params = [param.detach() for param in var_post_params]
		# [b,s]
		log_qz = self.variational_posterior.log_prob(z_samples.transpose(0,1), \
				*detached_params)
		assert log_pz.shape[1] == self.k # assert k samples
		assert log_qz.shape[1] == self.k # assert k samples
		# Decode.
		log_likes = self.decode(z_samples, xs, nan_mask) # [b,s]
		assert log_likes.shape[1] == self.k # assert k samples
		# Define importance weights.
		log_ws = log_pz + log_likes - log_qz - np.log(self.k) # [b,k]
		# Doubly reparameterized gradients
		with torch.no_grad():
			weights = log_ws - torch.logsumexp(log_ws, dim=1, keepdim=True)
			weights = torch.exp(weights) # [b,k]
			if z_samples.requires_grad:
				z_samples.register_hook(lambda grad: weights.unsqueeze(-1)*grad)
		# Evaluate loss.
		loss = -torch.mean(torch.sum(weights * log_ws, dim=1))
		return loss



class MmvaeQuadraticElbo(VaeObjective):

	def __init__(self, vae, K=10, **kwargs):
		super(MmvaeQuadraticElbo, self).__init__(vae)
		self.k = args.K
		self.M = args.dataset.n_modalities

	def forward(self, xs):
		"""
		Stratified sampling ELBO from Shi et al. (2019), Eq. 3.

		K samples are drawn from every mixture component. Uses DReG gradients.

		Parameters
		----------
		xs : torch.Tensor
		"""
		# Get missingness pattern, replace with zeros to prevent NaN gradients.
		xs, nan_mask = apply_nan_mask(xs)
		# Encode data.
		var_dist_params = self.encoder(xs) # [n_params][b,m,param_dim]
		# Combine evidence.
		var_post_params = self.variational_strategy(*var_dist_params, \
				nan_mask=nan_mask)
		# Make a variational posterior and sample.
		z_samples, _ = self.variational_posterior(*var_post_params, \
				n_samples=self.k) # [b,s,m,z]
		# Evaluate prior.
		log_pz = self.prior(z_samples) # [b,s,m]
		# Now stop gradients through the encoder when evaluating log q(z|x).
		detached_params = [param.detach() for param in var_post_params]
		# [b,s,m]
		log_qz = self.variational_posterior.log_prob(z_samples, \
				*detached_params)
		assert log_pz.shape[1] == self.k # assert k samples
		assert log_qz.shape[1] == self.k # assert k samples
		# Decode.
		z_samples = z_samples.contiguous() # not ideal!
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
			weights = torch.exp(weights) # [b,k,m]
			def hook_func(grad):
				return weights.view(grad.shape[0],-1).unsqueeze(-1) * grad
			z_samples.register_hook(hook_func)
		# Evaluate loss.
		elbo = torch.mean(torch.sum(weights * log_ws, dim=1), dim=1)
		assert len(elbo.shape) == 1 and elbo.shape[0] == log_ws.shape[0]
		loss = -torch.mean(elbo)
		return loss



class MvaeElbo(VaeObjective):

	def __init__(self, vae, **kwargs):
		super(MvaeElbo, self).__init__(vae)

	def forward(self, xs):
		"""
		Evaluate the subsampling ELBO from Wu & Goodman (2018).

		Parameters
		----------
		xs : torch.Tensor
		"""
		raise NotImplementedError



class SnisElbo(VaeObjective):

	def __init__(self, vae, K=10, **kwargs):
		super(SnisElbo, self).__init__(vae)
		self.k = K

	def forward(self, xs):
		"""
		Evaluate a self-normalized importance sampling ELBO.

		Parameters
		----------
		xs : torch.Tensor
		"""
		raise NotImplementedError



class ArElbo(VaeObjective):

	def __init__(self, vae, **kwargs):
		super(ArElbo, self).__init__(vae)

	def forward(self, xs):
		"""
		Evaluate an autoregressive ELBO.

		Parameters
		----------
		xs : torch.Tensor
		"""
		raise NotImplementedError



def apply_nan_mask(xs):
	"""
	Find out where the data is missing, replace these entries with zeros.

	Parameters
	----------
	xs : torch.Tensor or tuple of torch.Tensors
		Shape:
			[batch,modalities,m_dim] if vectorized
			[modalities][batch,m_dim] otherwise

	Returns
	-------
	xs : torch.Tensor or tuple of torch.Tensors
		With NaNs replaced with zeros.
	nan_mask : torch.Tensor or tuple of torch.Tensor
		Shape:
			[batch,modalities] if vectorized
			[modalities][b] otherwise
	"""
	if isinstance(xs, (tuple,list)): # non-vectorized modalities
		nan_mask = tuple(torch.isnan(x[:,0]) for x in xs)
		for i in range(len(xs)):
			xs[i][nan_mask[i]] = 0
	else: # vectorized modalities
		nan_mask = torch.isnan(xs[:,:,0]) # [b,m]
		xs[nan_mask.unsqueeze(-1)] = 0
	return xs, nan_mask



if __name__ == '__main__':
	pass



###
