"""
Different objectives for VAEs.

Objectives play a central role in this VAE abstraction. They take a VAE object,
which isn't anything more than a collection of various modular parts, and, given
a batch of data, determine how to route the data through these parts to
calculate a loss.

Representing objectives as torch.nn.Modules instead of functions leaves the door
open for objectives to have trainable parameters and other state. For the
objectives so far, though, we don't need this.
"""
__date__ = "January 2021"


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
		Evaluate loss on a minibatch x.

		Parameters
		----------
		x : torch.Tensor or list of torch.Tensor

		Returns
		-------
		loss : torch.Tensor
		"""
		raise NotImplementedError


	def encode(self, x, nan_mask, n_samples=1):
		"""
		Encode data.

		Parameters
		----------
		x : list of torch.Tensor or torch.Tensor
			Shape: [modalities][batch,m_dim] or [batch,modalities,m_dim]
		nan_mask : list of torch.Tensor

		Returns
		-------
		...
		"""
		# Encode data.
		var_dist_params = self.encoder(x) # [n_params][b,m,param_dim]
		# Combine evidence.
		var_post_params = self.variational_strategy(*var_dist_params, \
				nan_mask=nan_mask)
		# Make a variational posterior and sample.
		z_samples, log_qz = self.variational_posterior(*var_post_params, \
				n_samples=n_samples)
		# Evaluate prior.
		log_pz = self.prior(z_samples)
		return z_samples, log_qz, log_pz


	def decode(self, z_samples, x, nan_mask):
		"""

		Parameters
		----------
		z_samples : torch.Tensor
			Shape: [batch,n_samples,z_dim]
		x : list of torch.Tensor or torch.Tensor
			Shape: [modalities][batch,m_dim] or [batch,modalities,m_dim]

		Returns
		-------
		log_likes : torch.Tensor
			Shape: [batch,samples]
		"""
		# Decode samples to get likelihood parameters.
		# likelihood_params shape:
		# [n_params][b,m,m_dim] if vectorized, otherwise [m][n_params][b,s,x]
		likelihood_params = self.decoder(z_samples)
		# Evaluate likelihoods, sum over modalities.
		log_likes = self.likelihood(x, likelihood_params, \
				nan_mask=nan_mask)
		# Sum over modality dimension.
		if type(log_likes) == type([]): # not vectorized
			log_likes = sum(log_like for log_like in log_likes)
		else:
			log_likes = torch.sum(log_likes, dim=2)
		return log_likes


	def estimate_marginal_log_like(self, x, n_samples=1000, keepdim=False):
		"""

		Parameters
		----------
		x : list of torch.Tensor
		n_samples : int
		 	Number of importance-weighted samples.
		keepdim : bool
			Keep the sample dimension.

		Returns
		-------
		est_log_m : torch.Tensor
			If `keepdim`: Likelihoods of all samples. Shape: [b,k]
			Else: Estimated log marginal. Shape: [b]
		"""
		# Get missingness pattern, replace with zeros to prevent NaN gradients.
		nan_mask = get_nan_mask(x)
		if type(x) == type([]): # not vectorized, shape: [m][batch]
			for i in range(len(x)):
				x[i][nan_mask[i]] = 0.0
		else: # vectorized modalities, shape: [batch,m]
			x[nan_mask.unsqueeze(-1)] = 0.0
		# [b,k,z], [b,k], [b,k]
		z_samples, log_qz, log_pz = \
				self.encode(x, nan_mask, n_samples=n_samples)
		log_likes = self.decode(z_samples, x, nan_mask) # [b,k]
		if keepdim: # Keep the sample dimension.
			return log_pz - log_qz + log_likes
		est_log_m = torch.logsumexp(log_pz - log_qz + log_likes \
				- np.log(n_samples), dim=1)
		return est_log_m



class StandardElbo(VaeObjective):

	def __init__(self, vae):
		super(StandardElbo, self).__init__(vae)

	def forward(self, x):
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
		x : list of torch.Tensor or torch.Tensor
			Shape: [modalities][batch,m_dim] or [batch,modalities,m_dim]

		Returns
		-------
		loss : torch.Tensor
		"""
		# Get missingness pattern, replace with zeros to prevent NaN gradients.
		nan_mask = get_nan_mask(x)
		if type(x) == type([]): # not vectorized, shape: [m][batch]
			for i in range(len(x)):
				x[i][nan_mask[i]] = 0.0
		else: # vectorized modalities, shape: [batch,m]
			x[nan_mask.unsqueeze(-1)] = 0.0
		# Encode.
		z_samples, log_qz, log_pz = self.encode(x, nan_mask) # [b,s,z], [b,s]
		assert log_pz.shape[1] == 1 # assert single sample
		log_pz = log_pz.sum(dim=1) # [b] Sum over sample dimension.
		# Evaluate KL.
		kld = self.variational_posterior.kld(self.prior).sum(dim=1) # [b]
		# Decode.
		log_likes = self.decode(z_samples, x, nan_mask) # [b,1]
		# assert log_likes.shape[1] == 1
		log_likes = log_likes.sum(dim=1) # sum over sample dimension
		# Evaluate loss.
		assert len(log_pz.shape) == 1
		assert len(log_likes.shape) == 1
		assert len(kld.shape) == 1
		loss = -torch.mean(log_pz + log_likes - kld)
		return loss



class IwaeElbo(VaeObjective):

	def __init__(self, vae, k):
		super(IwaeElbo, self).__init__(vae)
		self.k = k

	def forward(self, x):
		"""
		Evaluate the multisample IWAE ELBO.

		Parameters
		----------
		x : torch.Tensor
		"""
		# Get missingness pattern, replace with zeros to prevent NaN gradients.
		nan_mask = get_nan_mask(x)
		if type(x) == type([]): # not vectorized, shape: [m][batch]
			for i in range(len(x)):
				x[i][nan_mask[i]] = 0.0
		else: # vectorized modalities, shape: [batch,m]
			x[nan_mask.unsqueeze(-1)] = 0.0
		# Encode.
		# [b,s,z], [b,s], [b,s]
		z_samples, log_qz, log_pz = self.encode(x, nan_mask, n_samples=self.k)
		assert log_pz.shape[1] == self.k # assert k samples
		# log_pz = log_pz.sum(dim=1) # [b] Sum over sample dimension
		# Decode.
		log_likes = self.decode(z_samples, x, nan_mask) # [b,1]
		# assert log_likes.shape[1] == 1
		# log_likes = log_likes.sum(dim=1) # sum over sample dimension
		ws = log_pz + log_likes - log_qz - np.log(self.k)
		print("ws", ws.shape)
		quit()
		# Evaluate loss.
		assert len(log_pz.shape) == 1
		assert len(log_likes.shape) == 1
		assert len(kld.shape) == 1
		loss = -torch.mean(log_pz + log_likes - kld)
		return loss



class DregIwaeElbo(VaeObjective):

	def __init__(self, vae, k):
		super(DregIwaeElbo, self).__init__(vae)
		self.k = k

	def forward(self, x):
		"""
		Evaluate the multisample IWAE ELBO with the DReG estimator.

		Parameters
		----------
		x : torch.Tensor
		"""
		raise NotImplementedError



class WuGoodmanElbo(VaeObjective):

	def __init__(self, vae):
		super(WuGoodmanElbo, self).__init__(vae)

	def forward(self, x):
		"""
		Evaluate the subsampling ELBO from Wu & Goodman (2018?).

		Parameters
		----------
		x : torch.Tensor
		"""
		raise NotImplementedError



class SnisElbo(VaeObjective):

	def __init__(self, vae):
		super(SnisElbo, self).__init__(vae)

	def forward(self, x):
		"""
		Evaluate a self-normalized importance sampling ELBO.

		Parameters
		----------
		x : torch.Tensor
		"""
		raise NotImplementedError



class ArElbo(VaeObjective):

	def __init__(self, vae):
		super(ArElbo, self).__init__(vae)

	def forward(self, x):
		"""
		Evaluate an autoregressive ELBO.

		Parameters
		----------
		x : torch.Tensor
		"""
		raise NotImplementedError



def get_nan_mask(xs):
	"""Return a mask indicating which minibatch items are NaNs."""
	if type(xs) == type([]):
		return [torch.isnan(x[:,0]) for x in xs]
	else:
		return torch.isnan(xs[:,:,0]) # [b,m]


if __name__ == '__main__':
	pass



###
