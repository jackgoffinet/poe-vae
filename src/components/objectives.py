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
		x : list of torch.Tensor

		Returns
		-------
		loss : torch.Tensor
		"""
		# Get missingness pattern, replace with zeros to prevent NaN gradients.
		nan_mask = get_nan_mask(x)
		if type(x) == type([]): # not vectorized
			for i in range(len(x)):
				x[i][nan_mask[i]] = 0.0
		else: # vectorized modalities
			x[nan_mask.unsqueeze(-1)] = 0.0
		# Encode.
		z_samples, _, log_pz = self.encode(x, nan_mask) # [b,1,z], [b,1]
		print("z_samples", z_samples.shape)
		print("log_pz", log_pz.shape)
		log_pz = log_pz.sum(dim=1) # [b] Sum over sample dimension.
		# Evaluate KL.
		kld = self.variational_posterior.kld(self.prior).sum(dim=1) # [b]
		print("kld", kld.shape)
		# Decode.
		log_likes = self.decode(z_samples, x, nan_mask) # [b,1]
		assert log_likes.shape[1] == 1
		log_likes = log_likes.sum(dim=1)
		# Evaluate loss.
		assert len(log_pz.shape) == 1
		assert len(log_likes.shape) == 1
		assert len(kld.shape) == 1
		loss = -torch.mean(log_pz + log_likes - kld)
		return loss


	def encode(self, x, nan_mask, n_samples=1):
		"""
		Encode data.

		Parameters
		----------
		x : list of torch.Tensor
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
			Shape: [batch,sampls,z_dim]

		"""
		# Decode samples to get likelihood parameters.
		likelihood_params = self.decoder(z_samples) # [m][l_params][b,s,x]
		# Evaluate likelihoods, sum over modalities.
		log_likes = self.likelihood(x, likelihood_params, \
				nan_mask=nan_mask)
		log_likes = sum(log_like for log_like in log_likes)
		return log_likes


	def estimate_log_marginal(self, x, k=1000):
		"""

		Parameters
		----------
		x : list of torch.Tensor
		k : int
		 	Number of importance-weighted samples.

		Returns
		-------
		est_log_m : torch.Tensor
			Estimated log marginal. Shape: [b]
		"""
		# Get missingness pattern, replace with zeros to prevent NaN gradients.
		nan_mask = get_nan_mask(x)
		for i in range(len(x)):
			x[i][nan_mask[i]] = 0.0
		# [b,k,z], [b,k], [b,k]
		z_samples, log_qz, log_pz = self.encode(x, nan_mask, n_samples=k)
		log_likes = self.decode(z_samples, x, nan_mask) # [b,k]
		est_log_m = torch.logsumexp(log_pz - log_qz + log_likes - np.log(k), \
				dim=1)
		return est_log_m



class IwaeElbo(VaeObjective):

	def __init__(self, vae):
		super(IwaeElbo, self).__init__(vae)

	def forward(self, x):
		"""
		Evaluate the multisample IWAE ELBO.

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
