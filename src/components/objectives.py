"""
Different objectives for VAEs.

Objectives play a central role in this VAE abstraction. They take a VAE object,
which isn't anything more than a collection of various modular parts, and, given
a batch of data, determine how to route the data through these parts to
calculate a loss.

Defining objectives as Modules instead of functions leaves the door open for the
objectives to have trainable parameters and other state. But for the objectives
so far, we don't need this.
"""
__date__ = "January 2021"


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
		nan_mask = _get_nan_mask(x)
		for i in range(len(x)):
			x[i][nan_mask[i]] = 0.0
		# Encode data.
		var_dist_params = self.encoder(x)
		# Combine evidence.
		var_post_params = self.variational_strategy(*var_dist_params, \
				nan_mask=nan_mask)
		# Make a variational posterior and sample.
		z_samples, log_qz = self.variational_posterior(*var_post_params)
		# Evaluate prior.
		log_pz = self.prior(z_samples).sum(dim=1)
		# Evaluate KL.
		kld = self.variational_posterior.kld(self.prior).sum(dim=1)
		# Decode samples to get likelihood parameters.
		likelihood_params = self.decoder(z_samples)
		# Evaluate likelihoods, sum over modalities.
		log_likes = self.likelihood(x, likelihood_params, \
				nan_mask=nan_mask)
		log_likes = sum(log_like.sum(dim=1) for log_like in log_likes)
		# Evaluate loss.
		assert len(log_pz.shape) == 1 and len(log_likes.shape) == 1 and len(kld.shape) == 1
		loss = -torch.mean(log_pz + log_likes - kld)
		return loss



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



def _get_nan_mask(xs):
	"""Return a mask indicating which minibatch items are NaNs."""
	if type(xs) == type([]):
		return [torch.isnan(x[:,0]) for x in xs]
	else:
		raise NotImplementedError



if __name__ == '__main__':
	pass



###
