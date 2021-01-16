"""
Different objectives for VAEs.

Objectives play a central role in this VAE abstraction. They take a VAE object,
which isn't much more than a collection of various modular parts, and, given a
batch of data, determine how to route the data through these parts to calculate
a loss.
"""
__date__ = "January 2021"


import torch



class VaeObjective(torch.nn.Module):
	"""Abstract VAE Objective class"""

	def __init__(self, vae):
		super(VaeObjective, self).__init__()
		self.vae = vae
		print("VaeObjective", sum(p.numel() for p in self.parameters() if p.requires_grad))
		print("VaeObjective encoder", sum(p.numel() for p in self.vae.encoder.parameters() if p.requires_grad))

	def forward(self, x):
		"""
		Evaluate loss on a minibatch x.

		Parameters
		----------
		x : torch.Tensor
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
		  with a single sample.

		Parameters
		----------
		x : torch.Tensor
		"""
		# Encode data.
		var_dist_params = self.vae.encoder(x)
		# Combine evidence.
		var_post_params = self.vae.variational_strategy(*var_dist_params)
		# Make a variational posterior and sample.
		z_samples, log_qz = self.vae.variational_posterior(*var_post_params)
		# Evaluate prior.
		log_pz = self.vae.prior(z_samples)
		# Decode samples to get likelihood parameters.
		likelihood_params = self.vae.decoder(z_samples)
		# Evaluate likelihood.
		log_like = self.vae.likelihood(x, likelihood_params)
		# Evaluate loss.
		loss = torch.mean(log_pz + log_like - kld)
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




if __name__ == '__main__':
	pass



###
