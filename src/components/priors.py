"""
Define latent priors.

"""
__date__ = "January 2021"


import numpy as np
import torch
from torch.distributions import Normal



class AbstractPrior(torch.nn.Module):
	"""Abstract prior distribution class"""

	def __init__(self):
		super(AbstractPrior, self).__init__()

	def forward(self, samples):
		"""
		Evaluate log probability of the samples under the prior.

		Parameters
		----------
		samples : torch.Tensor
			Shape: [batch,n_samples,z_dim]

		Returns
		-------
		log_prob : torch.Tensor
			Shape: [batch,n_samples]
		"""
		raise NotImplementedError



class StandardGaussianPrior(AbstractPrior):
	n_parameters = 0


	def __init__(self):
		"""Standard Gaussian prior distribution."""
		super(StandardGaussianPrior, self).__init__()
		self.dist = None
		self.dim = -1

	def forward(self, x):
		""" """
		if x.shape[-1] != self.dim or self.dist is None:
			loc = torch.zeros(x.shape[-1], device=x.device)
			scale = torch.ones_like(loc)
			self.dist = Normal(loc, scale)
		log_prob = self.dist.log_prob(x)
		return torch.sum(log_prob, dim=2)


	def rsample(self):
		""" """
		raise NotImplementedError

	def log_prob(self, samples):
		"""

		Parameters
		----------
		samples : torch.Tensor [...]
		"""
		raise NotImplementedError



if __name__ == '__main__':
	pass



###
