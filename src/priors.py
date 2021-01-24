"""
Define latent priors.

"""
__date__ = "January 2021"


import numpy as np
import torch
from torch.distributions import Normal

from .distributions.hyperspherical_uniform import HypersphericalUniform



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

	def rsample(self, n_samples=1, n_batches=1):
		"""
		Get reparameterized samples from the prior.

		Parameters
		----------
		n_samples : int

		Returns
		-------
		samples : torch.Tensor
		"""
		raise NotImplementedError



class StandardGaussianPrior(AbstractPrior):
	# n_parameters = 0 # Not needed, right?

	def __init__(self, args):
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
		return torch.sum(log_prob, dim=-1)


	def rsample(self, n_samples=1, n_batches=1):
		"""
		Get reparameterized samples from the prior.

		Parameters
		----------
		n_samples : int

		Returns
		-------
		samples : torch.Tensor
			Shape: [batch,n_samples,z_dim]
		"""
		assert self.dist is not None
		return self.dist.rsample(sample_shape=(n_batches,n_samples))


	def log_prob(self, samples):
		"""

		Parameters
		----------
		samples : torch.Tensor [...]
		"""
		raise NotImplementedError



class UniformHypersphericalPrior(AbstractPrior):

	def __init__(self, args):
		"""Uniform hyperspherical prior distribution."""
		super(UniformHypersphericalPrior, self).__init__()
		self.dim = -1
		self.vmf_dim = args.vmf_dim # vMFs are defined on the (vmf_dim-1)-sphere
		self.latent_dim = args.latent_dim
		assert self.latent_dim % (self.vmf_dim - 1) == 0, \
				"Incompatible z_dim and vmf_dim!"
		self.n_vmfs = self.latent_dim // (self.vmf_dim - 1)
		self.dist = HypersphericalUniform(self.vmf_dim-1, device=args.device)

	def forward(self, x, keep_nvmf_dim=False):
		"""
		Evaluate log probability under the prior.

		Parameters
		----------
		x : torch.Tensor
			Shape: [b,s,n_vmf*d]
		keep_nvmf_dim : bool

		Returns
		-------
		log_prob : torch.Tensor
			Shape: [b,s,n_vmf] if `keep_nvmf_dim`, else: [b,s]
		"""
		x = x.view(x.shape[:2]+(self.n_vmfs,self.vmf_dim))
		log_prob = self.dist.log_prob(x)
		if not keep_nvmf_dim:
			log_prob = log_prob.sum(dim=-1)
		return log_prob


	def rsample(self, n_samples=1, n_batches=1):
		"""
		Get reparameterized samples from the prior.

		Parameters
		----------
		n_samples : int

		Returns
		-------
		samples : torch.Tensor
			Shape: [batch,n_samples,z_dim]
		"""
		assert self.dist is not None
		sample_shape = torch.Size([n_batches,n_samples,self.n_vmfs])
		samples = self.dist.sample(shape=sample_shape)
		return samples.view(n_batches, n_samples, -1)


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
