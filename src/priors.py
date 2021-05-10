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
			Shape: ???
		"""
		raise NotImplementedError



class StandardGaussianPrior(AbstractPrior):

	def __init__(self, device='cpu', latent_dim=20, **kwargs):
		"""
		Standard Gaussian prior distribution.

		Parameters
		----------
		device : str or torch.device
		latent_dim : int, optional
		"""
		super(StandardGaussianPrior, self).__init__()
		self.latent_dim = latent_dim
		self.dist = Normal(
			torch.zeros(latent_dim, device=device),
			torch.ones(latent_dim, device=device),
		)


	def forward(self, x):
		""" """
		return self.dist.log_prob(x).sum(dim=-1)


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
		return self.dist.rsample(sample_shape=(n_batches,n_samples))


	def log_prob(self, samples):
		"""

		Parameters
		----------
		samples : torch.Tensor [...]
		"""
		raise NotImplementedError



class UniformHypersphericalPrior(AbstractPrior):

	def __init__(self, device='cpu', n_vmfs=5, vmf_dim=4, **kwargs):
		"""
		Uniform hyperspherical prior distribution.

		Parameters
		-----------
		device : str or torch.device
		n_vmfs : int, optional
		vmf_dim : int, optional
		"""
		super(UniformHypersphericalPrior, self).__init__()
		self.n_vmfs = n_vmfs
		self.vmf_dim = vmf_dim
		self.dist = HypersphericalUniform(self.vmf_dim-1, device=device)


	def forward(self, x):
		"""
		Evaluate log probability under the prior.

		Parameters
		----------
		x : torch.Tensor
			Shape: [b,s,n_vmfs*(vmf_dim+1)]

		Returns
		-------
		log_prob : torch.Tensor
			Shape: [b,s]
		"""
		x = x.view(x.shape[:2]+(self.n_vmfs,self.vmf_dim+1)) #
		return self.dist.log_prob(x).sum(dim=-1) # [b,s]


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
		raise NotImplementedError
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
