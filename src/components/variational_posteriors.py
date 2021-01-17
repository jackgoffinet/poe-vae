"""
Define variational posteriors.

"""
__date__ = "January 2021"


import torch
from torch.distributions import Normal



class AbstractVariationalPosterior(torch.nn.Module):
	"""Abstract class for variational posteriors."""

	def __init__(self):
		super(AbstractVariationalPosterior, self).__init__()
		self.dist = None

	def forward(self, *dist_parameters, n_samples=1):
		"""
		Produce reparamaterizable samples and evaluate their log probability.

		Parameters
		----------
		dist_parameters: tuple
			Distribution parameters, probably containing torch.Tensors.
		n_samples : int, optional
			Number of samples to draw.

		Returns
		-------
		samples : torch.Tensor
			Samples from the distribution. Shape: TO DO
		log_prob : torch.Tensor
			Log probability of samples under the distribution.
		"""
		raise NotImplementedError


	def kld(self, other):
		"""


		"""
		type_tuple = (type(self.dist), type(other.dist))
		if type_tuple in torch.distributions.kl._KL_REGISTRY:
			return torch.distributions.kl_divergence(self.dist, other.dist)



class DiagonalGaussianPosterior(AbstractVariationalPosterior):
	n_parameters = 2 # mean and precision vectors: TO DO needed?
	parameter_dim_func = lambda d: (d,d) # latent_dim -> parameter dimensions
	# CONSTANT = -0.5 * np.log(2 * np.pi)


	def __init__(self):
		"""Diagonal Normal varitional posterior."""
		super(DiagonalGaussianPosterior, self).__init__()


	def forward(self, mean, precision, n_samples=1):
		"""
		Produce reparamaterizable samples and evaluate their log probability.

		Parameters
		----------
		mean : torch.Tensor
			Shape [batch,z_dim]
		precision : torch.Tensor
			Shape [batch, z_dim]
		n_samples : int

		Returns
		-------
		samples : torch.Tensor
			Samples from the distribution. Shape: [batch,n_samples,z_dim]
		log_prob : torch.Tensor
			Log probability of samples under the distribution.
			Shape: [batch,n_samples,z_dim]
		"""
		assert mean.shape == precision.shape, \
				"{} != {}".format(mean.shape,precision.shape)
		assert len(mean.shape) == 2, \
				"len(mean.shape) == {}".format(len(mean).shape)
		std_dev = torch.sqrt(torch.reciprocal(precision))
		self.dist = Normal(mean, std_dev)
		samples = self.dist.rsample(sample_shape=(n_samples,))
		log_prob = self.dist.log_prob(samples)
		return samples.transpose(0,1), log_prob.transpose(0,1)


	def rsample(self):
		""" """
		raise NotImplementedError


	def log_prob(self, samples):
		"""
		...

		Parameters
		----------
		samples : torch.Tensor [...]
		"""
		raise NotImplementedError



if __name__ == '__main__':
	pass



###
