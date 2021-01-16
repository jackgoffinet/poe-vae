"""
Define observation likelihoods.

"""
__date__ = "January 2021"


import numpy as np
import torch


class SphericalGaussianLikelihood(torch.nn.Module):
	n_parameters = 1 # mean parameter
	parameter_dim_func = lambda d: (d,) # latent_dim -> parameter dimensions
	magic_constant = None # TO DO


	def __init__(self, std_dev):
		"""
		Spherical Gaussian likelihood distribution.

		"""
		super(SphericalGaussianLikelihood, self).__init__()
		self.std_dev = std_dev


	def forward(self, samples):
		"""
		Evaluate log probability of samples.

		Parameters
		----------
		samples : torch.Tensor [...]
		"""
		raise NotImplementedError



if __name__ == '__main__':
	pass



###
