"""
Define latent priors.

"""
__date__ = "January 2021"


import numpy as np
import torch



class StandardGaussianPrior(torch.nn.Module):
	n_parameters = 0
	magic_constant = None # TO DO

	def __init__(self):
		"""Standard Gaussian prior distribution."""
		super(StandardGaussianPrior, self).__init__()

	def forward(self, x):
		""" """
		raise NotImplementedError

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
