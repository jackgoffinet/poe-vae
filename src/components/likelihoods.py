"""
Define observation likelihoods.

TO DO
-----
* generalize to multi-parameter likelihoods, like in variational_posteriors.py
"""
__date__ = "January 2021"


import numpy as np
import torch


class SphericalGaussianLikelihood(torch.nn.Module):
	magic_constant = None # TO DO

	def __init__(self, std_dev):
		"""
		Spherical Gaussian likelihood distribution.

		"""
		super(SphericalGaussianLikelihood, self).__init__()
		self.std_dev = std_dev

	def forward(self, x):
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
