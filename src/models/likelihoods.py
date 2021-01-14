"""
Define observation likelihoods.

"""
__date__ = "January 2021"


import numpy as np
import torch


class SphericalNormalLikelihood():

	def __init__(self, x_dim):
		"""
		Spherical normal likelihood distribution.

		Parameters
		----------
		x_dim : int
		"""
		self.x_dim = x_dim
		self.constant = None # TO DO


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
