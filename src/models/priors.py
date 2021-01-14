"""
Define latent priors.

"""
__date__ = "January 2021"


import numpy as np
import torch



class StandardNormalPrior():

	def __init__(self, z_dim):
		"""
		Standard normal prior distribution.

		Parameters
		----------
		z_dim : int
		"""
		self.z_dim = z_dim
		self.constant = None # TO DO


	def rsample(self):
		""" """
		return torch.randn()


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
