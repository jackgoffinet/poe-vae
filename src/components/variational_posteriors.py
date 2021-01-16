"""
Define variational posteriors.

"""
__date__ = "January 2021"


import torch



class DiagonalGaussianPosterior(torch.nn.Module):
	n_parameters = 2 # mean and precision vectors
	parameter_dim_func = lambda d: (d,d) # latent_dim -> parameter dimensions
	magic_constant = None # TO DO

	def __init__(self):
		"""Diagonal Normal varitional posterior."""
		super(DiagonalGaussianPosterior, self).__init__()

	def forward(self, mean, log_precision):
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
