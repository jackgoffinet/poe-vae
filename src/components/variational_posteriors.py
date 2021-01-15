"""
Define variational posteriors.

"""
__date__ = "January 2021"


import torch



class DiagonalGaussianPosterior():

	def __init__(self, log_precisions):
		"""
		Diagonal Normal varitional posterior.

		Parameters
		----------
		log_precisions : torch.Tensor
		"""
		self.log_precisions = log_precisions


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
