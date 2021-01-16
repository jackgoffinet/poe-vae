"""
Define strategies for combining evidence into variational distributions.

These strategies all subclass torch.nn.Module. Their job is to convert
parameter values straight out of the encoder into a variational posterior,
combining evidence across the different modalities in some way.
"""
__date__ = "January 2021"


import torch



class GaussianPoeStrategy(torch.nn.Module):

	def __init__(self):
		"""
		Gaussian product of experts strategy

		Parameters
		----------
		...
		"""
		super(GaussianPoeStrategy, self).__init__()
		pass


	def forward(self, means, log_precisions):
		"""
		Given means and log precisions, output the product mean and precision.

		Parameters
		----------
		means : list of torch.Tensor
			means[modality] = [...fill in dimensions...]
		log_precisions : list of torch.Tensor

		Returns
		-------
		mean : torch.Tensor
		precision : torch.Tensor
		"""
		raise NotImplementedError



class GaussianMoeStrategy(torch.nn.Module):

	def __init__(self):
		"""
		Gaussian mixture of experts strategy

		"""
		super(GaussianPoeStrategy, self).__init__()
		pass


	def forward(self, means, log_precisions):
		"""
		Given means and log precisions, output mixture parameters.

		Parameters
		----------
		means : list of torch.Tensor
			means[modality] = [...fill in dimensions...]
		log_precisions : list of torch.Tensor
		
		Returns
		-------
		"""
		raise NotImplementedError



if __name__ == '__main__':
	pass



###
