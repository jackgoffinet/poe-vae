"""
Define strategies for combining evidence into variational distributions.

These strategies all subclass torch.nn.Module. Their job is to convert
parameter values straight out of the encoder into a variational posterior,
combining evidence across the different modalities in some way.
"""
__date__ = "January 2021"


import torch



class GaussianPoeStrategy(torch.nn.Module):
	EPS = 1e-5

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
			means[modality] shape: [batch, z_dim]
		log_precisions : list of torch.Tensor
			log_precisions[modality] shape: [batch, z_dim]

		Returns
		-------
		mean : torch.Tensor
			Shape: [batch, z_dim]
		precision : torch.Tensor
			Shape: [batch, z_dim]
		"""
		means = torch.stack(means, dim=1) # [b,m,z]
		precisions = torch.stack(log_precisions, dim=1).exp() # [b,m,z]
		precision = torch.sum(precisions, dim=1)
		mean = torch.sum(means * precisions, dim=1) / (precision + self.EPS)
		return mean.squeeze(1), precision.squeeze(1)



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
