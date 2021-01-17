"""
Define strategies for combining evidence into variational distributions.

These strategies all subclass torch.nn.Module. Their job is to convert
parameter values straight out of the encoder into a variational posterior,
combining evidence across the different modalities in some way.
"""
__date__ = "January 2021"


import torch



class AbstractVariationalStrategy(torch.nn.Module):
	"""Abstract variational strategy class"""

	def __init__(self):
		super(AbstractVariationalStrategy, self).__init__()

	def forward(self, *modality_params, nan_mask=None):
		"""
		Combine the information from each modality into prior parameters.

		Parameters
		----------
		modality_params : ...
		nan_mask : torch.Tensor or list of torch.Tensor
			Indicates where data is missing.

		Returns
		-------
		prior_parameters : ...
		"""
		raise NotImplementedError



class GaussianPoeStrategy(AbstractVariationalStrategy):
	EPS = 1e-5

	def __init__(self):
		"""
		Gaussian product of experts strategy

		Parameters
		----------
		...
		"""
		super(GaussianPoeStrategy, self).__init__()


	def forward(self, means, log_precisions, nan_mask=None):
		"""
		Given means and log precisions, output the product mean and precision.

		Parameters
		----------
		means : list of torch.Tensor
			means[modality] shape: [batch, z_dim]
		log_precisions : list of torch.Tensor
			log_precisions[modality] shape: [batch, z_dim]
		nan_mask : torch.Tensor or list of torch.Tensor
			Indicates where data is missing.

		Returns
		-------
		mean : torch.Tensor
			Shape: [batch, z_dim]
		precision : torch.Tensor
			Shape: [batch, z_dim]
		"""
		means = torch.stack(means, dim=1) # [b,m,z]
		precisions = torch.stack(log_precisions, dim=1).exp() # [b,m,z]
		if nan_mask is not None:
			temp_mask = (~torch.stack(nan_mask, dim=1)).float().unsqueeze(-1)
			precisions = precisions * temp_mask
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


	def forward(self, means, log_precisions, nan_mask=None):
		"""
		Given means and log precisions, output mixture parameters.

		Parameters
		----------
		means : list of torch.Tensor
			means[modality] = [...fill in dimensions...]
		log_precisions : list of torch.Tensor
		nan_mask : torch.Tensor or list of torch.Tensor
			Indicates where data is missing.

		Returns
		-------
		"""
		raise NotImplementedError



if __name__ == '__main__':
	pass



###
