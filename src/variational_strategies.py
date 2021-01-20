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
		list_flag = type(means) == type([]) # not vectorized
		if list_flag:
			means = torch.stack(means, dim=1) # [b,m,z]
			precisions = torch.stack(log_precisions, dim=1)
			# precisions = torch.nn.functional.softplus(precisions) # [b,m,z]
			precisions = torch.exp(precisions) # [b,m,z]
		else:
			precisions = log_precisions.exp()
		# print("mean prec:", torch.mean(precisions), torch.min(precisions), torch.max(precisions))
		if nan_mask is not None:
			if list_flag:
				temp_mask = torch.stack(nan_mask, dim=1)
			else:
				temp_mask = nan_mask
			assert len(precisions.shape) == 3
			temp_mask = (~temp_mask).float().unsqueeze(-1)
			temp_mask = temp_mask.expand(-1,-1,precisions.shape[2])
			precisions = precisions * temp_mask
		# print("precisions", precisions.shape)
		# quit()
		# Combine all the experts. Include the 1.0 for the prior expert!
		precision = torch.sum(precisions, dim=1) + 1.0 # [b,z_dim]
		prec_mean = torch.sum(means * precisions, dim=1) # [b,z_dim]
		mean = prec_mean / (precision + self.EPS)
		return mean, precision



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
