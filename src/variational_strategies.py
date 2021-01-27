"""
Define strategies for combining evidence into variational distributions.

These strategies all subclass torch.nn.Module. Their job is to convert
parameter values straight out of the encoder into a variational posterior,
combining evidence across the different modalities in some way.

TO DO
-----
* The GaussianPoeStrategy assumes a unit normal prior. Generalize this.
"""
__date__ = "January 2021"


import torch
import torch.nn.functional as F



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

	def __init__(self, args):
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
			precisions = torch.exp(precisions) # [b,m,z]
		else:
			precisions = log_precisions.exp()
		if nan_mask is not None:
			if list_flag:
				temp_mask = torch.stack(nan_mask, dim=1)
			else:
				temp_mask = nan_mask
			assert len(precisions.shape) == 3
			temp_mask = (~temp_mask).float().unsqueeze(-1)
			temp_mask = temp_mask.expand(-1,-1,precisions.shape[2])
			precisions = precisions * temp_mask
		# Combine all the experts. Include the 1.0 for the prior expert!
		precision = torch.sum(precisions, dim=1) + 1.0 # [b,z_dim]
		prec_mean = torch.sum(means * precisions, dim=1) # [b,z_dim]
		mean = prec_mean / (precision + self.EPS)
		return mean, precision



class GaussianMoeStrategy(torch.nn.Module):

	def __init__(self, args):
		"""
		Gaussian mixture of experts strategy

		"""
		super(GaussianMoeStrategy, self).__init__()


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
		list_flag = type(means) == type([]) # not vectorized
		if list_flag:
			means = torch.stack(means, dim=1) # [b,m,z]
			precisions = torch.stack(log_precisions, dim=1)
			precisions = torch.exp(precisions) # [b,m,z]
		else:
			precisions = log_precisions.exp()
		# Where things are NaNs, replace mixture components with the prior.
		if nan_mask is not None:
			if list_flag:
				temp_mask = torch.stack(nan_mask, dim=1)
			else:
				temp_mask = nan_mask
			assert len(precisions.shape) == 3
			temp_mask = (~temp_mask).float().unsqueeze(-1)
			temp_mask = temp_mask.expand(-1,-1,precisions.shape[2])
			precisions = precisions * temp_mask
			means = means * temp_mask
		precisions = precisions + 1.0 # Add the prior expert.
		return means, precisions



class VmfPoeStrategy(AbstractVariationalStrategy):
	EPS = 1e-5

	def __init__(self, args):
		"""
		von Mises Fisher product of experts strategy

		Parameters
		----------
		...
		"""
		super(VmfPoeStrategy, self).__init__()


	def forward(self, locs, scales, nan_mask=None):
		"""
		scale is the \kappa parameter in the vMF

		Parameters
		----------
		locs : list of torch.Tensor
			Shape: [b,m,d*n_vmf]
		scales : list of torch.Tensor
			Shape: [b,m,n_vmf]
		nan_mask : torch.Tensor or list of torch.Tensor
			Indicates where data is missing.

		Returns
		-------
		loc : torch.Tensor
			Shape: [batch, z_dim]
		scale : torch.Tensor
			Shape: [batch, z_dim]
		"""
		list_flag = type(locs) == type([]) # not vectorized
		if list_flag:
			locs = torch.stack(locs, dim=1) # [b,m,d*n_vmf]
			scales = torch.stack(scales, dim=1) # [b,m,n_vmf]
		scales = F.softplus(scales) # [b,m,n_vmf]
		assert len(locs.shape) == 3 and len(scales.shape) == 3
		assert locs.shape[-1] % scales.shape[-1] == 0
		d = locs.shape[-1] // scales.shape[-1]
		locs = locs.view(scales.shape[:3]+(d,)) # [b,m,n_vmf,d]
		scales = scales.unsqueeze(-1) # [b,m,n_vmf,1]
		if nan_mask is not None:
			if list_flag:
				temp_mask = torch.stack(nan_mask, dim=1) # [b,m]
			else:
				temp_mask = nan_mask # [b,m]
			temp_mask = (~temp_mask).float().unsqueeze(-1).unsqueeze(-1)
			scales = scales * temp_mask.expand(-1,-1,scales.shape[2],-1)
		# Combine all the experts.
		kappa_mus = torch.sum(locs * scales,dim=1) # [b,n_vmf,d]
		kappa = kappa_mus.norm(dim=-1, keepdim=True) # [b,n_vmf,1]
		mus = kappa_mus / (kappa + self.EPS) # [b,n_vmf,d]
		return mus, kappa



class EbmStrategy(AbstractVariationalStrategy):
	EPS = 1e-5

	def __init__(self, args):
		"""
		EBM strategy: just pass the energy function parameters to the EBM.

		Parameters
		----------
		...
		"""
		super(EbmStrategy, self).__init__()

	def forward(self, thetas, nan_mask=None):
		"""


		Parameters
		----------
		thetas (vectorized): list of torch.Tensor
			Shape: ???
		thetas (non-vectorized): list of torch.Tensor
			Shape: [m][b,theta_dim]
		nan_mask : torch.Tensor or list of torch.Tensor
			Indicates where data is missing.

		Returns
		-------
		thetas : torch.Tensor
			Passed from input.
		nan_mask : torch.Tensor
			Passed from input.
		"""
		list_flag = type(thetas) == type([]) # not vectorized
		if list_flag:
			thetas = torch.stack(thetas, dim=1)
			nan_mask = torch.stack(nan_mask, dim=1)
		return thetas, nan_mask



class LocScaleEbmStrategy(AbstractVariationalStrategy):
	EPS = 1e-5

	def __init__(self, args):
		"""
		Location/Scale EBM strategy: multiply the Gaussian proposals

		The other EBM parameters, the thetas, are simply passed. The NaN mask
		is also passed.

		Parameters
		----------
		...
		"""
		super(LocScaleEbmStrategy, self).__init__()

	def forward(self, thetas, means, log_precisions, nan_mask=None):
		"""
		Multiply the Gaussian proposals. The other EBM parameters, the thetas,
		are simply passed. The NaN mask is also passed.

		Parameters
		----------
		thetas (vectorized): list of torch.Tensor
			Shape: ???
		thetas (non-vectorized): list of torch.Tensor
			Shape: [m][b,theta_dim]
		means : list of torch.Tensor
			means[modality] shape: [batch, z_dim]
		log_precisions : list of torch.Tensor
			log_precisions[modality] shape: [batch, z_dim]
		nan_mask : torch.Tensor or list of torch.Tensor
			Indicates where data is missing.

		Returns
		-------
		thetas : torch.Tensor
			Shape: [batch, m, theta_dim]
		mean : torch.Tensor
			Shape: [batch, z_dim]
		precision : torch.Tensor
			Shape: [batch, z_dim]
		means : torch.Tensor
			Shape: [batch, m, z_dim]
		precisions : torch.Tensor
			Shape: [batch, m, z_dim]
		nan_mask : torch.Tensor
			Shape : [b,m]
		"""
		list_flag = type(means) == type([]) # not vectorized
		if list_flag:
			thetas = torch.stack(thetas, dim=1)
			nan_mask = torch.stack(nan_mask, dim=1)
			means = torch.stack(means, dim=1) # [b,m,z]
			log_precisions = torch.stack(log_precisions, dim=1)
			precisions = log_precisions.exp() # [b,m,z]
		else:
			precisions = log_precisions.exp()
		if nan_mask is not None:
			assert len(precisions.shape) == 3
			temp_mask = (~nan_mask).float().unsqueeze(-1)
			temp_mask = temp_mask.expand(-1,-1,precisions.shape[2])
			precisions = precisions * temp_mask
		# Combine all the experts. Include the 1.0 for the prior expert!
		precision = torch.sum(precisions, dim=1) + 1.0 # [b,z_dim]
		prec_mean = torch.sum(means * precisions, dim=1) # [b,z_dim]
		mean = prec_mean / (precision + self.EPS)
		return thetas, mean, precision, means, precisions, nan_mask



if __name__ == '__main__':
	pass



###
