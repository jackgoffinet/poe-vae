"""
Define strategies for combining evidence into variational distributions.

These strategies all subclass `torch.nn.Module`. Their job is to convert
parameter values straight out of the encoder into a variational posterior,
combining evidence across the different modalities in some way.

TO DO
-----
* The GaussianPoeStrategy assumes a unit normal prior. Generalize this.
"""
__date__ = "January - May 2021"


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
		nan_mask : torch.Tensor
			Indicates where data is missing.
			Shape: [b,m]

		Returns
		-------
		prior_parameters : ...
		"""
		raise NotImplementedError



class GaussianPoeStrategy(AbstractVariationalStrategy):
	EPS = 1e-5

	def __init__(self, **kwargs):
		"""
		Gaussian product of experts strategy

		Note
		----
		* Assumes a standard normal prior!

		"""
		super(GaussianPoeStrategy, self).__init__()


	def forward(self, means, log_precisions, nan_mask=None):
		"""
		Given means and log precisions, output the product mean and precision.

		Parameters
		----------
		means : torch.Tensor or list of torch.Tensor
			Shape:
				[batch,modality,z_dim] if vectorized
				[modality][batch,z_dim] otherwise
		log_precisions : torch.Tensor or list of torch.Tensor
			Shape:
				[batch,modality,z_dim] if vectorized
				[modality][batch,z_dim] otherwise
		nan_mask : torch.Tensor
			Indicates where data is missing.
			Shape: [batch,modality]

		Returns
		-------
		mean : torch.Tensor
			Shape: [batch, z_dim]
		precision : torch.Tensor
			Shape: [batch, z_dim]
		"""
		if isinstance(means, (tuple,list)): # not vectorized
			means = torch.stack(means, dim=1) # [b,m,z]
			log_precisions = torch.stack(log_precisions, dim=1) # [b,m,z]
		precisions = torch.exp(log_precisions) # [b,m,z]
		if nan_mask is not None:
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

	def __init__(self, **kwargs):
		"""
		Gaussian mixture of experts strategy

		Note
		----
		* Assumes a standard normal prior!

		"""
		super(GaussianMoeStrategy, self).__init__()


	def forward(self, means, log_precisions, nan_mask=None):
		"""
		Given means and log precisions, output mixture parameters.

		Parameters
		----------
		means : torch.Tenosr or tuple of torch.Tensor
			Shape:
				[b,m,z] if vectorized
				[m][b,z] otherwise
		log_precisions : torch.Tensor ot tuple of torch.Tensor
			Shape:
				[b,m,z] if vectorized
				[m][b,z] otherwise
		nan_mask : torch.Tensor
			Indicates where data is missing.
			Shape: [batch,modality]

		Returns
		-------
		mean : torch.Tensor
			Shape: [batch, m, z_dim]
		precision : torch.Tensor
			Shape: [batch, m, z_dim]
		"""
		tuple_flag = isinstance(means, (tuple,list)) # not vectorized
		if tuple_flag:
			means = torch.stack(means, dim=1) # [b,m,z]
			log_precisions = torch.stack(log_precisions, dim=1)
		precisions = torch.exp(log_precisions) # [b,m,z]
		# Where things are NaNs, replace mixture components with the prior.
		if nan_mask is not None:
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

	def __init__(self, n_vmfs=5, vmf_dim=4, **kwargs):
		"""
		von Mises Fisher product of experts strategy

		Parameters
		----------
		...
		"""
		super(VmfPoeStrategy, self).__init__()
		self.n_vmfs = n_vmfs
		self.vmf_dim = vmf_dim


	def forward(self, kappa_mus, nan_mask=None):
		"""

		Parameters
		----------
		kappa_mus : torch.Tensor or list of torch.Tensor
			Shape:
				[b,m,n_vmfs*(vmf_dim+1)] if vectorized
				[m][b,n_vmfs*(vmf_dim+1)] otherwise
		nan_mask : torch.Tensor
			Indicates where data is missing.
			Shape: [b,m]

		Returns
		-------
		kappa_mu : tuple of torch.Tensor
			Shape: [1][b,n_vmfs,vmf_dim+1]
		"""
		tuple_flag = isinstance(kappa_mus, tuple) # not vectorized
		if tuple_flag:
			kappa_mus = torch.stack(kappa_mus, dim=1) # [b,m,n_vmf*(vmf_dim+1)]
		assert len(kappa_mus.shape) == 3, f"len({kappa_mus.shape}) != 3"
		assert kappa_mus.shape[2] == self.n_vmfs * (self.vmf_dim+1)
		new_shape = kappa_mus.shape[:2]+(self.n_vmfs, self.vmf_dim+1)
		kappa_mus = kappa_mus.view(new_shape) # [b,m,n_vmfs,vmf_dim+1]
		if nan_mask is not None:
			temp_mask = nan_mask # [b,m]
			temp_mask = (~temp_mask).float().unsqueeze(-1).unsqueeze(-1)
			temp_mask = temp_mask.expand(
					-1,
					-1,
					kappa_mus.shape[2],
					kappa_mus.shape[3],
			) # [b,m,n_vmfs,vmf_dim+1]
			kappa_mus = kappa_mus * temp_mask
		# Combine all the experts.
		kappa_mu = torch.sum(kappa_mus, dim=1) # [b,n_vmfs,vmf_dim+1]
		return (kappa_mu,)



class LocScaleEbmStrategy(AbstractVariationalStrategy):
	EPS = 1e-5

	def __init__(self, **kwargs):
		"""
		Location/Scale EBM strategy: multiply the Gaussian proposals

		The other EBM parameters, the thetas, are simply passed. The NaN mask
		is also passed.

		"""
		super(LocScaleEbmStrategy, self).__init__()

	def forward(self, thetas, means, log_precisions, nan_mask=None):
		"""
		Multiply the Gaussian proposals.

		The other EBM parameters, the thetas,
		are simply passed. The NaN mask is also passed.

		Parameters
		----------
		thetas: torch.Tensor or tuple of torch.Tensor
			Shape:
				[b,m,theta_dim] if vectorized
				[m][b,theta_dim] otherwise
		means : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,m,z_dim] if vectorized
				[m][batch,z_dim] otherwise
		log_precisions : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,m,z_dim] if vectorized
				[m][batch,z_dim] otherwise
		nan_mask : torch.Tensor
			Indicates where data is missing.
			Shape: [b,m]

		Returns
		-------
		thetas : torch.Tensor
			Shape: [b,m,theta_dim]
		mean : torch.Tensor
			Shape: [b,z_dim]
		precision : torch.Tensor
			Shape: [b,z_dim]
		means : torch.Tensor
			Shape: [b,m,z_dim]
		precisions : torch.Tensor
			Shape: [b,m,z_dim]
		nan_mask : torch.Tensor
			Shape : [b,m]
		"""
		if isinstance(means, (tuple,list)):
			thetas = torch.stack(thetas, dim=1)
			means = torch.stack(means, dim=1) # [b,m,z]
			log_precisions = torch.stack(log_precisions, dim=1) # [b,m,z]
		precisions = log_precisions.exp() # [b,m,z]
		if nan_mask is not None:
			assert len(precisions.shape) == 3, f"len({precisions.shape}) != 3"
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
