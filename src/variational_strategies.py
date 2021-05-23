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


	def forward(self, means, log_precisions, nan_mask=None, collapse=True):
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
		collapse : bool, optional
			Whether to collapse across modalities.

		Returns
		-------
		if `collapse`:
			prec_mean : torch.Tensor
				Shape: [batch, z_dim]
			precision : torch.Tensor
				Shape: [batch, z_dim]
		else:
			prec_means : torch.Tensor
				Shape: [b,m,z]
			precisions : torch.Tensor
				Does not include the prior expert!
				Shape: [b,m,z]
		"""
		if isinstance(means, (tuple,list)): # not vectorized
			means = torch.stack(means, dim=1) # [b,m,z]
			log_precisions = torch.stack(log_precisions, dim=1) # [b,m,z]
		precisions = torch.exp(log_precisions) # [b,m,z]
		if nan_mask is not None:
			temp_mask = nan_mask
			assert len(precisions.shape) == 3, f"len({precisions.shape}) != 3"
			temp_mask = (~temp_mask).float().unsqueeze(-1)
			temp_mask = temp_mask.expand(-1,-1,precisions.shape[2])
			precisions = precisions * temp_mask
		prec_means = means * precisions
		if collapse:
			return self.collapse(prec_means, precisions)
		return prec_means, precisions


	def collapse(self, prec_means, precisions, include_prior=True):
		"""
		Collapse across modalities, combining evidence.

		Parameters
		----------
		prec_means : torch.Tensor
			Shape: [b,m,z]
		precisions : torch.Tensor
			Shape: [b,m,z]
		include_prior : bool, optional
			Whether to include the effect of the prior expert.

		Returns
		-------
		prec_mean : torch.Tensor
			Shape: [b,z]
		precision : torch.Tensor
			Shape: [b,z]
		"""
		precision = torch.sum(precisions, dim=1) # [b,m,z] -> [b,z]
		if include_prior:
			precision = precision + 1.0
		prec_mean = torch.sum(prec_means, dim=1) # [b,m,z] -> [b,z]
		return prec_mean, precision



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
		# Where modalities are missing, sample from the prior.
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
		n_vmfs : int, optional
		vmf_dim : int, optional
		"""
		super(VmfPoeStrategy, self).__init__()
		self.n_vmfs = n_vmfs
		self.vmf_dim = vmf_dim


	def forward(self, kappa_mus, nan_mask=None):
		"""
		Multiply the vMF's given by the kappa_mus.

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
		assert kappa_mus.shape[2] == self.n_vmfs * (self.vmf_dim+1), \
				f"error: {kappa_mus.shape}, {self.n_vmfs}, {self.vmf_dim}"
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

		"""
		super(LocScaleEbmStrategy, self).__init__()


	def forward(self, thetas, means, log_precisions, nan_mask=None, \
		collapse=True):
		"""
		Mostly just pass the parameters and apply NaN mask.

		Parameters
		----------
		thetas: torch.Tensor or tuple of torch.Tensor
			Describes deviations from the Gaussian proposal
			Shape:
				[b,m,theta_dim] if vectorized
				[m][b,theta_dim] otherwise
		means : torch.Tensor or tuple of torch.Tensor
			Means of the Gaussian proposals
			Shape:
				[batch,m,z_dim] if vectorized
				[m][batch,z_dim] otherwise
		log_precisions : torch.Tensor or tuple of torch.Tensor
			log precisions of the Gaussian proposals
			Shape:
				[batch,m,z_dim] if vectorized
				[m][batch,z_dim] otherwise
		nan_mask : torch.Tensor
			Indicates where data is missing.
			Shape: [b,m]
		collapse : bool, optional
			Doesn't do anything. Here because AR-ELBO expects it.

		Returns
		-------
		thetas : torch.Tensor
			Shape: [b,m,theta_dim]
		means : torch.Tensor
			Shape: [b,m,z]
		prec_means : torch.Tensor
			Shape: [b,m,z]
		precisions : torch.Tensor
			Shape: [b,m,z]
		nan_mask : torch.Tensor
			Shape : [b,m]
		"""
		if isinstance(means, (tuple,list)):
			thetas = torch.stack(thetas, dim=1) # [b,m,theta]
			means = torch.stack(means, dim=1) # [b,m,z]
			log_precisions = torch.stack(log_precisions, dim=1) # [b,m,z]
		precisions = log_precisions.exp() # [b,m,z]
		precisions = torch.clamp(precisions, max=50.0)
		if nan_mask is not None:
			assert len(precisions.shape) == 3, f"len({precisions.shape}) != 3"
			temp_mask = (~nan_mask).float().unsqueeze(-1)
			temp_mask = temp_mask.expand(-1,-1,precisions.shape[2])
			precisions = precisions * temp_mask
		prec_means = means * precisions
		if torch.isnan(precisions).sum() > 0:
			print("LocScaleEbmStrategy NaN")
			print("prec_means", torch.isnan(prec_means).sum())
			print("thetas", torch.isnan(thetas).sum())
			print("means", torch.isnan(means).sum())
			print("precisions", torch.isnan(precisions).sum())
			print("log_precisions", torch.isnan(log_precisions).sum())
			print()
		return thetas, means, prec_means, precisions, nan_mask



if __name__ == '__main__':
	pass



###
