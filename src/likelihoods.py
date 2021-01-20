"""
Define observation likelihoods.

"""
__date__ = "January 2021"


import numpy as np
import torch
from torch.distributions import Normal



class AbstractLikelihood(torch.nn.Module):

	def __init__(self):
		super(AbstractLikelihood, self).__init__()

	def forward(self, sample, like_params, nan_mask=None):
		"""
		Evaluate log probability of samples.

		Parameters
		----------
		sample : torch.Tensor
			Shape: [...]
		like_params : torch.Tensor
			Likelihood parameters. Shape: [...]
		nan_mask : torch.Tensor or list of torch.Tensor
			Indicates where data is missing.

		Returns
		-------
		log_prob : torch.Tensor
			Shape: [...]
		"""
		raise NotImplementedError



class SphericalGaussianLikelihood(AbstractLikelihood):
	n_parameters = 1 # mean parameter
	parameter_dim_func = lambda d: (d,) # latent_dim -> parameter dimensions


	def __init__(self, std_dev):
		"""
		Spherical Gaussian likelihood distribution.

		"""
		super(SphericalGaussianLikelihood, self).__init__()
		self.std_dev = std_dev
		self.dist = None
		self.dim = -1


	def forward(self, xs, decoder_xs, nan_mask=None):
		"""
		Evaluate log probability of samples.

		Parameters
		----------
		xs : list of torch.Tensor or torch.Tensor
			Shape: [modalities][batch,m_dim] or [batch,modalities,m_dim]
		decoder_xs (vectorized): list of single torch.Tensor
			Shape: [1][batch,n_samples,m_dim]
		decoder_xs (not vectorized): list of lists of single torch.Tensor
			Shape: [modalities][1][batch,n_samples,m_dim]
		nan_mask (vectorized) : None or torch.Tensor
			Indicates where data is missing. Shape: [batch,modalities]
		nan_mask (vectorized) : None or list of torch.Tensor
			Indicates where data is missing. Shape [modalities][...]

		Returns
		-------
		log_probs (vectorized): torch.Tensor
			Shape: [batch,n_samples,modalities]
		log_probs (not vectorized): list of torch.Tensor
			Shape: [modalities][batch,n_samples]
		"""
		if type(xs) == type([]): # not vectorized
			return self._forward_non_vectorized(xs, decoder_xs, \
					nan_mask=nan_mask)
		# Unwrap the single parameter lists.
		decoder_xs = decoder_xs[0] # [b,s,m,m_dim]
		# We also know: xs.shape = [b,m,m_dim]
		# Reset the distribution if necessary.
		if xs.shape[-1] != self.dim or self.dist is None:
			loc = torch.zeros(xs.shape[-1], device=xs.device)
			scale = self.std_dev * torch.ones_like(loc)
			self.dist = Normal(loc, scale)
		log_probs = xs.unsqueeze(1) - decoder_xs
		log_probs = self.dist.log_prob(log_probs).sum(dim=3) # [b,s,m]
		if nan_mask is not None:
			temp_mask = (~nan_mask).float().unsqueeze(1).expand(log_probs.shape)
			assert temp_mask.shape == log_probs.shape, \
					"{}!={}".format(temp_mask.shape, log_probs.shape)
			log_probs = log_probs * temp_mask # [b,s,m]
		return log_probs


	def _forward_non_vectorized(self, xs, decoder_xs, nan_mask=None):
		""" """
		# Unwrap the single parameter lists.
		assert len(xs) == len(decoder_xs), \
				"{} != {}".format(len(xs), len(decoder_xs))
		decoder_xs = [decoder_x[0] for decoder_x in decoder_xs]
		# Reset the distribution if necessary.
		if xs[0].shape[-1] != self.dim or self.dist is None:
			loc = torch.zeros(xs[0].shape[-1], device=xs[0].device)
			scale = self.std_dev * torch.ones_like(loc)
			self.dist = Normal(loc, scale)
		log_probs = [self.dist.log_prob(x.unsqueeze(1) - decoder_x) for \
				x, decoder_x in zip(xs, decoder_xs)]
		if nan_mask is not None:
			temp_masks = [(~mask).float().unsqueeze(-1).unsqueeze(-1) \
					for mask in nan_mask]
			log_probs = [log_prob*temp_mask for log_prob,temp_mask in \
					zip(log_probs,temp_masks)]
		return [torch.sum(log_prob, dim=2) for log_prob in log_probs]


	def mean(self, like_params, n_samples):
		"""

		Parameters
		----------
		like_params : list of list of torch.tensor
			like_params[modality][param_num] shape: [batch,n_samples,z_dim]
		"""
		if type(like_params[0]) == type([]): # not vectorized
			return [like_param[0] for like_param in like_params]
		return like_params[0]


	def rsample(self, like_params, n_samples):
		"""
		Test this!

		"""
		assert self.dist is not None
		loc = torch.zeros(like_params[0].shape[-1], \
				device=like_params[0].device)
		scale = self.std_dev * torch.ones_like(loc)
		self.dist = Normal(loc, scale)
		return self.dist.rsample(sample_shape=(n_samples,)).transpose(0,1)



if __name__ == '__main__':
	pass



###
