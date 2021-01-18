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
	magic_constant = None # TO DO


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
		xs : list of torch.Tensor
			xs[modality] shape: [batch,x_dim]
		decoder_xs : list of lists of single torch.Tensor
			xs[modality][0] shape: [batch,n_samples,x_dim]
		nan_mask : torch.Tensor or list of torch.Tensor
			Indicates where data is missing.

		Returns
		-------
		log_probs : list of torch.Tensor
			log_probs[modality] shape: [batch,n_samples]
		"""
		assert len(xs) == len(decoder_xs)
		# Unwrap the single parameter lists.
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
		return [like_param[0] for like_param in like_params]


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
