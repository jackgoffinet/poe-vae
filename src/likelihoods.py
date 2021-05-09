"""
Define observation likelihoods.

Notes
-----
* Explain `like_params`...
* GroupedLikelihood is the only likelihood that handles non-vectorized
  modalities.
"""
__date__ = "January - May 2021"


from itertools import product
import numpy as np
import torch
from torch.distributions import Normal



class AbstractLikelihood(torch.nn.Module):

	def __init__(self):
		super(AbstractLikelihood, self).__init__()

	def forward(self, xs, like_params, nan_mask=None):
		"""
		Evaluate the log probability of data under the likelihood.

		Parameters
		----------
		xs : torch.Tensor or tuple of torch.Tensor
			Shape:
			 	[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise
		decoder_xs : tuple of torch.Tensor or tuple of tuples of torch.Tensor
			Shape:
				[1][batch,n_samples,m_dim] if vectorized
				[1][modalities][batch,n_samples,m_dim] otherwise
		nan_mask : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities] if vectorized
				[modalities][b] otherwise

		Returns
		-------
		log_probs : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,n_samples,modalities] if vectorized
				[modalities][batch,n_samples] otherwise
		"""
		raise NotImplementedError



class GroupedLikelihood(AbstractLikelihood):

	def __init__(self, likelihoods):
		"""
		Group several likelihoods together.

		Useful for non-vectorized data.

		Parameters
		----------
		likelihoods : list of AbstractLikelihood
		"""
		super(GroupedLikelihood, self).__init__()
		self.likelihoods = torch.nn.ModuleList(likelihoods)

	def forward(self, xs, like_params, nan_mask=None):
		"""
		Evaluate the log probability of data under the likelihood.

		Parameters
		----------
		xs : tuple of torch.Tensor
			Shape: [modalities][batch,m_dim]
		like_params : tuple of tuple of torch.Tensor
			Shape: [n_params][m][b,s,x]
		nan_mask : None or tuple of torch.Tensor
			Shape: [modalities][b]

		Returns
		-------
		log_probs : tuple of torch.Tensor
			Shape: [batch,n_samples,modalities]
		"""
		if nan_mask is None:
			nan_mask = [None for _ in self.likelihoods]
		else:
			nan_mask = [mask.unsqueeze(1) for mask in nan_mask]
		# Transpose first two dimensions of like_params.
		like_params = tuple(tuple(p) for p in zip(*like_params))
		# Send each modality to its own likelihood.
		gen = zip(self.likelihoods, xs, like_params, nan_mask)
		return torch.cat([like(i,j,k) for like,i,j,k in gen], dim=2)



class SphericalGaussianLikelihood(AbstractLikelihood):

	def __init__(self, obs_std_dev=0.1, **kwargs):
		"""
		Spherical Gaussian likelihood distribution.

		"""
		super(SphericalGaussianLikelihood, self).__init__()
		self.std_dev = obs_std_dev
		self.dist = None
		self.dim = -1
		self.parameter_dim_func = lambda d: (d,)


	def forward(self, xs, decoder_xs, nan_mask=None):
		"""
		Evaluate the log probability of data under the likelihood.

		Parameters
		----------
		xs : torch.Tensor or tuple of torch.Tensor
			Shape:
			 	[batch,modalities,m_dim] if vectorized
				[modalities][batch,m_dim] otherwise
		decoder_xs : tuple of torch.Tensor or tuple of tuples of torch.Tensor
			Shape:
				[1][batch,n_samples,m_dim] if vectorized
				[1][modalities][batch,n_samples,m_dim] otherwise
		nan_mask : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,modalities] if vectorized
				[modalities][b] otherwise

		Returns
		-------
		log_probs : torch.Tensor or tuple of torch.Tensor
			Shape:
				[batch,n_samples,modalities] if vectorized
				[modalities][batch,n_samples] otherwise
		"""
		# assert len(decoder_xs) == 1, f"SphericalGaussianLikelihood only takes" \
		# 		+ f" a single parameter. Found {len(decoder_xs)}."
		if isinstance(xs, (tuple,list)): # not vectorized
			raise NotImplementedError
			print("not vectorized", len(xs))
			return self._forward_non_vectorized(
					xs,
					decoder_xs,
					nan_mask=nan_mask,
			)
		# Unwrap the single parameter tuple.
		decoder_xs = decoder_xs[0]
		if len(decoder_xs.shape) == 3:
			decoder_xs = decoder_xs.unsqueeze(2) # [b,s,m,m_dim]
			xs = xs.unsqueeze(1) # [b,m,m_dim]
		# Reset the distribution if necessary.
		if xs.shape[-1] != self.dim or self.dist is None:
			loc = torch.zeros(xs.shape[-1], device=xs.device)
			scale = self.std_dev * torch.ones_like(loc)
			self.dist = Normal(loc, scale)
		log_probs = xs.unsqueeze(1) - decoder_xs # [b,s,m,m_dim]
		log_probs = self.dist.log_prob(log_probs).sum(dim=3) # [b,s,m]
		if nan_mask is not None:
			temp_mask = (~nan_mask).float().unsqueeze(1).expand(log_probs.shape)
			assert temp_mask.shape == log_probs.shape, \
					f"{temp_mask.shape} != {log_probs.shape}"
			log_probs = log_probs * temp_mask # [b,s,m]
		return log_probs


	def _forward_non_vectorized(self, xs, decoder_xs, nan_mask=None):
		"""
		Non-vectorized version of `forward`

		Parameters
		----------
		xs : tuple of torch.Tensor
			Shape: [m][m_dim]
		decoder_xs : tuple of torch.Tensor
			Shape : [m][m_dim]
		"""
		assert len(decoder_xs) == len(xs), f"{len(decoder_xs)} != {len(xs)}"
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
		NOTE: is this used?

		Parameters
		----------
		like_params (non-vectorized): list of list of torch.tensor
			like_params[param_num][modality] shape: [batch,n_samples,z_dim]
		like_params (vectorized): list of torch.tensor
			Shape: ...
		"""
		if isinstance(like_params[0], (tuple,list)): # not vectorized
			return [like_param for like_param in like_params[0]]
		return like_params[0]



if __name__ == '__main__':
	pass



###
