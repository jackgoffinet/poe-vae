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
from torch.distributions import Bernoulli, Normal



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
		like_params : tuple of torch.Tensor or tuple of tuples of torch.Tensor
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


	def mean(self, like_params):
		"""
		Return the mean value of the likelihood.

		Parameters
		----------
		like_params: tuple of torch.tensor
			Shape: [1][batch,n_samples,m_dim]

		Returns
		-------
		mean : torch.Tensor
			Shape: [batch,n_samples,m_dim]
		"""
		raise NotImplementedError


	def sample(self, like_params):
		"""
		Return a sample from the likelihood.

		Parameters
		----------
		like_params: tuple of torch.tensor
			Shape: [1][batch,n_samples,m_dim]

		Returns
		-------
		sample : torch.Tensor
			Shape: [batch,n_samples,m_dim]
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


	def mean(self, like_params):
		"""
		Return the mean value of the likelihood.

		Parameters
		----------
		like_params: tuple of torch.tensor
			Shape: [n_params][modalities][batch,n_samples,m_dim]

		Returns
		-------
		mean : tuple of torch.Tensor
			Shape: [modalities][batch,n_samples,m_dim]
		"""
		# Transpose first two dimensions of like_params.
		like_params = tuple(tuple(p) for p in zip(*like_params))
		gen = zip(self.likelihoods,like_params)
		return tuple(like.mean(p)[0] for like,p in gen)


	def sample(self, like_params):
		"""
		Return a sample from the likelihood.

		Parameters
		----------
		like_params: tuple of torch.tensor
			Shape: [n_params][modalities][batch,n_samples,m_dim]

		Returns
		-------
		sample : torch.Tensor
			Shape: [modalities][batch,n_samples,m_dim]
		"""
		# Transpose first two dimensions of like_params.
		like_params = tuple(tuple(p) for p in zip(*like_params))
		gen = zip(self.likelihoods,like_params)
		return tuple(like.sample(p)[0] for like,p in gen)



class SphericalGaussianLikelihood(AbstractLikelihood):

	def __init__(self, obs_std_dev=0.1, **kwargs):
		"""
		Spherical Gaussian likelihood with a fixed covariance.

		Parameters
		----------
		obs_std_dev : float, optional
			Observation standard deviation.
		"""
		super(SphericalGaussianLikelihood, self).__init__()
		self.std_dev = obs_std_dev


	def forward(self, xs, like_params, nan_mask=None):
		"""
		Evaluate the log probability of data under the likelihood.

		NOTE: check like_params shape for vectorized input!

		Parameters
		----------
		xs : torch.Tensor
			Shape: [batch,modalities,m_dim]
		like_params : tuple of torch.Tensor
			Shape: [1][batch,n_samples,m_dim]
		nan_mask : torch.Tensor
			Shape: [batch,modalities]

		Returns
		-------
		log_probs : torch.Tensor or tuple of torch.Tensor
			Shape: [batch,n_samples,modalities]
		"""
		assert len(like_params) == 1, f"SphericalGaussianLikelihood only takes"\
				+ f" a single parameter. Found {len(like_params)}."
		# Unwrap the single parameter tuple.
		like_params = like_params[0] # [b,s,m]
		if len(like_params.shape) == 3:
			like_params = like_params.unsqueeze(2) # [b,s,m,m_dim]
			xs = xs.unsqueeze(1) # [b,m,m_dim]
		# Make a Gaussian distribution.
		dist = Normal(
				torch.zeros(1, device=xs.device),
				self.std_dev*torch.ones(1, device=xs.device),
		)
		log_probs = xs.unsqueeze(1) - like_params # [b,s,m,m_dim]
		log_probs = dist.log_prob(log_probs).sum(dim=3) # [b,s,m]
		if nan_mask is not None:
			temp_mask = (~nan_mask).float().unsqueeze(1).expand(log_probs.shape)
			assert temp_mask.shape == log_probs.shape, \
					f"{temp_mask.shape} != {log_probs.shape}"
			log_probs = log_probs * temp_mask # [b,s,m]
		return log_probs


	def mean(self, like_params):
		"""
		Return the mean value of the likelihood.

		Parameters
		----------
		like_params: tuple of torch.tensor
			Shape: [1][batch,n_samples,m_dim]

		Returns
		-------
		mean : tuple of torch.Tensor
			Shape: [1][batch,n_samples,m_dim]
		"""
		assert len(like_params) == 1, f"SphericalGaussianLikelihood only takes"\
				+ f" a single parameter. Found {len(like_params)}."
		return (like_params[0],)


	def sample(self, like_params):
		"""
		Return a sample from the likelihood.

		Parameters
		----------
		like_params: tuple of torch.tensor
			Shape: [1][batch,n_samples,m_dim]

		Returns
		-------
		sample : tuple of torch.Tensor
			Shape: [1][batch,n_samples,m_dim]
		"""
		assert len(like_params) == 1, f"SphericalGaussianLikelihood only takes"\
				+ f" a single parameter. Found {len(like_params)}."
		# Unwrap the single parameter tuple.
		like_params = like_params[0] # [b,s,m]
		# Make a Gaussian distribution.
		dist = Normal(
				torch.zeros(1, device=like_params.device),
				self.std_dev*torch.ones(1, device=like_params.device),
		)
		samples = dist.sample()
		return (samples,)



class BernoulliLikelihood(AbstractLikelihood):

	def __init__(self, **kwargs):
		"""
		Bernoulli likelihood.

		"""
		super(BernoulliLikelihood, self).__init__()


	def forward(self, xs, like_params, nan_mask=None):
		"""
		Evaluate the log probability of data under the likelihood.

		Parameters
		----------
		xs : torch.Tensor
			Shape: [batch,modalities,m_dim]
		like_params : tuple of torch.Tensor
			Shape: [1][batch,n_samples,m_dim]
		nan_mask : torch.Tensor
			Shape: [batch,modalities]

		Returns
		-------
		log_probs : torch.Tensor or tuple of torch.Tensor
			Shape: [batch,n_samples,modalities]
		"""
		assert len(like_params) == 1, f"BernoulliLikelihood only takes" \
				+ f" a single parameter. Found {len(like_params)}."
		# Unwrap the single parameter tuple.
		like_params = like_params[0] # [b,s,m]
		if len(like_params.shape) == 3:
			like_params = like_params.unsqueeze(2) # [b,s,m,m_dim]
			xs = xs.unsqueeze(1) # [b,m,m_dim]
		dist = Bernoulli(logits=like_params)
		log_probs = dist.log_prob(xs.unsqueeze(1)).sum(dim=3) # [b,s,m]
		if nan_mask is not None:
			temp_mask = (~nan_mask).float().unsqueeze(1).expand(log_probs.shape)
			assert temp_mask.shape == log_probs.shape, \
					f"{temp_mask.shape} != {log_probs.shape}"
			log_probs = log_probs * temp_mask # [b,s,m]
		return log_probs


	def mean(self, like_params):
		"""
		Return the mean value of the likelihood.

		Parameters
		----------
		like_params: tuple of torch.tensor
			Shape: [1][batch,n_samples,m_dim]

		Returns
		-------
		mean : tuple of torch.Tensor
			Shape: [1][batch,n_samples,m_dim]
		"""
		assert len(like_params) == 1, f"BernoulliLikelihood only takes" \
				+ f" a single parameter. Found {len(like_params)}."
		return (torch.sigmoid(like_params[0]),)


	def sample(self, like_params):
		"""
		Return a sample from the likelihood.

		Parameters
		----------
		like_params: tuple of torch.tensor
			Shape: [1][batch,n_samples,m_dim]

		Returns
		-------
		sample : tuple of torch.Tensor
			Shape: [1][batch,n_samples,m_dim]
		"""
		assert len(like_params) == 1, f"BernoulliLikelihood only takes" \
				+ f" a single parameter. Found {len(like_params)}."
		# Unwrap the single parameter tuple.
		like_params = like_params[0] # [b,s,m]
		dist = Bernoulli(logits=like_params)
		samples = dist.sample()
		return (samples,)



if __name__ == '__main__':
	pass



###
