"""
Define variational posteriors.

"""
__date__ = "January 2021"


from math import log
import torch
from torch.distributions import Normal, OneHotCategorical

from .distributions.von_mises_fisher import VonMisesFisher

EPS = 1e-5



class AbstractVariationalPosterior(torch.nn.Module):
	"""Abstract class for variational posteriors."""

	def __init__(self):
		super(AbstractVariationalPosterior, self).__init__()
		self.dist = None

	def forward(self, *dist_parameters, n_samples=1):
		"""
		Produce reparamaterized samples and evaluate their log probability.

		Parameters
		----------
		dist_parameters: tuple
			Distribution parameters, probably containing torch.Tensors.
		n_samples : int, optional
			Number of samples to draw.

		Returns
		-------
		samples : torch.Tensor
			Samples from the distribution. Shape: TO DO
		log_prob : torch.Tensor
			Log probability of samples under the distribution.
		"""
		raise NotImplementedError


	def kld(self, other):
		"""
		Return KL-divergence.

		"""
		type_tuple = (type(self.dist), type(other.dist))
		if type_tuple in torch.distributions.kl._KL_REGISTRY:
			return torch.distributions.kl_divergence(self.dist, other.dist)
		raise NotImplementedError



class DiagonalGaussianPosterior(AbstractVariationalPosterior):

	def __init__(self, args):
		"""Diagonal Gaussian varitional posterior."""
		super(DiagonalGaussianPosterior, self).__init__()
		# latent_dim -> parameter dimensions
		self.parameter_dim_func = lambda d: (d,d)


	def forward(self, mean, precision, n_samples=1, transpose=True):
		"""
		Produce reparamaterized samples and evaluate their log probability.

		Parameters
		----------
		mean : torch.Tensor
			Shape [batch,z_dim]
		precision : torch.Tensor
			Shape [batch, z_dim]
		n_samples : int
		transpose : bool

		Returns
		-------
		samples : torch.Tensor
			Samples from the distribution. Shape: [batch,n_samples,z_dim]
		log_prob : torch.Tensor
			Log probability of samples under the distribution.
			Shape: [batch,n_samples]
		"""
		assert mean.shape == precision.shape, \
				"{} != {}".format(mean.shape,precision.shape)
		assert len(mean.shape) == 2, \
				"len(mean.shape) == len({})".format(mean.shape)
		std_dev = torch.sqrt(torch.reciprocal(precision + EPS))
		self.dist = Normal(mean, std_dev)
		samples = self.dist.rsample(sample_shape=(n_samples,)) # [s,b,z]
		log_prob = self.dist.log_prob(samples).sum(dim=2) # Sum over latent dim
		if transpose:
			return samples.transpose(0,1), log_prob.transpose(0,1)
		return samples, log_prob


	def rsample(self):
		""" """
		raise NotImplementedError


	def log_prob(self, samples, mean, precision, transpose=True):
		"""
		...

		Parameters
		----------
		samples : torch.Tensor [...]
		"""
		std_dev = torch.sqrt(torch.reciprocal(precision + EPS))
		self.dist = Normal(mean, std_dev)
		log_prob = self.dist.log_prob(samples).sum(dim=2) # Sum over latent dim
		if transpose:
			return log_prob.transpose(0,1)
		return log_prob



class DiagonalGaussianMixturePosterior(AbstractVariationalPosterior):

	def __init__(self, args):
		"""
		Mixture of diagonal Gaussians variational posterior.

		The component weights are assumed to be equal.
		"""
		super(DiagonalGaussianMixturePosterior, self).__init__()
		# latent_dim -> parameter dimensions
		self.parameter_dim_func = lambda d: (d,d)


	def forward(self, means, precisions, n_samples=1):
		"""
		Produce stratified samples and evaluate their log probability.

		Parameters
		----------
		means : torch.Tensor
			Shape [batch, modalities, z_dim]
		precisions : torch.Tensor
			Shape [batch, modalities, z_dim]
		n_samples : int
			Samples per modality.

		Returns
		-------
		samples : torch.Tensor
			Stratified samples from the distribution.
			Shape: [batch,n_samples,modalities,z_dim]
		log_prob : torch.Tensor
			Log probability of samples under the mixture distribution.
			Shape: [batch,n_samples,modalities]
		"""
		return self._helper(means, precisions, n_samples=n_samples)


	def log_prob(self, samples, means, precisions):
		"""

		"""
		return self._helper(means, precisions, samples=samples)[1]


	def _helper(self, means, precisions, samples=None, n_samples=1):
		"""
		If `samples` is given, evaluate their log probability. Otherwise, make
		samples and evaluate their log probability.
		"""
		if type(means) == type([]): # not vectorized
			raise NotImplementedError
		else: # vectorized
			std_devs = torch.sqrt(torch.reciprocal(precisions + EPS))
			means = means.unsqueeze(1) # [b,1,m,z]
			std_devs = std_devs.unsqueeze(1) # [b,1,m,z]
			self.dist = Normal(means, std_devs) # [b,1,m,z]
			if samples is None:
				# [s,b,1,m,z]
				samples = self.dist.rsample(sample_shape=(n_samples,))
				samples = samples.squeeze(2).transpose(0,1) # [b,s,m,z]
			# [b,s,m]
			log_probs = torch.zeros(samples.shape[:3], device=means.device)
			# For each modality m...
			M = log_probs.shape[2]
			for m in range(M):
				# Take the samples produced by expert m.
				temp_samples = samples[:,:,m:m+1] # [b,s,1,z]
				# And evaluate their log probability under all the experts.
				temp_logp = self.dist.log_prob(temp_samples).sum(dim=3)
				# Convert to a log probability under the MoE.
				temp_logp = torch.logsumexp(temp_logp - log(M), dim=2)
				log_probs[:,:,m] = temp_logp
			return samples, log_probs


	def non_stratified_forward(self, means, precisions, n_samples=1):
		"""
		Standard (non-stratified) sampling version of `forward`.

		* useful for MLL estimation
		"""
		if type(means) == type([]): # not vectorized
			raise NotImplementedError
		else: # vectorized
			std_devs = torch.sqrt(torch.reciprocal(precisions + EPS))
			means = means.unsqueeze(1) # [b,1,m,z]
			std_devs = std_devs.unsqueeze(1) # [b,1,m,z]
			self.dist = Normal(means, std_devs) # [b,1,m,z]
			# if samples is None:
			# [s,b,1,m,z]
			samples = self.dist.rsample(sample_shape=(n_samples,))
			samples = samples.squeeze(2).transpose(0,1) # [b,s,m,z]
			ohc_probs = torch.ones(means.shape[0],means.shape[2], \
					device=means.device)
			ohc_dist = OneHotCategorical(probs=ohc_probs)
			ohc_sample = ohc_dist.sample(sample_shape=(n_samples,))
			ohc_sample = ohc_sample.unsqueeze(-1).transpose(0,1)
			# [b,s,1,z]
			samples = torch.sum(samples * ohc_sample, dim=2, keepdim=True)
			# [b,s,m,z]
			log_probs = self.dist.log_prob( \
					samples.expand(-1,-1,means.shape[2],-1))
			log_probs = torch.sum(log_probs, dim=3)
			# [b,s]
			log_probs = torch.logsumexp(log_probs - log(means.shape[2]), dim=2)
			return samples.squeeze(2), log_probs



class VmfProductPosterior(AbstractVariationalPosterior):

	def __init__(self, args):
		"""Product of von Mises Fishers varitional posterior."""
		super(VmfProductPosterior, self).__init__()
		self.vmf_dim = args.vmf_dim # vMFs are defined on the (vmf_dim-1)-sphere
		self.latent_dim = args.latent_dim
		assert self.latent_dim % (self.vmf_dim - 1) == 0, \
				"Incompatible z_dim and vmf_dim!"
		self.n_vmfs = self.latent_dim // (self.vmf_dim - 1)
		self.input_dim = self.n_vmfs * self.vmf_dim

		def parameter_dim_func(z):
			assert z == self.latent_dim, "Incompatible shapes!"
			return (self.input_dim, self.n_vmfs)

		self.parameter_dim_func = parameter_dim_func


	def forward(self, loc, scale, n_samples=1, transpose=True):
		"""
		Produce reparamaterized samples and evaluate their log probability.

		Parameters
		----------
		loc : torch.Tensor
			Shape: [b,n_vmfs,d]
		scale : torch.Tensor
			Shape: [b,n_vmfs,1]
		n_samples : int
		transpose : bool

		Returns
		-------
		samples : torch.Tensor
			Samples from the distribution. Shape: [batch,n_samples,z_dim]
		log_prob : torch.Tensor
			Log probability of samples under the distribution.
			Shape: [batch,n_samples]
		"""
		assert len(loc.shape) == 3, \
				"len(loc.shape) == {}".format(len(loc).shape)
		assert loc.shape[:-1] == scale.shape[:-1]
		assert scale.shape[-1] == 1
		assert loc.shape[-1] == self.vmf_dim
		self.dist = VonMisesFisher(loc, scale)
		samples = self.dist.rsample(shape=n_samples) # [s,b,n_vmfs,vmf_dim]
		log_prob = self.dist.log_prob(samples).sum(dim=-1) #[s,b,n_vmfs*vmf_dim]
		samples = samples.view(samples.shape[:2]+(-1,)) # [s,b,n_]
		if transpose:
			# [s,b,*] -> [b,s,*]
			return samples.transpose(0,1), log_prob.transpose(0,1)
		return samples, log_prob


	def rsample(self):
		""" """
		raise NotImplementedError


	def log_prob(self, samples, loc, scale, transpose=True):
		"""
		...

		Parameters
		----------
		samples : torch.Tensor [...]
		"""
		raise NotImplementedError



if __name__ == '__main__':
	pass



###
