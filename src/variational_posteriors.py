"""
Define variational posteriors.

TO DO
-----
* EnergyBasedModelPosterior assumes a standard Gaussian prior. Generalize this!
"""
__date__ = "January 2021"


from math import log
import torch
from torch.distributions import Normal, OneHotCategorical
import torch.nn.functional as F

from .distributions.von_mises_fisher import VonMisesFisher
from .gumbel_softmax import gumbel_softmax

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



class EnergyBasedModelPosterior(AbstractVariationalPosterior):

	def __init__(self, args):
		"""
		EBM varitional posterior.

		Includes a proposal network and a network mapping thetas and zs to
		energies.
		"""
		super(EnergyBasedModelPosterior, self).__init__()
		self.k = args.ebm_samples
		self.e_1 = torch.nn.Linear(args.theta_dim+args.latent_dim, 16)
		self.e_2 = torch.nn.Linear(16, 16)
		self.e_3 = torch.nn.Linear(16, 1)
		# proposal layer 1
		self.pi_1 = torch.nn.Linear(args.theta_dim, 16)
		# proposal layer 2a: mean
		self.pi_mu = torch.nn.Linear(16, args.latent_dim)
		# proposal layer2b: log std dev
		self.pi_log_prec = torch.nn.Linear(16, args.latent_dim)
		self.parameter_dim_func = lambda d: (args.theta_dim,)


	def proposal_network(self, thetas, n_samples=1):
		"""
		Map thetas to a proposal distribution, sample, and evaluate log prob.

		Parameters
		----------
		thetas : torch.Tensor
			Shape: [b,m,theta_dim]
		k : int

		Returns
		-------
		samples : torch.Tensor
			Shape: [b,s,k,z]
		pi_log_prob : torch.Tensor
			Shape: [b,s,k]
		prior_log_prob : torch.Tensor
			Shape: [b,s,k]
		"""
		# Sum over the modality dimension for exchangeability.
		h = thetas.sum(dim=-2)
		if torch.isnan(h).sum() > 0:
			print("p h NaN!")
		h = F.relu(self.pi_1(h))
		mu = self.pi_mu(h)
		prec = torch.exp(self.pi_log_prec(h)) + 1e-2
		if torch.isnan(prec).sum() > 0:
			print("p prec NaN!")
		var = torch.reciprocal(1.0 + prec)
		if torch.isnan(var).sum() > 0:
			print("p var NaN!")
		mu = var * prec * mu
		if torch.isnan(mu).sum() > 0:
			print("p mu NaN!")
		std_dev = torch.sqrt(var)
		if torch.isnan(std_dev).sum() > 0:
			print("p std_dev NaN!")
		dist = Normal(mu, torch.sqrt(var))
		samples = dist.rsample(sample_shape=(n_samples,self.k))
		if torch.isnan(samples).sum() > 0:
			print("p samples NaN!")
		pi_log_prob = dist.log_prob(samples).transpose(1,2).transpose(0,1)
		samples = samples.transpose(1,2).transpose(0,1)
		loc = torch.zeros(samples.shape[-1], device=samples.device)
		scale = torch.ones_like(loc)
		prior_log_prob = Normal(loc, scale).log_prob(samples).sum(dim=-1)
		return samples, pi_log_prob.sum(dim=-1), prior_log_prob


	def energy_network(self, thetas, z_samples):
		"""
		Map thetas and z's to scalar energies.

		Parameters
		----------
		thetas : torch.Tensor
			Shape: [b,m,theta_dim]
		z_samples : torch.Tensor
			Shape: [b,s,k,z]

		Returns
		-------
		energies : torch.Tensor
			Shape: [b,m,s,k]
		"""
		# Turn both into [b,m,s,k,*]
		thetas = thetas.unsqueeze(2).unsqueeze(2)
		thetas = thetas.expand(-1,-1,z_samples.shape[1],z_samples.shape[2],-1)
		z_samples = z_samples.unsqueeze(1).expand(-1,thetas.shape[1],-1,-1,-1)
		# Concatenate to [b,m,s,k,z+theta].
		h = torch.cat([thetas,z_samples], dim=-1)
		h = F.relu(self.e_1(h))
		h = F.relu(self.e_2(h))
		h = self.e_3(h).squeeze(-1)
		return h


	def forward(self, thetas, nan_mask, n_samples=1):
		"""
		Produce reparamaterized samples and evaluate their log probability.

		Parameters
		----------
		thetas : torch.Tensor
			Shape: [b,m,theta]
		nan_mask : torch.Tensor
			Shape: [b,m]
		n_samples : int
		transpose : bool

		Returns
		-------
		samples : torch.Tensor
			Samples from the distribution. Shape: [batch,s,z_dim]
		log_prob : torch.Tensor
			Log probability of samples under the distribution.
			Shape: [batch,s]
		"""
		if torch.isnan(thetas).sum() > 0:
			print("vp forward theta NaN!")
		# Get proposal samples and log probs. [b,s,k,z], [b,s,k]
		pi_samples, pi_log_prob, prior_log_prob = \
				self.proposal_network(thetas, n_samples)
		if torch.isnan(pi_samples).sum() > 0:
			print("pi_samples NaN!")
		if torch.isnan(pi_log_prob).sum() > 0:
			print("pi_log_prob NaN!")
		if torch.isnan(prior_log_prob).sum() > 0:
			print("prior_log_prob NaN!")
		# Get energies.
		energies = self.energy_network(thetas, pi_samples) # [b,m,s,k]
		# Sum over the modality energies, applying the missingness mask.
		temp_mask = (~nan_mask).float().unsqueeze(-1).unsqueeze(-1)
		energies = energies * temp_mask # [b,m,s,k]
		energies = energies.sum(dim=1) # [b,s,k]
		# Calculate importance weights: [b,s,k]
		# Following the EBM convention: p(x) \propto exp[-E(x)]
		log_iws = prior_log_prob - energies - pi_log_prob
		# Sample.
		z_one_hot = gumbel_softmax(log_iws) # [b,s,k], last dim is one-hot
		# [b,s,k] -> [b,s,k,z]
		z_mask = z_one_hot.unsqueeze(-1).expand(-1,-1,-1,pi_samples.shape[-1])
		z_samples = torch.sum(pi_samples * z_mask, dim=2) # [b,s,z]
		# Estimate log probability under the EBM.
		# We're making the approximation q(z_i|x) \approx EBM(z_i) / avg(w_j)
		# => log q(z_i|x) \approx -E(z_i) - logavgexp(w_j)
		log_avg_w = torch.logsumexp(log_iws, dim=-1) - log(self.k) # [b,s]
		e_zi = torch.sum(energies * z_one_hot, dim=-1) # [b,s]
		log_probs = - e_zi - log_avg_w
		if torch.isnan(log_probs).sum() > 0:
			print("ebm log_prob NaN!")
		# Return.
		return z_samples, log_probs


if __name__ == '__main__':
	pass



###
