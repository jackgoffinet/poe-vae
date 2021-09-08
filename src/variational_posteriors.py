"""
Define variational posteriors.

TO DO
-----
* LocScaleEbmPosterior assumes a standard Gaussian prior. Generalize this!
* Implement LocScaleEbmPosterior log_prob, check the other log probs
* Double check shapes for DiagonalGaussianMixturePosterior
"""
__date__ = "January - May 2021"


from math import log
import torch
from torch.distributions import Normal, OneHotCategorical, Bernoulli
import torch.nn.functional as F

from .distributions.von_mises_fisher import VonMisesFisher
from .gumbel_softmax import gumbel_softmax

EPS = 1e-5
ENERGY_REG = log(4.0)
MIN_PRECISION_FACTOR = 0.7



class AbstractVariationalPosterior(torch.nn.Module):
	"""Abstract class for variational posteriors."""

	def __init__(self):
		super(AbstractVariationalPosterior, self).__init__()
		self.dist = None

	def forward(self, *dist_parameters, n_samples=1, samples=None):
		"""
		Produce reparamaterized samples and evaluate their log probability.

		Parameters
		----------
		dist_parameters: tuple
			Distribution parameters, probably containing torch.Tensors.
		n_samples : int, optional
			Number of samples to draw.
		samples : torch.Tensor or None, optional
			Pass to use these samples instead of drawing new samples.

		Returns
		-------
		samples : torch.Tensor
			Samples from the distribution.
			Shape: ???
		log_prob : torch.Tensor
			Log probability of samples under the distribution.
			Shape: ???
		"""
		raise NotImplementedError


	def kld(self, other):
		"""
		Return KL-divergence.

		"""
		type_tuple = (type(self.dist), type(other.dist))
		if type_tuple in torch.distributions.kl._KL_REGISTRY:
			return torch.distributions.kl_divergence(self.dist, other.dist)
		err_str = f"({type(self.dist)},{type(other.dist)}) not in KL registry!"
		raise NotImplementedError(err_str)


	def log_prob(self, samples, *dist_parameters):
		"""
		Estimate the log probability of the samples on the distribution.

		Parameters
		----------
		samples : torch.Tensor
			Shape: ???

		Returns
		-------
		log_probs : torch.Tensor
			Shape: ???
		"""
		return self(*dist_parameters, samples=samples)[0]



class DiagonalGaussianPosterior(AbstractVariationalPosterior):
	EPS = 1e-5

	def __init__(self, **kwargs):
		"""Diagonal Gaussian varitional posterior."""
		super(DiagonalGaussianPosterior, self).__init__()
		self.dist = None
		self.prec_mean = None
		self.precision = None


	def forward(self, prec_mean, precision, n_samples=1, transpose=True, \
		samples=None):
		"""
		Produce reparamaterized samples and evaluate their log probability.

		Note: is `transpose` used?

		Parameters
		----------
		prec_mean : torch.Tensor
			Precision/mean product. Shape: [batch,z_dim]
		precision : torch.Tensor
			Precision (inverse variance). Shape: [batch, z_dim]
		n_samples : int
		transpose : bool
		samples : torch.Tensor or None

		Returns
		-------
		samples : torch.Tensor
			Samples from the distribution. Shape: [batch,n_samples,z_dim]
		log_prob : torch.Tensor
			Log probability of samples under the distribution.
			Shape: [batch,n_samples]
		"""
		assert prec_mean.shape == precision.shape, \
				f"{mean.shape} != {precision.shape}"
		assert len(prec_mean.shape) == 2, f"len({prec_mean.shape}) != 2"
		mean = prec_mean / (precision + self.EPS)
		std_dev = torch.sqrt(torch.reciprocal(precision + self.EPS))
		self.dist = Normal(mean, std_dev)
		if samples is None:
			samples = self.dist.rsample(sample_shape=(n_samples,)) # [s,b,z]
		log_prob = self.dist.log_prob(samples).sum(dim=2) # sum over z dim
		if transpose:
			return samples.transpose(0,1), log_prob.transpose(0,1)
		return samples, log_prob


	def log_prob(self, samples, prec_mean, precision, transpose=False):
		"""
		Estimate the log probability of the samples on the distribution.

		Note: is `transpose` used?

		Parameters
		----------
		samples : torch.Tensor
			Shape: [b,s,z]
		prec_mean : torch.Tensor
			Shape: [b,z]
		precision : torch.Tensor
			Shape: [b,z]
		transpose : bool, optional

		Returns
		-------
		log_prob : torch.Tensor
			Shape: [b,s]
		"""
		mean = prec_mean / (precision + self.EPS)
		std_dev = torch.sqrt(torch.reciprocal(precision + self.EPS))
		self.dist = Normal(mean.unsqueeze(1), std_dev.unsqueeze(1))
		log_prob = self.dist.log_prob(samples).sum(dim=2) # Sum over latent dim
		if transpose:
			return log_prob.transpose(0,1)
		return log_prob


	def rsample(self, n_samples=1):
		"""
		Return reparameterized samples.

		Parameters
		----------
		n_samples : int, optional

		Returns
		-------
		z_samples : torch.Tensor
			Shape: [b,s,z]
		"""
		assert self.dist is not None, "self.dist is None!"
		return self.dist.rsample(sample_shape=(n_samples,)).transpose(0,1)



class DiagonalGaussianMixturePosterior(AbstractVariationalPosterior):

	def __init__(self, **kwargs):
		"""
		Mixture of diagonal Gaussians variational posterior.

		The component weights are assumed to be equal.
		"""
		super(DiagonalGaussianMixturePosterior, self).__init__()


	def forward(self, means, precisions, n_samples=1, samples=None):
		"""
		Produce reparamaterized samples and evaluate their log probability.

		If `samples` is given, evaluate their log probability. Otherwise, make
		samples and evaluate their log probability.

		Parameters
		----------
		samples : None or torch.Tensor
			Shape: [b,s,z]

		Returns
		-------
		samples : torch.Tensor
			Shape: [b,s,z]
		log_prob : torch.Tensor
			Shape: [b,s]
		"""
		std_devs = torch.sqrt(torch.reciprocal(precisions + EPS))
		means = means.unsqueeze(1) # [b,1,m,z]
		std_devs = std_devs.unsqueeze(1) # [b,1,m,z]
		self.dist = Normal(means, std_devs) # [b,1,m,z]
		if samples is None:
			samples = self.dist.rsample(sample_shape=(n_samples,)) # [s,b,1,m,z]
			samples = samples.squeeze(2).transpose(0,1) # [b,s,m,z]
		else:
			# [b,s,m,z]
			samples = samples.unsqueeze(2).expand(-1,-1,means.shape[2],-1)
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


	def stratified_forward(self, means, precisions, n_samples=1, samples=None):
		"""
		Produce stratified samples and evaluate their log probability.

		Parameters
		----------
		means : torch.Tensor
			Shape [batch, modalities, z_dim]
		precisions : torch.Tensor
			Shape [batch, modalities, z_dim]
		n_samples : int, optional
			Samples per modality.
		samples : torch.Tensor or None, optional

		Returns
		-------
		samples : torch.Tensor
			Stratified samples from the distribution.
			Shape: [batch,n_samples,modalities,z_dim]
		log_prob : torch.Tensor
			Log probability of samples under the mixture distribution.
			Shape: [batch,n_samples,modalities]
		"""
		std_devs = torch.sqrt(torch.reciprocal(precisions + EPS)) # [b,m,z]
		means = means.unsqueeze(1) # [b,1,m,z]
		std_devs = std_devs.unsqueeze(1) # [b,1,m,z]
		self.dist = Normal(means, std_devs) # [b,1,m,z]
		if samples is None:
			samples = self.dist.rsample(sample_shape=(n_samples,)) # [s,b,1,m,z]
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


	def log_prob(self, samples, means, precisions, stratified=False):
		"""
		Evaluate the log probability of the samples.

		Parameters
		----------
		samples : torch.Tensor
			Shape: [b,s,z]
		means : torch.Tensor
			Shape: [b,m,z]
		precisions : torch.Tensor
			Shape: [b,m,z]
		stratified : bool, optional

		Returns
		-------
		log_prob : torch.Tensor
			Shape:
				[b,s,m] if stratified
				[b,s] otherwise
		"""
		if stratified:
			return self.stratified_forward(
					means,
					precisions,
					samples=samples,
			)[1]
		return self.forward(
				means,
				precisions,
				samples=samples,
		)[1]



class VmfProductPosterior(AbstractVariationalPosterior):

	def __init__(self, n_vmfs=5, vmf_dim=4, **kwargs):
		"""Product of von Mises Fishers varitional posterior."""
		super(VmfProductPosterior, self).__init__()
		self.n_vmfs = n_vmfs
		self.vmf_dim = vmf_dim
		self.dist = None

	def forward(self, kappa_mu, n_samples=1, samples=None):
		"""
		Produce reparamaterized samples and evaluate their log probability.

		Parameters
		----------
		kappa_mu : torch.Tensor
			Shape : [b,n_vmfs,vmf_dim+1]
		n_samples : int, optional
		samples : torch.Tensor or None, optional

		Returns
		-------
		samples : torch.Tensor
			Samples from the distribution. Shape: [batch,n_samples,z_dim]
		log_prob : torch.Tensor
			Log probability of samples under the distribution.
			Shape: [batch,n_samples]
		"""
		assert len(kappa_mu.shape) == 3, f"len({kappa_mu.shape}) != 3"
		# Calculate loc and scale parameterization from kappa_mu.
		scale = torch.norm(kappa_mu, dim=2, keepdim=True)
		loc = kappa_mu / (scale + EPS)
		self.dist = VonMisesFisher(loc, scale)
		if samples is None:
			samples = self.dist.rsample(shape=n_samples) #[s,b,n_vmfs,vmf_dim+1]
		log_prob = self.dist.log_prob(samples).sum(dim=-1) #[s,b]
		# samples shape: [s,b,n_vmfs*(vmf_dim+1)]
		samples = samples.view(samples.shape[:2]+(-1,))
		# [s,b,*] -> [b,s,*]
		return samples.transpose(0,1), log_prob.transpose(0,1)



class LocScaleEbmPosterior(AbstractVariationalPosterior):
	EPS = 1e-5

	def __init__(self, ebm_samples=10, theta_dim=2, latent_dim=20, **kwargs):
		"""
		Location/scale EBM varitional posterior in Euclidean space.

		"""
		super(LocScaleEbmPosterior, self).__init__()
		self.k = ebm_samples
		# Energy network: (theta,z) -> scalar energy
		self.e_1 = torch.nn.Linear(theta_dim+1, 128)
		self.e_2 = torch.nn.Linear(128, 1)
		# Proposal network: (means,precisions) -> mean,precision
		self.pi_1 = torch.nn.Linear(2, 128)
		self.pi_2 = torch.nn.Linear(128, 1)
		# Book-keeping for adding evidence.
		self.pi_dist = None


	def proposal_network(self, means, precisions):
		"""
		Map individual expert proposal parameters to PoE proposal parameters.

		Parameters
		----------
		means : torch.Tensor
			Shape: [b,m,z]
		precisions : torch.Tensor
			Shape: [b,m,z]
		"""
		cat = torch.cat(
				(means.unsqueeze(-1), precisions.unsqueeze(-1)),
				dim=3,
		) # [b,m,z,2]
		h = F.relu(self.pi_1(cat)) # [b,m,z,128]
		h = torch.sum(h, dim=1) # [b,z,128]
		h = torch.clamp(self.pi_2(h), min=MIN_PRECISION_FACTOR).squeeze(2)#[b,z]
		precisions = (precisions * h.unsqueeze(1)) # [b,m,z]
		prec_mean = (precisions * means).sum(dim=1) # [b,z]
		precision = torch.sum(precisions, dim=1) + 1.0 # [b,z]
		mean = prec_mean / precision
		std_dev = torch.reciprocal(torch.sqrt(precision) + self.EPS)
		self.pi_dist = Normal(mean, std_dev)


	def forward(self, thetas, means, prec_means, precisions, nan_mask, \
		n_samples=1, samples=None):
		"""
		Produce reparamaterized samples and evaluate their log probability.

		Parameters
		----------
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
		n_samples : int, optional
		samples : torch.Tensor or None, optional
			Shape: [b,s,z]

		Returns
		-------
		samples : torch.Tensor
			Samples from the distribution. Shape: [b,s,z]
		log_probs : torch.Tensor
			Estimated log probability of samples under the distribution.
			Shape: [b,s]
		"""
		# Make a proposal distribution.
		self.proposal_network(means, precisions)
		return self.rsample(
				thetas,
				means,
				prec_means,
				precisions,
				nan_mask,
				n_samples=n_samples,
				samples=samples,
				return_log_probs=True
		) # [b,s,z], [b,s]


	def rsample(self, thetas, means, prec_means, precisions, nan_mask,
		n_samples=1, samples=None, return_log_probs=False):
		"""
		Return reparameterized samples.

		Parameters
		----------
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
		n_samples : int, optional
		return_log_probs : bool, optional

		Returns
		-------
		z_samples : torch.Tensor
			Shape: [b,s,z]
		log_probs : torch.Tensor
			Returned if `return_log_probs`. Shape: [b,s]
		"""
		assert self.pi_dist is not None
		sample_flag = samples is not None
		if thetas.requires_grad:
			samp = self.k
		else:
			samp = 2 * self.k # a bit more careful for test estimates
		# Get samples. [s,k,b,z]
		pi_samples = self.pi_dist.rsample(sample_shape=(n_samples,samp))
		if sample_flag:
			samp += 1 # k += 1
			print("pi_samples", pi_samples.shape) # [s,k,b,z]
			print("samples", samples.shape)
			samples = samples.unsqueeze(1) # [s,1,b,z]
			pi_samples = torch.cat((samples,pi_samples), dim=1) # [s,k,b,z]
			print("pi_samples", pi_samples.shape)
		# Get the log prob of the samples under the soft product proposal.
		pi_log_prob = self.pi_dist.log_prob(pi_samples)
		# Get the log prob of the samples under the full product proposal.
		precision = torch.sum(precisions, dim=1) + 1.0 # [b,z]
		mean = prec_means.sum(dim=1) / precision # [b,z]
		std_dev = torch.sqrt(torch.reciprocal(precision + self.EPS))
		full_pi_log_prob = Normal(mean, std_dev).log_prob(pi_samples)
		# [s,k,b,z] -> [b,s,k,z]
		pi_samples = pi_samples.transpose(1,2).transpose(0,1)
		# [s,k,b,z] -> [b,s,k]
		pi_log_prob = pi_log_prob.transpose(1,2).transpose(0,1).sum(dim=-1)
		if sample_flag:
			print("pi_log_prob", pi_log_prob.shape)
		full_pi_log_prob = full_pi_log_prob.transpose(1,2).transpose(0,1)
		full_pi_log_prob = full_pi_log_prob.sum(dim=-1) # [b,s,k]
		# Get standardized versions of the proposal samples by representing
		# them in each modality-specific reference frame. [b,m,s,k,z]
		std_pi_samples = pi_samples.unsqueeze(1)-means.unsqueeze(2).unsqueeze(2)
		# [b,m,s,k,z]
		std_pi_samples = std_pi_samples * \
				torch.sqrt(precisions.unsqueeze(2).unsqueeze(2) + self.EPS)
		std_pi_samples = torch.pow(std_pi_samples, 2).sum(dim=4) # [b,m,s,k]
		# Get energies by passing the thetas and standardized proposal samples
		# through a network: [b,m,s,k]
		energies =  self.energy_network(thetas, std_pi_samples)
		# Sum over the modality energies, applying the missingness mask.
		temp_mask = (~nan_mask).float().unsqueeze(-1).unsqueeze(-1) # [b,m,1,1]
		energies = energies * temp_mask # [b,m,s,k]
		energies = energies.sum(dim=1) # [b,m,s,k] -> [b,s,k]
		energies = energies - full_pi_log_prob
		# Calculate importance weights.
		log_iws = - energies - pi_log_prob # [b,s,k]
		if sample_flag:
			print("energies", energies.shape)
			print("log_iws", log_iws.shape)
		# Sample.
		if sample_flag:
			z_one_hot = torch.zeros_like(log_iws)
			z_one_hot[:,:] = 1.0
			print("z_one_hot", z_one_hot.shape)
		else:
			z_one_hot = gumbel_softmax(log_iws) # [b,s,k], last dim is one-hot
		# [b,s,k] -> [b,s,k,z]
		z_mask = z_one_hot.unsqueeze(-1).expand(-1,-1,-1,pi_samples.shape[-1])
		z_samples = torch.sum(pi_samples * z_mask, dim=2) # [b,s,z]
		if not return_log_probs:
			return z_samples
		# Estimate log probability under the EBM.
		# We're making the approximation q(z_i|x) \approx EBM(z_i) / avg(w_j)
		# => log q(z_i|x) \approx -E(z_i) - logavgexp(w_j)
		log_avg_w = torch.logsumexp(log_iws, dim=2) - log(samp) # [b,s]
		e_zi = torch.sum(energies * z_one_hot, dim=2) # [b,s]
		log_probs = - e_zi - log_avg_w # [b,s]
		if sample_flag:
			print(torch.min(log_probs), torch.max(log_probs))
			quit()
		return z_samples, log_probs


	def energy_network(self, thetas, samples):
		"""
		Map thetas and z's to scalar energies.

		Parameters
		----------
		thetas : torch.Tensor
			Shape: [b,m,theta_dim]
		samples : torch.Tensor
			Standardized latent samples. Shape: [b,m,s,k]

		Returns
		-------
		energies : torch.Tensor
			Shape: [b,m,s,k]
		"""
		# Turn theta into [b,m,s,k,theta_dim]
		thetas = thetas.unsqueeze(2).unsqueeze(2) # [b,m,1,1,theta]
		thetas = thetas.expand(-1,-1,samples.shape[2],samples.shape[3],-1)
		# Concatenate to [b,m,s,k,theta+1]
		h = torch.cat([thetas, samples.unsqueeze(-1)], dim=-1)
		h = F.relu(self.e_1(h))
		h = self.e_2(h).squeeze(-1) # [b,m,s,k]
		return ENERGY_REG * torch.sigmoid(h)



if __name__ == '__main__':
	pass



###
