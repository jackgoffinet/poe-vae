"""
Import different VAE components. Map component names to classes.

"""
__date__ = "January - May 2021"



from .datasets import MnistHalvesDataset
DATASET_MAP = {
	'mnist_halves': MnistHalvesDataset,
}
DATASET_KEYS = sorted(list(DATASET_MAP.keys()))


# Variational strategies
from .variational_strategies import GaussianPoeStrategy, GaussianMoeStrategy, \
		VmfPoeStrategy, EbmStrategy, LocScaleEbmStrategy
VAR_STRATEGY_MAP = {
	'gaussian_poe': GaussianPoeStrategy,
	'gaussian_moe': GaussianMoeStrategy,
	'vmf_poe': VmfPoeStrategy,
	'ebm': EbmStrategy,
	'loc_scale_ebm': LocScaleEbmStrategy,
}


# Variational posteriors
from .variational_posteriors import DiagonalGaussianPosterior, \
		DiagonalGaussianMixturePosterior, VmfProductPosterior, \
		EbmPosterior, LocScaleEbmPosterior
VAR_POSTERIOR_MAP = {
	'diag_gaussian': DiagonalGaussianPosterior,
	'diag_gaussian_mixture': DiagonalGaussianMixturePosterior,
	'vmf_product': VmfProductPosterior,
	'ebm': EbmPosterior,
	'loc_scale_ebm': LocScaleEbmPosterior,
}


# Priors
from .priors import StandardGaussianPrior, UniformHypersphericalPrior
PRIOR_MAP = {
	'standard_gaussian': StandardGaussianPrior,
	'uniform_hyperspherical': UniformHypersphericalPrior,
}


# Likelihoods
from .likelihoods import SphericalGaussianLikelihood, BernoulliLikelihood
LIKELIHOOD_MAP = {
	'spherical_gaussian': SphericalGaussianLikelihood,
	'bernoulli': BernoulliLikelihood,
}


# Objectives
from .objectives import StandardElbo, IwaeElbo, DregIwaeElbo, MmvaeQuadraticElbo
OBJECTIVE_MAP = {
	'elbo': StandardElbo,
	'iwae': IwaeElbo,
	'dreg_iwae': DregIwaeElbo,
	'mmvae_quadratic': MmvaeQuadraticElbo,
}


# Models
from .mnist_halves_model import get_vae as mnist_halves_get_vae
MODEL_MAP = {
	'mnist_halves': mnist_halves_get_vae,
}



if __name__ == '__main__':
	pass



###
