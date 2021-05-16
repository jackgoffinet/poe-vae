"""
Import different VAE components. Map component names to classes.

"""
__date__ = "January - May 2021"


# Datasets
from .datasets import MnistHalvesDataset, MnistPixelsDataset
DATASET_MAP = {
	'mnist_halves': MnistHalvesDataset,
	'mnist_pixels': MnistPixelsDataset
}


# Variational strategies
from .variational_strategies import GaussianPoeStrategy, GaussianMoeStrategy, \
		VmfPoeStrategy, LocScaleEbmStrategy
VAR_STRATEGY_MAP = {
	'gaussian_poe': GaussianPoeStrategy,
	'gaussian_moe': GaussianMoeStrategy,
	'vmf_poe': VmfPoeStrategy,
	'loc_scale_ebm': LocScaleEbmStrategy,
}


# Variational posteriors
from .variational_posteriors import DiagonalGaussianPosterior, \
		DiagonalGaussianMixturePosterior, VmfProductPosterior, \
		LocScaleEbmPosterior
VAR_POSTERIOR_MAP = {
	'diag_gaussian': DiagonalGaussianPosterior,
	'diag_gaussian_mixture': DiagonalGaussianMixturePosterior,
	'vmf_product': VmfProductPosterior,
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
from .objectives import StandardElbo, IwaeElbo, DregIwaeElbo, MmvaeElbo, \
		MvaeElbo, ArElbo
OBJECTIVE_MAP = {
	'elbo': StandardElbo,
	'iwae': IwaeElbo,
	'dreg_iwae': DregIwaeElbo,
	'mmvae_elbo': MmvaeElbo,
	'mvae_elbo': MvaeElbo,
	'ar_elbo': ArElbo,
}


# Models
from .mnist_halves_model import get_vae as mnist_halves_get_vae
from .mnist_pixels_model import get_vae as mnist_pixels_get_vae
MODEL_MAP = {
	'mnist_halves': mnist_halves_get_vae,
	'mnist_pixels': mnist_pixels_get_vae,
}



if __name__ == '__main__':
	pass



###
