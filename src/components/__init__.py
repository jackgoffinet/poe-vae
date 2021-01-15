"""
Import different VAE components, map component names to classes.
"""
__date__ = "January 2021"


# Datasets
from .datasets import MnistHalvesDataset, MnistMcarDataset
DATASET_MAP = {
	'mnist_halves': MnistHalvesDataset,
	'mnist_mcar': MnistMcarDataset,
}
DATASET_KEYS = sorted(list(DATASET_MAP.keys()))


# Encoders/Decoders
from .encoders_decoders import MLP
ENCODER_DECODER_MAP = {
	'mlp': MLP,
}
ENCODER_DECODER_KEYS = sorted(list(ENCODER_DECODER_MAP.keys()))


# Variational strategies
from .variational_strategies import GaussianPoeStrategy, GaussianMixtureStrategy
VARIATIONAL_STRATEGY_MAP = {
	'gaussian_poe': GaussianPoeStrategy,
	'gaussian_moe': GaussianMixtureStrategy,
}
VARIATIONAL_STRATEGY_KEYS = sorted(list(VARIATIONAL_STRATEGY_MAP.keys()))


# Variational posteriors
from .variational_posteriors import DiagonalGaussianPosterior
VARIATIONAL_POSTERIOR_MAP = {
	'diag_gaussian': DiagonalGaussianPosterior,
}
VARIATIONAL_POSTERIOR_KEYS = sorted(list(VARIATIONAL_POSTERIOR_MAP.keys()))


# Priors
from .priors import StandardGaussianPrior
PRIOR_MAP = {
	'standard_gaussian': StandardGaussianPrior,
}
PRIOR_KEYS = sorted(list(PRIOR_MAP.keys()))


# Likelihoods
from .likelihoods import SphericalGaussianLikelihood
LIKELIHOOD_MAP = {
	'spherical_gaussian': SphericalGaussianLikelihood,
}
LIKELIHOOD_KEYS = sorted(list(LIKELIHOOD_MAP.keys()))


# Objectives
from .objectives import elbo, iwae
OBJECTIVE_MAP = {
	'elbo': elbo,
	'iwae': iwae,
}
OBJECTIVE_KEYS = sorted(list(OBJECTIVE_MAP.keys()))



if __name__ == '__main__':
	pass



###
