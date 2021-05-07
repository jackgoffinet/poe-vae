"""
Import different VAE components, map component names to classes.

"""
__date__ = "January - May 2021"


# Datasets
# from .datasets import MnistHalvesDataset, MnistMcarDataset
# 	'mnist_mcar': MnistMcarDataset,

from .datasets import MnistHalvesDataset
DATASET_MAP = {
	'mnist_halves': MnistHalvesDataset,
}
DATASET_KEYS = sorted(list(DATASET_MAP.keys()))


from .models import mnist_halves_get_vae
MODEL_MAP = {
	'mnist_halves': mnist_halves_get_vae,
}


# # Encoders/Decoders
# from .encoders_decoders import MLP
# ENCODER_DECODER_MAP = {
# 	'mlp': MLP,
# }
# ENCODER_DECODER_KEYS = sorted(list(ENCODER_DECODER_MAP.keys()))


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
VAR_STRATEGY_KEYS = sorted(list(VAR_STRATEGY_MAP.keys()))


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
VAR_POSTERIOR_KEYS = sorted(list(VAR_POSTERIOR_MAP.keys()))


# Priors
from .priors import StandardGaussianPrior, UniformHypersphericalPrior
PRIOR_MAP = {
	'standard_gaussian': StandardGaussianPrior,
	'uniform_hypershperical': UniformHypersphericalPrior,
}
PRIOR_KEYS = sorted(list(PRIOR_MAP.keys()))


# Likelihoods
from .likelihoods import SphericalGaussianLikelihood
LIKELIHOOD_MAP = {
	'spherical_gaussian': SphericalGaussianLikelihood,
}
LIKELIHOOD_KEYS = sorted(list(LIKELIHOOD_MAP.keys()))


# Objectives
from .objectives import StandardElbo, IwaeElbo, DregIwaeElbo, MmvaeQuadraticElbo
OBJECTIVE_MAP = {
	'elbo': StandardElbo,
	'iwae': IwaeElbo,
	'dreg_iwae': DregIwaeElbo,
	'mmvae_quadratic': MmvaeQuadraticElbo,
}
OBJECTIVE_KEYS = sorted(list(OBJECTIVE_MAP.keys()))



if __name__ == '__main__':
	pass



###
