# PoE VAE

## Product of experts variational autoencoders for multimodal data

This repo contains code for quickly and easily implementing multimodal
variational autoencoders (VAEs). This is a work in progress!


### Usage

```bash
$ python main.py --help
```


### Modular Multimodal VAE Abstraction

```python
import torch
import torch.nn as nn

from src.encoders_decoders import GatherLayer, NetworkList, SplitLinearLayer
from src.likelihoods import GroupedLikelihood, BernoulliLikelihood
from src.objectives import StandardElbo
from src.priors import StandardGaussianPrior
from src.variational_posteriors import DiagonalGaussianPosterior
from src.variational_strategies import GaussianPoeStrategy

# Make a VAE with two modalities, both with 392 dimensions, and a 20-dimensional
# latent space. The VAE is simply a collection of different pieces, with each
# piece subclassing `torch.nn.Module`.
latent_dim, m_dim = 20, 392
vae = nn.ModuleDict({
  'encoder': NetworkList(
    nn.ModuleList([
      nn.Sequential(
        nn.Linear(m_dim,200),
        nn.ReLU(),
        SplitLinearLayer(200, (latent_dim,latent_dim)),
      ),
      nn.Sequential(
        nn.Linear(m_dim,200),
        nn.ReLU(),
        SplitLinearLayer(200, (latent_dim,latent_dim)),
      ),
    ])
  ),
  'variational_strategy': GaussianPoeStrategy(),
  'variational_posterior': DiagonalGaussianPosterior(),
  'decoder': nn.Sequential(
    nn.Linear(latent_dim,200),
    nn.ReLU(),
    SplitLinearLayer(200, (m_dim,m_dim)),
    GatherLayer(),
  ),
  'likelihood': GroupedLikelihood(
    BernoulliLikelihood(),
    BernoulliLikelihood(),
  ),
  'prior': StandardGaussianPrior(),
})

# Feed the VAE to an objective. The objective determines how data is routed
# through the various VAE pieces to determine a loss. Objectives also subclass
# `torch.nn.Module`.
objective = StandardElbo(vae)

# Train the VAE like any other PyTorch model.
loader = make_dataloader(...)
optimizer = torch.optim.Adam(objective)
for epoch in range(100):
  for batch in loader:
    objective.zero_grad()
    loss = objective(batch)
    loss.backward()
    optimizer.step()

```

### Methods Implemented
* [MVAE](https://arxiv.org/abs/1802.05335)
   `--variational-strategy=gaussian_poe`
   `--variational-posterior=diag_gaussian`
   `--prior=standard_gaussian`
   `--objective=mvae_elbo`
* [MMVAE](https://arxiv.org/abs/1911.03393)
   `--variational-strategy=gaussian_moe`
   `--variational-posterior=diag_gaussian_mixture`
   `--prior=standard_gaussian`
   `--objective=mmvae_elbo`
* [s-VAE](https://arxiv.org/abs/1804.00891) (originally a single modality VAE)
   `--variational-strategy=vmf_poe`
   `--variational-posterior=vmf_product`
   `--prior=uniform_hyperspherical`
   `--objective=elbo`
* [MIWAE](https://arxiv.org/abs/1812.02633)
   `--unstructured-encoder=True`
   `--variational-posterior=diag_gaussian`
   `--prior=standard_gaussian`
   `--objective=elbo`
* [partial VAE](https://arxiv.org/abs/1809.11142) TO DO
   `--variational-strategy=permutation_invariant`
   `--variational-posterior=diag_gaussian`
   `--prior=standard_gaussian`
   `--objective=elbo`
* [VAEVAE](https://arxiv.org/abs/1912.05075)?
* [MoPoE VAE](https://arxiv.org/abs/2105.02470)?


### Applying this to your own data
Check out `src/datasets/` for some examples of how to do this. To use the
existing training framework, you will also have to modify `DATASET_MAP` and
`MODEL_MAP` in `src/param_maps.py`.


#### Dependencies
* [Python3](https://www.python.org/) (3.6+)
* [PyTorch](https://pytorch.org) (1.6+)
* [Python Fire](https://github.com/google/python-fire) (only used in `main.py`)


#### See also:
* [MVAE repo](https://github.com/mhw32/multimodal-vae-public), uses a product of
  experts strategy for combining evidence across modalities.
* [MMVAE repo](https://github.com/iffsid/mmvae), uses a mixture of experts
  strategy for combining evidence across modalities.
* [Hyperspherical VAE repo](https://github.com/nicola-decao/s-vae-pytorch), a
  VAE with a latent space defined on an n-sphere with von
  Mises-Fisher-distributed approximate posteriors.

#### TO DO

12. Validation set for early stopping
16. Implement STL gradients?
20. Student experts?
21. Compare network architectures w/ other papers
22. partial-VAE implementation
25. Add a documentation markdown file
28. Implement jackknife variational inference?
30. AR-ELBO for vMF
31. Double check unstructured recognition models work
35. Is there an easy way for Encoder and DecoderModalityEmbeddings to share parameters?
36. Test vMF KL divergence
37. Why is MVAE performing poorly?
