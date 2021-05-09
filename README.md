# PoE VAE

## product of experts variational autoencoders for multimodal data


This repo contains ...


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

# Make a VAE with two modalities, both with 392 dimensions. The VAE is simply
# a collection of different pieces, with each piece subclassing
# `torch.nn.Module`.
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

# Train.
loader = make_dataloader(...)
optimizer = torch.optim.Adam(objective)
for epoch in range(100):
  for batch in loader:
    objective.zero_grad()
    loss = objective(batch)
    loss.backward()
    optimizer.step()

```

### Applying this to your own data
...

#### Dependencies
* [Python3](https://www.python.org/) (3.5+)
* [PyTorch](https://pytorch.org) (1.1+)
* [Python Fire](https://github.com/google/python-fire) (only used in `main.py`)


#### See also:
* [MVAE repo](https://github.com/mhw32/multimodal-vae-public)
* [MMVAE repo](https://github.com/iffsid/mmvae)
* [Hyperspherical VAE Repo](https://github.com/nicola-decao/s-vae-pytorch)

#### TO DO

12. Validation set for early stopping
15. Double check DReG gradients
16. STL gradients
17. AR-ELBO
18. Well-defined proposals for EBM MLL
19. Implement EBM IWAE variants
20. Student experts
21. Compare network architectures w/ other papers
22. partial-VAE implementation
