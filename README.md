# PoE VAE

## product of experts variational autoencoder for multimodal data


This repo contains ...


Test vMF PoE:
`python main.py -q vmf_product --latent-dim 20 --vmf-dim 3 -v vmf_poe -p uniform_hypershperical`

Test EBM PoE:
`python main.py -q ebm -v ebm -o iwae --K 1`


### Usage

```bash
$ python main.py -h
```


### Modular Multimodal VAE Abstraction

```python
import torch

# ...
```

### Applying this to your own data
...

#### Dependencies
* [Python3](https://www.python.org/) (3.5+)
* [PyTorch](https://pytorch.org)
* [Python Fire](https://github.com/google/python-fire)


#### See also:
* [MVAE repo](https://github.com/mhw32/multimodal-vae-public)
* [MMVAE repo](https://github.com/iffsid/mmvae)
* Hyperspherical VAE Repo

#### TO DO

2. Figure out how to enforce component compatibility.
12. Validation set for early stopping
14. Lightweight VAE class?
15. Double check DReG gradients
16. STL gradients
17. AR-ELBO
18. Well-defined proposals for EBM MLL
19. Implement EBM IWAE variants
20. Student experts
21. Compare network architectures w/ other papers
22. Simplify encoder/decoders
