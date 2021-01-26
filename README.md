# PoE VAE

## product of experts variational autoencoder for multimodal data


This repo contains ...


Test vMF PoE:
python main.py -q vmf_product --latent-dim 20 --vmf-dim 3 -v vmf_poe -p uniform_hypershperical

Test EBM PoE:
python main.py -q ebm -v ebm -o iwae --K 1



See also:

* [MVAE repo](https://github.com/mhw32/multimodal-vae-public)
* [MMVAE repo](https://github.com/iffsid/mmvae)


### Modular Multimodal VAE Abstraction

```python
import torch

# ...
```



### TO DO

2. Figure out how to enforce component compatibility.
12. Validation set for early stopping
14. Lightweight VAE class?
15. Double check DReG gradients
16. STL gradients
17. AR-ELBO
