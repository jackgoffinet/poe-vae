# PoE VAE

## Product of Experts VAE for multimodal data


This repo contains ...

This repository is based on code from the
[MMVAE repo](https://github.com/iffsid/mmvae).

### Modular Multimodal VAE Abstraction:

```python
import torch

class VAE(torch.nn.Module):
	"""Abstract VAE class."""

	# def __init__(self, *args):
	#		...

	def forward(self, x):
		# Encode data.
		var_dist_params = self.encoder(x)
		# Combine evidence.
		var_post_params = self.variational_strategy(*var_dist_params)
		# Make a variational posterior.
		var_post = self.variational_posterior(*var_post_params)
		# Sample from posterior.
		z_samples, log_qz = var_post.rsample(*self.sample_params)
		# Decode samples to get likelihood parameters.
		likelihood_params = self.decoder(z_samples)
		# Evaluate likelihood.
		log_like = self.likelihood.log_prob(x, likelihood_params)
		# Return the relevant things.
		return log_qz, log_like
```



### TO DO

- Add abstract classes for each component with string representations.
- Figure out how to enforce component compatibility.
