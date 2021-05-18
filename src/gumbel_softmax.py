"""
Define a Gumbel-Softmax sampler.

Adapted from: https://github.com/YongfeiYan/Gumbel_Softmax_VAE
"""
__date__ = "January 2021"


import torch

EPS = 1e-20



def gumbel_softmax(logits, temperature=1.0, hard=True):
	"""
	Get reparamaterized samples from a Gumbel-Softmax distribution.

	Parameters
	----------
	logits : torch.Tensor
		Shape: [*,categories]
	temperature : float, optional
	hard : bool, optional
		If `hard`, the sample is pushed to a corner of the simplex, but
		gradients are passed as if the sample were still in the simplex. This
		is called the "Straight Through" Gumbel estimator in Jang, Gu, and Poole
		(2017).

	Returns
	-------
	sample : torch.Tensor
		Shape: [*,categories], same as `logits`.
	"""
	gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + EPS) + EPS)
	y = logits + gumbel_noise
	y = torch.nn.functional.softmax(y / temperature, dim=-1)
	if hard:
		if len(y.shape) == 2:
			_, ind = y.max(dim=-1)
			y_hard = torch.zeros_like(y)
			y_hard.scatter_(1, ind.view(-1, 1), 1)
		elif len(y.shape) == 3: # Pytorch doesn't have a batched maximum!
			res = []
			for i in range(y.shape[1]):
				_, ind = y[:,i].max(dim=-1)
				y_hard = torch.zeros_like(y[:,i])
				y_hard.scatter_(1, ind.view(-1, 1), 1)
				res.append(y_hard)
			y_hard = torch.stack(res, dim=1)
		else:
			raise NotImplementedError(f"len({y.shape}) > 3 not implemented!")
		# Set gradients w.r.t. y_hard equal to gradients w.r.t. y
		y_hard = (y_hard - y).detach() + y
		return y_hard
	return y



if __name__ == '__main__':
	pass



###
