"""
Copied from the hyperspherical VAE repository:
https://github.com/nicola-decao/s-vae-pytorch


MIT License

Copyright (c) 2018 Nicola De Cao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import torch


class HypersphericalUniform(torch.distributions.Distribution):

	support = torch.distributions.constraints.real
	has_rsample = False
	_mean_carrier_measure = 0

	@property
	def dim(self):
		return self._dim

	@property
	def device(self):
		return self._device

	@device.setter
	def device(self, val):
		self._device = val if isinstance(val, torch.device) else torch.device(val)

	def __init__(self, dim, validate_args=None, device="cpu"):
		super(HypersphericalUniform, self).__init__(
			torch.Size([dim]), validate_args=validate_args
		)
		self._dim = dim
		self.device = device

	def sample(self, shape=torch.Size()):
		output = (
			torch.distributions.Normal(0, 1)
			.sample(
				(shape if isinstance(shape, torch.Size) else torch.Size([shape]))
				+ torch.Size([self._dim + 1])
			)
			.to(self.device)
		)

		return output / output.norm(dim=-1, keepdim=True)

	def entropy(self):
		return self.__log_surface_area()

	def log_prob(self, x):
		return -torch.ones(x.shape[:-1], device=self.device) * self.__log_surface_area()

	def __log_surface_area(self):
		if torch.__version__ >= "1.0.0":
			lgamma = torch.lgamma(torch.tensor([(self._dim + 1) / 2]).to(self.device))
		else:
			lgamma = torch.lgamma(
				torch.Tensor([(self._dim + 1) / 2], device=self.device)
			)
		return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - lgamma
