"""
VAE has been replaced with torch.nn.ModuleDict...
"""
__date__ = "January 2021"


import torch



class VAE():

	def __init__(self, encoder, variational_strategy, variational_posterior, \
		prior, decoder, likelihood):
		"""
		Abstract VAE class

		Parameters
		----------
		...
		"""
		self.encoder = encoder
		self.variational_strategy = variational_strategy
		self.variational_posterior = variational_posterior
		self.prior = prior
		self.decoder = decoder
		self.likelihood = likelihood
		# print("VAE", sum(p.numel() for p in self.parameters() if p.requires_grad))


	# def generate(self, N, K):
	# 	with torch.no_grad():
	# 		pz = self.pz(*self.pz_params)
	# 		latents = pz.rsample(torch.Size([N]))
	# 		px_z = self.px_z(*self.dec(latents))
	# 		data = px_z.sample(torch.Size([K]))
	# 	return data.view(-1, *data.size()[3:])


	# def reconstruct(self, data):
	# 	with torch.no_grad():
	# 		qz_x = self.qz_x(*self.enc(data))
	# 		latents = qz_x.rsample()  # no dim expansion
	# 		px_z = self.px_z(*self.dec(latents))
	# 		recon = get_mean(px_z)
	# 	return recon



if __name__ == '__main__':
	pass



###
