"""
Different datasets are defined here.

"""
__date__ = "January 2021"


import contextlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
import torch
from torch.utils.data import Dataset

# from .encoders_decoders import MLP, SplitLinearLayer, EncoderModalityEmbedding,\
# 		DecoderModalityEmbedding



GENERATE_FN = 'generations.pdf'
TRAIN_RECONSTRUCT_FN = 'train_reconstructions.pdf'
TEST_RECONSTRUCT_FN = 'test_reconstructions.pdf'



@contextlib.contextmanager
def temp_seed(seed):
	"""https://stackoverflow.com/questions/49555991/"""
	state = np.random.get_state()
	np.random.seed(seed)
	try:
		yield
	finally:
		np.random.set_state(state)


class MnistMcarEncoder(torch.nn.Module):

	def __init__(self, n_hidden_layers, hidden_dim, out_dims, embed_dim,\
		activation=torch.nn.ReLU):
		""" """
		super(MnistMcarEncoder, self).__init__()
		assert n_hidden_layers >= 1
		self.m = MnistMcarDataset.n_modalities
		self.m_dim = MnistMcarDataset.modality_dim
		self.embed_layer = EncoderModalityEmbedding(self.m, embed_dim)
		self.mlp = MLP([self.m_dim+embed_dim]+[hidden_dim]*n_hidden_layers, \
				activation=activation, last_activation=True)
		self.split_layer = SplitLinearLayer(hidden_dim+embed_dim, out_dims) # NOTE: HERE!


	def forward(self, x):
		x = self.embed_layer(x)
		x = self.mlp(x)
		x = self.embed_layer(x)
		x = self.split_layer(x)
		return x



class MnistMcarDecoder(torch.nn.Module):

	def __init__(self, n_hidden_layers, hidden_dim, out_dims, z_dim, embed_dim,\
		activation=torch.nn.ReLU):
		""" """
		super(MnistMcarDecoder, self).__init__()
		assert n_hidden_layers >= 1
		self.m = MnistMcarDataset.n_modalities
		self.m_dim = MnistMcarDataset.modality_dim
		self.z_dim = z_dim
		self.embed_layer = DecoderModalityEmbedding(self.m, embed_dim)
		self.mlp = MLP([self.z_dim+embed_dim]+[hidden_dim]*n_hidden_layers, \
				activation=activation, last_activation=True)
		self.split_layer = SplitLinearLayer(hidden_dim+embed_dim, out_dims)


	def forward(self, z):
		assert z.shape[-1] == self.z_dim, \
				"{}[-1]!={}".format(z.shape,self.z_dim)
		z = self.embed_layer(z)
		z = self.mlp(z)
		z = self.embed_layer(z)
		return self.split_layer(z)


class MnistMcarDataset(Dataset):
	n_modalities = 784
	modality_dim = 1
	vectorized_modalities = True
	encoder_c = MnistMcarEncoder
	decoder_c = MnistMcarDecoder

	def __init__(self, data_fn, device, missingness=0.5, \
		digits=(0,1,2,3,4,5,6,7,8,9), mode='train', seed=0):
		"""
		MNIST data with pixels missing completely at random.

		Parameters
		----------
		data_fn : str
		missingness : float
		digits : tuple of ints
		"""
		self.data_fn = data_fn
		self.device = device
		self.missingness = missingness
		self.digits = digits
		self.seed = seed
		images, labels = load_mnist_data(data_fn, mode)
		idx = [np.argwhere(labels == digit).flatten() for digit in digits]
		idx = np.concatenate(idx)
		images = images[idx]
		with temp_seed(self.seed):
			images = images[np.random.permutation(images.shape[0])]
			n, d = images.shape[0], images.shape[1]
			if missingness > 0.0:
				num_missing = int(round(missingness * d))
				for i in range(n):
					idx = np.random.permutation(d)[:num_missing]
					images[i,idx] = np.nan
		self.images = torch.tensor(images, dtype=torch.float).unsqueeze(-1)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		return self.images[index].to(self.device)

	def make_plots(self, model, datasets, dataloaders, exp_dir):
		# Make filenames.
		generate_fn = os.path.join(exp_dir, GENERATE_FN)
		train_reconstruct_fn = os.path.join(exp_dir, TRAIN_RECONSTRUCT_FN)
		test_reconstruct_fn = os.path.join(exp_dir, TEST_RECONSTRUCT_FN)
		# Plot generated data.
		generated_data = generate(model, n_samples=5) # [m,b,s,x]
		self.plot(generated_data, generate_fn)
		# Plot train reconstructions.
		for batch in dataloaders['train']:
			data = batch[:5]
			recon_data = reconstruct(model, data) # [m,b,s,x]
			recon_data = np.swapaxes(recon_data, 0, 1)
			break
		data = data.detach().cpu().numpy()
		data = data.reshape(recon_data.shape) # [m,b,s,x]
		self.plot([data,recon_data], train_reconstruct_fn)
		# Plot test reconstructions.
		for batch in dataloaders['test']:
			data = batch[:5]
			recon_data = reconstruct(model, data) # [m,b,s,x]
			recon_data = np.swapaxes(recon_data, 0, 1)
			break
		data = data.detach().cpu().numpy()
		data = data.reshape(recon_data.shape) # [m,b,s,x]
		self.plot([data,recon_data], test_reconstruct_fn)


	def plot(self, data, fn):
		"""
		MnistMCAR plotting

		data : numpy.ndarray
			Shape: [m,b,samples,x_dim]
		fn : str
		"""
		if type(data) != type([]):
			data = [data]
		data_cols = []
		for data_col in data:
			b, s, m, m_dim = data_col.shape # [1, 5, 784, 1]
			assert b == 1
			assert m_dim == 1
			assert m == 784
			data_col = data_col.reshape(s,m)
			data_col = data_col.reshape(-1,28)
			data_cols.append(data_col)
		data = np.concatenate(data_cols, axis=1)
		plt.subplots(figsize=(2,6))
		plt.imshow(data, cmap='Greys', vmin=-0.1,vmax=1.1)
		plt.axis('off')
		plt.savefig(fn)
		plt.close('all')



def generate(vae, n_samples=9, decoder_noise=False):
	"""
	Generate data with a VAE.

	Parameters
	----------
	vae :
	n_samples : int
	decoder_noise : bool

	Returns
	-------
	generated : numpy.ndarray
	"""
	vae.eval()
	with torch.no_grad():
		z_samples = vae.prior.rsample(n_samples=n_samples) # [1,n,z]
		like_params = vae.decoder(z_samples) # [param_num][m][1,n,x]
		if decoder_noise:
			assert hasattr(vae.likelihood, 'rsample'), \
					f"type {type(vae.likelihood)} has no rsample attribute!"
			generated = vae.likelihood.rsample(like_params, n_samples=n_samples)
		else:
			assert hasattr(vae.likelihood, 'mean'), \
					f"type {type(vae.likelihood)} has no mean attribute!"
			generated = vae.likelihood.mean(like_params, n_samples=n_samples)
	return np.array([g.detach().cpu().numpy() for g in generated])


def reconstruct(vae, x, decoder_noise=False):
	"""
	Reconstruct data with a VAE.

	Parameters
	----------

	Returns
	-------

	"""
	vae.eval()
	with torch.no_grad():
		nan_mask = get_nan_mask(x)
		if isinstance(x, (tuple, list)): # not vectorized, shape: [m][batch]
			for i in range(len(x)):
				x[i][nan_mask[i]] = 0.0
		else: # vectorized modalities, shape: [batch,m]
			x[nan_mask.unsqueeze(-1)] = 0.0
		# Encode data.
		var_dist_params = vae.encoder(x) # [n_params][b,m,param_dim]
		# Combine evidence.
		var_post_params = vae.variational_strategy(*var_dist_params, \
				nan_mask=nan_mask)
		# Make a variational posterior and sample.
		z_samples, _ = vae.variational_posterior(*var_post_params)
		like_params = vae.decoder(z_samples)
		if decoder_noise:
			assert hasattr(vae.likelihood, 'rsample'), \
					f"type {type(vae.likelihood)} has no rsample attribute!"
			generated = vae.likelihood.rsample(like_params, n_samples=1)
		else:
			assert hasattr(vae.likelihood, 'mean'), \
					f"type {type(vae.likelihood)} has no mean attribute!"
			generated = vae.likelihood.mean(like_params, n_samples=1)
	if isinstance(x, (tuple, list)):
		return np.array([g.detach().cpu().numpy() for g in generated])
	return generated.detach().cpu().numpy()



def get_nan_mask(xs):
	"""Return a mask indicating which minibatch items are NaNs."""
	if type(xs) == type([]):
		return [torch.isnan(x[:,0]) for x in xs]
	else:
		return torch.isnan(xs[:,:,0]) # [b,m]



if __name__ == '__main__':
	pass



###
