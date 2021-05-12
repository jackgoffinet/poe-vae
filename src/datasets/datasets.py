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



GENERATE_FN = 'generations.pdf'
TRAIN_RECONSTRUCT_FN = 'train_reconstructions.pdf'
TEST_RECONSTRUCT_FN = 'test_reconstructions.pdf'



@contextlib.contextmanager
def local_seed(seed):
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






if __name__ == '__main__':
	pass



###
