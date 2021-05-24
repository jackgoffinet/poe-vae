"""
MNIST halves dataset.

"""
__date__ = "January - May 2021"


import contextlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


GENERATE_FN = 'generations.pdf'
TRAIN_RECONSTRUCT_FN = 'train_reconstructions.pdf'
TEST_RECONSTRUCT_FN = 'test_reconstructions.pdf'



class MnistHalvesDataset(Dataset):
	n_modalities = 2
	modality_dim = 392
	vectorized_modalities = False

	def __init__(self, device, missingness=0.5, data_dir='data/', mode='train',
		restrict_to_label=2):
		"""
		Binary MNIST data with image halves treated as two modalities.

		The bottom half is missing at a rate given by the `missingness`
		parameter. The top half is always observed.

		Parameters
		----------
		data_fn : str
		missingness : float
		data_dir : str
		mode : {'train', 'valid', 'test'}
		restrict_to_label : None or int
		"""
		assert mode in ['train', 'valid', 'test']
		self.missingness = missingness
		train = mode in ['train', 'valid']
		dataset = datasets.MNIST(data_dir, train=train, download=True)
		data = dataset.data.view(-1,784)
		if restrict_to_label is not None:
			print("TEMP RESTRICTING TO SINGLE DIGIT!")
			labels = np.array([i[1] for i in dataset], dtype=int)
			idx = np.argwhere(labels == restrict_to_label).flatten()
			data = data[idx]
		if mode == 'train':
			data = data[:int(round(5/6 *len(data)))]
		elif mode == 'valid':
			data = data[int(round(5/6 *len(data))):]
		self.view_1 = torch.zeros(data.shape[0], 392, dtype=torch.uint8)
		self.view_2 = torch.zeros(data.shape[0], 392, dtype=torch.uint8)
		self.view_1[data[:,:392] > 127] = 1
		self.view_2[data[:,392:] > 127] = 1
		self.view_1 = self.view_1.to(device=device, dtype=torch.float32)
		self.view_2 = self.view_2.to(device=device, dtype=torch.float32)
		if missingness > 0.0:
			num_missing = int(round(missingness * len(self.view_1)))
			self.view_2[:num_missing] = np.nan


	def __len__(self):
		return len(self.view_1)


	def __getitem__(self, index):
		return self.view_1[index], self.view_2[index]


	def make_plots(self, objective, loaders, exp_dir):
		# Make filenames.
		generate_fn = os.path.join(exp_dir, GENERATE_FN)
		train_reconstruct_fn = os.path.join(exp_dir, TRAIN_RECONSTRUCT_FN)
		test_reconstruct_fn = os.path.join(exp_dir, TEST_RECONSTRUCT_FN)
		# Plot generated data.
		gen_data = objective.generate(n_samples=5) # [2][1,5,1,392]
		gen_data = np.concatenate(gen_data, axis=-1).reshape(1,1,5,784)
		self.plot(gen_data, generate_fn)
		# Plot train reconstructions.
		for batch in loaders['train']:
			recon_data = objective.reconstruct(batch) # [m][b,s,m,m_dim]
			recon_data = np.stack(recon_data, axis=-2)
			recon_data = recon_data.reshape(-1,784)[:5].reshape(1,1,5,784)
			break
		# [2,5,392]
		data = np.array([d.detach().cpu().numpy() for d in batch[:5]])[:,:5]
		data = np.concatenate([data[0], data[1]], axis=1)
		data = data.reshape(recon_data.shape) # [1,1,5,784]
		self.plot([data,recon_data], train_reconstruct_fn)
		# Plot test reconstructions.
		for batch in loaders['test']:
			recon_data = objective.reconstruct(batch) # [m][b,s,m,m_dim]
			recon_data = np.stack(recon_data, axis=-2)
			recon_data = recon_data.reshape(-1,784)[:5].reshape(1,1,5,784)
			break
		# [2,5,392]
		data = np.array([d.detach().cpu().numpy() for d in batch[:5]])[:,:5]
		data = np.concatenate([data[0], data[1]], axis=1)
		data = data.reshape(recon_data.shape) # [1,1,5,784]
		self.plot([data,recon_data], test_reconstruct_fn)


	def plot(self, data, fn):
		"""
		MnistHalves plotting

		Parameters
		----------
		data : numpy.ndarray or tuple of numpy.ndarray
			Shape: [1,1,samples,x_dim] or [n_cols][1,1,samples,x_dim]
		fn : str
		"""
		if not isinstance(data, (list,tuple)):
			data = [data]
		data_cols = []
		# For each column...
		for data_col in data:
			m, b, s, x_dim = data_col.shape
			assert (m,b,s,x_dim) == (1,1,5,784), str(data_col.shape)
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
