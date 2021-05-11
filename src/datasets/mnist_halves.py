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

from .datasets import GENERATE_FN, TRAIN_RECONSTRUCT_FN, TEST_RECONSTRUCT_FN



class MnistHalvesDataset(Dataset):
	n_modalities = 2
	modality_dim = 392
	vectorized_modalities = False

	def __init__(self, device, missingness=0.5, data_dir='data/', train=True):
		"""
		Binary MNIST data with image halves treated as two modalities.

		The bottom half is missing at a rate given by the `missingness`
		parameter. The top half is always observed.

		Parameters
		----------
		data_fn : str
		missingness : float
		digit : tuple of ints
		"""
		self.missingness = missingness
		dataset = datasets.MNIST(data_dir, train=train, download=True)
		data = dataset.data.view(-1,784)
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
		generated_data = objective.generate(n_samples=5, likelihood_noise=True) # [m,1,s,m_dim]
		self.plot(generated_data, generate_fn)
		# Plot train reconstructions.
		for batch in loaders['train']:
			data = [b[:5] for b in batch]
			recon_data = objective.reconstruct(data) # [m,b,s,x]
			if len(recon_data.shape) == 5: # stratified sampling
				recon_data = recon_data[:,:,:,np.random.randint(recon_data.shape[3])]
			# recon_data = np.swapaxes(recon_data, 0, 1)
			break
		data = np.array([d.detach().cpu().numpy() for d in data])
		data = data.reshape(recon_data.shape) # [m,b,s,x]
		self.plot([data,recon_data], train_reconstruct_fn)
		# Plot test reconstructions.
		for batch in loaders['test']:
			data = [b[:5] for b in batch]
			recon_data = objective.reconstruct(data) # [m,b,s,x]
			if len(recon_data.shape) == 5: # stratified sampling
				recon_data = recon_data[:,:,:,np.random.randint(recon_data.shape[3])]
			break
		data = np.array([d.detach().cpu().numpy() for d in data])
		data = data.reshape(recon_data.shape) # [m,b,s,x]
		self.plot([data,recon_data], test_reconstruct_fn)


	def plot(self, data, fn):
		"""
		MnistHalves plotting

		data : numpy.ndarray or tuple of numpy.ndarray
			Shape: [m,b,samples,x_dim] or [n_cols][m,b,samples,x_dim]
		fn : str
		"""
		if not isinstance(data, (list,tuple)):
			data = [data]
		data_cols = []
		# For each column...
		for data_col in data:
			m, b, s, x_dim = data_col.shape
			assert m == 2 and x_dim == 392, str(data_col.shape)
			data_col = data_col.reshape(m,b*s,x_dim)
			data_col = np.concatenate([data_col[0],data_col[1]], axis=-1)
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
