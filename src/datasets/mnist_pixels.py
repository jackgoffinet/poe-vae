"""
MNIST pixels dataset.

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



class MnistPixelsDataset(Dataset):
	n_modalities = 784
	modality_dim = 1
	vectorized_modalities = True

	def __init__(self, device, missingness=0.5, data_dir='data/', mode='train',
		seed=42):
		"""
		Binary MNIST data with image pixels treated as modalities.

		Pixels are missing completely at random at a rate given by the
		`missingness` parameter.

		Parameters
		----------
		device : torch.device
		missingness : float, optional
		data_dir : str, optional
		mode : {'train', 'valid', 'test'}, optional
		seed : int, optional
		"""
		assert mode in ['train', 'valid', 'test']
		self.missingness = missingness
		self.num_missing = int(round(missingness * 784)) # per image
		train = mode in ['train', 'valid']
		orig_data = datasets.MNIST(data_dir, train=train, download=True).data
		orig_data = orig_data.reshape(-1,784)
		if mode == 'train':
			orig_data = orig_data[:int(round(5/6 *len(orig_data)))]
		elif mode == 'valid':
			orig_data = orig_data[int(round(5/6 *len(orig_data))):]
		self.data = torch.zeros(orig_data.shape[0], 784, dtype=torch.uint8)
		self.data[orig_data > 127] = 1
		self.data = self.data.view(-1,784,1)
		self.data = self.data.to(device=device, dtype=torch.float32)
		if self.num_missing > 0:
			with local_seed(seed):
				for i in range(self.data.shape[0]):
					perm = np.random.permutation(784)[:self.num_missing]
					self.data[i,perm] = np.nan


	def __len__(self):
		return len(self.data)


	def __getitem__(self, index):
		return self.data[index]


	def make_plots(self, objective, loaders, exp_dir):
		# Make filenames.
		generate_fn = os.path.join(exp_dir, GENERATE_FN)
		train_reconstruct_fn = os.path.join(exp_dir, TRAIN_RECONSTRUCT_FN)
		test_reconstruct_fn = os.path.join(exp_dir, TEST_RECONSTRUCT_FN)
		# Plot generated data.
		gen_data = objective.generate(n_samples=5) # [m][1,s,sub_m,m_dim]
		gen_data = np.array(gen_data) # [1,1,5,784,1]
		gen_data = gen_data.reshape(1,1,5,784)
		self.plot(gen_data, generate_fn)
		# Plot train reconstructions.
		for batch in loaders['train']:
			recon_data = objective.reconstruct(batch) # [m][b,s,m,m_dim]
			recon_data = np.array(recon_data)[:,:5] # [1,5,1,784,1]
			recon_data = recon_data.reshape(1,1,5,784)
			break
		# [5,784,1]
		data = np.array([d.detach().cpu().numpy() for d in batch[:5]])
		data = data.reshape(recon_data.shape) # [1,1,5,784]
		self.plot([data,recon_data], train_reconstruct_fn)
		# Plot test reconstructions.
		for batch in loaders['test']:
			recon_data = objective.reconstruct(batch) # [m][b,s,m,m_dim]
			recon_data = np.array(recon_data)[:,:5] # [1,5,1,784,1]
			recon_data = recon_data.reshape(1,1,5,784)
			break
		# [5,784,1]
		data = np.array([d.detach().cpu().numpy() for d in batch[:5]])
		data = data.reshape(recon_data.shape) # [1,1,5,784]
		self.plot([data,recon_data], test_reconstruct_fn)


	def plot(self, data, fn):
		"""
		MnistPixels plotting

		Parameters
		----------
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
			assert (m,b,s,x_dim) == (1,1,5,784), str(data_col.shape)
			data_col = data_col.reshape(-1,28)
			data_cols.append(data_col)
		data = np.concatenate(data_cols, axis=1)
		plt.subplots(figsize=(2,6))
		plt.imshow(data, cmap='Greys', vmin=-0.1,vmax=1.1)
		plt.axis('off')
		plt.savefig(fn)
		plt.close('all')



@contextlib.contextmanager
def local_seed(seed):
	"""https://stackoverflow.com/questions/49555991/"""
	state = np.random.get_state()
	np.random.seed(seed)
	try:
		yield
	finally:
		np.random.set_state(state)



if __name__ == '__main__':
	pass



###
