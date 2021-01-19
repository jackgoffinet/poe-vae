"""
Different datasets are defined here.

"""
__date__ = "January 2021"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import torch
from torch.utils.data import Dataset


# MNIST_DEFAULT_FN = '/media/jackg/Jacks_Animal_Sounds/datasets/mnist_train.csv'
MNIST_DEFAULT_FN = '/media/jackg/Jacks_Animal_Sounds/datasets/mnist_test.csv'
print("Using temporary MNIST data!")



def load_mnist_data(data_fn):
	"""
	Load MNIST from a .csv file.

	TO DO: toggle between binarized and real-valued input
	"""
	if data_fn == '':
		data_fn = MNIST_DEFAULT_FN
	d = np.loadtxt(data_fn, delimiter=',')
	images, labels = d[:,1:]/255, np.array(d[:,0], dtype='int')
	return images, labels



class MnistHalvesDataset(Dataset):
	n_modalities = 2
	modality_dim = 392
	vectorized_modalities = False

	def __init__(self, data_fn, device, missingness=0.5, digits=(2,)):
		"""
		MNIST data with the top and bottom halves treated as two modalities.

		Parameters
		----------
		data_fn : str
		missingness : float
		digit : tuple of ints
		"""
		self.data_fn = data_fn
		self.device = device
		self.missingness = missingness
		self.digits = digits
		images, labels = load_mnist_data(data_fn)
		idx = [np.argwhere(labels == digit).flatten() for digit in digits]
		idx = np.concatenate(idx)
		images = images[idx]
		images = images[np.random.permutation(images.shape[0])]
		n, d = images.shape[0], images.shape[1]//2
		self.view_1 = np.zeros((n,d))
		self.view_2 = np.zeros((n,d))
		self.view_1[:n] = images[:,:d]
		self.view_2[:n] = images[:,d:]
		if missingness > 0.0:
			num_missing = int(round(missingness * n))
			self.view_2[:num_missing] = np.nan
		# To torch tensors.
		self.view_1 = torch.tensor(self.view_1, dtype=torch.float)
		self.view_2 = torch.tensor(self.view_2, dtype=torch.float)

	def __len__(self):
		return len(self.view_1)

	def __getitem__(self, index):
		return [ \
				self.view_1[index].to(self.device),
				self.view_2[index].to(self.device),
		]

	def plot(self, data, fn, grid_shape=None):
		"""
		TO DO: make this better!

		data : numpy.ndarray
			Shape: [m,b,samples,x_dim]
		fn : str
		"""
		if type(data) != type([]):
			data = [data]
		data_cols = []
		for data_col in data:
			m, b, s, x_dim = data_col.shape
			assert m == 2 and x_dim == 392
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



class MnistMcarDataset(Dataset):
	n_modalities = 784
	modality_dim = 1
	vectorized_modalities = True

	def __init__(self, data_fn, device, missingness=0.5, digits=(2,4,6)):
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
		images, labels = load_mnist_data(data_fn)
		idx = [np.argwhere(labels == digit).flatten() for digit in digits]
		idx = np.concatenate(idx)
		images = images[idx]
		images = images[np.random.permutation(images.shape[0])]
		n, d = images.shape[0], images.shape[1]
		if missingness > 0.0:
			num_missing = int(round(missingness * d))
			idx = np.random.randint(d, size=(n,num_missing))
			images[idx] = np.nan
		self.images = torch.tensor(images, dtype=torch.float).unsqueeze(-1)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		return self.images[index].to(self.device)



if __name__ == '__main__':
	pass



###
