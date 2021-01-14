"""
Define the datasets here.

"""
__date__ = "January 2021"


import numpy as np
from torch.utils.data import Dataset



def get_mnist_data(data_fn):
	"""Load MNIST from a .csv file."""
	d = np.loadtxt(data_fn, delimiter=',')
	images, labels = d[:,1:]/255, np.array(d[:,0], dtype='int')
	return images, labels


class MnistHalvesDataset(Dataset):

	def __init__(self, data_fn, missingness=0.0, digit=2):
		"""
		MNIST data with the top and bottom halves treated as two modalities.

		Parameters
		----------
		data_fn : str
		missingness : float
		digit : int
		"""
		self.data_fn = data_fn
		self.missingness = missingness
		self.digit = digit
		images, labels = get_mnist_data(data_fn)
		idx = np.argwhere(labels == digit).flatten()
		images = images[idx]
		images = images[np.random.permutation(images.shape[0])]
		print("\tx:", images.shape)
		n, d = images.shape[0], images.shape[1]//2
		self.view_1 = np.zeros((n,d))
		self.view_2 = np.zeros((n,d))
		# Double view.
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
		return [self.view_1[index], self.view_2[index]]



class MnistMcarDataset(Dataset):

	def __init__(self, data_fn, missingness=0.0, digit=2):
		"""
		MNIST data with pixels missing completely at random.

		Parameters
		----------
		data_fn : str
		missingness : float
		digit : int
		"""
		raise NotImplementedError

	def __len__(self):
		return 0

	def __getitem__(self, index):
		return None



if __name__ == '__main__':
	pass



###
