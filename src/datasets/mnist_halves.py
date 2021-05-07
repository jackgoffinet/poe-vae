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

# from .encoders_decoders import MLP, SplitLinearLayer, EncoderModalityEmbedding,\
# 		DecoderModalityEmbedding


# GENERATE_FN = 'generations.pdf'
# TRAIN_RECONSTRUCT_FN = 'train_reconstructions.pdf'
# TEST_RECONSTRUCT_FN = 'test_reconstructions.pdf'
# MNIST_TRAIN_FN = '/media/jackg/Jacks_Animal_Sounds/datasets/mnist_train.csv'
# MNIST_TEST_FN = '/media/jackg/Jacks_Animal_Sounds/datasets/mnist_test.csv'


#
# @contextlib.contextmanager
# def temp_seed(seed):
# 	"""https://stackoverflow.com/questions/49555991/"""
# 	state = np.random.get_state()
# 	np.random.seed(seed)
# 	try:
# 		yield
# 	finally:
# 		np.random.set_state(state)


# def load_mnist_data(data_fn, mode):
# 	"""
# 	Load MNIST from a .csv file.
#
# 	TO DO: toggle between binarized and real-valued input
# 	"""
# 	if data_fn == '':
# 		if mode == 'train':
# 			data_fn = MNIST_TRAIN_FN
# 		elif mode == 'test':
# 			data_fn = MNIST_TEST_FN
# 		else:
# 			raise NotImplementedError
# 	d = np.loadtxt(data_fn, delimiter=',')
# 	images, labels = d[:,1:]/255, np.array(d[:,0], dtype='int')
# 	return images, labels



class MnistHalvesDataset(Dataset):
	n_modalities = 2
	modality_dim = 392
	vectorized_modalities = False

	def __init__(self, missingness=0.5, data_dir='data/', train=True):
		"""
		MNIST data with the top and bottom halves treated as two modalities.

		Parameters
		----------
		data_fn : str
		missingness : float
		digit : tuple of ints
		"""
		self.missingness = missingness
		dataset = datasets.MNIST(data_dir, train=train, download=True)
		data = dataset.data.view(-1,784)
		self.view_1 = torch.zeros(data.shape[0], 384, dtype=torch.uint8)
		self.view_2 = torch.zeros(data.shape[0], 384, dtype=torch.uint8)
		self.view_1[data[:,:384] > 127] = 0
		self.view_2[data[:,384:] > 127] = 0
		if missingness > 0.0:
			num_missing = int(round(missingness * len(self.view_1)))
			self.view_2[:num_missing] = 255


	def __len__(self):
		return len(self.view_1)


	def __getitem__(self, index):
		return self.view_1[index], self.view_2[index],


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
			data = [b[:5] for b in batch]
			recon_data = reconstruct(model, data) # [m,b,s,x]
			if len(recon_data.shape) == 5: # stratified sampling
				recon_data = recon_data[:,:,:,np.random.randint(recon_data.shape[3])]
			# recon_data = np.swapaxes(recon_data, 0, 1)
			break
		data = np.array([d.detach().cpu().numpy() for d in data])
		data = data.reshape(recon_data.shape) # [m,b,s,x]
		self.plot([data,recon_data], train_reconstruct_fn)
		# Plot test reconstructions.
		for batch in dataloaders['test']:
			data = [b[:5] for b in batch]
			recon_data = reconstruct(model, data) # [m,b,s,x]
			if len(recon_data.shape) == 5: # stratified sampling
				recon_data = recon_data[:,:,:,np.random.randint(recon_data.shape[3])]
			break
		data = np.array([d.detach().cpu().numpy() for d in data])
		data = data.reshape(recon_data.shape) # [m,b,s,x]
		self.plot([data,recon_data], test_reconstruct_fn)


	def plot(self, data, fn):
		"""
		MnistHalves plotting

		data : numpy.ndarray
			Shape: [m,b,samples,x_dim]
		fn : str
		"""
		if type(data) != type([]):
			data = [data]
		data_cols = []
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
			assert hasattr(vae.likelihood, 'rsample')
			generated = vae.likelihood.rsample(like_params, n_samples=n_samples)
		else:
			assert hasattr(vae.likelihood, 'mean')
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
		if type(x) == type([]): # not vectorized, shape: [m][batch]
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
			assert hasattr(vae.likelihood, 'rsample')
			generated = vae.likelihood.rsample(like_params, n_samples=1)
		else:
			assert hasattr(vae.likelihood, 'mean')
			generated = vae.likelihood.mean(like_params, n_samples=1)
	if type(x) == type([]):
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
