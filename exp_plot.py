"""
Plot the results of an experiment.

"""
__date__ = "January 2021"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
import sys
import torch


LOGGING_DIR = 'logs'
AGG_FN = 'agg.pt'



EXP_1 = { \
	'title': 'MNIST Halves 90% Missing',
	'dirs': ['46583323', '58292551', '58554696', '69011097', '05033611', '85227924'],
	'names': ['5vMF IWAE', '10vMF IWAE', '20vMF IWAE', 'Gaussian IWAE', 'EBM', 'Gaussian'],
	'colors': ['gray', 'b', 'firebrick', 'darkorchid', 'mediumseagreen', 'goldenrod'],
	'min_val': 350.0,
}

EXP_2 = { \
	'title': 'MNIST MCAR 90% Missing',
	'dirs': ['51048121', '34649524', '52082347'],
	'names': ['Gaussian IWAE', 'Gaussian', 'EBM'],
	'colors': ['darkorchid', 'goldenrod', 'mediumseagreen'],
	'min_val': -1000,
}




if __name__ == '__main__':
	EXP = EXP_2
	fig, ax = plt.subplots(figsize=(5,3))
	min_value = EXP['min_val']

	for exp_dir, exp_name, exp_color in zip(EXP['dirs'], EXP['names'], EXP['colors']):
		if exp_name in ['1vMF PoE', '2vMF PoE', '4vMF PoE']:
			continue
		# Load run.
		fn = os.path.join(LOGGING_DIR, exp_dir, AGG_FN)
		if not os.path.isfile(fn):
			print('File {} does not exist!'.format(fn))
			continue
		agg = torch.load(fn)

		# Collect data.
		train_elbo = -np.array(agg['train_loss'])
		train_epoch = agg['train_epoch']
		test_elbo = -np.array(agg['test_loss'])
		test_epoch = agg['test_epoch']
		train_mll = agg['train_mll']
		train_mll_epoch = agg['train_mll_epoch']
		test_mll = agg['test_mll']
		test_mll_epoch = agg['test_mll_epoch']

		# Plot.
		# plt.plot(train_epoch, train_elbo, c='mediumseagreen', alpha=0.3, label='train ELBO')
		plt.plot(test_epoch, test_elbo, c=exp_color, alpha=0.7, label=exp_name)

		# plt.scatter(train_mll_epoch, train_mll, c='mediumseagreen', alpha=0.8, label='train MLL')
		plt.scatter(test_mll_epoch, test_mll, c=exp_color, alpha=0.8)

	plt.title(EXP['title'])
	plt.ylabel("MLL/ELBO")
	plt.xlabel("Epoch")
	plt.legend(loc='best')
	plt.ylim(min_value, None)

	plt.tight_layout()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.savefig('exp.pdf')
	plt.close('all')




###
