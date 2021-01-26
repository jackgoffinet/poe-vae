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
	'title': 'MNIST Halves',
	'dirs': ['67044527', '67241136', '67634354', '67830963', '77089584', '68548139', '13474357', '46032319'],
	'names': ['1vMF PoE', '2vMF PoE', '3vMF PoE', '4vMF PoE', 'Gaussian PoE', 'MMVAE MoE', 'IWAE Gaussian PoE', 'EBM PoE'],
	'colors': ['firebrick', 'orchid', 'mediumseagreen', 'hotpink', 'goldenrod', 'steelblue', 'peru', 'orange'],
	'min_val': 100.0,
}

EXP_2 = { \
	'title': 'MNIST MCAR',
	'dirs': ['27367119', '27563728', '27956946', '28153555', '08052048', '77052245'],
	'names': ['1vMF PoE', '2vMF PoE', '3vMF PoE', '4vMF PoE', 'Gaussian PoE', 'IWAE Gaussian PoE'],
	'colors': ['firebrick', 'orchid', 'mediumseagreen', 'hotpink', 'goldenrod', 'peru'],
	'min_val': -700,
}

EXP_3 = { \
	'title': 'MNIST MCAR',
	'dirs': ['91139835', '91336444', '91729662', '91926271', '77591932', '54994305', '16284639'],
	'names': ['1vMF PoE', '2vMF PoE', '3vMF PoE', '4vMF PoE', 'Gaussian PoE', 'IWAE Gaussian PoE', 'EBM PoE'],
	'colors': ['firebrick', 'orchid', 'mediumseagreen', 'hotpink', 'goldenrod', 'peru', 'orange'],
	'min_val': -500,
}



if __name__ == '__main__':
	EXP = EXP_1
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
