"""
Plot training run.

"""
__date__ = "January 2021"



import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
import sys
import torch


USAGE_STR = "Usage:   $ python plot.py log_subdirectory [min_value]\n" + \
		"Example: $ python plot.py 42282539 -1000.0"
LOGGING_DIR = 'logs'
AGG_FN = 'agg.pt'



if __name__ == '__main__':
	if len(sys.argv) not in [2,3]:
		print(USAGE_STR)
		quit()
	min_value = None
	if len(sys.argv) == 3:
		min_value = float(sys.argv[2])
	fn = os.path.join(LOGGING_DIR, sys.argv[1], AGG_FN)
	if not os.path.isfile(fn):
		print('File {} does not exist!'.format(fn))
		quit()
	agg = torch.load(fn)

	# Plot.
	fig, ax = plt.subplots(figsize=(5,3))
	train_elbo = -np.array(agg['train_loss'])
	train_epoch = agg['train_epoch']
	test_elbo = -np.array(agg['test_loss'])
	test_epoch = agg['test_epoch']
	train_mll = agg['train_mll']
	train_mll_epoch = agg['train_mll_epoch']
	test_mll = agg['test_mll']
	test_mll_epoch = agg['test_mll_epoch']

	plt.plot(train_epoch, train_elbo, c='mediumseagreen', alpha=0.3, label='train ELBO')
	plt.plot(test_epoch, test_elbo, c='orchid', alpha=0.3, label='test ELBO')

	plt.scatter(train_mll_epoch, train_mll, c='mediumseagreen', alpha=0.8, label='train MLL')
	plt.scatter(test_mll_epoch, test_mll, c='orchid', alpha=0.8, label='test MLL')

	plt.title("Run {}".format(sys.argv[1]))
	plt.ylabel("MLL/ELBO")
	plt.xlabel("Epoch")
	plt.legend(loc='best')
	plt.ylim(min_value, None)

	plt.tight_layout()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.savefig('training_run_'+sys.argv[1]+'.pdf')
	plt.close('all')




###
