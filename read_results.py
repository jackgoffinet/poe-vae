"""
Read agg.pt

"""

import sys
import torch



if __name__ == '__main__':
	if len(sys.argv) != 2:
		quit("Usage: python read_results.py <log_directory>")
	fn = 'logs/' + sys.argv[1] + '/agg.pt'
	d = torch.load(fn)
	print(len(d['test_mll_epoch']))
	for i, epoch in enumerate(d['test_mll_epoch']):
		valid_mll = d['valid_mll'][d['valid_mll_epoch'].index(epoch)]
		test_mll = d['test_mll'][i]
		temp = [epoch, valid_mll, test_mll]
		if valid_mll == max(d['valid_mll']):
			temp.append("Best!")
		print(temp)


###
