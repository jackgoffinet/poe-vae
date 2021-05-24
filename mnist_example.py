"""
Make plots for MNIST example.

"""
__date__ = "May 2021"

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
import seaborn as sns
import torch
import json
from scipy.interpolate import interp2d
from PIL import Image

from src.misc import AGG_FN
from src.utils import make_objective
from src.datasets.mnist_halves import MnistHalvesDataset
from src.objectives import apply_nan_mask


GAUSSIAN_FN = 'logs/44918334/state.tar'
GAUSSIAN_ARGS = 'logs/44918334/args.json'
EBM_FN = 'logs/75980341/state.tar'
EBM_ARGS = 'logs/75980341/args.json'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def load_state(objective, state_fn):
	checkpoint = torch.load(state_fn, map_location=device)
	objective.load_state_dict(checkpoint['objective_state_dict'])


def get_post_params(xs, obj, nan_mask):
	encoding = obj.encoder(xs)
	# Transpose first two dimensions: [n_params][m][b,param_dim]
	if isinstance(xs, (tuple,list)):
		encoding = tuple(tuple(e) for e in zip(*encoding))
	var_post_params = obj.variational_strategy(
			*encoding,
			nan_mask=nan_mask,
	)
	return var_post_params


def make_grid(extent, nx=350, ny=350):
	x = np.linspace(extent[0], extent[1], nx)
	y = np.linspace(extent[2], extent[3], ny)
	xv, yv = np.meshgrid(x, y)
	z = np.stack([xv,yv], axis=-1)
	return torch.tensor(z).to(device, torch.float32), x, y


def make_plot(val_1, val_2):
	rgbArray = np.zeros((val_1.shape[0],val_1.shape[1],3), dtype='int')
	rgbArray[..., 0] = val_1.clip(0,1)*255 # R
	rgbArray[..., 1] = val_2.clip(0,1)*255 # G
	rgbArray[..., 2] = val_1.clip(0,1)*255 # B
	return rgbArray



if __name__ == '__main__':
	with open(GAUSSIAN_ARGS, 'r') as f:
		lines = f.read()
	args = json.loads(lines)
	args['device'] = device
	gaussian_obj = make_objective(**args)
	gaussian_obj.to(device)
	load_state(gaussian_obj, GAUSSIAN_FN)

	with open(EBM_ARGS, 'r') as f:
		lines = f.read()
	args = json.loads(lines)
	args['device'] = device
	ebm_obj = make_objective(**args).to(device)
	ebm_obj.to(device)
	load_state(ebm_obj, EBM_FN)
	# ebm_obj.variational_posterior.k = 350//2

	dset = MnistHalvesDataset(
			device,
			missingness=0.0,
			data_dir=args['data_dir'],
			mode='train',
			restrict_to_label=2,
	)


	view_1 = dset.view_1[0:1]
	view_2 = dset.view_2[0:1]
	print(torch.isnan(view_1).sum(), torch.isnan(view_2).sum())
	xs = (view_1.to(device), view_2.to(device))
	xs, nan_mask = apply_nan_mask(xs)

	obj = ebm_obj


	# x_min, x_max = 100, -100
	# y_min, y_max = 100, -100

	def foo(xs, obj, nan_mask, true_post_extent, approx_post_extent):
		with torch.no_grad():
			var_post_params = get_post_params(xs, obj, nan_mask)
			z_samples, log_qz = obj.variational_posterior(
					*var_post_params,
					n_samples=100,
			) # [1,s,2]
			z_mean = torch.mean(z_samples, dim=1).squeeze(0).cpu().numpy()
			print("z_mean", z_mean)
			grid, x_2, y_2 = make_grid(extent=approx_post_extent)
			_, log_qz = obj.variational_posterior(
					*var_post_params,
					samples=grid,
					n_samples=10,
			) # [10,100]
			log_qz = log_qz.transpose(0,1)
			approx_post = log_qz.cpu().numpy()
			approx_post -= np.mean(approx_post)
			approx_post = np.exp(approx_post)
			approx_post /= np.sum(approx_post) + 1e-5

			grid, x_1, y_1 = make_grid(extent=true_post_extent)
			log_pz = gaussian_obj.prior(grid)
			log_likes = gaussian_obj.decode(grid, xs, nan_mask)
			true_post = log_pz + log_likes
			true_post = true_post.cpu().numpy()
			true_post -= np.mean(true_post)
			true_post = np.exp(true_post)
			true_post /= np.sum(true_post) + 1e-5

		return true_post, approx_post, x_1, y_1, x_2, y_2


	def bar(xs, obj, nan_mask, extent):
		with torch.no_grad():
			var_post_params = get_post_params(xs, obj, nan_mask)
			z_samples, log_qz = obj.variational_posterior(
					*var_post_params,
					n_samples=100,
			) # [1,s,2]
			z_mean = torch.mean(z_samples, dim=1).squeeze(0).cpu().numpy()
			print("z_mean", z_mean)
			grid, _, _ = make_grid(extent=extent)
			grid_len = grid.shape[0]
			grid = grid.view(grid.shape[0]**2,1,-1)
			_, log_qz = obj.variational_posterior(
					*var_post_params,
					samples=grid,
					n_samples=1,
			) # [10,100]
			print("log_qz", log_qz.shape)
			log_qz = log_qz.view(grid_len,grid_len).transpose(0,1)
			log_pz = gaussian_obj.prior(grid).view(grid_len,grid_len)
			log_likes = gaussian_obj.decode(grid, xs, nan_mask).view(grid_len,grid_len)
			approx_post = log_qz.cpu().numpy() #.clip(-50,None)
			approx_post -= np.max(approx_post)
			approx_post = np.exp(approx_post)
			approx_post /= np.max(approx_post) + 1e-3
			true_post = (log_likes + log_pz).cpu().numpy() #.clip(-50,None)
			true_post -= np.max(true_post)
			true_post = np.exp(true_post)
			true_post /= np.max(true_post) + 1e-3
		return true_post, approx_post



	# Second attempt.
	extent = [-3, 3, -3, 3]
	true_post, approx_post = bar(xs, obj, nan_mask, extent)
	print("true post", np.min(true_post), np.max(true_post))
	print("approx post", np.min(true_post), np.max(true_post))
	plot_1 = make_plot(true_post, approx_post)
	temp_nan_mask = torch.zeros_like(nan_mask)
	temp_nan_mask[:,1] = 1
	true_post_1, approx_post_1 = bar(xs, obj, temp_nan_mask, extent)
	print("true post_1", np.min(true_post_1), np.max(true_post_1))
	print("approx post_1", np.min(true_post_1), np.max(true_post_1))
	plot_2 = make_plot(true_post_1, approx_post_1)
	temp_nan_mask = torch.zeros_like(nan_mask)
	temp_nan_mask[:,0] = 1
	true_post_2, approx_post_2 = bar(xs, obj, temp_nan_mask, extent)
	print("true post_2", np.min(true_post_2), np.max(true_post_2))
	print("approx post_2", np.min(true_post_2), np.max(true_post_2))
	plot_3 = make_plot(true_post_2, approx_post_2)
	_, axarr = plt.subplots(ncols=3)
	axarr[0].imshow(plot_3, extent=extent)
	axarr[0].set_title("Lower Half Observed")
	axarr[1].imshow(plot_2, extent=extent)
	axarr[1].set_title("Upper Half Observed")
	axarr[2].imshow(plot_1, extent=extent)
	axarr[2].set_title("Both Halves Observed")
	plt.tight_layout()
	plt.savefig('temp.pdf')
	quit()

	# Attempt 1
	extent = [0.5,0.9,-1.2,-0.8]
	extent = [-3.5, 3.5, -3.5, 3.5]
	true_post, approx_post, x_1, y_1, x_2, y_2 = foo(xs, obj, nan_mask, extent, extent)
	interp_true = interp2d(x_1,y_1,true_post,bounds_error=False,fill_value=0.0)
	interp_approx = interp2d(x_2,y_2,approx_post,bounds_error=False,fill_value=0.0)

	temp_nan_mask = torch.zeros_like(nan_mask)
	temp_nan_mask[:,1] = 1
	true_post_1, approx_post_1, x_1, y_1, x_2, y_2 = foo(xs, obj, temp_nan_mask, extent, extent)
	interp_true_1 = interp2d(x_1,y_1,true_post_1)
	interp_approx_1 = interp2d(x_2,y_2,approx_post_1,bounds_error=False,fill_value=0.0)

	temp_nan_mask = torch.zeros_like(nan_mask)
	temp_nan_mask[:,0] = 1
	# extent_2 = [-0.2, 0.2, -3.5, -3.2]
	true_post_2, approx_post_2, x_1, y_1, x_2, y_2 = foo(xs, obj, temp_nan_mask, extent, extent)
	interp_true_2 = interp2d(x_1,y_1,true_post_2,bounds_error=False,fill_value=0.0)
	interp_approx_2 = interp2d(x_2,y_2,approx_post_2,bounds_error=False,fill_value=0.0)

	# x_min = min(np.min(x), np.min(x_1), np.min(x_2))
	# x_max = max(np.max(x), np.max(x_1), np.max(x_2))
	# y_min = min(np.min(y), np.min(y_1), np.min(y_2))
	# y_max = max(np.max(y), np.max(y_1), np.max(y_2))
	x_min, x_max = -4.0, 4.0
	y_min, y_max = -4.0, 4.0

	xx = np.linspace(extent[0], extent[1], 100)
	yy = np.linspace(extent[2], extent[3], 100)

	print("here")
	val_11 = interp_true(xx,yy)
	print(np.min(val_11), np.max(val_11))
	val_11 /= np.max(val_11) + 1e-4
	val_12 = interp_true_1(xx,yy)
	val_12 /= np.max(val_12) + 1e-4
	val_13 = interp_true_2(xx,yy)
	val_13 /= np.max(val_13) + 1e-4

	val_21 = interp_approx(xx,yy)
	val_21 /= np.max(val_21) + 1e-4
	val_22 = interp_approx_1(xx,yy)
	val_22 /= np.max(val_22) + 1e-4
	val_23 = interp_approx_2(xx,yy)
	val_23 /= np.max(val_23) + 1e-4

	print(np.min(val_11), np.max(val_11))
	print(np.min(val_21), np.max(val_21))
	plot_1 = make_plot(val_11, val_21)
	print(np.min(val_12), np.max(val_12))
	print(np.min(val_22), np.max(val_22))
	plot_2 = make_plot(val_12, val_22)
	print(np.min(val_13), np.max(val_13))
	print(np.min(val_23), np.max(val_23))
	plot_3 = make_plot(val_13, val_23)


	_, axarr = plt.subplots(ncols=3)

	axarr[0].imshow(plot_1, extent=extent)
	axarr[1].imshow(plot_2, extent=extent)
	axarr[2].imshow(plot_3, extent=extent)

	# rgbArray = np.concatenate([plot_1, plot_2, plot_3], axis=1)

	# plt.imshow(rgbArray)
	plt.savefig('temp.pdf')

	# img = Image.fromarray(np.array(rgbArray, 'uint8'))
	# img.save('temp.jpeg')

	# new_extent = (z_mean[0]+extent[0], z_mean[0]+extent[1], z_mean[1]+extent[2], z_mean[1]+extent[3])
	# _, axarr = plt.subplots(ncols=2)
	# axarr[0].imshow(
	# 		true_post,
	# 		vmin=np.min(true_post),
	# 		vmax=np.max(true_post),
	# 		extent=new_extent,
	# )
	# axarr[1].imshow(
	# 		approx_post,
	# 		vmin=np.min(approx_post),
	# 		vmax=np.max(approx_post),
	# 		extent=new_extent,
	# )
	# plt.savefig('temp.pdf')




###
