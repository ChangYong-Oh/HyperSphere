from copy import deepcopy

import torch
from torch.autograd import Variable

from HyperSphere.GP.inference.inference import Inference
from HyperSphere.feature_map.functionals import id_transform


class ShadowInference(Inference):
	def __init__(self, train_data, model):
		super(ShadowInference, self).__init__(train_data, model)
		self.weight = 3

	def predict(self, pred_x, hyper=None):
		if hyper is not None:
			param_original = self.model.param_to_vec()
			self.matrix_update(hyper)

		train_x_input_map = self.model.kernel.input_map(self.train_x)
		pred_x_input_map = self.model.kernel.input_map(pred_x)
		assert torch.sum(pred_x_input_map.data[:, 0] == 0) == 0
		kernel_on_input_map = deepcopy(self.model.kernel)
		kernel_on_input_map.input_map = id_transform

		k_pred_train = kernel_on_input_map(pred_x_input_map, train_x_input_map)

		shared_part = k_pred_train.mm(self.K_noise_inv)

		pred_mean = torch.mm(shared_part, self.mean_vec) + self.model.mean(pred_x)
		pred_var = self.model.kernel.forward_on_identical() - (shared_part * k_pred_train).sum(1, keepdim=True)

		slide_boundary = pred_x_input_map.clone()
		slide_boundary[:, 0] = pred_x.size(1) ** 0.5

		K_train_boundary = kernel_on_input_map(train_x_input_map, slide_boundary)
		Ainv_B = self.K_noise_inv.mm(K_train_boundary)
		k_pred_boundary = torch.cat([kernel_on_input_map(pred_x_input_map[i:i + 1], slide_boundary[i:i + 1]) for i in range(pred_x.size(0))], 0)
		k_boundary = self.model.kernel.forward_on_identical() + self.model.likelihood(slide_boundary[:, 1:] * slide_boundary[:, :1]).view(-1, 1)
		pT_Ainv_B = (k_pred_train.t() * Ainv_B).sum(0).view(-1, 1)
		slide_boundary_quad_adjustment = (pT_Ainv_B - k_pred_boundary) ** 2 / (k_boundary - (K_train_boundary * Ainv_B).sum(0).view(-1, 1))

		if hyper is not None:
			self.matrix_update(param_original)
		return pred_mean, pred_var - slide_boundary_quad_adjustment


if __name__ == '__main__':
	import math
	import numpy as np
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	from HyperSphere.GP.kernels.modules.matern52 import Matern52
	from HyperSphere.GP.models.gp_regression import GPRegression
	from HyperSphere.BO.acquisition.acquisition_maximization import acquisition
	from HyperSphere.feature_map.functionals import id_transform

	ndata = 10
	ndim = 2
	search_radius = ndim ** 0.5
	x_input = Variable(torch.FloatTensor(ndata, ndim).uniform_(-1, 1))
	x_input.data[0, :] = 0
	x_input.data[1, :] = 1
	output = torch.cos(x_input[:, 0:1] + (x_input[:, 1:2] / math.pi * 0.5) + torch.prod(x_input, 1, keepdim=True))
	reference = torch.min(output).data.squeeze()[0]
	train_data = (x_input, output)

	model_rect = GPRegression(kernel=Matern52(ndim, id_transform))
	kernel_input_map = x_radial
	model_sphere1 = GPRegression(kernel=Matern52(kernel_input_map.dim_change(ndim), kernel_input_map))
	model_sphere2 = GPRegression(kernel=Matern52(kernel_input_map.dim_change(ndim), kernel_input_map))

	inference_rect = Inference((x_input, output), model_rect)
	inference_sphere1 = Inference((x_input, output), model_sphere1)
	inference_sphere2 = ShadowInference((x_input, output), model_sphere2)
	inference_rect.model_param_init()
	inference_sphere1.model_param_init()
	inference_sphere2.model_param_init()

	params_rect = inference_rect.learning(n_restarts=10)
	params_sphere1 = inference_sphere1.learning(n_restarts=10)
	inference_sphere2.matrix_update(model_sphere1.param_to_vec())

	if ndim == 2:
		x1_grid, x2_grid = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
		x_pred_points = Variable(torch.from_numpy(np.vstack([x1_grid.flatten(), x2_grid.flatten()]).astype(np.float32)).t())
		pred_mean_rect, pred_var_rect = inference_rect.predict(x_pred_points)
		pred_std_rect = pred_var_rect ** 0.5
		acq_rect = acquisition(x_pred_points, inference_rect, params_rect, reference=reference)

		pred_mean_sphere1, pred_var_sphere1 = inference_sphere1.predict(x_pred_points)
		pred_mean_sphere2, pred_var_sphere2 = inference_sphere2.predict(x_pred_points)
		pred_std_sphere1 = pred_var_sphere1 ** 0.5
		pred_std_sphere2 = pred_var_sphere2 ** 0.5
		acq_sphere1 = acquisition(x_pred_points, inference_sphere1, params_sphere1, reference=reference)
		acq_sphere2 = acquisition(x_pred_points, inference_sphere2, params_sphere1, reference=reference)

		# ShadowInference unit test
		# x_pred_points_input_map = model_sphere2.kernel.input_map(x_pred_points)
		# slide_boundary = x_pred_points_input_map.clone()
		# slide_boundary[:, 0] = ndim ** 0.5
		# x_input_input_map = model_sphere2.kernel.input_map(x_input)
		# model_input_map = deepcopy(model_sphere2)
		# model_input_map.kernel.input_map = id_transform
		# var_input_map_list = []
		# output = torch.cat([output, output[:1]], 0)
		# for i in range(x_pred_points.size(0)):
		# 	inference_input_map = Inference((torch.cat([slide_boundary[i:i + 1], x_input_input_map], 0), output), model_input_map)
		# 	_, var_input_map = inference_input_map.predict(x_pred_points_input_map[i:i + 1])
		# 	var_input_map_list.append(var_input_map)
		# print(torch.cat([pred_var_sphere2, torch.cat(var_input_map_list, 0)], 1))
		# print(torch.dist(pred_var_sphere2, torch.cat(var_input_map_list, 0)))
		# exit()

		fig = plt.figure()
		acq_list = [acq_rect, acq_sphere1, acq_sphere2]
		pred_mean_list = [pred_mean_rect, pred_mean_sphere1, pred_mean_sphere2]
		pred_std_list = [pred_std_rect, pred_std_sphere1, pred_std_sphere2]
		for i in range(3):
			ax = fig.add_subplot(3, 6, 6 * i + 1)
			if torch.min(acq_list[i].data) < torch.max(acq_list[i].data):
				ax.contour(x1_grid, x2_grid, acq_list[i].data.numpy().reshape(x1_grid.shape))
			ax.plot(x_input.data.numpy()[:, 0], x_input.data.numpy()[:, 1], '*')
			if i == 0:
				ax.set_ylabel('rect')
			elif i == 1:
				ax.set_ylabel('sphere')
			elif i == 2:
				ax.set_ylabel('sphere & shadow')
			ax = fig.add_subplot(3, 6, 6 * i + 2, projection='3d')
			ax.plot_surface(x1_grid, x2_grid, acq_list[i].data.numpy().reshape(x1_grid.shape))
			if i == 0:
				ax.set_title('acquistion')
			ax = fig.add_subplot(3, 6, 6 * i + 3)
			ax.contour(x1_grid, x2_grid, pred_mean_list[i].data.numpy().reshape(x1_grid.shape))
			ax = fig.add_subplot(3, 6, 6 * i + 4, projection='3d')
			ax.plot_surface(x1_grid, x2_grid, pred_mean_list[i].data.numpy().reshape(x1_grid.shape))
			if i == 0:
				ax.set_title('pred mean')
			ax = fig.add_subplot(3, 6, 6 * i + 5)
			ax.contour(x1_grid, x2_grid, pred_std_list[i].data.numpy().reshape(x1_grid.shape))
			ax.plot(x_input.data.numpy()[:, 0], x_input.data.numpy()[:, 1], '*')
			ax = fig.add_subplot(3, 6, 6 * i + 6, projection='3d')
			ax.plot_surface(x1_grid, x2_grid, pred_std_list[i].data.numpy().reshape(x1_grid.shape))
			if i == 0:
				ax.set_title('pred std')

	plt.show()
