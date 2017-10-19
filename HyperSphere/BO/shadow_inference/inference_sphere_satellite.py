import numpy as np

import torch
from torch.autograd import Variable

from HyperSphere.GP.inference.inference import Inference
from HyperSphere.BO.acquisition.acquisition_maximization import deepcopy_inference


class ShadowInference(Inference):
	def __init__(self, train_data, model):
		super(ShadowInference, self).__init__(train_data, model)

	def predict(self, pred_x, hyper=None):
		if hyper is not None:
			param_original = self.model.param_to_vec()
			self.matrix_update(hyper)
		k_pred_train = self.model.kernel(pred_x, self.train_x)
		input_satellite = pred_x / torch.sqrt(torch.sum(pred_x ** 2, 1, keepdim=True)) * pred_x.size(1) ** 0.5
		K_train_satellite = self.model.kernel(self.train_x, input_satellite)

		linear_solver = torch.gesv(torch.cat([k_pred_train.t(), K_train_satellite], 1), self.K_noise)[0]
		shared_part = linear_solver[:, :pred_x.size(0)].t()
		Ainv_B = linear_solver[:, pred_x.size(0):]

		pred_mean = torch.mm(shared_part, self.mean_vec) + self.model.mean(pred_x)
		pred_var = (self.model.kernel.forward_on_identical() - (shared_part * k_pred_train).sum(1, keepdim=True).clamp(min=0)).clamp(min=1e-8)

		k_satellite_pred_diag = torch.cat([self.model.kernel(pred_x[i:i + 1], input_satellite[i:i + 1]) for i in range(pred_x.size(0))], 0)
		K_satellite = self.model.kernel.forward_on_identical()
		reduction_numer = ((Ainv_B * k_pred_train.t()).sum(0).view(-1, 1) - k_satellite_pred_diag) ** 2
		Bt_Ainv_B = (Ainv_B * K_train_satellite).sum(0).view(-1, 1)
		reduction_denom = (K_satellite - Bt_Ainv_B.clamp(min=0)).clamp(min=1e-8) + self.model.likelihood(pred_x).view(-1, 1)
		reduction = reduction_numer / reduction_denom
		pred_var_reduced = (pred_var - reduction).clamp(min=1e-8)

		if hyper is not None:
			self.matrix_update(param_original)
		return pred_mean, pred_var_reduced


if __name__ == '__main__':
	import math
	from mpl_toolkits.mplot3d import Axes3D
	from copy import deepcopy
	import matplotlib.pyplot as plt
	from HyperSphere.GP.kernels.modules.radialization import RadializationKernel
	from HyperSphere.GP.models.gp_regression import GPRegression
	from HyperSphere.BO.acquisition.acquisition_maximization import acquisition

	ndata = 10
	ndim = 2
	search_radius = ndim ** 0.5
	x_input = Variable(torch.FloatTensor(ndata, ndim).uniform_(-1, 1))
	x_input.data[0, :] = 0
	x_input.data[1, :] = 1
	output = torch.cos(x_input[:, 0:1] + (x_input[:, 1:2] / math.pi * 0.5) + torch.prod(x_input, 1, keepdim=True))
	reference = torch.min(output).data.squeeze()[0]
	train_data = (x_input, output)

	model_normal = GPRegression(kernel=RadializationKernel(3, search_radius))
	model_shadow = GPRegression(kernel=RadializationKernel(3, search_radius))

	inference_normal = Inference((x_input, output), model_normal)
	inference_shadow = ShadowInference((x_input, output), model_shadow)
	inference_normal.init_parameters()
	inference_shadow.init_parameters()

	params_normal = inference_normal.learning(n_restarts=5)
	inference_shadow.matrix_update(model_normal.param_to_vec())

	if ndim == 2:
		x1_grid, x2_grid = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
		x_pred_points = Variable(torch.from_numpy(np.vstack([x1_grid.flatten(), x2_grid.flatten()]).astype(np.float32)).t())
		pred_mean_normal, pred_var_normal = inference_normal.predict(x_pred_points)
		pred_std_normal = pred_var_normal ** 0.5
		acq_normal = acquisition(x_pred_points, deepcopy_inference(inference_normal, params_normal), reference=reference)

		pred_mean_shadow, pred_var_shadow = inference_shadow.predict(x_pred_points)
		pred_std_shadow = pred_var_shadow ** 0.5
		acq_shadow = acquisition(x_pred_points, deepcopy_inference(inference_shadow, params_normal), reference=reference)

		# ShadowInference unit test
		# satellite = x_pred_points / torch.sqrt(torch.sum(x_pred_points ** 2, dim=1)).view(-1, 1) * ndim ** 0.5
		# var_input_map_list = []
		# model_sanity = deepcopy(model_shadow)
		# output = torch.cat([output, output[:1]], 0)
		# for i in range(x_pred_points.size(0)):
		# 	inference_input_map = Inference((torch.cat([satellite[i:i + 1], x_input], 0), output), model_sanity)
		# 	inference_input_map.matrix_update()
		# 	_, var_input_map = inference_input_map.predict(x_pred_points[i:i + 1])
		# 	var_input_map_list.append(var_input_map)
		# print(torch.min(pred_var_shadow).data[0], torch.max(pred_var_shadow).data[0])
		# assert (pred_var_shadow > 0).data.all()
		# assert (torch.cat(var_input_map_list, 0).data > 0).all()
		# print('l2 distance', torch.dist(pred_var_shadow, torch.cat(var_input_map_list, 0)).data[0])
		# print('l inf distance', torch.max(torch.abs(pred_var_shadow - torch.cat(var_input_map_list, 0))).data[0])
		# exit()

		fig = plt.figure()
		acq_list = [acq_normal, acq_shadow]
		pred_mean_list = [pred_mean_normal, pred_mean_shadow]
		pred_std_list = [pred_std_normal, pred_std_shadow]
		for i in range(2):
			ax = fig.add_subplot(2, 6, 6 * i + 1)
			if torch.min(acq_list[i].data) < torch.max(acq_list[i].data):
				ax.contour(x1_grid, x2_grid, acq_list[i].data.numpy().reshape(x1_grid.shape))
			ax.plot(x_input.data.numpy()[:, 0], x_input.data.numpy()[:, 1], '*')
			if i == 0:
				ax.set_ylabel('normal')
			elif i == 1:
				ax.set_ylabel('shadow')
			ax = fig.add_subplot(2, 6, 6 * i + 2, projection='3d')
			ax.plot_surface(x1_grid, x2_grid, acq_list[i].data.numpy().reshape(x1_grid.shape))
			if i == 0:
				ax.set_title('acquistion')
			ax = fig.add_subplot(2, 6, 6 * i + 3)
			ax.contour(x1_grid, x2_grid, pred_mean_list[i].data.numpy().reshape(x1_grid.shape))
			ax = fig.add_subplot(2, 6, 6 * i + 4, projection='3d')
			ax.plot_surface(x1_grid, x2_grid, pred_mean_list[i].data.numpy().reshape(x1_grid.shape))
			if i == 0:
				ax.set_title('pred mean')
			ax = fig.add_subplot(2, 6, 6 * i + 5)
			ax.contour(x1_grid, x2_grid, pred_std_list[i].data.numpy().reshape(x1_grid.shape))
			ax.plot(x_input.data.numpy()[:, 0], x_input.data.numpy()[:, 1], '*')
			ax = fig.add_subplot(2, 6, 6 * i + 6, projection='3d')
			ax.plot_surface(x1_grid, x2_grid, pred_std_list[i].data.numpy().reshape(x1_grid.shape))
			if i == 0:
				ax.set_title('pred std')

	plt.show()
