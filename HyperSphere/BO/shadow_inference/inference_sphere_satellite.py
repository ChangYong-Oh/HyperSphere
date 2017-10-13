import torch
from torch.autograd import Variable

from HyperSphere.GP.inference.inference import Inference


class ShadowInference(Inference):
	def __init__(self, train_data, model):
		super(ShadowInference, self).__init__(train_data, model)

	def predict(self, pred_x, hyper=None):
		if hyper is not None:
			param_original = self.model.param_to_vec()
			self.matrix_update(hyper)
		k_pred_train = self.model.kernel(pred_x, self.train_x)

		shared_part = k_pred_train.mm(self.K_noise_inv)
		kernel_on_identical = torch.cat([self.model.kernel(pred_x[[i], :]) for i in range(pred_x.size(0))])

		pred_mean = torch.mm(shared_part, self.mean_vec) + self.model.mean(pred_x)
		pred_var = kernel_on_identical - (shared_part * k_pred_train).sum(1, keepdim=True)

		satellite_weight = 1.0
		satellite_weight_sqrt = satellite_weight ** 0.5
		input_satellite = pred_x / torch.sqrt(torch.sum(pred_x ** 2, 1, keepdim=True)) * pred_x.size(1) ** 0.5

		K_train_satellite = self.model.kernel(self.train_x, input_satellite)
		Ainv_B = self.K_noise_inv.mm(K_train_satellite)
		k_satellite_pred_diag = torch.cat([self.model.kernel(pred_x[i:i + 1], input_satellite[i:i + 1]) for i in range(pred_x.size(0))], 0)
		k_satellite_diag = torch.cat([self.model.kernel(input_satellite[i:i + 1]) for i in range(pred_x.size(0))], 0) + self.model.likelihood(input_satellite).view(-1, 1)
		quad_satellite_reduction = ((Ainv_B * k_pred_train.t()).sum(0).view(-1, 1) - k_satellite_pred_diag * satellite_weight_sqrt) ** 2 / (k_satellite_diag - (Ainv_B * K_train_satellite).sum(0).view(-1, 1))
		if hyper is not None:
			self.matrix_update(param_original)
		return pred_mean, pred_var - quad_satellite_reduction


if __name__ == '__main__':
	import math
	import numpy as np
	import matplotlib.pyplot as plt
	from copy import deepcopy
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

	model_normal = GPRegression(kernel=RadializationKernel(3))
	model_shadow = GPRegression(kernel=RadializationKernel(3))

	inference_normal = Inference((x_input, output), model_normal)
	inference_shadow = ShadowInference((x_input, output), model_shadow)
	inference_normal.init_parameters()
	inference_shadow.init_parameters()

	params_normal = inference_normal.learning(n_restarts=10)
	inference_shadow.matrix_update(model_normal.param_to_vec())

	if ndim == 2:
		x1_grid, x2_grid = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
		x_pred_points = Variable(torch.from_numpy(np.vstack([x1_grid.flatten(), x2_grid.flatten()]).astype(np.float32)).t())
		pred_mean_normal, pred_var_normal = inference_normal.predict(x_pred_points)
		pred_std_normal = pred_var_normal ** 0.5
		acq_normal = acquisition(x_pred_points, inference_normal, params_normal, reference=reference)

		pred_mean_shadow, pred_var_shadow = inference_shadow.predict(x_pred_points)
		pred_std_shadow = pred_var_shadow ** 0.5
		acq_shadow = acquisition(x_pred_points, inference_shadow, params_normal, reference=reference)

		# ShadowInference unit test
		satellite = x_pred_points / torch.sqrt(torch.sum(x_pred_points ** 2, dim=1)).view(-1, 1) * ndim ** 0.5
		var_input_map_list = []
		model_sanity = deepcopy(model_shadow)
		output = torch.cat([output, output[:1]], 0)
		for i in range(x_pred_points.size(0)):
			inference_input_map = Inference((torch.cat([satellite[i:i + 1], x_input], 0), output), model_sanity)
			_, var_input_map = inference_input_map.predict(x_pred_points[i:i + 1])
			var_input_map_list.append(var_input_map)
		print(pred_var_shadow.size())
		print(torch.cat([pred_var_shadow, torch.cat(var_input_map_list, 0)], 1))
		print(torch.dist(pred_var_shadow, torch.cat(var_input_map_list, 0)))
		exit()

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
