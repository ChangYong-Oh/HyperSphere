import time

import torch
from torch.autograd import Variable
from HyperSphere.GP.inference.inference import Inference


class ShadowInference(Inference):
	def __init__(self, train_data, model):
		super(ShadowInference, self).__init__(train_data, model)
		self.weight = 3

	def predict(self, pred_x, hyper=None):
		if hyper is not None:
			self.model.vec_to_param(hyper)

		zero_radius_mask = (self.train_x[:, 0] == 0).data
		n_zero_radius = torch.sum(zero_radius_mask)
		n_nonzero_radius = torch.sum(~zero_radius_mask)
		_, zero_radius_ind = torch.sort(zero_radius_mask, 0, descending=True)
		zero_radius_ind = zero_radius_ind[:n_zero_radius]
		_, nonzero_radius_ind = torch.sort(~zero_radius_mask, 0, descending=True)
		nonzero_radius_ind = nonzero_radius_ind[:n_nonzero_radius]
		input_nonzero_radius = self.train_x[nonzero_radius_ind]

		K_nonzero_noise = self.model.kernel(input_nonzero_radius) + torch.diag(self.model.likelihood(input_nonzero_radius))
		K_nonzero_noise_inv, _ = torch.gesv(Variable(torch.eye(n_nonzero_radius).type_as(pred_x.data)), K_nonzero_noise)

		adjusted_y_nonzero = self.train_y[nonzero_radius_ind] - self.model.mean(input_nonzero_radius)
		y_zero = self.train_y[zero_radius_ind]

		k_star_nonzero = self.model.kernel(pred_x, input_nonzero_radius)
		Ainv_p = K_nonzero_noise_inv.mm(k_star_nonzero.t())
		Ainv_q = K_nonzero_noise_inv.mm(adjusted_y_nonzero)
		pt_Ainv_q = k_star_nonzero.mm(Ainv_q) + self.model.mean(pred_x)
		diag_pt_Ainv_p = (k_star_nonzero * Ainv_p.t()).sum(1, keepdim=True)

		input_zero_relocated = pred_x.clone()
		input_zero_relocated.data[:, 0] = 0
		input_stationary_satellite = pred_x.clone()
		input_stationary_satellite.data[:, 0] = 1

		K_nonzero_zero_relocated = self.model.kernel(input_nonzero_radius, input_zero_relocated)
		K_nonzero_stationary_satellite = self.model.kernel(input_nonzero_radius, input_stationary_satellite)

		pred_mean_list = [None] * pred_x.size(0)
		pred_var_list = [None] * pred_x.size(0)
		for i in range(pred_x.size(0)):
			mu_added_input = input_zero_relocated[i:i + 1]
			adjusted_y_zero = y_zero - (self.model.mean(mu_added_input)).view(1, 1).repeat(n_zero_radius, 1)
			mu_K_nonzero_zero = K_nonzero_zero_relocated[:, i:i + 1].repeat(1, n_zero_radius)
			mu_k_star = self.model.kernel(pred_x[i:i + 1], mu_added_input).view(1, 1).repeat(1, n_zero_radius)
			mu_K_noise = torch.diag((self.model.kernel(mu_added_input) + torch.diag(self.model.likelihood(mu_added_input))).view(-1).repeat(n_zero_radius))
			mu_BtAinvp_p0 = mu_K_nonzero_zero.t().mm(Ainv_p[:, i:i+1]) - mu_k_star.t()
			mu_BtAinvq_q0 = mu_K_nonzero_zero.t().mm(Ainv_q) - adjusted_y_zero
			mu_D_BtAinvB = mu_K_noise - mu_K_nonzero_zero.t().mm(K_nonzero_noise_inv).mm(mu_K_nonzero_zero)

			mu_linear_solver, _ = torch.gesv(mu_BtAinvp_p0, mu_D_BtAinvB)
			pred_mean_list[i] = pt_Ainv_q[i].view(1, 1) + mu_BtAinvq_q0.t().mm(mu_linear_solver) + self.model.mean(pred_x[i:i + 1])

			quad_added_input = torch.cat([mu_added_input.repeat(n_zero_radius, 1), input_stationary_satellite[i:i + 1]], 0)
			quad_K_nonzero_zero = torch.cat([mu_K_nonzero_zero, K_nonzero_stationary_satellite[:, i:i + 1]], 1)
			quad_k_star = torch.cat([mu_k_star, self.model.kernel(pred_x[i:i + 1], input_stationary_satellite[i:i + 1])], 1)
			quad_K_noise = self.model.kernel(quad_added_input) + torch.diag(self.model.likelihood(quad_added_input))
			quad_BtAinvp_p0 = quad_K_nonzero_zero.t().mm(Ainv_p[:, i:i+1]) - quad_k_star.t()
			quad_D_BtAinvB = quad_K_noise - quad_K_nonzero_zero.t().mm(K_nonzero_noise_inv).mm(quad_K_nonzero_zero)

			quad_linear_solver, _ = torch.gesv(quad_BtAinvp_p0, quad_D_BtAinvB)
			pred_var_list[i] = self.model.kernel(pred_x[i:i + 1]) - (diag_pt_Ainv_p[i:i + 1] + quad_BtAinvp_p0.t().mm(quad_linear_solver))

		return torch.cat(pred_mean_list, 0), torch.cat(pred_var_list, 0)


if __name__ == '__main__':
	import math
	import numpy as np
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	from HyperSphere.GP.kernels.modules.matern52 import Matern52
	from HyperSphere.GP.models.gp_regression import GPRegression
	from HyperSphere.BO.acquisition_maximization import acquisition
	from HyperSphere.coordinate.transformation import rect2spherical, spherical2rect, phi2rphi, rphi2phi
	from HyperSphere.feature_map.functionals import phi_reflection, phi_smooth, id_transform

	ndata = 10
	ndim = 2
	search_radius = ndim ** 0.5
	x_input = Variable(torch.FloatTensor(ndata, ndim).uniform_(-1, 1))
	x_input.data[0, :] = 0
	x_input.data[1, :] = 1
	output = torch.cos(x_input[:, 0:1] + (x_input[:, 1:2] / math.pi * 0.5) + torch.prod(x_input, 1, keepdim=True))
	reference = torch.min(output).data.squeeze()[0]
	train_data = (x_input, output)

	rphi_input = rect2spherical(x_input)
	phi_input = rphi2phi(rphi_input, search_radius)

	model_rect = GPRegression(kernel=Matern52(ndim, id_transform))
	kernel_input_map = phi_reflection
	model_sphere1 = GPRegression(kernel=Matern52(kernel_input_map.dim_change(ndim), phi_reflection))
	model_sphere2 = GPRegression(kernel=Matern52(kernel_input_map.dim_change(ndim), phi_reflection))

	inference_rect = Inference((x_input, output), model_rect)
	inference_sphere1 = Inference((phi_input, output), model_sphere1)
	inference_sphere2 = ShadowInference((phi_input, output), model_sphere2)
	inference_rect.model_param_init()
	inference_sphere1.model_param_init()
	inference_sphere2.model_param_init()

	params_rect = inference_rect.learning(n_restarts=10)
	params_sphere1 = inference_sphere1.learning(n_restarts=10)
	model_sphere2.vec_to_param(model_sphere1.param_to_vec())

	if ndim == 2:
		x1_grid, x2_grid = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
		x_pred_points = Variable(torch.from_numpy(np.vstack([x1_grid.flatten(), x2_grid.flatten()]).astype(np.float32)).t())
		pred_mean_rect, pred_var_rect = inference_rect.predict(x_pred_points)
		pred_std_rect = pred_var_rect ** 0.5
		acq_rect = acquisition(x_pred_points, inference_rect, params_rect, reference=reference)

		rphi_pred_points = rect2spherical(x_pred_points)
		phi_pred_points = rphi2phi(rphi_pred_points, search_radius)
		pred_mean_sphere1, pred_var_sphere1 = inference_sphere1.predict(phi_pred_points)
		pred_mean_sphere2, pred_var_sphere2 = inference_sphere2.predict(phi_pred_points)
		pred_std_sphere1 = pred_var_sphere1 ** 0.5
		pred_std_sphere2 = pred_var_sphere2 ** 0.5
		acq_sphere1 = acquisition(phi_pred_points, inference_sphere1, params_sphere1, reference=reference)
		acq_sphere2 = acquisition(phi_pred_points, inference_sphere2, params_sphere1, reference=reference)

		fig = plt.figure()
		acq_list = [acq_rect, acq_sphere1, acq_sphere2]
		pred_mean_list = [pred_mean_rect, pred_mean_sphere1, pred_mean_sphere2]
		pred_std_list = [pred_std_rect, pred_std_sphere1, pred_std_sphere2]
		for i in range(3):
			ax = fig.add_subplot(3, 6, 6 * i + 1)
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
