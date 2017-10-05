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
		pt_Ainv_q = k_star_nonzero.mm(Ainv_q)
		diag_pt_Ainv_p = (k_star_nonzero * Ainv_p.t()).sum(1, keepdim=True)

		input_zero_relocated = pred_x.clone()
		input_zero_relocated.data[:, 0] = 0
		# input_stationary_satellite = pred_x.clone()
		# input_stationary_satellite.data[:, 0] = 1

		K_nonzero_zero_relocated = self.model.kernel(input_nonzero_radius, input_zero_relocated)
		# K_nonzero_stationary_satellite = self.model.kernel(input_nonzero_radius, input_stationary_satellite)

		pred_mean_list = [None] * pred_x.size(0)
		pred_var_list = [None] * pred_x.size(0)
		for i in range(pred_x.size(0)):
			mu_added_input = input_zero_relocated[i:i + 1]
			adjusted_y_zero = y_zero - (self.model.mean(mu_added_input)).view(1, 1).repeat(1, n_zero_radius)
			mu_K_nonzero_zero = K_nonzero_zero_relocated[:, i:i + 1].repeat(1, n_zero_radius)
			mu_k_star = self.model.kernel(pred_x[i:i + 1], mu_added_input).view(1, 1).repeat(1, n_zero_radius)
			mu_K_noise = torch.diag((self.model.kernel(mu_added_input) + torch.diag(self.model.likelihood(mu_added_input))).view(-1).repeat(n_zero_radius))
			mu_BtAinvp_p0 = mu_K_nonzero_zero.t().mm(Ainv_p[:, i:i+1]) - mu_k_star.t()
			mu_BtAinvq_q0 = mu_K_nonzero_zero.t().mm(Ainv_q) - adjusted_y_zero
			mu_D_BtAinvB = mu_K_noise - mu_K_nonzero_zero.t().mm(K_nonzero_noise_inv).mm(mu_K_nonzero_zero)

			mu_linear_solver, _ = torch.gesv(mu_BtAinvp_p0, mu_D_BtAinvB)
			pred_mean_list[i] = pt_Ainv_q[i].view(1, 1) + mu_BtAinvq_q0.t().mm(mu_linear_solver)

			# quad_added_input = torch.cat([mu_added_input, input_stationary_satellite[i:i + 1]], 0)
			# quad_K_nonzero_zero = torch.cat([mu_K_nonzero_zero, K_nonzero_stationary_satellite[:, i:i + 1]], 1)
			# quad_k_star = torch.cat([mu_k_star, self.model.kernel(pred_x[i:i + 1], input_stationary_satellite[i:i + 1])], 1)
			# quad_K_noise = self.model.kernel(quad_added_input) + torch.diag(self.model.likelihood(quad_added_input))
			# quad_BtAinvp_p0 = quad_K_nonzero_zero.t().mm(Ainv_p[:, i:i+1]) - quad_k_star.t()
			# quad_D_BtAinvB = quad_K_noise - quad_K_nonzero_zero.t().mm(K_nonzero_noise_inv).mm(quad_K_nonzero_zero)
			#
			# quad_linear_solver, _ = torch.gesv(quad_BtAinvp_p0, quad_D_BtAinvB)
			# pred_var_list[i] = self.model.kernel(pred_x[i:i + 1]) - (diag_pt_Ainv_p[i:i + 1] + quad_BtAinvp_p0.t().mm(quad_linear_solver))
			pred_var_list[i] = self.model.kernel(pred_x[i:i + 1]) - (diag_pt_Ainv_p[i:i + 1] + mu_BtAinvp_p0.t().mm(mu_linear_solver))

		return torch.cat(pred_mean_list, 0), torch.cat(pred_var_list, 0)


if __name__ == '__main__':
	import math
	import numpy as np
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	from torch.autograd import Variable
	from HyperSphere.GP.kernels.modules.matern52 import Matern52
	from HyperSphere.GP.models.gp_regression import GPRegression
	from HyperSphere.BO.acquisition_maximization import acquisition
	from HyperSphere.feature_map.functionals import phi_reflection, phi_reflection_threshold

	ndata = 10
	train_x = Variable(torch.FloatTensor(ndata, 2).uniform_(0, 1))
	train_x.data[0, :] = 0
	train_y = torch.cos(train_x[:, 0:1] + (train_x[:, 1:2] / math.pi * 0.5) + torch.prod(train_x, 1, keepdim=True))
	reference = torch.min(train_y).data.squeeze()[0]
	train_data = (train_x, train_y)

	model1 = GPRegression(kernel=Matern52(3, phi_reflection))
	model2 = GPRegression(kernel=Matern52(3, phi_reflection))

	# n_added = 5
	# added_x = Variable(torch.stack([torch.zeros(n_added), torch.linspace(0, 1, n_added)], 0).t())
	# added_y = train_y[0:1, 0:1].repeat(n_added, 1)
	# added_data = (torch.cat([train_x[1:], added_x], 0), torch.cat([train_y[1:], added_y], 0))

	inference1 = Inference(train_data, model1)
	inference2 = ShadowInference(train_data, model2)
	model1.mean.const_mean.data[:] = train_y[0].data.squeeze()[0]
	model1.kernel.log_amp.data[:] = torch.log(torch.std(train_y)).data.squeeze()[0]
	learned_params = inference1.learning(n_restarts=10)
	model2.vec_to_param(model1.param_to_vec())
	x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
	pred_points = Variable(torch.from_numpy(np.vstack([x_grid.flatten(), y_grid.flatten()]).astype(np.float32)).t())
	# acq1 = acquisition(pred_points, inference1, learned_params, reference=reference)
	start_time = time.time()
	_, acq1 = inference1.predict(pred_points, learned_params.squeeze())
	print(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
	start_time = time.time()
	_, acq2 = inference2.predict(pred_points, learned_params.squeeze())
	print(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
	acq1 = acq1.data.numpy().reshape(x_grid.shape)
	acq2 = acq2.data.numpy().reshape(x_grid.shape)

	ax1 = plt.subplot(221)
	ax1.contour(x_grid, y_grid, acq1)
	ax1.plot(train_x.data.numpy()[:, 0], train_x.data.numpy()[:, 1], '*')
	ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
	ax2.contour(x_grid, y_grid, acq2)
	ax2.plot(train_x.data.numpy()[:, 0], train_x.data.numpy()[:, 1], '*')
	ax3 = plt.subplot(223, projection='3d')
	ax3.plot_surface(x_grid, y_grid, acq1)
	ax3.set_title('inference')
	ax3.set_zlim(0, np.max(acq1))
	ax4 = plt.subplot(224, projection='3d')
	ax4.plot_surface(x_grid, y_grid, acq2)
	ax4.set_title('shadow inference')
	ax4.set_zlim(0, np.max(acq1))
	plt.show()