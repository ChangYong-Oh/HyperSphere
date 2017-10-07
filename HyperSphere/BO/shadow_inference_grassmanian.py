import time
from copy import deepcopy

import torch
from torch.autograd import Variable
from HyperSphere.GP.inference.inference import Inference
from HyperSphere.feature_map.functionals import id_transform


class ShadowInference(Inference):
	def __init__(self, train_data, model):
		super(ShadowInference, self).__init__(train_data, model)

	def predict(self, pred_x, hyper=None):
		if hyper is not None:
			self.model.vec_to_param(hyper)

		n_pred, ndim = pred_x.size()

		K_noise = self.model.kernel(self.train_x) + torch.diag(self.model.likelihood(self.train_x))
		K_noise_inv, _ = torch.gesv(Variable(torch.eye(self.train_x.size(0)).type_as(pred_x.data)), K_noise)

		adjusted_y = self.train_y - self.model.mean(self.train_x)

		k_star = self.model.kernel(pred_x, self.train_x)
		Ainv_p = K_noise_inv.mm(k_star.t())
		Ainv_q = K_noise_inv.mm(adjusted_y)
		pt_Ainv_q = k_star.mm(Ainv_q)
		diag_pt_Ainv_p = (k_star * Ainv_p.t()).sum(1, keepdim=True)

		pred_x_radius = torch.sqrt(torch.sum(pred_x ** 2, 1, keepdim=True))
		normalized_pred_x = pred_x / pred_x_radius
		input_stationary_satellite = normalized_pred_x * ndim ** 0.5

		K_nonzero_stationary_satellite = self.model.kernel(self.train_x, input_stationary_satellite)

		pred_var_list = [None] * n_pred
		for i in range(n_pred):

			quad_added_input = input_stationary_satellite[i:i + 1]
			quad_K_nonzero_zero = K_nonzero_stationary_satellite[:, i:i + 1]
			quad_k_star = self.model.kernel(pred_x[i:i + 1], input_stationary_satellite[i:i + 1])
			quad_K_noise = self.model.kernel(quad_added_input) + torch.diag(self.model.likelihood(quad_added_input))
			quad_BtAinvp_p0 = quad_K_nonzero_zero.t().mm(Ainv_p[:, i:i+1]) - quad_k_star.t()
			quad_D_BtAinvB = quad_K_noise - quad_K_nonzero_zero.t().mm(K_noise_inv).mm(quad_K_nonzero_zero)

			quad_linear_solver, _ = torch.gesv(quad_BtAinvp_p0, quad_D_BtAinvB)
			pred_var_list[i] = self.model.kernel(pred_x[i:i + 1]) - (diag_pt_Ainv_p[i:i + 1] + quad_BtAinvp_p0.t().mm(quad_linear_solver))

		return pt_Ainv_q, torch.cat(pred_var_list, 0)


if __name__ == '__main__':
	import math
	import numpy as np
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
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