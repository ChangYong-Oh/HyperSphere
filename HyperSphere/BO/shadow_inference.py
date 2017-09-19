import torch
from HyperSphere.GP.inference.inference import Inference


class ShadowInference(Inference):
	def __init__(self, train_data, model):
		super(ShadowInference, self).__init__(train_data, model)

	def predict(self, pred_x, hyper=None):
		assert pred_x.size(0) == 1
		if hyper is not None:
			self.model.vec_to_param(hyper)

		shadow = pred_x.repeat(1, 1)
		shadow[0, 0] = pred_x[0, 0] * 0
		# shadow[1, 0] = 2 - pred_x[0, 0]

		k_pred_train = self.model.kernel(pred_x, self.train_x)
		k_pred_shadow = self.model.kernel(pred_x, shadow)

		K_train_noise = self.model.kernel(self.train_x) + torch.diag(self.model.likelihood(self.train_x))
		K_shadow_noise = self.model.kernel(shadow) + torch.diag(self.model.likelihood(shadow))
		K_train_shadow = self.model.kernel(self.train_x, shadow)

		shared_part, _ = torch.gesv(k_pred_train.t(), K_train_noise)
		kernel_on_identical = torch.cat([self.model.kernel(pred_x[[i], :]) for i in range(pred_x.size(0))])
		adjusted_y = self.train_y - self.model.mean(self.train_x)
		pred_mean = torch.mm(shared_part.t(), adjusted_y) + self.model.mean(pred_x)
		pred_var = kernel_on_identical - (shared_part.t() * k_pred_train).sum(1, keepdim=True)

		K_Ainv_B, _ = torch.gesv(K_train_shadow, K_train_noise)
		var_adjust_mat = K_shadow_noise - K_Ainv_B.t().mm(K_train_shadow)
		var_adjust_vec = k_pred_train.mm(K_Ainv_B) - k_pred_shadow
		var_adjust_sol, _ = torch.gesv(var_adjust_vec.t(), var_adjust_mat)

		pred_var_adjustment = var_adjust_vec.mm(var_adjust_sol)

		_, ind = torch.max(self.train_x[:, 0] == 0, 0)
		ind = ind.data.squeeze()[0]

		mean_adjust_mat = var_adjust_mat[0:1, 0:1]
		k_star_shadow = k_pred_train.mm(K_Ainv_B[:, 0:1]) - k_pred_shadow[:, 0:1]
		adjusted_y_shadow = (adjusted_y.t().mm(K_Ainv_B[:, 0:1]) - self.train_y[ind:ind+1, 0:1])

		pred_mean_adjustment = k_star_shadow * 1.0/mean_adjust_mat * adjusted_y_shadow

		return pred_mean + pred_mean_adjustment, pred_var + pred_var_adjustment


if __name__ == '__main__':
	import math
	import numpy as np
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	from torch.autograd import Variable
	from HyperSphere.GP.kernels.modules.squared_exponential import SquaredExponentialKernel
	from HyperSphere.GP.kernels.modules.matern52 import Matern52
	from HyperSphere.GP.models.gp_regression import GPRegression
	from HyperSphere.BO.acquisition_maximization import acquisition
	from HyperSphere.feature_map.functionals import periodize, periodize_lp, periodize_one, periodize_sin

	ndata = 10
	train_x = Variable(torch.FloatTensor(ndata, 2).uniform_(0, 1))
	train_x.data[0, :] = 0
	train_y = torch.cos(train_x[:, 0:1] + (train_x[:, 1:2] / math.pi * 0.5) + torch.prod(train_x, 1, keepdim=True))
	reference = torch.min(train_y).data.squeeze()[0]
	train_data = (train_x, train_y)

	model1 = GPRegression(kernel=Matern52(3, periodize))
	model2 = GPRegression(kernel=Matern52(3, periodize_one))

	# n_added = 5
	# added_x = Variable(torch.stack([torch.zeros(n_added), torch.linspace(0, 1, n_added)], 0).t())
	# added_y = train_y[0:1, 0:1].repeat(n_added, 1)
	# added_data = (torch.cat([train_x[1:], added_x], 0), torch.cat([train_y[1:], added_y], 0))

	inference1 = Inference(train_data, model1)
	inference2 = Inference(train_data, model2)
	model1.mean.const_mean.data[:] = train_y[0].data.squeeze()[0]
	model1.kernel.log_amp.data[:] = torch.log(torch.std(train_y)).data.squeeze()[0]
	learned_params = inference1.learning(n_restarts=10)
	model2.vec_to_param(model1.param_to_vec())
	x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
	pred_points = Variable(torch.from_numpy(np.vstack([x_grid.flatten(), y_grid.flatten()]).astype(np.float32)).t())
	# acq1 = acquisition(pred_points, inference1, learned_params, reference=reference)
	_, acq1 = inference1.predict(pred_points, learned_params.squeeze())
	_, acq2 = inference2.predict(pred_points, learned_params.squeeze())
	acq1 = acq1.data.numpy().reshape(x_grid.shape)
	acq2 = acq2.data.numpy().reshape(x_grid.shape)

	ax1 = plt.subplot(221)
	ax1.contour(x_grid, y_grid, acq1)
	ax1.plot(train_x.data.numpy()[:, 0], train_x.data.numpy()[:, 1], '*')
	ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
	ax2.contour(x_grid, y_grid, acq2)
	ax2.plot(train_x.data.numpy()[:, 0], train_x.data.numpy()[:, 1], '*')
	ax4 = plt.subplot(224, projection='3d')
	ax4.plot_surface(x_grid, y_grid, acq2)
	ax3 = plt.subplot(223, projection='3d')
	ax3.plot_surface(x_grid, y_grid, acq1)
	plt.show()