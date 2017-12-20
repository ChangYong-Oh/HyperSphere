import numpy as np

import torch
from torch.autograd import Variable
from torch.autograd._functions.linalg import Potrf

from HyperSphere.GP.inference.inference import Inference
from HyperSphere.BO.acquisition.acquisition_maximization import deepcopy_inference


class ShadowInference(Inference):
	def __init__(self, train_data, model):
		super(ShadowInference, self).__init__(train_data, model)
		origin_mask = torch.sum(self.train_x ** 2, 1) == 0
		n_origin = torch.sum(origin_mask).data[0]
		assert n_origin == 1
		self.zero_radius_ind = None
		ind_origin = torch.sort(origin_mask, 0, descending=True)[1]
		self.ind_origin = ind_origin[:n_origin]
		self.ind_nonorigin = ind_origin[n_origin:]
		self.train_x_origin = self.train_x.index_select(0, self.ind_origin)
		self.train_x_nonorigin = self.train_x.index_select(0, self.ind_nonorigin)
		self.train_x_nonorigin_radius = torch.sum(self.train_x_nonorigin ** 2, 1, keepdim=True) ** 0.5
		self.train_x_nonorigin_sphere = self.train_x_nonorigin / self.train_x_nonorigin_radius
		self.cholesky_nonorigin = None
		self.cholesky_nonorigin_inverse = None

	def cholesky_update(self, hyper):
		self.gram_mat_update(hyper)
		gram_mat_nonorigin = self.gram_mat.index_select(0, self.ind_nonorigin).index_select(1, self.ind_nonorigin)
		eye_mat = torch.eye(gram_mat_nonorigin.size(0)).type_as(gram_mat_nonorigin.data)
		chol_jitter = 0
		while True:
			try:
				self.cholesky_nonorigin = Potrf.apply(gram_mat_nonorigin + Variable(eye_mat) * chol_jitter, False)
				torch.gesv(gram_mat_nonorigin[:, :1], self.cholesky_nonorigin)
				break
			except RuntimeError:
				chol_jitter = gram_mat_nonorigin.data[0, 0] * 1e-6 if chol_jitter == 0 else chol_jitter * 10
		self.jitter = chol_jitter

	def predict(self, pred_x, hyper=None, in_optimization=False):
		if hyper is not None:
			param_original = self.model.param_to_vec()
			self.cholesky_update(hyper)
		kernel_max = self.model.kernel.forward_on_identical().data[0]
		n_pred, n_dim = pred_x.size()
		pred_x_radius = torch.sqrt(torch.sum(pred_x ** 2, 1, keepdim=True))
		assert (pred_x_radius.data > 0).all()
		pred_x_sphere = pred_x / pred_x_radius
		satellite = pred_x_sphere * pred_x.size(1) ** 0.5

		one_radius = Variable(torch.ones(1, 1)).type_as(self.train_x)
		K_non_ori_radius = self.model.kernel.radius_kernel(self.train_x_nonorigin_radius, one_radius * 0)
		K_non_ori_sphere = self.model.kernel.sphere_kernel(self.train_x_nonorigin_sphere, pred_x_sphere)
		K_non_ori = K_non_ori_radius.view(-1, 1) * K_non_ori_sphere
		K_non_pre = self.model.kernel(self.train_x_nonorigin, pred_x)
		K_non_sat = self.model.kernel(self.train_x_nonorigin, satellite)
		K_ori_pre_diag = self.model.kernel.radius_kernel(pred_x_radius, one_radius * 0)
		K_ori_sat_diag = self.model.kernel.radius_kernel(one_radius * 0, one_radius * n_dim ** 0.5).repeat(n_pred, 1)
		K_sat_pre_diag = self.model.kernel.radius_kernel(pred_x_radius, one_radius * n_dim ** 0.5)

		chol_B = torch.cat([K_non_ori, K_non_pre, self.mean_vec.index_select(0, self.ind_nonorigin), K_non_sat], 1)
		chol_solver = torch.gesv(chol_B, self.cholesky_nonorigin)[0]
		chol_solver_q = chol_solver[:, :n_pred]
		chol_solver_k = chol_solver[:, n_pred:n_pred * 2]
		chol_solver_y = chol_solver[:, n_pred * 2:n_pred * 2 + 1]
		chol_solver_q_bar_0 = chol_solver[:, n_pred * 2 + 1:]

		sol_p_sqr = kernel_max + self.model.likelihood(pred_x).view(-1, 1) + self.jitter - (chol_solver_q ** 2).sum(0).view(-1, 1)
		if not (sol_p_sqr.data >= 0).all():
			if not in_optimization:
				neg_mask = sol_p_sqr.data < 0
				neg_val = sol_p_sqr.data[neg_mask]
				min_neg_val = torch.min(neg_val)
				max_neg_val = torch.max(neg_val)
				kernel_max = self.model.kernel.forward_on_identical().data[0]
				print('p')
				print('negative %d/%d pred_var range %.4E(%.4E) ~ %.4E(%.4E)' % (torch.sum(neg_mask), sol_p_sqr.numel(), min_neg_val, min_neg_val / kernel_max, max_neg_val, max_neg_val / kernel_max))
				print('kernel max %.4E / noise variance %.4E' % (kernel_max, torch.exp(self.model.likelihood.log_noise_var.data)[0]))
				print('jitter %.4E' % self.jitter)
				print('-' * 50)
		sol_p = torch.sqrt(sol_p_sqr.clamp(min=1e-12))
		sol_k_bar = (K_ori_pre_diag - (chol_solver_q * chol_solver_k).sum(0).view(-1, 1)) / sol_p
		sol_y_bar = (self.mean_vec.index_select(0, self.ind_origin) - torch.mm(chol_solver_q.t(), chol_solver_y)) / sol_p
		sol_q_bar_1 = (K_ori_sat_diag - (chol_solver_q * chol_solver_q_bar_0).sum(0).view(-1, 1)) / sol_p

		sol_p_bar_sqr = kernel_max + self.model.likelihood(pred_x).view(-1, 1) + self.jitter - (chol_solver_q_bar_0 ** 2).sum(0).view(-1, 1) - (sol_q_bar_1 ** 2)
		if not (sol_p_bar_sqr.data >= 0).all():
			if not in_optimization:
				neg_mask = sol_p_bar_sqr.data < 0
				neg_val = sol_p_bar_sqr.data[neg_mask]
				min_neg_val = torch.min(neg_val)
				max_neg_val = torch.max(neg_val)
				kernel_max = self.model.kernel.forward_on_identical().data[0]
				print('p bar')
				print('negative %d/%d pred_var range %.4E(%.4E) ~ %.4E(%.4E)' % (torch.sum(neg_mask), sol_p_bar_sqr.numel(), min_neg_val, min_neg_val / kernel_max, max_neg_val, max_neg_val / kernel_max))
				print('kernel max %.4E / noise variance %.4E' % (kernel_max, torch.exp(self.model.likelihood.log_noise_var.data)[0]))
				print('jitter %.4E' % self.jitter)
				print('-' * 50)
		sol_p_bar = torch.sqrt(sol_p_bar_sqr.clamp(min=1e-12))

		sol_k_tilde = (K_sat_pre_diag - (chol_solver_q_bar_0 * chol_solver_k).sum(0).view(-1, 1) - sol_k_bar * sol_q_bar_1) / sol_p_bar

		pred_mean = torch.mm(chol_solver_k.t(), chol_solver_y) + sol_k_bar * sol_y_bar + self.model.mean(pred_x)
		pred_var = self.model.kernel.forward_on_identical() - (chol_solver_k ** 2).sum(0).view(-1, 1) - sol_k_bar ** 2 - sol_k_tilde ** 2

		if not (pred_var.data >= 0).all():
			if not in_optimization:
				neg_mask = pred_var.data < 0
				neg_val = pred_var.data[neg_mask]
				min_neg_val = torch.min(neg_val)
				max_neg_val = torch.max(neg_val)
				kernel_max = self.model.kernel.forward_on_identical().data[0]
				print('predictive variance')
				print('negative %d/%d pred_var range %.4E(%.4E) ~ %.4E(%.4E)' % (torch.sum(neg_mask), pred_var.numel(), min_neg_val, min_neg_val / kernel_max, max_neg_val, max_neg_val / kernel_max))
				print('kernel max %.4E / noise variance %.4E' % (kernel_max, torch.exp(self.model.likelihood.log_noise_var.data)[0]))
				print('jitter %.4E' % self.jitter)
				print('-' * 50)
		numerically_stable = (pred_var.data >= 0).all()
		zero_pred_var = (pred_var.data <= 0).all()

		if hyper is not None:
			self.cholesky_update(param_original)
		return pred_mean, pred_var.clamp(min=1e-12), numerically_stable, zero_pred_var

	def negative_log_likelihood(self, hyper=None):
		if hyper is not None:
			param_original = self.model.param_to_vec()
			self.cholesky_update(hyper)
		kernel_max = self.model.kernel.forward_on_identical().data[0]
		n_nonorigin = self.train_x_nonorigin_radius.size(0)
		one_radius = Variable(torch.ones(1, 1)).type_as(self.train_x)
		K_non_ori_rel_radius = self.model.kernel.radius_kernel(self.train_x_nonorigin_radius, one_radius * 0).repeat(1, n_nonorigin)
		K_non_ori_rel_sphere = self.model.kernel.sphere_kernel(self.train_x_nonorigin_sphere, self.train_x_nonorigin_sphere)
		K_non_ori_rel = K_non_ori_rel_radius * K_non_ori_rel_sphere

		chol_solver = torch.gesv(torch.cat([self.mean_vec.index_select(0, self.ind_nonorigin), K_non_ori_rel], 1), self.cholesky_nonorigin)[0]
		chol_solver_y = chol_solver[:, :1]
		chol_solver_q = chol_solver[:, 1:]
		sol_p_sqr = kernel_max + self.model.likelihood(self.train_x_origin).repeat(n_nonorigin, 1) + self.jitter - (chol_solver_q ** 2).sum(0).view(-1, 1)
		if not (sol_p_sqr.data >= 0).all():
			neg_mask = sol_p_sqr.data < 0
			neg_val = sol_p_sqr.data[neg_mask]
			min_neg_val = torch.min(neg_val)
			max_neg_val = torch.max(neg_val)
			kernel_max = self.model.kernel.forward_on_identical().data[0]
			print('nll p')
			print('negative %d/%d pred_var range %.4E(%.4E) ~ %.4E(%.4E)' % (torch.sum(neg_mask), sol_p_sqr.numel(), min_neg_val, min_neg_val / kernel_max, max_neg_val, max_neg_val / kernel_max))
			print('kernel max %.4E / noise variance %.4E' % (kernel_max, torch.exp(self.model.likelihood.log_noise_var.data)[0]))
			print('jitter %.4E' % self.jitter)
			print('-' * 50)
		sol_p = torch.sqrt(sol_p_sqr.clamp(min=1e-12))
		sol_y_i = (self.mean_vec.index_select(0, self.ind_origin) - chol_solver_q.t().mm(chol_solver_y)) / sol_p

		nll = 0.5 * (torch.sum(chol_solver_y ** 2) + torch.mean(sol_y_i ** 2)) + torch.sum(torch.log(torch.diag(self.cholesky_nonorigin))) + torch.mean(torch.log(sol_p)) + 0.5 * self.train_y.size(0) * np.log(2 * np.pi)
		if hyper is not None:
			self.cholesky_update(param_original)
		return nll


if __name__ == '__main__':
	import math
	from mpl_toolkits.mplot3d import Axes3D
	from copy import deepcopy
	import matplotlib.pyplot as plt
	from HyperSphere.GP.kernels.modules.radialization import RadializationKernel
	from HyperSphere.GP.models.gp_regression import GPRegression
	from HyperSphere.BO.acquisition.acquisition_maximization import acquisition

	ndata = 3
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
	inference_shadow.cholesky_update(model_normal.param_to_vec())

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
		satellite = x_pred_points / torch.sqrt(torch.sum(x_pred_points ** 2, dim=1)).view(-1, 1) * ndim ** 0.5
		var_input_map_list = []
		jitter_list = []
		model_sanity = deepcopy(model_shadow)
		output = torch.cat([output, output[:1]], 0)
		for i in range(x_pred_points.size(0)):
			inference_input_map = Inference((torch.cat([satellite[i:i + 1], x_input], 0), output), model_sanity)
			inference_input_map.cholesky_update(model_normal.param_to_vec())

			# inference_input_map.gram_mat_update()
			# inference_input_map.jitter = inference_shadow.jitter
			# eye_mat = Variable(torch.eye(inference_input_map.gram_mat.size(0)).type_as(inference_input_map.gram_mat.data))
			# inference_input_map.cholesky = Potrf.apply(inference_input_map.gram_mat + eye_mat * inference_input_map.jitter, False)

			jitter_list.append(inference_input_map.jitter)
			_, var_input_map = inference_input_map.predict(x_pred_points[i:i + 1])
			var_input_map_list.append(var_input_map)
		pred_var_input_map = torch.cat(var_input_map_list, 0)
		jitter = torch.from_numpy(np.array(jitter_list)).view(-1, 1).type_as(x_pred_points.data)
		shadow_jitter = jitter.clone().fill_(inference_shadow.jitter)
		data = torch.cat([pred_var_shadow.data, pred_var_input_map.data, jitter, shadow_jitter], 1)
		print(torch.min(pred_var_shadow).data[0], torch.max(pred_var_shadow).data[0])
		print('l2 distance', torch.dist(pred_var_shadow, pred_var_input_map).data[0])
		print('l-inf distance', torch.max(torch.abs(pred_var_shadow - pred_var_input_map)).data[0])

		mask_more = (pred_var_shadow < pred_var_input_map).data
		print('fake data var < element wise var', torch.sum(mask_more))
		ind_differ = torch.sort(mask_more, 0, descending=True)[1][:torch.sum(mask_more)].squeeze()
		print('decreased jitter', torch.sum(jitter < shadow_jitter))

		mask_less = (pred_var_shadow > pred_var_input_map).data
		print('fake data var > element wise var', torch.sum(mask_less))
		if torch.sum(mask_less) > 0:
			ind_less = torch.sort(mask_less, 0, descending=True)[1][:torch.sum(mask_less)].squeeze()
		# exit()

		fig = plt.figure()
		acq_list = [acq_normal, acq_shadow]
		pred_mean_list = [pred_mean_normal, pred_mean_shadow]
		pred_std_list = [pred_std_normal, pred_std_shadow]
		for i in range(2):
			ax = fig.add_subplot(2, 6, 6 * i + 1)
			if torch.min(acq_list[i].data) < torch.max(acq_list[i].data):
				ax.contour(x1_grid, x2_grid, acq_list[i].data.numpy().reshape(x1_grid.shape))
			ax.plot(x_input.data.numpy()[:, 0], x_input.data.numpy()[:, 1], 'rx')
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
			if torch.min(pred_std_list[i].data) != torch.max(pred_std_list[i].data):
				ax.contour(x1_grid, x2_grid, pred_std_list[i].data.numpy().reshape(x1_grid.shape))
			ax.plot(x_input.data.numpy()[:, 0], x_input.data.numpy()[:, 1], 'rx')
			ax = fig.add_subplot(2, 6, 6 * i + 6, projection='3d')
			ax.plot_surface(x1_grid, x2_grid, pred_std_list[i].data.numpy().reshape(x1_grid.shape))
			if i == 0:
				ax.set_title('pred std')

	plt.show()
