import torch
from HyperSphere.GP.inference.inference import Inference


class ShadowInference(Inference):
	def __init__(self, train_data, model):
		super(ShadowInference, self).__init__(train_data, model)

	def predict(self, pred_x, hyper=None):
		assert pred_x.size(0) == 1
		if hyper is not None:
			self.model.vec_to_param(hyper)

		shadow = pred_x.repeat(2, 1)
		shadow[0, 0] = pred_x[0, 0] * 0
		shadow[1, 0] = 2 - pred_x[0, 0]

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

		mean_adjust_mat = var_adjust_mat[0:1, 0:1]
		mean_adjust_vec = k_pred_train.mm(K_Ainv_B[:, 0:1]) - k_pred_shadow[:, 0:1]
		mean_adjust_sol, _ = torch.gesv(mean_adjust_vec.t(), mean_adjust_mat)

		ind = torch.max(self.train_x[:, 0] == 0)
		pred_mean_adjustment = mean_adjust_sol.mm(torch.cat([adjusted_y, self.train_y[ind:ind+1, 0:1] - self.model.mean(shadow[0:1])], 0))

		return pred_mean - pred_mean_adjustment, pred_var - pred_var_adjustment