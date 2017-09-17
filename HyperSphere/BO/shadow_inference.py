import math
import numpy as np
import sampyl as smp

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from HyperSphere.GP.inference.inverse_bilinear_form import InverseBilinearForm
from HyperSphere.GP.inference.log_determinant import LogDeterminant
from HyperSphere.GP.inference.inference import Inference


class ShadowInference(Inference):
	def __init__(self, train_data, model):
		super(ShadowInference, self).__init__(train_data, model)

	def predict(self, pred_x, hyper=None):
		assert pred_x.size(0) == 1
		if hyper is not None:
			self.model.vec_to_param(hyper)

		shadow = None
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
		shadow_adjust_mat = K_shadow_noise - K_Ainv_B.t().mm(K_train_shadow)
		shadow_adjust_vec = k_pred_train.mm(K_Ainv_B) - k_pred_shadow
		shadow_adjust_sol, _ = torch.gesv(shadow_adjust_vec.t(), shadow_adjust_mat)

		pred_var_shadow_adjustment = shadow_adjust_vec.mm(shadow_adjust_sol)

		return pred_mean, pred_var - pred_var_shadow_adjustment