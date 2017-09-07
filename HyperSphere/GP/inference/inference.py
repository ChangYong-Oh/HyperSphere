import torch
from torch.autograd import Variable
import torch.nn as nn


class Inference(nn.Module):

	def __init__(self, train_data, model):
		super(Inference, self).__init__()
		self.model = model
		self.train_x = train_data[0]
		self.train_y = train_data[1]

	def predict(self, pred_x):
		k_pred_train = self.model.kernel(pred_x, self.train_x)
		K_noise_inv = self.model.kernel(self.train_x) + torch.diag(self.model.likelihood(self.train_x)).inverse()

		shared_part = torch.mm(k_pred_train, K_noise_inv)
		kernel_on_identical = torch.cat([self.model.kernel(self.train_x[[i], :]) for i in range(self.train_x.size(0))]).squeeze()

		pred_mean = torch.mm(shared_part, self.train_y - self.model.mean(self.train_x)) + self.model.mean(pred_x)
		pred_var = kernel_on_identical - (shared_part * k_pred_train.t()).sum(1)
		return pred_mean, pred_var


