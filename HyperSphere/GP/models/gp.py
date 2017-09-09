import torch
import torch.nn as nn


class GP(nn.Module):
	def __init__(self, **kwargs):
		super(GP, self).__init__()

	def n_param(self):
		cnt = 0
		for param in self.parameters():
			cnt += param.numel()
		return cnt

	def param_to_vec(self):
		flat_param_list = []
		for param in self.parameters():
			flat_param_list.append(param.data.clone().view(-1))
		return torch.cat(flat_param_list)

	def vec_to_param(self, vec):
		ind = 0
		for param in self.parameters():
			param.data = vec[ind:ind+param.numel()].clone().view(param.size())
			ind += param.numel()
