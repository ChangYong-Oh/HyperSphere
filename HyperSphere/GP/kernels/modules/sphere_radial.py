import sampyl as smp

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from HyperSphere.GP.modules.gp_modules import GPModule, log_lower_bnd, log_upper_bnd
from HyperSphere.feature_map.functionals import id_transform


class SphereRadialKernel(GPModule):

	def __init__(self, max_power, input_map=None):
		super(SphereRadialKernel, self).__init__()
		self.max_power = max_power
		self.log_amp_const = Parameter(torch.FloatTensor(1))
		self.log_amp_power = Parameter(torch.FloatTensor(max_power))
		if input_map is not None:
			self.input_map = input_map
		else:
			self.input_map = id_transform

	def reset_parameters(self):
		self.log_amp_const.data.normal_()
		self.log_amp_power.data.normal_()
		if isinstance(self.input_map, GPModule):
			self.input_map.reset_parameters()

	def init_parameters(self):
		self.log_amp_const.data.fill_(0)
		self.log_amp_power.data.fill_(0)
		if isinstance(self.input_map, GPModule):
			self.input_map.init_parameters()

	def out_of_bounds(self, vec=None):
		if vec is None:
			if (self.log_amp_power.data < log_lower_bnd).any() or (self.log_amp_power.data > log_upper_bnd).any():
				return True
			if isinstance(self.input_map, GPModule):
				return self.input_map.out_of_bounds()
			return (self.log_amp_const.data < log_lower_bnd).any() or (self.log_amp_const.data > log_upper_bnd).any()
		else:
			if isinstance(self.input_map, GPModule):
				return self.input_map.out_of_bounds(vec[1+self.max_power:])
			return (vec[:1+self.max_power] < log_lower_bnd).any() or (vec[:1+self.max_power] > log_upper_bnd).any()

	def n_params(self):
		cnt = self.max_power + 1
		if isinstance(self.input_map, GPModule):
			for p in self.input_map.parameters():
				cnt += p.numel()
		return cnt

	def param_to_vec(self):
		flat_param_list = [self.log_amp_const.data, self.log_amp_power.data]
		if isinstance(self.input_map, GPModule):
			flat_param_list.append(self.input_map.param_to_vec())
		return torch.cat(flat_param_list)

	def vec_to_param(self, vec):
		self.log_amp_const.data = vec[:1]
		self.log_amp_power.data = vec[1:1+self.max_power]
		if isinstance(self.input_map, GPModule):
			self.input_map.vec_to_param(vec[1+self.max_power:])

	def prior(self, vec):
		log_lik = smp.normal(vec[:1], 0, 2) + smp.normal(vec[1:], 0, 2)
		if isinstance(self.input_map, GPModule):
			log_lik += self.input_map.prior(vec[1:])
		return log_lik

	def forward(self, input1, input2=None):
		stabilizer = 0
		if input2 is None:
			input2 = input1
			stabilizer = Variable(torch.diag(input1.data.new(input1.size(0)).fill_(1e-6)))
		inner_prod = input1.mm(input2.t())
		sum_exp = torch.exp(self.log_amp_const) + torch.sum(torch.exp(self.log_amp_power))
		gram_mat = (torch.exp(self.log_amp_const) / sum_exp) + (torch.exp(self.log_amp_power[0]) / sum_exp) * inner_prod
		for p in range(1, self.max_power):
			gram_mat += (torch.exp(self.log_amp_power[p]) / sum_exp) * inner_prod ** (p + 1)
		return gram_mat + stabilizer

	def __repr__(self):
		return self.__class__.__name__ + ' (max_power=' + str(self.max_power) + ')'


if __name__ == '__main__':
	kernel = SphereRadialKernel(max_power=3)
	vec = torch.ones(4)
	vec[2:] *= 80
	print(kernel.out_of_bounds(vec))