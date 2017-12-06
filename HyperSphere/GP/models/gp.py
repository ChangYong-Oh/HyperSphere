import torch
from HyperSphere.GP.modules.gp_modules import GPModule


class GP(GPModule):
	def __init__(self, **kwargs):
		super(GP, self).__init__()

	def reset_parameters(self):
		for m in self.children():
			if hasattr(m, 'reset_parameters'):
				m.reset_parameters()

	def kernel_amp(self):
		return self.kernel.kernel_amp()

	def out_of_bounds(self, vec=None):
		if vec is None:
			for m in self.children():
				if m.out_of_bounds():
					return True
			return False
		else:
			ind = 0
			for m in self.children():
				jump = m.n_params()
				if m.out_of_bounds(vec[ind:ind + jump]):
					return True
				ind += jump
			return False

	def n_params(self):
		cnt = 0
		for param in self.parameters():
			cnt += param.numel()
		return cnt

	def param_to_vec(self):
		flat_param_list = []
		for m in self.children():
			flat_param_list.append(m.param_to_vec())
		return torch.cat(flat_param_list)

	def vec_to_param(self, vec):
		ind = 0
		for m in self.children():
			jump = m.n_params()
			m.vec_to_param(vec[ind:ind+jump])
			ind += jump

	def elastic_vec_to_param(self, vec, func):
		ind = 0
		for name, m in self.named_children():
			jump = m.n_params()
			if name == 'kernel':
				m.elastic_vec_to_param(vec[ind:ind + jump], func)
			else:
				m.vec_to_param(vec[ind:ind + jump])
			ind += jump

	def prior(self, vec):
		prior_lik = 0
		ind = 0
		for m in self.children():
			jump = m.n_params()
			prior_lik += m.prior(vec[ind:ind + jump])
			ind += jump
		return prior_lik
