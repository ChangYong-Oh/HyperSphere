import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from HyperSphere.GP.modules.gp_modules import GPModule, log_lower_bnd, log_upper_bnd
from HyperSphere.feature_map.functionals import id_transform


class Kernel(GPModule):

	def __init__(self, input_map=None, trainable_amp=True):
		super(Kernel, self).__init__()
		self.trainable_amp = trainable_amp
		if trainable_amp:
			self.log_amp = Parameter(torch.FloatTensor(1))
		if input_map is not None:
			self.input_map = input_map
		else:
			self.input_map = id_transform

	def reset_parameters(self):
		if self.trainable_amp:
			self.log_amp.data.normal_()
		if isinstance(self.input_map, GPModule):
			self.input_map.reset_parameters()

	def init_parameters(self, amp):
		if self.trainable_amp:
			self.log_amp.data.fill_(amp).log_()
		if isinstance(self.input_map, GPModule):
			self.input_map.init_parameters()

	def kernel_amp(self):
		return torch.exp(self.log_amp) if self.trainable_amp else Variable(torch.ones(1))

	def out_of_bounds(self, vec=None):
		if vec is None:
			if self.trainable_amp:
				if not (log_lower_bnd <= self.log_amp.data <= log_upper_bnd).all():
					return True
			if isinstance(self.input_map, GPModule):
				return self.input_map.out_of_bounds()
			return False
		else:
			if self.trainable_amp:
				if not (log_lower_bnd <= vec[:self.trainable_amp] <= log_upper_bnd).all():
					return True
			if isinstance(self.input_map, GPModule):
				return self.input_map.out_of_bounds(vec[self.trainable_amp:])
			return False

	def n_params(self):
		cnt = self.trainable_amp
		if isinstance(self.input_map, GPModule):
			for p in self.input_map.parameters():
				cnt += p.numel()
		return cnt

	def param_to_vec(self):
		flat_param_list = [self.log_amp.data] if self.trainable_amp else []
		if isinstance(self.input_map, GPModule):
			flat_param_list.append(self.input_map.param_to_vec())
		return torch.cat(flat_param_list) if len(flat_param_list) > 0 else torch.FloatTensor(0)

	def vec_to_param(self, vec):
		if self.trainable_amp:
			self.log_amp.data = vec[:self.trainable_amp]
		if isinstance(self.input_map, GPModule):
			self.input_map.vec_to_param(vec[self.trainable_amp:])

	def prior(self, vec):
		if isinstance(self.input_map, GPModule):
			return self.input_map.prior(vec[self.trainable_amp:])
		return 0

	def forward(self, input1, input2=None):
		raise NotImplementedError
