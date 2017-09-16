from torch.nn.modules.module import Module


class GPModule(Module):

	def __init__(self):
		super(GPModule, self).__init__()

	def reset_parameters(self):
		raise NotImplementedError

	def out_of_bounds(self, vec=None):
		raise NotImplementedError

	def n_params(self):
		raise NotImplementedError

	def param_to_vec(self):
		raise NotImplementedError

	def vec_to_param(self, vec):
		raise NotImplementedError

	def prior(self, vec):
		raise NotImplementedError

	def __repr__(self):
		return self.__class__.__name__
