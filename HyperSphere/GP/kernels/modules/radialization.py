import torch
from torch.autograd import Variable
from HyperSphere.GP.modules.gp_modules import GPModule, log_upper_bnd
from HyperSphere.GP.kernels.modules.sphere_radial import SphereRadialKernel
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.feature_map.functionals import x2radial


class RadializationKernel(GPModule):

	def __init__(self, max_power, search_radius, radius_input_map=None):
		super(RadializationKernel, self).__init__()
		self.search_radius = search_radius
		self.radius_kernel = Matern52(ndim=1, max_ls=2.0 * search_radius, input_map=radius_input_map)
		self.sphere_kernel = SphereRadialKernel(max_power)

	def reset_parameters(self):
		self.radius_kernel.reset_parameters()
		self.sphere_kernel.reset_parameters()

	def init_parameters(self, amp):
		self.radius_kernel.init_parameters(amp)
		self.sphere_kernel.init_parameters()

	def log_kernel_amp(self):
		return self.radius_kernel.log_amp

	def out_of_bounds(self, vec=None):
		if vec is None:
			return self.radius_kernel.out_of_bounds() or self.sphere_kernel.out_of_bounds()
		else:
			n_param_super = self.radius_kernel.n_params()
			return self.radius_kernel.out_of_bounds(vec[:n_param_super]) or self.sphere_kernel.out_of_bounds(vec[n_param_super:])

	def n_params(self):
		return self.radius_kernel.n_params() + self.sphere_kernel.n_params()

	def param_to_vec(self):
		vec = torch.cat([self.radius_kernel.param_to_vec(), self.sphere_kernel.param_to_vec()])
		return vec

	def vec_to_param(self, vec):
		n_param_radius = self.radius_kernel.n_params()
		self.radius_kernel.vec_to_param(vec[:n_param_radius])
		self.sphere_kernel.vec_to_param(vec[n_param_radius:])

	def prior(self, vec):
		n_param_radius = self.radius_kernel.n_params()
		return self.radius_kernel.prior(vec[:n_param_radius]) + self.sphere_kernel.prior(vec[n_param_radius:])

	def forward_on_identical(self):
		return torch.exp(self.radius_kernel.log_amp) * (1 + 1e-6)

	def forward(self, input1, input2=None):
		radial1 = x2radial(input1)
		stabilizer = 0
		if input2 is None:
			input2 = input1
			stabilizer = Variable(torch.diag(input1.data.new(input1.size(0)).fill_(1e-6 * torch.exp(self.log_kernel_amp()).data.squeeze()[0])))
		radial2 = x2radial(input2)
		r1 = radial1[:, :1]
		d1 = radial1[:, 1:]
		r2 = radial2[:, :1]
		d2 = radial2[:, 1:]
		radial_gram = self.radius_kernel(r1, r2)
		sphere_gram = self.sphere_kernel(d1, d2)
		# only_radial_ind = ((r1 == 0) + (r2 == 0).t()).clamp(max=1)
		# sphere_gram[only_radial_ind] = 1.0
		return radial_gram * sphere_gram + stabilizer

	def __repr__(self):
		return self.__class__.__name__ + ' (max_power=' + str(self.sphere_kernel.max_power) + ')'
