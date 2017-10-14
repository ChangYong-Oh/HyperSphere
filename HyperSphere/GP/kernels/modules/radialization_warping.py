import math

import torch
from torch.autograd import Variable
from HyperSphere.GP.modules.gp_modules import GPModule, log_upper_bnd
from HyperSphere.GP.kernels.modules.sphere_radial import SphereRadialKernel
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.feature_map.functionals import x2radial
from HyperSphere.feature_map.modules.kumaraswamy import Kumaraswamy


class RadializationWarpingKernel(GPModule):

	def __init__(self, max_power, search_radius):
		super(RadializationWarpingKernel, self).__init__()
		self.search_radius = search_radius
		self.radius_kernel = Matern52(1, input_map=Kumaraswamy(ndim=1, max_input=search_radius), max_ls=2.0 * search_radius)
		self.sphere_kernel = SphereRadialKernel(max_power)
		self.n_param_radial = self.radius_kernel.n_params()

	def reset_parameters(self):
		self.radius_kernel.reset_parameters()
		self.sphere_kernel.reset_parameters()

	def init_parameters(self, amp):
		self.radius_kernel.init_parameters(amp ** 0.5)
		self.sphere_kernel.init_parameters(amp ** 0.5)

	def log_amp(self):
		return self.radius_kernel.log_amp + self.sphere_kernel.log_amp

	def out_of_bounds(self, vec=None):
		if vec is None:
			return self.radius_kernel.out_of_bounds() or self.sphere_kernel.out_of_bounds() or (self.log_amp().data < -6).any() or (self.log_amp().data > log_upper_bnd).any()
		else:
			sum_log_amp = vec[0] + vec[self.n_param_radial]
			return self.radius_kernel.out_of_bounds(vec[:self.n_param_radial]) or self.sphere_kernel.out_of_bounds(vec[self.n_param_radial:]) or (sum_log_amp < -6).any() or (sum_log_amp > log_upper_bnd).any()

	def n_params(self):
		return self.radius_kernel.n_params() + self.sphere_kernel.n_params()

	def param_to_vec(self):
		vec = torch.cat([self.radius_kernel.param_to_vec(), self.sphere_kernel.param_to_vec()])
		return vec

	def vec_to_param(self, vec):
		self.radius_kernel.vec_to_param(vec[:self.n_param_radial])
		self.sphere_kernel.vec_to_param(vec[self.n_param_radial:])

	def prior(self, vec):
		return self.radius_kernel.prior(vec[:self.n_param_radial]) + self.sphere_kernel.prior(vec[self.n_param_radial:])

	def forward(self, input1, input2=None):
		radial1 = x2radial(input1)
		stabilizer = 0
		if input2 is None:
			input2 = input1
			stabilizer = Variable(torch.diag(input1.data.new(input1.size(0)).fill_(1e-6 * math.exp(self.radius_kernel.log_amp.data[0]) * self.sphere_kernel.forward_on_identity())))
		radial2 = x2radial(input2)
		r1 = radial1[:, :1]
		d1 = radial1[:, 1:]
		r2 = radial2[:, :1]
		d2 = radial2[:, 1:]
		radial_gram = self.radius_kernel.forward(r1, r2)
		sphere_gram = self.sphere_kernel.forward(d1, d2)
		only_radial_ind = ((r1 == 0) + (r2 == 0).t()).clamp(max=1)
		sphere_gram[only_radial_ind] = self.sphere_kernel.forward_on_identity()
		return radial_gram * sphere_gram + stabilizer

	def __repr__(self):
		return self.__class__.__name__ + ' (max_power=' + str(self.sphere_kernel.max_power) + ')'
