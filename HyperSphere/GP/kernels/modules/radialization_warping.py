import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from HyperSphere.GP.modules.gp_modules import GPModule, log_lower_bnd, log_upper_bnd
from HyperSphere.GP.kernels.modules.sphere_radial import SphereRadialKernel
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.GP.kernels.modules.neural_network import NeuralNetworkKernel
from HyperSphere.feature_map.functionals import x2radial
from HyperSphere.feature_map.modules.kumaraswamy import Kumaraswamy
from HyperSphere.feature_map.modules.kumaraswamy_periodize import KumaraswamyPeriodize
from HyperSphere.feature_map.modules.radius_periodize import RadiusPeriodize


class RadializationWarpingKernel(GPModule):

	def __init__(self, max_power, search_radius):
		super(RadializationWarpingKernel, self).__init__()
		self.search_radius = search_radius

		# input_warping = Kumaraswamy
		# input_warping = RadiusPeriodize
		input_warping = KumaraswamyPeriodize

		self.radius_kernel = Matern52(ndim=1, input_map=input_warping(ndim=1, max_input=search_radius), max_ls=search_radius, trainable_amp=False)
		self.product_kernel_radius = Matern52(ndim=1, input_map=input_warping(ndim=1, max_input=search_radius), max_ls=search_radius, trainable_amp=False)

		# self.radius_kernel = NeuralNetworkKernel(ndim=1, input_map=input_warping(ndim=1, max_input=search_radius), trainable_amp=False)
		# self.product_kernel_radius = NeuralNetworkKernel(ndim=1, input_map=input_warping(ndim=1, max_input=search_radius), trainable_amp=False)

		self.sphere_kernel = SphereRadialKernel(max_power=max_power, trainable_amp=False)
		self.product_kernel_sphere = SphereRadialKernel(max_power=max_power, trainable_amp=False)

		self.log_amps = Parameter(torch.FloatTensor(3))

	def reset_parameters(self):
		self.log_amps.data.normal_(std=2.0)
		self.radius_kernel.reset_parameters()
		self.sphere_kernel.reset_parameters()

	def init_parameters(self, amp):
		self.log_amps.data.fill_(amp / self.log_amps.numel()).log_()
		self.radius_kernel.init_parameters()
		self.sphere_kernel.init_parameters()
		self.radius_kernel.log_ls.data.fill_(self.search_radius * 0.5).log_()
		self.product_kernel_radius.log_ls.data.fill_(self.search_radius * 0.5).log_()

	def kernel_amp(self):
		return torch.sum(torch.exp(self.log_amps))

	def out_of_bounds(self, vec=None):
		n_own_param = self.log_amps.numel()
		if vec is None:
			if not (log_lower_bnd <= vec[:n_own_param] <= log_upper_bnd).all():
				return True
			return self.radius_kernel.out_of_bounds() or self.sphere_kernel.out_of_bounds()
		else:
			if not ((log_lower_bnd <= self.log_amps.data[:n_own_param]).all() and (self.log_amps.data[:n_own_param] <= log_upper_bnd).all()):
				return True
			n_param_radial = self.radius_kernel.n_params()
			return self.radius_kernel.out_of_bounds(vec[n_own_param:n_own_param + n_param_radial]) or self.sphere_kernel.out_of_bounds(vec[n_own_param + n_param_radial:])

	def n_params(self):
		return self.log_amps.numel() + self.radius_kernel.n_params() + self.sphere_kernel.n_params()

	def param_to_vec(self):
		vec = torch.cat([self.log_amps.data, self.radius_kernel.param_to_vec(), self.sphere_kernel.param_to_vec()])
		return vec

	def vec_to_param(self, vec):
		n_own_param = self.log_amps.numel()
		self.log_amps.data = vec[:n_own_param]
		n_param_radial = self.radius_kernel.n_params()
		self.radius_kernel.vec_to_param(vec[n_own_param:n_own_param + n_param_radial])
		self.sphere_kernel.vec_to_param(vec[n_own_param + n_param_radial:])

	def prior(self, vec):
		# Uniformative improper prior for 3 log_amps
		n_own_param = self.log_amps.numel()
		n_param_radial = self.radius_kernel.n_params()
		return self.radius_kernel.prior(vec[n_own_param:n_own_param + n_param_radial]) + self.sphere_kernel.prior(vec[n_own_param + n_param_radial:])

	def forward(self, input1, input2=None):
		radial1 = x2radial(input1)
		stabilizer = 0
		if input2 is None:
			input2 = input1
			stabilizer = Variable(torch.diag(input1.data.new(input1.size(0)).fill_(1e-6)))
		radial2 = x2radial(input2)
		r1 = radial1[:, :1]
		d1 = radial1[:, 1:]
		r2 = radial2[:, :1]
		d2 = radial2[:, 1:]
		radial_gram = self.radius_kernel(r1, r2)
		sphere_gram = self.sphere_kernel(d1, d2)
		product_gram_radius = self.product_kernel_radius(r1, r2)
		product_gram_sphere = self.product_kernel_sphere(d1, d2)

		return self.combine_kernel(radial_gram=radial_gram, sphere_gram=sphere_gram, prod_gram_radial=product_gram_radius, prod_gram_sphere=product_gram_sphere)\
		       + stabilizer * self.kernel_amp()
		# return self.combine_kernel(radial_gram=radial_gram, sphere_gram=sphere_gram) + stabilizer * self.kernel_amp()

	def forward_on_identical(self):
		return torch.sum(torch.exp(self.log_amps)) * (1 + 1e-6)

	def combine_kernel(self, radial_gram, sphere_gram, prod_gram_radial=None, prod_gram_sphere=None):
		if prod_gram_sphere is None:
			prod_gram_sphere = sphere_gram
		if prod_gram_radial is None:
			prod_gram_radial = radial_gram
		amps = torch.exp(self.log_amps)
		return prod_gram_radial * prod_gram_sphere * amps[0] + radial_gram * amps[1] + sphere_gram * amps[2]

	def __repr__(self):
		return self.__class__.__name__ + ' (max_power=' + str(self.sphere_kernel.max_power) + ')'
