from scipy.stats import norm

import torch
from torch.autograd import Function, Variable, gradcheck


class NormalCDF(Function):

	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		input_np = input.data if isinstance(input, Variable) else input
		input_np = (input_np.cpu() if input_np.is_cuda else input_np).numpy()

		output_np = norm.cdf(input_np)
		output = torch.from_numpy(output_np).type_as(input)
		if isinstance(input, Variable):
			output = Variable(output)

		return output

	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_variables
		grad_input = None

		input_np = input.data if isinstance(input, Variable) else input
		input_np = (input_np.cpu() if input_np.is_cuda else input_np).numpy()

		grad_np = norm.pdf(input_np)
		grad = Variable(torch.from_numpy(grad_np).type_as(input.data))

		if ctx.needs_input_grad[0]:
			grad_input = grad_output * grad

		return grad_input


def norm_cdf(input):
	if isinstance(input, Variable):
		return NormalCDF.apply(input)
	else:
		return NormalCDF.apply(input).data


if __name__ == '__main__':
	ndata = 10
	input = Variable(torch.randn(ndata), requires_grad=True)
	test = gradcheck(NormalCDF.apply, (input, ), eps=1e-4, atol=1e-3, rtol=1e-2)
	print(test)