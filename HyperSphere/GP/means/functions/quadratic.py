import torch
from torch.autograd import Function, Variable, gradcheck


class QuadraticMean(Function):

	@staticmethod
	def forward(ctx, input, a, b):
		ctx.save_for_backward(input, a, c)

		return a * input[:, :1] ** 2 + c

	@staticmethod
	def backward(ctx, grad_output):
		"""
		:param ctx: 
		:param grad_output: grad_output is assumed to be d[Scalar]/dK and size() is n_input 
		:return: 
		"""
		input, a, c = ctx.saved_variables
		grad_input = grad_a = grad_c = None

		if ctx.needs_input_grad[0]:
			grad_input = input.clone()
			grad_input.data.zero_()
			grad_input[:, :1] = 2 * a * input[:, :1] * grad_output
		if ctx.needs_input_grad[1]:
			grad_a = torch.sum(grad_output * input[:, :1] ** 2)
		if ctx.needs_input_grad[2]:
			grad_c = torch.sum(grad_output)

		return grad_input, grad_a, grad_c


if __name__ == '__main__':
	n1 = 3
	ndim = 5
	input_grad = False
	param_grad = not input_grad
	input = Variable(torch.randn(n1, ndim), requires_grad=input_grad)
	a = Variable(torch.abs(torch.randn(1)), requires_grad=param_grad)
	b = Variable(torch.randn(1) + 1.5, requires_grad=param_grad)
	eps = 1e-4
	# gradcheck doesn't have to pass all the time.
	test = gradcheck(QuadraticMean.apply, (input, a, b), eps=eps, atol=1e-3, rtol=1e-2)
	print(test)
	# eval0 = torch.sum(ConstantMean.apply(input, const_mean))
	# const_mean_perturb = Variable(const_mean.data + eps)
	# eval1 = torch.sum(ConstantMean.apply(input, const_mean_perturb))
	# print((eval1-eval0)/eps)
	# eval0.backward(gradient=torch.ones(n1))
	# print(const_mean.grad.data)
