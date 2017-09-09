import torch
from torch.autograd import Function, Variable, gradcheck


class ConstantMean(Function):

	@staticmethod
	def forward(ctx, input, const_mean):
		ctx.save_for_backward(input, const_mean)

		return const_mean.repeat(input.size(0), 1)

	@staticmethod
	def backward(ctx, grad_output):
		"""
		:param ctx: 
		:param grad_output: grad_output is assumed to be d[Scalar]/dK and size() is n_input 
		:return: 
		"""
		input, const_mean = ctx.saved_variables
		grad_input = grad_const_mean = None

		if ctx.needs_input_grad[0]:
			grad_input = input.clone()
			grad_input.data.zero_()
		if ctx.needs_input_grad[1]:
			grad_const_mean = torch.sum(grad_output)

		return grad_input, grad_const_mean


if __name__ == '__main__':
	n1 = 3
	ndim = 5
	input_grad = False
	param_grad = not input_grad
	input = Variable(torch.randn(n1, ndim), requires_grad=input_grad)
	const_mean = Variable(torch.randn(1) + 1.5, requires_grad=param_grad)
	eps = 1e-4
	# gradcheck doesn't have to pass all the time.
	test = gradcheck(ConstantMean.apply, (input, const_mean), eps=eps)
	print(test)
	# eval0 = torch.sum(ConstantMean.apply(input, const_mean))
	# const_mean_perturb = Variable(const_mean.data + eps)
	# eval1 = torch.sum(ConstantMean.apply(input, const_mean_perturb))
	# print((eval1-eval0)/eps)
	# eval0.backward(gradient=torch.ones(n1))
	# print(const_mean.grad.data)
