import torch
from torch.autograd import Function, Variable, gradcheck


class GaussianLikelihood(Function):

	@staticmethod
	def forward(ctx, input, log_noise_var):
		output = log_noise_var.exp().repeat(input.size(0))
		ctx.save_for_backward(input, log_noise_var, output)

		return output

	@staticmethod
	def backward(ctx, grad_output):
		input, log_noise_var, output = ctx.saved_variables
		grad_input = grad_log_noise_var = None

		if ctx.needs_input_grad[0]:
			grad_input = input.clone()
			grad_input.data.zero_()
		if ctx.needs_input_grad[1]:
			grad_log_noise_var = torch.sum(grad_output * output)

		return grad_input, grad_log_noise_var


if __name__ == '__main__':
	n1 = 3
	ndim = 5
	input_grad = False
	param_grad = not input_grad
	input = Variable(torch.randn(n1, ndim), requires_grad=input_grad)
	log_noise_var = Variable(torch.randn(1), requires_grad=param_grad)
	eps = 1e-4
	# gradcheck doesn't have to pass all the time.
	test = gradcheck(GaussianLikelihood.apply, (input, log_noise_var), eps=eps)
	print(test)
	# eval0 = torch.sum(GaussianLikelihood.apply(input, log_noise_var))
	# log_noise_var_perturb = Variable(log_noise_var.data + eps)
	# eval1 = torch.sum(GaussianLikelihood.apply(input, log_noise_var_perturb))
	# print((eval1-eval0)/eps)
	# eval0.backward(gradient=torch.ones(n1))
	# print(log_noise_var.grad.data)