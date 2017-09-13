import torch
from torch.autograd import Function, Variable, gradcheck


class SquaredExponentialKernel(Function):

	@staticmethod
	def forward(ctx, input1, input2, log_amp, log_ls):
		n1, ndim = input1.size()
		n2 = input2.size(0)

		diff = input1.unsqueeze(1).repeat(1, n2, 1) - input2.unsqueeze(0).repeat(n1, 1, 1)
		scaling = log_ls.exp().view(1, 1, log_ls.numel()).repeat(n1, n2, 1)

		output = log_amp.exp() * torch.exp(-torch.sum(0.5 * diff**2 / scaling, dim=2))
		ctx.save_for_backward(input1, input2, log_amp, log_ls, output)
		return output

	@staticmethod
	def backward(ctx, grad_output):
		"""
		:param ctx: 
		:param grad_output: grad_output is assumed to be d[Scalar]/dK and size() is n1 x n2 
		:return: 
		"""
		input1, input2, log_amp, log_ls, output = ctx.saved_variables
		grad_input1 = grad_input2 = grad_log_amp = grad_log_ls = None

		n1, ndim = input1.size()
		n2 = input2.size(0)

		diff = input1.unsqueeze(1).repeat(1, n2, 1) - input2.unsqueeze(0).repeat(n1, 1, 1)
		scaling = log_ls.exp().view(1, 1, log_ls.numel()).repeat(n1, n2, 1)

		if ctx.needs_input_grad[0]:
			kernel_grad_input1 = -diff / scaling * output.view(n1, n2, 1).repeat(1, 1, ndim)
			grad_input1 = (grad_output.unsqueeze(2).repeat(1, 1, ndim) * kernel_grad_input1).sum(1)
		if ctx.needs_input_grad[1]:
			kernel_grad_input2 = diff / scaling * output.view(n1, n2, 1).repeat(1, 1, ndim)
			grad_input2 = (grad_output.unsqueeze(2).repeat(1, 1, ndim) * kernel_grad_input2).sum(0)
		if ctx.needs_input_grad[2]:
			kernel_grad_log_amp = output
			grad_log_amp = (grad_output * kernel_grad_log_amp).sum()
		if ctx.needs_input_grad[3]:
			kernel_grad_log_ls = 0.5 * diff ** 2/scaling * output.unsqueeze(2).repeat(1, 1, ndim)
			grad_log_ls = (grad_output.unsqueeze(2).repeat(1, 1, ndim) * kernel_grad_log_ls).sum(0).sum(0)

		return grad_input1, grad_input2, grad_log_amp, grad_log_ls


if __name__ == '__main__':
	n1 = 3
	n2 = 4
	ndim = 5
	input_grad = False
	param_grad = not input_grad
	input1 = Variable(torch.randn(n1, ndim), requires_grad=input_grad)
	input2 = Variable(torch.randn(n2, ndim), requires_grad=input_grad)
	log_amp = Variable(torch.randn(1), requires_grad=param_grad)
	log_ls = Variable(torch.randn(ndim), requires_grad=param_grad)
	# gradcheck doesn't have to pass all the time.
	test = gradcheck(SquaredExponentialKernel.apply, (input1, input2, log_amp, log_ls), eps=1e-5, atol=1e-3, rtol=1e-2)
	print(test)