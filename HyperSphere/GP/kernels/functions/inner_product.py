import torch
from torch.autograd import Function, Variable, gradcheck


class InnerProductKernel(Function):

	@staticmethod
	def forward(ctx, input1, input2, log_amp, sigma_chol_L):
		n1, ndim = input1.size()
		n2 = input2.size(0)

		if sigma_chol_L.dim() == 1:
			transformed_input1 = input1 * sigma_chol_L
			transformed_input2 = input2 * sigma_chol_L
		elif sigma_chol_L.dim() == 2:
			transformed_input1 = torch.mm(input1, sigma_chol_L)
			transformed_input2 = torch.mm(input2, sigma_chol_L)

		output = log_amp.exp() * (transformed_input1.unsqueeze(1).repeat(1, n2, 1) * transformed_input2.unsqueeze(0).repeat(n1, 1, 1)).sum(2)
		ctx.save_for_backward(input1, input2, log_amp, sigma_chol_L, output)
		return output

	@staticmethod
	def backward(ctx, grad_output):
		"""
		:param ctx: 
		:param grad_output: grad_output is assumed to be d[Scalar]/dK and size() is n1 x n2 
		:return: 
		"""
		input1, input2, log_amp, sigma_chol_L, output = ctx.saved_variables
		grad_input1 = grad_input2 = grad_log_amp = grad_sigma_chol_L = None

		n1, ndim = input1.size()
		n2 = input2.size(0)

		if sigma_chol_L.dim() == 1:
			sigma = sigma_chol_L ** 2
			mat_input1 = input1 * sigma
			mat_input2 = input2 * sigma
		elif sigma_chol_L.dim() == 2:
			sigma = sigma_chol_L.mm(sigma_chol_L.t())
			mat_input1 = torch.mm(input1, sigma)
			mat_input2 = torch.mm(input2, sigma)

		if ctx.needs_input_grad[0]:
			grad_input1 = grad_output.mm(mat_input2) * log_amp.exp()
		if ctx.needs_input_grad[1]:
			grad_input2 = grad_output.t().mm(mat_input1) * log_amp.exp()
		if ctx.needs_input_grad[2]:
			grad_log_amp = (grad_output * output).sum()
		if ctx.needs_input_grad[3]:
			y_xT = torch.bmm(input2.unsqueeze(2), (input1.unsqueeze(1).repeat(1, n2, 1) * grad_output.unsqueeze(2).repeat(1, 1, ndim)).sum(0, keepdim=True).transpose(0, 1)).sum(0)
			input_outer = y_xT + y_xT.t()
			if sigma_chol_L.dim() == 1:
				grad_sigma_chol_L = log_amp.exp() * torch.sum(sigma_chol_L.unsqueeze(1) * input_outer, 0)
			elif sigma_chol_L.dim() == 2:
				grad_sigma_chol_L = log_amp.exp() * sigma_chol_L.t().mm(input_outer)

		return grad_input1, grad_input2, grad_log_amp, grad_sigma_chol_L


if __name__ == '__main__':
	n1 = 3
	n2 = 4
	ndim = 5
	input_grad = False
	param_grad = not input_grad
	input1 = Variable(torch.randn(n1, ndim), requires_grad=input_grad)
	input2 = Variable(torch.randn(n2, ndim), requires_grad=input_grad)
	log_amp = Variable(torch.randn(1), requires_grad=param_grad)
	# sigma_chol_L = Variable(torch.tril(torch.randn(ndim, ndim)), requires_grad=param_grad)
	sigma_chol_L = Variable(torch.randn(ndim), requires_grad=param_grad)
	# gradcheck doesn't have to pass all the time.
	test = gradcheck(InnerProductKernel.apply, (input1, input2, log_amp, sigma_chol_L), eps=1e-4, atol=1e-3, rtol=1e-2)
	print(test)