import math

import torch
from torch.autograd import Function, Variable, gradcheck


class ReduceThreshold(Function):

	@staticmethod
	def forward(ctx, input, threshold):
		ctx.save_for_backward(input, threshold)
		ratio = 0.5 * (1 - input[:, 0:1])
		reduction = ratio.clone() * 0 + 1
		ind_small = ratio < threshold
		if torch.sum(ind_small) > 0:
			# reduction[ind_small] = 0.5 * (1 - torch.cos(ratio[ind_small] * math.pi / threshold))
			reduction[ind_small] = torch.sin(ratio[ind_small] * 0.25 * math.pi / threshold)
		return torch.cat([input[:, :1], input[:, 1:] * reduction.view(-1, 1)], 1)

	@staticmethod
	def backward(ctx, grad_output):
		input, threshold = ctx.saved_variables
		grad_input = grad_threshold = None

		ratio = 0.5 * (1 - input[:, 0:1])
		ratio_derivative = -0.5
		reduction = ratio.clone() * 0 + 1
		ind_small = ratio < threshold
		if ind_small.data.any():
			# reduction[ind_small] = 0.5 * (1 - torch.cos(ratio[ind_small] * math.pi / threshold))
			reduction[ind_small] = torch.sin(ratio[ind_small] * 0.25 * math.pi / threshold)

		if ctx.needs_input_grad[0]:
			grad_f_phi0 = ratio.clone() * 0
			if ind_small.data.any():
				grad_f_phi0[ind_small] = 0.25 * math.pi / threshold * torch.cos(ratio[ind_small] * 0.25 * math.pi / threshold) * ratio_derivative
			grad_phi0 = grad_output[:, 0:1] + (input[:, 1:] * grad_f_phi0.view(-1, 1) * grad_output[:, 1:]).sum(1, keepdim=True)
			grad_input = torch.cat([grad_phi0, grad_output[:, 1:] * reduction.view(-1, 1)], 1)
		if ctx.needs_input_grad[1]:
			grad_reduction = ratio.clone() * 0
			if ind_small.data.any():
				grad_reduction[ind_small] = -0.25 * ratio[ind_small] * math.pi / threshold**2 * torch.cos(ratio[ind_small] * 0.25 * math.pi / threshold)
			grad_threshold = (grad_output[:, 1:] * input[:, 1:] * grad_reduction.view(-1, 1)).sum()

		return grad_input, grad_threshold


if __name__ == '__main__':
	n = 1
	ndim = 5
	input_grad = False
	param_grad = not input_grad
	input = Variable(torch.FloatTensor(n, ndim).uniform_(-1, 1), requires_grad=input_grad)
	threshold = Variable(torch.FloatTensor(1).fill_(0.5), requires_grad=param_grad)
	eps = 1e-4

	# d = 3
	# output = (ReduceThreshold.apply(input, threshold))[:, d]
	# fdm_grad = torch.zeros(n, ndim)
	# for i in range(n):
	# 	for j in range(ndim):
	# 		input_perturb = Variable(input.data.clone())
	# 		input_perturb.data[i, j] += eps
	# 		output_perturb = (ReduceThreshold.apply(input_perturb, threshold))[:, d]
	# 		fdm_grad[i, j] = (output_perturb - output).data.squeeze()[0] / eps
	# print(input.data)
	# ratio = 0.5 * (1 - torch.cos(input.data[:, 0:1] * math.pi))
	# reduction = ratio.clone() * 0 + 1
	# ind_small = ratio < threshold.data
	# if torch.sum(ind_small) > 0:
	# 	reduction[ind_small] = 0.5 * (1 - torch.cos(ratio[ind_small] * math.pi / threshold.data))
	# print('reduction', reduction.squeeze()[0])
	# print(fdm_grad)
	# output.backward()
	# print(input.grad.data)

	# gradcheck doesn't have to pass all the time.
	test = gradcheck(ReduceThreshold.apply, (input, threshold), eps=eps, atol=1e-3, rtol=1e-2)
	print(test)