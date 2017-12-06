import torch


def innerProductKernel(input1, input2, log_amp, sigma_sqrt):
	n1, ndim = input1.size()
	n2 = input2.size(0)

	if sigma_sqrt.dim() == 1:
		transformed_input1 = input1 * sigma_sqrt
		transformed_input2 = input2 * sigma_sqrt
	elif sigma_sqrt.dim() == 2:
		transformed_input1 = torch.mm(input1, sigma_sqrt)
		transformed_input2 = torch.mm(input2, sigma_sqrt)

	output = log_amp.exp() * (transformed_input1.unsqueeze(1).repeat(1, n2, 1) * transformed_input2.unsqueeze(0).repeat(n1, 1, 1)).sum(2)
	return output
