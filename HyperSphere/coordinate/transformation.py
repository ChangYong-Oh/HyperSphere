import math

import torch


def rect2spherical(x):
	_, n_dim = x.size()
	reverse_ind = torch.arange(n_dim-1, -1, -1).type_as(x.data if hasattr(x, 'data') else x).long()
	x_sq_accum = torch.sqrt(torch.cumsum((x**2)[:, reverse_ind], dim=1)[:, reverse_ind])
	rphi = torch.cat((x_sq_accum[:, [0]], x[:, 0:n_dim-1]), dim=1)
	if n_dim > 2:
		rphi[:, 1:n_dim-1] = torch.acos(x[:, 0:n_dim-2]/x_sq_accum[:, 0:n_dim-2])
	rphi[:, -1] = torch.acos(x[:, -2] / x_sq_accum[:, -2])
	if hasattr(x, 'data'):
		rphi.data[:, -1][x.data[:, -1] < 0] = 2 * math.pi - rphi.data[:, -1][x.data[:, -1] < 0]
	else:
		rphi[:, -1][x[:, -1] < 0] = 2 * math.pi - rphi[:, -1][x[:, -1] < 0]
	return rphi


def spherical2rect(rphi):
	_, n_dim = rphi.size()
	x = torch.cumprod(torch.cat((rphi[:, [0]], torch.sin(rphi[:, 1:n_dim])), dim=1), dim=1)
	x[:, 0:n_dim-1] = x[:, 0:n_dim-1] * torch.cos(rphi[:, 1:n_dim])
	return x


def check_rphi(rphi):
	assert torch.sum(rphi < 0) == 0
	assert torch.sum(rphi[:, 1:-1] > math.pi) == 0
	assert torch.sum(rphi[:, -1] > 2 * math.pi) == 0


if __name__ == '__main__':
	n_dim = 5
	n_data = 10
	phi = torch.FloatTensor(n_data, n_dim-1).uniform_(0, 2*math.pi)
	r_phi = torch.cat((torch.ones(n_data, 1) * 1.0, phi), dim=1)
	r_phi_domain = rect2spherical(spherical2rect(r_phi))
	in_domain = (r_phi[:, 1:-1] <= math.pi).type_as(r_phi)
	in_domain_accum = torch.cumsum(in_domain, dim=1)
	not_in_domain = (r_phi[:, 1:-1] > math.pi).type_as(r_phi)
	not_in_domain_accum = torch.cumsum(not_in_domain, dim=1)
	diff = (r_phi[:, 1:-1] - r_phi_domain[:, 1:-1]) / math.pi
	sum = (r_phi[:, 1:-1] + r_phi_domain[:, 1:-1]) / math.pi
	for i in range(n_data):
		print(torch.stack((in_domain[i], in_domain_accum[i], not_in_domain_accum[i], sum[i], diff[i]), dim=0))

