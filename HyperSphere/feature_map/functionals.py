import math

import torch


def phi_periodize(phi):
	return torch.cat([torch.cos(phi[:, :-1] * math.pi), torch.cos(phi[:, -1:] * 2 * math.pi), torch.sin(phi[:, -1:] * 2 * math.pi)], 1)

phi_periodize.dim_change = lambda x: x+1


def phi_periodize_lp(phi, p=3):
	ratio = 0.5 * (1 - torch.cos(phi[:, 0:1] * math.pi))
	return phi_periodize(phi) * (1 - (1 - ratio) ** p) ** (1.0 / p)

phi_periodize_lp.dim_change = phi_periodize.dim_change


def phi_periodize_sin(phi):
	return phi_periodize(phi) * torch.abs(torch.sin(phi[:, [0]] * math.pi * 0.5))


phi_periodize_sin.dim_change = phi_periodize.dim_change


def phi_periodize_one(phi, inflection=0.1):
	ratio = 0.5 * (1 - torch.cos(phi[:, 0:1] * math.pi))
	multiplier = ratio.clone() * 0 + 1
	ind_small = ratio < inflection
	multiplier[ind_small] = 0.5 * (1 - torch.cos(ratio[ind_small] * math.pi / inflection))
	return phi_periodize(phi) * multiplier.view(-1, 1)


phi_periodize_one.dim_change = phi_periodize.dim_change
