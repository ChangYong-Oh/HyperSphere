import math
import numpy as np
import torch
from functools import partial


def id_transform(phi):
	return phi

id_transform.dim_change = lambda x: x


def x2radial(x):
	radius = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
	normalizer = radius.clone()
	normalizer[(radius == 0).detach()] = 1
	return torch.cat([radius, x/normalizer], 1)

x2radial.dim_change = lambda x: x + 1


def in_sphere(x, radius):
	return torch.sum(x.data ** 2) > radius ** 2


def sphere_bound(radius):
	return partial(in_sphere, radius=radius)


def phi_reflection(phi):
	f_phi0 = torch.cos(phi[:, :1] * math.pi)
	f_phi_rest = torch.cat([torch.cos(phi[:, -1:] * 2 * math.pi), torch.sin(phi[:, -1:] * 2 * math.pi)], 1)
	if phi.size(1) > 2:
		f_phi_rest = torch.cat([torch.cos(phi[:, 1:-1] * math.pi), f_phi_rest], 1)
	return torch.cat([f_phi0, f_phi_rest], 1)

phi_reflection.dim_change = lambda x: x+1


def phi_smooth(phi):
	f_phi0 = torch.abs(torch.sin(phi[:, :1] * math.pi * 0.25))
	sin_part = torch.cumprod(torch.cat([torch.sin(phi[:, 1:-1] * math.pi), torch.sin(phi[:, -1:] * math.pi * 2.0)], 1), 1)
	cos_part = torch.cat([torch.cos(phi[:, 1:-1] * math.pi), torch.cos(phi[:, -1:] * math.pi * 2)], 1)
	f_phi_rest = torch.cat([cos_part[:, :1], sin_part[:, :-1] * cos_part[:, 1:], sin_part[:, -1:]], 1)
	return torch.cat([f_phi0, f_phi_rest * f_phi0.view(-1, 1)], 1)

phi_smooth.dim_change = lambda x: x+1


def phi_reflection_lp(phi, p=3):
	f_phi0 = torch.cos(phi[:, :1] * math.pi)
	ratio = 0.5 * (1 - f_phi0)
	reduction = (1 - (1 - ratio) ** p) ** (1.0 / p)
	f_phi_rest = torch.cat([torch.cos(phi[:, 1:-1] * math.pi), torch.cos(phi[:, -1:] * 2 * math.pi), torch.sin(phi[:, -1:] * 2 * math.pi)], 1)
	return torch.cat([f_phi0, f_phi_rest * reduction.view(-1, 1)], 1)

phi_reflection_lp.dim_change = lambda x: x+1


def phi_reflection_threshold(phi, threshold=0.1):
	f_phi0 = torch.cos(phi[:, :1] * math.pi)
	ratio = 0.5 * (1 - f_phi0)
	reduction = ratio.clone() * 0 + 1
	ind_small = ratio < threshold
	# reduction[ind_small] = 0.5 * (1 - torch.cos(ratio[ind_small] * math.pi / threshold))
	reduction[ind_small] = torch.sin(ratio[ind_small] * 0.25 * math.pi / threshold)
	f_phi_rest = torch.cat([torch.cos(phi[:, -1:] * 2 * math.pi), torch.sin(phi[:, -1:] * 2 * math.pi)], 1)
	if phi.size(1) > 2:
		f_phi_rest = torch.cat([torch.cos(phi[:, 1:-1] * math.pi), f_phi_rest], 1)
	return torch.cat([f_phi0, f_phi_rest * reduction.view(-1, 1)], 1)


phi_reflection_threshold.dim_change = lambda x: x+1


def sigmoid_numpy(x):
	return 1.0 / (1.0 + np.exp(-x))


def sigmoid_inv_numpy(x):
	return 1.0 - np.log(1.0/x - 1)


def sigmoid_inv(x):
	return 1.0 - torch.log(1.0/x - 1)