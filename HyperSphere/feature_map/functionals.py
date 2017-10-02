import math

import numpy as np
import torch


def id_transform(phi):
	return phi

id_transform.dim_change = lambda x: x


def phi_reflection(phi):
	return torch.cat([torch.cos(phi[:, :-1] * math.pi), torch.cos(phi[:, -1:] * 2 * math.pi), torch.sin(phi[:, -1:] * 2 * math.pi)], 1)

phi_reflection.dim_change = lambda x: x+1


def phi_smooth(phi):
	return torch.cat([torch.cos(phi[:, :-1] * math.pi), torch.cos(phi[:, -1:] * 2 * math.pi), torch.sin(phi[:, -1:] * 2 * math.pi)], 1)

phi_smooth.dim_change = lambda x: x+1


def phi_reflection_lp(phi, p=3):
	ratio = 0.5 * (1 - torch.cos(phi[:, 0:1] * math.pi))
	return phi_reflection(phi) * (1 - (1 - ratio) ** p) ** (1.0 / p)

phi_reflection_lp.dim_change = phi_reflection.dim_change


def phi_reflection_threshold(phi, threshold=0.1):
	ratio = 0.5 * (1 - torch.cos(phi[:, 0:1] * math.pi))
	multiplier = ratio.clone() * 0 + 1
	ind_small = ratio < threshold
	multiplier[ind_small] = 0.5 * (1 - torch.cos(ratio[ind_small] * math.pi / threshold))
	return phi_reflection(phi) * multiplier.view(-1, 1)


phi_reflection_threshold.dim_change = phi_reflection.dim_change


def sigmoid_numpy(x):
	return 1.0 / (1.0 + np.exp(-x))


def sigmoid_inv_numpy(x):
	return 1.0 - np.log(1.0/x - 1)


def sigmoid_inv(x):
	return 1.0 - torch.log(1.0/x - 1)