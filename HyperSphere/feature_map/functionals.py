import math

import torch


def periodize(rphi, radius=1.0):
	return torch.cat([torch.abs(torch.sin(rphi[:, [0]]/radius * math.pi * 0.5)), torch.cos(rphi[:, 1:] * 2 * math.pi), torch.sin(rphi[:, -1:] * 2 * math.pi)], 1)

periodize.dim_change = 1


def periodize_lp(rphi, radius=1.0, p=3):
	return periodize(rphi, radius=radius) * (radius ** p - (radius - rphi[:, 0:1]) ** p) ** (1.0 / p)

periodize_lp.dim_change = periodize.dim_change


def periodize_sin(rphi, radius=1.0):
	return periodize(rphi, radius=radius) * torch.abs(torch.sin(rphi[:, [0]]/radius * math.pi * 0.5))


periodize_sin.dim_change = periodize.dim_change


def periodize_one(rphi, radius=1.0, inflection=0.1):
	r = rphi[:, 0]
	multiplier = r.clone() * 0 + 1
	ind_small = r < radius * inflection
	multiplier[ind_small] = 1/inflection**2 * r[ind_small] / radius * (2 * inflection - r[ind_small] / radius)
	ind_large = r > radius * (2 - inflection)
	multiplier[ind_large] = 1/inflection**2 * (2 - r[ind_large] / radius) * (2 * inflection - 2 + r[ind_large] / radius)
	return periodize(rphi, radius=radius) * multiplier.view(-1, 1)


periodize_one.dim_change = periodize.dim_change
