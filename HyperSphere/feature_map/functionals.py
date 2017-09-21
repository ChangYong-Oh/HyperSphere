import math

import torch


def rphi_periodize(rphi, radius=1.0):
	return torch.cat([torch.abs(torch.sin(rphi[:, [0]]/radius * math.pi * 0.5)), torch.cos(rphi[:, 1:] * math.pi), torch.sin(rphi[:, -1:] * 2 * math.pi)], 1)

rphi_periodize.dim_change = 1


def rphi_periodize_lp(rphi, radius=1.0, p=3):
	return rphi_periodize(rphi, radius=radius) * (radius ** p - (radius - rphi[:, 0:1]) ** p) ** (1.0 / p)

rphi_periodize_lp.dim_change = rphi_periodize.dim_change


def rphi_periodize_sin(rphi, radius=1.0):
	return rphi_periodize(rphi, radius=radius) * torch.abs(torch.sin(rphi[:, [0]]/radius * math.pi * 0.5))


rphi_periodize_sin.dim_change = rphi_periodize.dim_change


def rphi_periodize_one(rphi, radius=1.0, inflection=0.1):
	r = rphi[:, 0]
	multiplier = r.clone() * 0 + 1
	ind_small = r < radius * inflection
	multiplier[ind_small] = 1/inflection**2 * r[ind_small] / radius * (2 * inflection - r[ind_small] / radius)
	ind_large = r > radius * (2 - inflection)
	multiplier[ind_large] = 1/inflection**2 * (2 - r[ind_large] / radius) * (2 * inflection - 2 + r[ind_large] / radius)
	return rphi_periodize(rphi, radius=radius) * multiplier.view(-1, 1)


rphi_periodize_one.dim_change = rphi_periodize.dim_change


def phi_periodize(phi):
	return torch.cat([torch.abs(torch.sin(phi[:, [0]] * math.pi * 0.5)), torch.cos(phi[:, 1:] * math.pi), torch.sin(phi[:, -1:] * 2 * math.pi)], 1)

phi_periodize.dim_change = 1


def phi_periodize_lp(phi, p=3):
	r = torch.abs(torch.sin(phi[:, 0:1] * math.pi * 0.5))
	return phi_periodize(phi) * (1 - (1 - r) ** p) ** (1.0 / p)

phi_periodize_lp.dim_change = phi_periodize.dim_change


def phi_periodize_sin(phi):
	return phi_periodize(phi) * torch.abs(torch.sin(phi[:, [0]] * math.pi * 0.5))


phi_periodize_sin.dim_change = phi_periodize.dim_change


def phi_periodize_one(phi, inflection=0.5):
	r = torch.abs(torch.sin(phi[:, 0] * math.pi * 0.5))
	multiplier = r.clone() * 0 + 1
	ind_small = r < inflection
	multiplier[ind_small] = torch.sin(r[ind_small] * math.pi * 0.25 / inflection)
	return phi_periodize(phi) * multiplier.view(-1, 1)


phi_periodize_one.dim_change = phi_periodize.dim_change
