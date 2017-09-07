import math

import torch


def abs_sin(rphi, radius=1.0):
	r_periodize = lambda x: torch.abs(torch.sin(2 * x))
	phi_periodize = lambda x: torch.cos(x)

	return torch.cat((r_periodize(rphi[:, [0]]/radius * math.pi), phi_periodize(rphi[:, 1:])), 1)
