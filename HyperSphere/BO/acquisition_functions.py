import math

import torch


def norm_cdf(x, mu=0.0, var=1.0):
	z = (x - mu) / var ** 0.5
	return 1.0 / (1.0 + torch.exp(-math.pi ** 0.5 * (-0.0004406 * z ** 5 + 0.0418198 * z ** 3 + 0.9 * z)))


def norm_pdf(x, mu=0.0, var=1.0):
	return torch.exp(-0.5 * (x-mu) ** 2 / var)/(2 * math.pi * var)**0.5


def expected_improvement(mean, var, reference):
	std = torch.sqrt(var + 1e-6)
	standardized = (-mean + reference) / std
	return std * norm_pdf(standardized) + (-mean + reference) * norm_cdf(standardized)


