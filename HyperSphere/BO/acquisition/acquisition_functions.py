import math

import torch

from HyperSphere.BO.utils.normal_cdf import norm_cdf


def norm_pdf(x, mu=0.0, var=1.0):
	return torch.exp(-0.5 * (x-mu) ** 2 / var)/(2 * math.pi * var)**0.5


def expected_improvement(mean, var, reference):
	std = torch.sqrt(var)
	standardized = (-mean + reference) / std
	return (std * norm_pdf(standardized) + (-mean + reference) * norm_cdf(standardized)).clamp(min=0)


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	from scipy.stats import norm
	x = torch.linspace(2, 3, 200)
	y1 = norm_cdf(x)
	y2 = norm.cdf(x.numpy())
	plt.plot(x.numpy(), y1.numpy(), label='approximate')
	plt.plot(x.numpy(), y2, ':', label='exact')
	z1 = norm_pdf(x)
	z2 = norm.pdf(x.numpy())
	plt.plot(x.numpy(), z1.numpy(), label='approximate')
	plt.plot(x.numpy(), z2, ':', label='exact')
	plt.legend()
	plt.show()
