import math
import torch


def branin(x):
	"""
	
	:param x: range : [-1, 1] 
	:return: 
	"""
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	if hasattr(x, 'data'):
		x.data = x.data * 7.5 + torch.FloatTensor([2.5, 7.5]).type_as(x.data)
	else:
		x = x * 7.5 + torch.FloatTensor([2.5, 7.5]).type_as(x)
	a = 1
	b = 5.1/(4 * math.pi**2)
	c = 5.0 / math.pi
	r = 6
	s = 10
	t = 1.0 / (8 * math.pi)
	output = a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - r) ** 2 + s * (1-t) * torch.cos(x[:, 0]) + s
	if flat:
		return output.squeeze(0)
	else:
		return output

branin.dim = 2