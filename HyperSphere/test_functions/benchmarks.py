import math
import torch
from torch.autograd import Variable


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


def hartmann6(x):
	alpha = torch.FloatTensor([1.0, 1.2, 3.0, 3.2]).type_as(x.data if hasattr(x, 'data') else x)
	A = torch.FloatTensor([[10.0, 3.00, 17.0, 3.50, 1.70, 8.00],
	                       [0.05, 10.0, 17.0, 0.10, 8.00, 14.0],
	                       [3.00, 3.50, 1.70, 10.0, 17.0, 8.00],
	                       [17.0, 8.00, 0.05, 10.0, 0.10, 14.0]]).type_as(x.data if hasattr(x, 'data') else x).t()
	P = torch.FloatTensor([[0.1312,0.1696,0.5569,0.0124,0.8283,0.5886],
	                       [0.2329,0.4135,0.8307,0.3736,0.1004,0.9991],
	                       [0.2348,0.1451,0.3522,0.2883,0.3047,0.6650],
	                       [0.4047,0.8828,0.8732,0.5743,0.1091,0.0381]]).type_as(x.data if hasattr(x, 'data') else x).t()
	if hasattr(x, 'data'):
		alpha = Variable(alpha)
		A = Variable(A)
		P = Variable(P)

	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	if hasattr(x, 'data'):
		x.data = (x.data + 1) * 0.5
	else:
		x = (x + 1) * 0.5

	output = -(alpha.view(1, -1).repeat(x.size(0), 1) * torch.exp(-(A.unsqueeze(0).repeat(x.size(0), 1, 1) * (x.unsqueeze(2).repeat(1, 1, 4) - P.unsqueeze(0).repeat(x.size(0), 1, 1)) ** 2).sum(1))).sum(1)
	if flat:
		return output.squeeze(0)
	else:
		return output


hartmann6.dim = 6


def levy(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	if hasattr(x, 'data'):
		x.data = x.data * 10
	else:
		x = x * 10

	w = 1 + (x - 1) / 4.0
	output = ((w[:, :-1] - 1) ** 2 * (1 + 10 * torch.sin(math.pi * w[:, :-1] + 1) ** 2)).sum(1, keepdim=True)
	output += torch.sin(math.pi * w[:, :1]) ** 2 + ((w[:, -1:] - 1) ** 2 * (1 + torch.sin(2 * math.pi * w[:, -1:]) ** 2))
	if flat:
		return output.squeeze(0)
	else:
		return output


levy.dim = 0