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

	w = (x - 1.0) / 4.0 + 1.0
	output = ((w[:, :-1] - 1) ** 2 * (1 + 10 * torch.sin(math.pi * w[:, :-1] + 1) ** 2)).sum(1, keepdim=True)
	output += torch.sin(math.pi * w[:, :1]) ** 2 + ((w[:, -1:] - 1) ** 2 * (1 + torch.sin(2 * math.pi * w[:, -1:]) ** 2))
	if flat:
		return output.squeeze(0)
	else:
		return output


levy.dim = 0


def rosenbrock(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	if hasattr(x, 'data'):
		x.data = x.data * 7.5 + 2.5
	else:
		x = x * 7.5 + 2.5

	normalizer = 50000.0 / ((90 ** 2 + 9 ** 2) * (x.size(1) - 1))
	output = (100.0 * ((x[:, 1:] - x[:, :-1] ** 2) ** 2) + (x[:, :-1] - 1) ** 2).sum(1, keepdim=True) * normalizer
	if flat:
		return output.squeeze(0)
	else:
		return output


rosenbrock.dim = 0


def styblinskitang(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	if hasattr(x, 'data'):
		x.data = x.data * 5
	else:
		x = x * 5

	output = ((x ** 4).sum(1, keepdim=True) - 16 * (x ** 2).sum(1, keepdim=True) + 5 * x.sum(1, keepdim=True)) / (2.0 * x.size(1))
	if flat:
		return output.squeeze(0)
	else:
		return output


styblinskitang.dim = 0


def rotatedstyblinskitang(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	ndim = x.size(1)
	orthogonal_matrix = generate_orthogonal_matrix(ndim)
	if hasattr(x, 'data'):
		x.data = x.data.mm(orthogonal_matrix) * 5.0
	else:
		x = x.mm(orthogonal_matrix) * 5.0

	output = ((x ** 4).sum(1, keepdim=True) - 16 * (x ** 2).sum(1, keepdim=True) + 5 * x.sum(1, keepdim=True)) / (2.0 * x.size(1))
	if flat:
		return output.squeeze(0)
	else:
		return output


rotatedstyblinskitang.dim = 0


def schwefel(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	if hasattr(x, 'data'):
		x.data = x.data * 500.0
	else:
		x = x * 500.0

	output = 418.9829 - torch.mean(x * torch.sin(torch.abs(x) ** 0.5), 1, keepdim=True)
	if flat:
		return output.squeeze(0)
	else:
		return output


schwefel.dim = 0


def rotatedschwefel(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	ndim = x.size(1)
	orthogonal_matrix = generate_orthogonal_matrix(ndim)
	if hasattr(x, 'data'):
		x.data = x.data.mm(orthogonal_matrix) * 500.0
	else:
		x = x.mm(orthogonal_matrix) * 500.0

	output = 418.9829 - torch.mean(x * torch.sin(torch.abs(x) ** 0.5), 1, keepdim=True)
	output += output.clone().normal_() * 20.0
	if flat:
		return output.squeeze(0)
	else:
		return output


rotatedschwefel.dim = 0


def michalewicz(x):
	pi = math.pi
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	ndim = x.size(1)
	indices = torch.arange(1, ndim + 1)
	if hasattr(x, 'data'):
		x.data = (x.data + 1) * 0.5 * pi
		indices = Variable(indices)
	else:
		x = (x + 1) * 0.5 * pi

	m = 10

	output = -torch.mean(torch.sin(x) * torch.sin(x ** 2 * indices / pi) ** (2 * m), 1, keepdim=True)
	if flat:
		return output.squeeze(0)
	else:
		return output


michalewicz.dim = 0


def generate_orthogonal_matrix(ndim):
	x = torch.exp(torch.sin(torch.linspace(-ndim ** 0.5, ndim ** 0.5, ndim)))
	gram_mat = torch.exp(-(x.unsqueeze(1).repeat(1, ndim) - x.unsqueeze(0).repeat(ndim, 1)) ** 2)
	return torch.qr(gram_mat)[0]
