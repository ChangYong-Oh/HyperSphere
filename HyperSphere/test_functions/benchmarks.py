import math
import torch
from torch.autograd import Variable


def bird(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	ndim = x.size(1)
	n_repeat = ndim / 2
	x = x * 2 * math.pi
	output = 0
	for i in range(n_repeat):
		output += (x[:, 2 * i] - x[:, 2 * i + 1]) ** 2 + torch.exp((1 - torch.sin(x[:, 2 * i])) ** 2) * torch.cos(x[:, 2 * i + 1]) + torch.exp((1 - torch.cos(x[:, 2 * i + 1])) ** 2) * torch.sin(x[:, 2 * i])
	output /= float(n_repeat)
	if flat:
		return output.squeeze(0)
	else:
		return output

bird.dim = 0


def branin(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	ndim = x.size(1)
	n_repeat = ndim / 2
	n_dummy = ndim % 2

	shift = torch.cat([torch.FloatTensor([2.5, 7.5]).repeat(n_repeat), torch.zeros(n_dummy)])

	if hasattr(x, 'data'):
		x.data = x.data * 7.5 + shift.type_as(x.data)
	else:
		x = x * 7.5 + shift.type_as(x)
	a = 1
	b = 5.1 / (4 * math.pi ** 2)
	c = 5.0 / math.pi
	r = 6
	s = 10
	t = 1.0 / (8 * math.pi)
	output = 0
	for i in range(n_repeat):
		output += a * (x[:, 2 * i + 1] - b * x[:, 2 * i] ** 2 + c * x[:, 2 * i] - r) ** 2 + s * (1 - t) * torch.cos(x[:, 2 * i]) + s
	output /= float(n_repeat)
	if flat:
		return output.squeeze(0)
	else:
		return output

branin.dim = 0


def camelback(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	ndim = x.size(1)
	n_repeat = ndim / 2
	n_dummy = ndim % 2

	scale = torch.cat([torch.FloatTensor([3.0, 2.0]).repeat(n_repeat), torch.zeros(n_dummy)])
	if flat:
		x = x.view(1, -1)
	if hasattr(x, 'data'):
		x.data = x.data * scale.type_as(x.data)
	else:
		x = x * scale.type_as(x)
	output = 0
	for i in range(n_repeat):
		output += (4.0 - 2.1 * x[:, 2 * i] ** 2 + x[:, 2 * i] ** 4 / 3.0) * x[:, 2 * i] ** 2 + x[:, 2 * i] * x[:, 2 * i + 1] + 4 * (x[:, 2 * i + 1] ** 2 - 1.0) * x[:, 2 * i + 1] ** 2
	output /= float(n_repeat)
	if flat:
		return output.squeeze(0)
	else:
		return output

camelback.dim = 0


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
	ndata, ndim = x.size()
	n_repeat = ndim / 6

	if hasattr(x, 'data'):
		x.data = (x.data + 1) * 0.5
	else:
		x = (x + 1) * 0.5

	output = 0
	for i in range(n_repeat):
		x_block = x[:, 6 * i:6 * (i + 1)]
		output += -(alpha.view(1, -1).repeat(ndata, 1) * torch.exp(-(A.unsqueeze(0).repeat(ndata, 1, 1) * (x_block.unsqueeze(2).repeat(1, 1, 4) - P.unsqueeze(0).repeat(ndata, 1, 1)) ** 2).sum(1))).sum(1, keepdim=True)
	output /= float(n_repeat)
	if flat:
		return output.squeeze(0)
	else:
		return output

hartmann6.dim = 0


def levy(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	x = x * 10

	w = (x - 1.0) / 4.0 + 1.0
	output = torch.sin(math.pi * w[:, :1]) ** 2.0
	for i in range(w.size(1) - 1):
		output += (w[:, i:i+1] - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(math.pi * w[:, i:i+1] + 1.0) ** 2.0)
	output += ((w[:, -1:] - 1.0) ** 2 * (1.0 + torch.sin(2 * math.pi * w[:, -1:]) ** 2.0))
	if flat:
		return output.squeeze(0)
	else:
		return output

levy.dim = 0


def michalewicz(x):
	pi = math.pi
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	ndim = x.size(1)
	indices = torch.arange(1, ndim + 1)
	if hasattr(x, 'data'):
		x.data = (x.data + 1) * 0.5 * pi
		indices = Variable(indices.type_as(x.data))
	else:
		x = (x + 1) * 0.5 * pi
		indices = indices.type_as(x)

	m = 10

	output = -torch.mean(torch.sin(x) * torch.sin(x ** 2 * indices / pi) ** (2 * m), 1, keepdim=True)
	if flat:
		return output.squeeze(0)
	else:
		return output

michalewicz.dim = 0


def qing(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	x = x * 500

	ndim = x.size(1)
	indices = torch.arange(1, ndim + 1)
	if hasattr(x, 'data'):
		indices = Variable(indices.type_as(x.data))
	else:
		indices = indices.type_as(x)
	output = ((x ** 2 - indices) ** 2).mean(1, keepdim=True)
	if flat:
		return output.squeeze(0)
	else:
		return output

qing.dim = 0


def rosenbrock(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	x = x * 7.5 + 2.5

	normalizer = 50000.0 / ((90 ** 2 + 9 ** 2) * (x.size(1) - 1))
	output = (100.0 * ((x[:, 1:] - x[:, :-1] ** 2) ** 2) + (x[:, :-1] - 1) ** 2).sum(1, keepdim=True) * normalizer
	if flat:
		return output.squeeze(0)
	else:
		return output

rosenbrock.dim = 0


def schwefel(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	x = x * 500.0

	output = 418.9829 - torch.mean(x * torch.sin(torch.abs(x) ** 0.5), 1, keepdim=True)
	if flat:
		return output.squeeze(0)
	else:
		return output

schwefel.dim = 0


def styblinskitang(x):
	flat = x.dim() == 1
	if flat:
		x = x.view(1, -1)
	x = x * 5

	output = ((x ** 4).sum(1, keepdim=True) - 16 * (x ** 2).sum(1, keepdim=True) + 5 * x.sum(1, keepdim=True)) / (2.0 * x.size(1))
	if flat:
		return output.squeeze(0)
	else:
		return output

styblinskitang.dim = 0


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
	if flat:
		return output.squeeze(0)
	else:
		return output

rotatedschwefel.dim = 0


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


def generate_orthogonal_matrix(ndim):
	x = torch.exp(torch.sin(torch.linspace(-ndim ** 0.5, ndim ** 0.5, ndim)))
	gram_mat = torch.exp(-(x.unsqueeze(1).repeat(1, ndim) - x.unsqueeze(0).repeat(ndim, 1)) ** 2)
	return torch.qr(gram_mat)[0]