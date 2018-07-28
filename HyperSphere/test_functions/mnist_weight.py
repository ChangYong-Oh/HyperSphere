import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import progressbar
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.cuda as cuda
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms


BATCH_SIZE = 64
EPOCH = 20
USE_VALIDATION = True


class Net(nn.Module):
	def __init__(self, n_hid, hid_weight=None):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(784, n_hid)
		if hid_weight is None:
			self.hid_weight = None
			self.fc2 = nn.Linear(n_hid, 10)
		else:
			self.hid_weight = hid_weight if hasattr(hid_weight, 'data') else Variable(hid_weight)
			self.hid_bias = Parameter(torch.Tensor(10))
			# hid_bias is also optimized by SGD
			# this is just for making problem 100, 200, 500 not 110, 210, 510

	def forward(self, x):
		x = x.view(-1, 784)
		x = F.relu(self.fc1(x))
		if self.hid_weight is None:
			x = self.fc2(x)
		else:
			if x.is_cuda:
				self.hid_weight = self.hid_weight.cuda()
			x = F.linear(x, self.hid_weight, self.hid_bias)
		return F.log_softmax(x)


def load_mnist(batch_size, use_cuda, use_validation):
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
	mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
	mnist_test = datasets.MNIST('../data', train=False, transform=transform)
	if use_validation:
		train_sampler = sampler.SubsetRandomSampler(range(45000))
		validation_sampler = sampler.SubsetRandomSampler(range(45000, 50000))
		train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=False, sampler=train_sampler, **kwargs)
		validation_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=False, sampler=validation_sampler, **kwargs)
	else:
		train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, **kwargs)
	if use_validation:
		return train_loader, validation_loader, test_loader
	else:
		return train_loader, test_loader


def train(train_loader, model, epoch, optimizer, use_cuda):
	model.train()
	progress = progressbar.ProgressBar(max_value=epoch)
	for e in range(epoch):
		for batch_idx, (data, target) in enumerate(train_loader):
			if use_cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)
			optimizer.zero_grad()
			output = model(data)
			loss = F.nll_loss(output, target)
			loss.backward()
			optimizer.step()
		progress.update(e)


def evaluation(evaluation_loader, model, use_cuda):
	model.eval()
	evaluation_loss = 0
	correct = 0
	n_eval_data = 0
	for data, target in evaluation_loader:
		n_eval_data += data.size(0)
		if use_cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		evaluation_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	evaluation_loss /= float(n_eval_data)
	evaluation_accuracy = correct / float(n_eval_data)
	return evaluation_loss, evaluation_accuracy


def mnist_weight(weight_vector, use_BO=True, use_validation=USE_VALIDATION, use_cuda=True):
	use_cuda = cuda.is_available() and use_cuda
	if use_BO:
		model = Net(n_hid=weight_vector.numel() / 10, hid_weight=weight_vector.view(10, -1))
	else:
		model = Net(n_hid=weight_vector.numel() / 10, hid_weight=None)
	for m in model.parameters():
		if m.dim() == 2:
			nn.init.xavier_normal(m.data)
		else:
			m.data.normal_()
	if use_cuda:
		model.cuda()
	if use_validation:
		train_loader, validation_loader, test_loader = load_mnist(BATCH_SIZE, use_cuda, use_validation)
	else:
		train_loader, test_loader = load_mnist(BATCH_SIZE, use_cuda, use_validation)
	if use_BO:
		optimizer = optim.Adam(model.parameters())
	else:
		optimizer = optim.Adam(model.parameters(), weight_decay=0.0001)
	train(train_loader, model, EPOCH, optimizer, use_cuda)
	if use_validation:
		loss, accuracy = evaluation(validation_loader, model, use_cuda)
	else:
		loss, accuracy = evaluation(test_loader, model, use_cuda)
	# if not use_BO:
	# 	print('Entirely with SGD(Adam)')
	# 	print(model.fc2.weight.data)
	print('\n%s loss : %f / Accuracy : %6.4f' % ('Validation' if use_validation else 'Test', loss, accuracy))

	if use_BO:
		return torch.FloatTensor([[loss]])
	if not use_BO:
		weight_radius = (torch.sum([elm[1] for elm in model.named_parameters() if elm[0]=='fc2.weight'][0] ** 2) ** 0.5).data[0]
		return torch.FloatTensor([[loss]]), weight_radius

mnist_weight.dim = 0


def mnist_weight_baseline(ndim, type='loss'):
	if ndim == 100:
		if type == 'loss':
			return [0.242009, 0.230133, 0.216998, 0.222007, 0.242975]
		elif type == 'accuracy':
			return [0.9322, 0.9349, 0.9388, 0.9406, 0.9321]
	elif ndim == 200:
		if type == 'loss':
			return [0.145960, 0.159507, 0.140117, 0.140135, 0.165476]
		elif type == 'accuracy':
			return [0.9585, 0.9559, 0.9605, 0.9619, 0.9545]
	elif ndim == 500:
		if type == 'loss':
			return [0.118356, 0.132987, 0.122135, 0.132618, 0.121591]
		elif type == 'accuracy':
			return [0.9726, 0.9729, 0.9732, 0.9728, 0.9729]

if __name__ == '__main__':
	weight_vector = torch.randn(500)
	print(mnist_weight(weight_vector))

# 10 by 10 case
# Loss : 0.242009 / Accuracy : 0.9322
# Loss : 0.230133 / Accuracy : 0.9349
# Loss : 0.216998 / Accuracy : 0.9388
# Loss : 0.222007 / Accuracy : 0.9406
# Loss : 0.242975 / Accuracy : 0.9321

# 10 by 20 case
# Loss : 0.145960 / Accuracy : 0.9585
# Loss : 0.159507 / Accuracy : 0.9559
# Loss : 0.140117 / Accuracy : 0.9605
# Loss : 0.140135 / Accuracy : 0.9619
# Loss : 0.165476 / Accuracy : 0.9545

# 10 by 50 case
# Loss : 0.118356 / Accuracy : 0.9726
# Loss : 0.132987 / Accuracy : 0.9729
# Loss : 0.122135 / Accuracy : 0.9732
# Loss : 0.132618 / Accuracy : 0.9728
# Loss : 0.121591 / Accuracy : 0.9729
