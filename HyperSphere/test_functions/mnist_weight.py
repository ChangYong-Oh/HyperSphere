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


def load_mnist(batch_size, use_cuda, use_validation=True):
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


def test(test_loader, model, use_cuda):
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		if use_cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	test_loss /= len(test_loader.dataset)
	test_accuracy = correct / float(len(test_loader.dataset))
	return test_loss, test_accuracy


def mnist_weight(weight_vector, use_BO=True):
	use_cuda = cuda.is_available()
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
	if USE_VALIDATION:
		train_loader, validation_loader, test_loader = load_mnist(BATCH_SIZE, use_cuda)
	else:
		train_loader, test_loader = load_mnist(BATCH_SIZE, use_cuda)
	optimizer = optim.Adam(model.parameters())
	train(train_loader, model, EPOCH, optimizer, use_cuda)
	if USE_VALIDATION:
		loss, accuracy = test(validation_loader, model, use_cuda)
	else:
		loss, accuracy = test(test_loader, model, use_cuda)
	if not use_BO:
		print('Entirely with SGD(Adam)')
		print(model.fc2.weight.data)
	print('\n%s loss : %f / Accuracy : %6.4f' % ('Validation' if USE_VALIDATION else 'Test', loss, accuracy))
	return torch.FloatTensor([[loss]])


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
	print(mnist_weight(weight_vector, use_BO=False))

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

