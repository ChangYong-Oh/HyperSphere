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
from torchvision import datasets, transforms


class Net(nn.Module):
	def __init__(self, n_hid, hid_weight=None):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(784, n_hid)
		self.hid_weight = hid_weight if hasattr(hid_weight, 'data') else Variable(hid_weight)
		if hid_weight is None:
			self.fc2 = nn.Linear(n_hid, 10)
		else:
			self.bias = Parameter(torch.Tensor(10))

	def forward(self, x):
		x = x.view(-1, 784)
		x = F.relu(self.fc1(x))
		if self.hid_weight is None:
			x = self.fc2(x)
		else:
			if x.is_cuda:
				self.hid_weight = self.hid_weight.cuda()
			x = F.linear(x, self.hid_weight, self.bias)
		return F.log_softmax(x)


def load_mnist(batch_size, use_cuda):
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	train_loader = torch.utils.data.DataLoader(
	    datasets.MNIST('../data', train=True, download=True,
	                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
	    batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
	    datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
	    batch_size=batch_size, shuffle=True, **kwargs)
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

batch_size = 64
epoch = 50


def mnist_weight(weight_vector):
	use_cuda = cuda.is_available()
	model = Net(n_hid=weight_vector.numel()/10, hid_weight=weight_vector.view(10, -1))
	for m in model.parameters():
		if m.dim() == 2:
			nn.init.xavier_normal(m.data)
		else:
			m.data.normal_()
	if use_cuda:
		model.cuda()
	train_loader, test_loader = load_mnist(batch_size, use_cuda)
	optimizer = optim.Adam(model.parameters())
	train(train_loader, model, epoch, optimizer, use_cuda)
	test_loss, test_accuracy = test(test_loader, model, use_cuda)
	print('\nLoss : %f / Accuracy : %6.4f' % (test_loss, test_accuracy))
	return test_loss


if __name__ == '__main__':
	weight_vector = torch.randn(100)
	print(mnist_weight(weight_vector))

