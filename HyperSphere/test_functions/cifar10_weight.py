import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import progressbar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.cuda as cuda
import torch.optim as optim
from torchvision import datasets, transforms

# DIM_LIST = [0, 1930, 4978, 9618]
DIM_LIST = [0, 1920, 4944, 9552]


class Net(nn.Module):
	def __init__(self, weight_vector):
		if hasattr(weight_vector, 'data'):
			weight_vector = weight_vector.data.clone()
		self.weight_dim = weight_vector.numel()

		assert self.weight_dim in DIM_LIST
		super(Net, self).__init__()

		self.block1_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
		self.block1_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
		self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

		n_BO_feed = 0

		if self.weight_dim in DIM_LIST[:3]:
			self.block2_conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
			self.block2_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
		else:
			begin_ind, end_ind = (n_BO_feed, n_BO_feed + 16 * 16 * 3 * 3)
			self.block2_conv1_weight = Variable(weight_vector[begin_ind:end_ind].view(16, 16, 3, 3))
			# begin_ind, end_ind = (end_ind, end_ind + 16)
			# self.block2_conv1_bias = Variable(weight_vector[begin_ind:end_ind])
			self.block2_conv1_bias = Parameter(torch.FloatTensor(16).type_as(weight_vector))

			begin_ind, end_ind = (end_ind, end_ind + 16 * 16 * 3 * 3)
			self.block2_conv2_weight = Variable(weight_vector[begin_ind:end_ind].view(16, 16, 3, 3))
			# begin_ind, end_ind = (end_ind, end_ind + 16)
			# self.block2_conv2_bias = Variable(weight_vector[begin_ind:end_ind])
			self.block2_conv2_bias = Parameter(torch.FloatTensor(16).type_as(weight_vector))

			# n_BO_feed += 2304 + 16 + 2304 + 16
			n_BO_feed += 2304 + 2304
		self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

		if self.weight_dim in DIM_LIST[:2]:
			self.block3_conv1 = nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3, padding=1)
			self.block3_conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
		else:
			begin_ind, end_ind = (n_BO_feed, n_BO_feed + 16 * 12 * 3 * 3)
			self.block3_conv1_weight = Variable(weight_vector[begin_ind:end_ind].view(12, 16, 3, 3))
			# begin_ind, end_ind = (end_ind, end_ind + 12)
			# self.block3_conv1_bias = Variable(weight_vector[begin_ind:end_ind])
			self.block3_conv1_bias = Parameter(torch.FloatTensor(12).type_as(weight_vector))

			begin_ind, end_ind = (end_ind, end_ind + 12 * 12 * 3 * 3)
			self.block3_conv2_weight = Variable(weight_vector[begin_ind:end_ind].view(12, 12, 3, 3))
			# begin_ind, end_ind = (end_ind, end_ind + 12)
			# self.block3_conv2_bias = Variable(weight_vector[begin_ind:end_ind])
			self.block3_conv2_bias = Parameter(torch.FloatTensor(12).type_as(weight_vector))

			# n_BO_feed += 1728 + 12 + 1296 + 12
			n_BO_feed += 1728 + 1296
		self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

		if self.weight_dim in DIM_LIST[:1]:
			self.fc = nn.Linear(in_features=12 * 4 * 4, out_features=10)
		else:
			begin_ind, end_ind = (n_BO_feed, n_BO_feed + 10 * 12 * 4 * 4)
			self.fc_weight = Variable(weight_vector[begin_ind:end_ind].view(10, 12 * 4 * 4))
			# begin_ind, end_ind = (end_ind, end_ind + 10)
			# self.fc_bias = Variable(weight_vector[begin_ind:end_ind])
			self.fc_bias = Parameter(torch.FloatTensor(10).type_as(weight_vector))

	def forward(self, x):
		using_cuda = x.is_cuda
		x = F.relu(self.block1_conv1(x))
		x = F.relu(self.block1_conv2(x))
		x = self.block1_pool(x)

		if self.weight_dim in DIM_LIST[:3]:
			x = F.relu(self.block2_conv1(x))
			x = F.relu(self.block2_conv2(x))
		else:
			if using_cuda and not self.block2_conv1_weight.is_cuda and isinstance(self.block2_conv1_weight, Variable):
				self.block2_conv1_weight = self.block2_conv1_weight.cuda()
				self.block2_conv2_weight = self.block2_conv2_weight.cuda()
			x = F.relu(F.conv2d(input=x, weight=self.block2_conv1_weight, bias=self.block2_conv1_bias, padding=1))
			x = F.relu(F.conv2d(input=x, weight=self.block2_conv2_weight, bias=self.block2_conv2_bias, padding=1))
		x = self.block2_pool(x)

		if self.weight_dim in DIM_LIST[:2]:
			x = F.relu(self.block3_conv1(x))
			x = F.relu(self.block3_conv2(x))
		else:
			if using_cuda and not self.block3_conv1_weight.is_cuda and isinstance(self.block3_conv1_weight, Variable):
				self.block3_conv1_weight = self.block3_conv1_weight.cuda()
				self.block3_conv2_weight = self.block3_conv2_weight.cuda()
			x = F.relu(F.conv2d(input=x, weight=self.block3_conv1_weight, bias=self.block3_conv1_bias, padding=1))
			x = F.relu(F.conv2d(input=x, weight=self.block3_conv2_weight, bias=self.block3_conv2_bias, padding=1))
		x = self.block3_pool(x)

		if self.weight_dim in DIM_LIST[:1]:
			x = self.fc(x.view(-1, 12 * 4 * 4))
		else:
			if using_cuda and not self.fc_weight.is_cuda and isinstance(self.fc_weight, Variable):
				self.fc_weight = self.fc_weight.cuda()
			x = F.linear(input=x.view(-1, 12 * 4 * 4), weight=self.fc_weight, bias=self.fc_bias)

		x = F.log_softmax(x)

		return x


def load_cifar10(batch_size, use_cuda):
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data', train=False, transform=transform), batch_size=batch_size, shuffle=True, **kwargs)
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


def cifar10_weight(weight_vector, train_result=False):
	use_cuda = cuda.is_available()
	model = Net(weight_vector=weight_vector)
	for m in model.parameters():
		if m.dim() >= 2:
			nn.init.xavier_normal(m.data)
		else:
			m.data.zero_()
	if use_cuda:
		model.cuda()
	train_loader, test_loader = load_cifar10(batch_size, use_cuda)
	optimizer = optim.Adam(model.parameters(), weight_decay=0.001)
	train(train_loader, model, epoch, optimizer, use_cuda)
	if train_result:
		train_loss, train_accuracy = test(train_loader, model, use_cuda)
		print('\nTRAIN - Loss : %f / Accuracy : %6.4f' % (train_loss, train_accuracy))
	test_loss, test_accuracy = test(test_loader, model, use_cuda)
	print('\nTEST  - Loss : %f / Accuracy : %6.4f' % (test_loss, test_accuracy))
	return torch.FloatTensor([[test_loss]])


cifar10_weight.dim = 0


def cifar10_weight_baseline(type='loss'):
	if type == 'loss':
		return [0.803153, 0.833086]
	elif type == 'accuracy':
		return [0.7269, 0.7215]


def architecture_SGD_trainable_check(n_BO_select):
	model = Net(torch.randn(n_BO_select))
	input_data = Variable(torch.randn(7, 3, 32, 32))
	n_param_total = 0
	for m in model.children():
		for p in m.parameters():
			n_param_total += p.numel()
	n_param_info = []
	n_param_accum = 0
	for m in model.named_children():
		n_param = 0
		for p in m[1].parameters():
			n_param += p.numel()
		n_param_accum += n_param
		n_param_info.append((m[0], n_param, n_param_accum, n_param_total - n_param_accum, m[1]))
	print('    name    /# w+b/#accum/# rest')
	for elm in n_param_info:
		print('%-12s %5d  %5d  %5d' % elm[:4])
	# for p in model.named_parameters():
	# 	print(p[0], p[1].size())
	# print(model(input_data).size())

if __name__ == '__main__':
	# architecture_SGD_trainable_check(DIM_LIST[0])
	argv = 0 if len(sys.argv) == 1 else int(sys.argv[1])
	n_BO_select = DIM_LIST[argv]
	weight_vector = torch.FloatTensor(n_BO_select).normal_()
	if torch.cuda.is_available():
		weight_vector = weight_vector.cuda()
	print('%d parameters are given by BO\n%d parameters are trained by SGD' % (n_BO_select, 12386 - n_BO_select))
	cifar10_weight(weight_vector, train_result=True)