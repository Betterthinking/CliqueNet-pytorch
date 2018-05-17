# -*- coding: utf-8 -*-
"""
Created on Mon May 14 20:25:51 2018

@author: Yuxi Li
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def global_pooling(x):
	# input x [n, c, h, w]
	# output l [n, c]
	s = torch.mean(x, dim=-1)
	s = torch.mean(s, dim=-1)

	return s

class CliqueNet(nn.Module):
	"""object of CliqueNet"""
	def __init__(self, nin, num_classes, layers, filters, attention=False, compression=False, dropout_prob=0.0):
		super(CliqueNet, self).__init__()
		self.conv = nn.Conv2d(nin, 64, kernel_size=3, padding=1, stride=1)
		self.bn = nn.BatchNorm2d(64)

		self.clique = nn.ModuleList([CliqueBlock(64, layers, filters, kernel=3, dropout_prob=dropout_prob)])
		for i in xrange(2):
			self.clique.append(CliqueBlock(layers*filters, layers, filters, kernel=3, dropout_prob=dropout_prob))
		self.transition = nn.ModuleList([Transition(layers*filters, layers*filters, attention, dropout_prob) for i in xrange(3)]) 

		feature_size = 0

		if compression:
			self.compression = nn.ModuleList()
			nout = 64+layers*filters
			self.compression.append(self.conv_bn_relu(nout, nout/2, dropout_prob))
			feature_size += nout/2
			nout = layers*filters*2
			self.compression.append(self.conv_bn_relu(nout, nout/2, dropout_prob))
			feature_size += nout/2
			self.compression.append(self.conv_bn_relu(nout, nout/2, dropout_prob))
			feature_size += nout/2
		else:
			self.compression = None
			feature_size += (64+layers*filters*4)

		self.predict = nn.Linear(feature_size, num_classes)

	def conv_bn_relu(self, nin, nout, dropout_prob):
		conv = nn.Sequential(
			nn.Conv2d(nin, nout, kernel_size=1, padding=0, stride=1),
			nn.BatchNorm2d(nout),
			nn.ReLU(),
			nn.Dropout2d(dropout_prob))

		return conv

	def forward(self, x):
		x = self.conv(x)
		x = F.relu(self.bn(x))
		count = 0
		features = []
		for c, t in zip(self.clique, self.transition):
			feature, s2 = c(x)
			x = t(s2)

			if self.compression is not None:
				feature = self.compression[count](feature)
				count += 1

			output = global_pooling(feature)
			features.append(output)

		output = torch.cat(features, dim=1)
		return self.predict(output)

class CliqueBlock(nn.Module):
	""" clique block for alternative cliques """
	def __init__(self, nin, layers, filters, kernel, dropout_prob=0.0):
		super(CliqueBlock, self).__init__()
		self.layers = layers
		self.channel = filters
		self.kernel = kernel
		self.nin = nin
		self.filters = filters

		num_kernels = layers*(layers-1) # A^2_layers
		num_norms = 2*layers

		# the organization of inside parameters
		# {W01, W02, .... , W0l}
		# {W12, W13, ... ,W1l, W21, W23,...., W2l, .... , Wl1, Wl2, ... W(l-1)l}

		self.W0 = nn.Parameter(torch.rand(self.layers, filters, nin, kernel, kernel))
		self.W = nn.Parameter(torch.rand(num_kernels, filters, filters, kernel, kernel))
		self.b = nn.Parameter(torch.rand(2*self.layers, filters))
		self.activates = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(filters), nn.ReLU(), nn.Dropout2d(dropout_prob)) 
			for i in xrange(num_norms)])

		self.reset_parameters(0.01)

	def reset_parameters(self, std):
		for weight in self.parameters():
			weight.data.normal_(mean=0, std=std)

	def stage1(self, x0):

		# input {X0}
		# return {X2, X3, X4, .... Xl}

		output = None
		for i in xrange(self.layers):
			if i == 0:
				data = x0
				weight = self.W0[i]
			else:
				data = torch.cat([data, output], dim=1)
			
				weight = torch.cat([self.W0[i]]+[self.W[self.coordinate2idx(j, i)] for j in xrange(i)], dim=1) 

			bias = self.b[i]

			conv = F.conv2d(data, weight, bias, stride=1, padding=self.kernel/2) 
			output = self.activates[i](conv)

		return torch.cat([data[:, (self.nin+self.filters):, :, :], output], dim=1)

	def stage2(self, x):

		# input {X2, X3, ... , Xl}
		# output {X1', X2',..., Xl'}
		output = None
		from_layers = range(1, self.layers) # from layer index

		for i in xrange(self.layers):
			if i == 0:
				data = x
			else:
				data = torch.cat([data[:, self.filters:, :, :], output], dim=1)
			
			weight = torch.cat([self.W[self.coordinate2idx(j, i)] for j in from_layers], dim=1)
			bias = self.b[self.layers+i]
			from_layers = from_layers[1:] + [self.recurrent_index(from_layers[-1]+1)]

			conv = F.conv2d(data, weight, bias, stride=1, padding=self.kernel/2) 
			output = self.activates[self.layers+i](conv)

		s2 = torch.cat([data, output], dim=1) 

		return s2

	def coordinate2idx(self, from_idx, to_idx):

		# input:  idx (from, to) excluding the x0 pairs
		# output: the linear index in self.W matrix

		assert from_idx != to_idx
		return from_idx*(self.layers-1)+to_idx-1

	def recurrent_index(self, a):
		return a % self.layers

	def forward(self, x0):

		s1 = self.stage1(x0)
		s2 = self.stage2(s1)

		feature = torch.cat([x0, s2], dim=1)

		return feature, s2

class Transition(nn.Module):
	"""docstring for Transition"""
	def __init__(self, nin, nout, attention=False, dropout_prob=0.0):
		super(Transition, self).__init__()
		self.trans = nn.Sequential(
			nn.Conv2d(nin, nout, kernel_size=1, padding=0, stride=1),
			nn.BatchNorm2d(nout),
			nn.ReLU(),
			nn.Dropout2d(dropout_prob))

		self.pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
		
		if attention:
			self.attention = nn.Sequential(
				nn.Linear(nout, nout/2),
				nn.ReLU(),
				nn.Linear(nout/2, nout),
				nn.Sigmoid())
		else:
			self.attention = None

	def forward(self, x):

		s = self.trans(x)

		if self.attention is not None:
			# global pooling
			w = global_pooling(s) # [n, c]
			w = self.attention(w) # [n, nout]
			s = w[:, :, None, None]*s

		s = self.pool(s)
		return s


if __name__ == '__main__':
	pass