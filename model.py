import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.distributions as dist

class fc_net(nn.Module):
	def __init__(self, in_size, layers, out_layers, activation=F.relu):
		super(fc_net, self).__init__()

		self.activation = activation

		if layers:
			self.input = nn.Linear(in_size, layers[0])
			self.hidden_layers = nn.ModuleList()
			for i in range(1, len(layers)):
				self.hidden_layers.append(nn.Linear(layers[i], layers[i]))
			self.output_layers = nn.ModuleList()
			self.output_activations = []
			for i, (outdim, activation) in enumerate(out_layers):
				self.output_layers.append(nn.Linear(layers[-1], outdim))
				self.output_activations.append(activation)
		else:
			self.output_layers = nn.ModuleList()
			self.output_activations = []
			for i, (outdim, activation) in enumerate(out_layers):
				self.output_layers.append(nn.Linear(in_size, outdim))
				self.output_activations.append(activation)

	def forward(self, x):

		x = self.activation( self.input(x) )
		try:
			for layer in self.hidden_layers:
				x = self.activation( layer(x) )
		except AttributeError:
			pass
		if self.output_layers:
			outputs = []
			for output_layer, output_activation in zip(self.output_layers, self.output_activations):
				if output_activation:
					outputs.append( output_activation( output_layer(x) ) )
				else:
					#outputs.append( F.hardtanh(output_layer(x), 0.0, 1.0) )
					#outputs.append( F.sigmoid(output_layer(x)) )
					outputs.append( output_layer(x) )
			return outputs if len(outputs) > 1 else outputs[0]
		else:
			return x

class decoder(nn.Module):
	def __init__(self, n_z, nh, h, binfeats, contfeats, activation):
		super(decoder, self).__init__()

		# p(x|z)
		self.hx = fc_net(n_z, (nh - 1) * [h], [], activation=activation)
		self.logits_1 = fc_net(h, [h], [[binfeats, F.sigmoid]], activation=activation)

		self.mu_sigma = fc_net(h, [h], [[contfeats, None], [contfeats, F.softplus]], activation=activation)

		# p(t|z)
		self.logits_2 = fc_net(n_z, [h], [[1, F.sigmoid]], activation=activation)

		# p(y|t,z)
		self.mu2_t0 = fc_net(n_z, nh * [h], [[1, None]], activation=activation)
		self.mu2_t1 = fc_net(n_z, nh * [h], [[1, None]], activation=activation)

	def forward(self, z, x_bin, x_cont, t_, y_):
		# p(x|z)

		hx = self.hx.forward(z)

		logits_1 = self.logits_1.forward(hx)

		x1 = pyro.sample('x1', dist.bernoulli, logits_1, obs=x_bin)

		mu, sigma = self.mu_sigma.forward(hx)
		x2 = pyro.sample('x2', dist.normal, mu, sigma, obs=x_cont)

		# p(t|z)
		logits_2 = self.logits_2(z)
		t = pyro.sample('t', dist.bernoulli,logits_2, obs=t_.contiguous().view(-1, 1))

		# p(y|t,z)
		mu2_t0 = self.mu2_t0(z)
		mu2_t1 = self.mu2_t1(z)

		sig = Variable(torch.ones(mu2_t0.size()))
		if mu2_t0.is_cuda:
			sig = sig.cuda()

		y = pyro.sample('y', dist.normal, t * mu2_t1 + (1. - t) * mu2_t0, sig, obs=y_.contiguous().view(-1, 1))

		return x1, x2, t, y

	def p_y_zt(self, z, t):

		# p(y|t,z)
		mu2_t0 = self.mu2_t0(z)
		mu2_t1 = self.mu2_t1(z)

		sig = Variable(torch.ones(mu2_t0.size()))
		if mu2_t0.is_cuda:
			sig = sig.cuda()

		if t:
			y = dist.normal(mu2_t1, sig)
		else:
			y = dist.normal(mu2_t0, sig)
		return y

class encoder(nn.Module):
	def __init__(self, in_size, in2_size, d, nh, h, n_pseudo_inputs, binfeats, contfeats, activation):
		super(encoder, self).__init__()

		# q(t|x)
		self.logits_t = fc_net(in_size, [d], [[1, F.sigmoid]], activation=activation)

		# q(y|x,t)
		self.hqy = fc_net(in_size, (nh - 1) * [h], [], activation=activation)
		self.mu_qy_t0 = fc_net(h, [h], [[1, None]], activation=activation)
		self.mu_qy_t1 = fc_net(h, [h], [[1, None]], activation=activation)

		# q(z|x,t,y)
		self.hqz = fc_net(in2_size, (nh - 1) * [h], [], activation=activation)
		self.muq_t0_sigmaq_t0 = fc_net(h, [h], [[d, None], [d, F.softplus]], activation=activation)
		self.muq_t1_sigmaq_t1 = fc_net(h, [h], [[d, None], [d, F.softplus]], activation=activation)

		# pseudo-inputs generation
		self.h_idle_input_cont = nn.Linear(n_pseudo_inputs, contfeats, bias=None)
		self.h_idle_input_bin = nn.Linear(n_pseudo_inputs, binfeats, bias=None)

	def forward(self, x):

		# q(t|x)
		logits_t = self.logits_t.forward(x)

		qt = dist.bernoulli(logits_t)

		# q(y|x,t)
		hqy = self.hqy.forward(x)
		mu_qy_t0 = self.mu_qy_t0.forward(hqy)
		mu_qy_t1 = self.mu_qy_t1.forward(hqy)

		sig = Variable(torch.ones(mu_qy_t0.size()))
		if mu_qy_t0.is_cuda:
			sig = sig.cuda()

		qy = dist.normal(qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, sig )

		# q(z|x,t,y)
		hqz = self.hqz.forward(torch.cat((x, qy), 1))
		muq_t0, sigmaq_t0 = self.muq_t0_sigmaq_t0.forward(hqz)
		muq_t1, sigmaq_t1 = self.muq_t1_sigmaq_t1.forward(hqz)

		return muq_t0, sigmaq_t0, muq_t1, sigmaq_t1, qt

	def forward_pseudo_inputs(self, x_idle):
		'''
		Specific forward pass to generate pseudo inputs given idle_input
		'''

		pseudo_input_cont = F.sigmoid(self.h_idle_input_cont(x_idle))
		pseudo_input_bin = F.hardtanh(self.h_idle_input_bin(x_idle), 0, 1)
		pseudo_input = torch.cat([pseudo_input_cont, pseudo_input_bin], 1)

		return pseudo_input
