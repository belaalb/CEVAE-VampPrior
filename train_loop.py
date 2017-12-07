import torch
from torch.autograd import Variable

import pyro
from pyro.infer import SVI
from pyro.util import ng_zeros, ng_ones
import pyro.distributions as dist
import shutil
import numpy as np
import time
from tqdm import tqdm
import os
import sys
from evaluation import Evaluator
import model as model_

class TrainLoop(object):
	def __init__(self, in_size, d, nh, h, n_pseudo_inputs, activation, optimizer, dataset, z_dim, checkpoint_path=None, cuda=True):

		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.in_size = in_size
		self.d = d
		self.nh = nh
		self.h = h
		self.activation = activation
		self.save_fmt = os.path.join(self.checkpoint_path, 'best_model_rep{}.pt')
		self.cuda_mode = cuda
		self.dataset = dataset
		self.optimizer = optimizer
		self.history = {'train_loss': [], 'valid_loss': []}
		self.z_dim = z_dim
		self.n_pseudo_inputs = n_pseudo_inputs

	def train(self, n_epochs=10, n_reps=1, batch_size=100, save_every=1):

		best_val_loss_gen = np.inf

		train_iter = tqdm(enumerate(self.dataset.get_train_valid_test()))

		## Replications loop
		for t, (train, valid, test, contfeats, binfeats) in train_iter:

			best_val_loss_rep = np.inf

			## Setup nem models/optimizers

			self.encoder = model_.encoder(self.in_size, self.in_size + 1, self.d, self.nh, self.h, self.n_pseudo_inputs, len(self.dataset.binfeats), len(self.dataset.contfeats), self.activation)
			self.decoder = model_.decoder(self.d, self.nh, self.h, len(self.dataset.binfeats), len(self.dataset.contfeats), self.activation)

			if self.cuda_mode:
				self.encoder = self.encoder.cuda()
				self.decoder = self.decoder.cuda()

			self.svi = SVI(self.model, self.guide, self.optimizer, loss="ELBO")

			train_loss = 0.0

			## Prepare data

			print('\nReplication {}/{} \n'.format(t + 1, n_reps))
			(xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
			(xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
			(xte, tte, yte), (y_cfte, mu0te, mu1te) = test
			xtr, ttr, ytr, y_cftr, mu0tr, mu1tr = torch.from_numpy(xtr).float(), torch.from_numpy(ttr).float(), torch.from_numpy(ytr).float(), torch.from_numpy(y_cftr).float(), torch.from_numpy(mu0tr).float(), torch.from_numpy(mu1tr).float()
			xva, tva, yva, y_cfva, mu0va, mu1va = torch.from_numpy(xva).float(), torch.from_numpy(tva).float(), torch.from_numpy(yva).float(), torch.from_numpy(y_cfva).float(), torch.from_numpy(mu0va).float(), torch.from_numpy(mu1va).float()
			xte, tte, yte, y_cfte, mu0te, mu1te = torch.from_numpy(xte).float(), torch.from_numpy(tte).float(), torch.from_numpy(yte).float(), torch.from_numpy(y_cfte).float(), torch.from_numpy(mu0te).float(), torch.from_numpy(mu1te).float()

			perm = binfeats + contfeats
			xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]

			xalltr, talltr, yalltr = torch.cat((xtr, xva), 0), torch.cat((ttr, tva), 0), torch.cat((ytr, yva), 0)

			# zero mean, unit variance for y during training
			self.ym, self.ys = torch.mean(ytr), torch.std(ytr)
			ytr, yva = (ytr - self.ym) / self.ys, (yva - self.ym) / self.ys

			train_data = Variable(torch.cat([xtr, ttr, ytr], 1).float())
			val_data = Variable(torch.cat([xva, yva, tva], 1).float())
			test_data = Variable(xte.float())

			if self.cuda_mode:
				train_data = train_data.cuda()
				val_data = val_data.cuda()
				test_data = test_data.cuda()
				self.evaluator_test = Evaluator(yte.cuda(), tte.cuda(), y_cf=y_cfte.cuda(), mu0=mu0te.cuda(), mu1=mu1te.cuda())
				self.evaluator_train = Evaluator(yalltr.cuda(), talltr.cuda(), y_cf=torch.cat([y_cftr, y_cfva], 0).cuda(), mu0=torch.cat([mu0tr, mu0va], 0).cuda(), mu1=torch.cat([mu1tr, mu1va], 0).cuda())
			else:
				self.evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)
				self.evaluator_train = Evaluator(yalltr, talltr, y_cf=torch.cat([y_cftr, y_cfva], 0), mu0=torch.cat([mu0tr, mu0va], 0), mu1=torch.cat([mu1tr, mu1va], 0))
			n_iter_per_epoch, idx = 10 * int(xtr.shape[0] / batch_size), np.arange(xtr.shape[0])

			## Train loop - batches

			for epoch in range(n_epochs):
				avg_loss = 0.0
				np.random.shuffle(idx)

				## Iterations loop - minibatches
				for j in range(n_iter_per_epoch):
					batch = Variable(torch.LongTensor(np.random.choice(idx, batch_size)) )
					if self.cuda_mode:
						batch = batch.cuda()

					minibatch_train_data = train_data.index_select(0, batch)

					info_dict = self.svi.step(minibatch_train_data)
					avg_loss += info_dict
				avg_loss = avg_loss / (n_iter_per_epoch)

				## evaluation - each save_every epochs

				if epoch%save_every == 0:
					train_loss = self.svi.evaluate_loss(train_data)
					val_loss = self.svi.evaluate_loss(val_data)

					self.history['train_loss'].append(train_loss)
					self.history['valid_loss'].append(val_loss)

					print("average train loss in epoch {}/{}: {} ".format(epoch+1, n_epochs, avg_loss))
					print("train loss in epoch {}/{}: {} ".format(epoch+1, n_epochs, train_loss))
					print("validation loss in epoch {}/{}: {}".format(epoch+1, n_epochs, val_loss))

					metrics_data = torch.cat([train_data[:,:-2], val_data[:,:-2] ],0)

					y0, y1 = self.get_y0_y1(metrics_data, L=1)
					y0, y1 = y0 * self.ys + self.ym, y1 * self.ys + self.ym

					score = self.evaluator_train.calc_stats(y1, y0)

					y0t, y1t = self.get_y0_y1(test_data, L=1)
					y0t, y1t = y0t * self.ys + self.ym, y1t * self.ys + self.ym

					score_test = self.evaluator_test.calc_stats(y1t, y0t)

					print('\nTrain and test metrics in epoch {}/{} ==> tr_ite: {:0.3f}, tr_ate: {:0.3f}, tr_pehe: {:0.3f}, te_ite: {:0.3f}, te_ate: {:0.3f}, te_pehe: {:0.3f}'.format(epoch+1, n_epochs, score[0], score[1], score[2], score_test[0], score_test[1], score_test[2]))

					if val_loss <= best_val_loss_rep:
						print('Improved validation bound, old: {:0.3f}, new: {:0.3f}\n '.format(best_val_loss_rep, val_loss))
						best_val_loss_rep = val_loss
						self.cur_epoch = epoch
						self.checkpointing(t+1)
						if val_loss < best_val_loss_gen:
							best_val_loss_gen = val_loss
							self.checkpointing()
		
			## evaluation - each replication - Load best model within the replications
			self.load_checkpoint(self.save_fmt.format(t+1))

			scores = np.zeros((n_reps, 3))
			scores_test = np.zeros((n_reps, 3))

			y0, y1 = self.get_y0_y1(metrics_data, L=100)
			y0, y1 = y0 * self.ys + self.ym, y1 * self.ys + self.ym

			score = self.evaluator_train.calc_stats(y1, y0)
			scores[t,:] = score

			y0t, y1t = self.get_y0_y1(test_data, L=100)
			y0t, y1t = y0t * self.ys + self.ym, y1t * self.ys + self.ym

			score_test = self.evaluator_test.calc_stats(y1t, y0t)
			scores_test[t,:] = score_test

			scores_mu = np.mean(scores, axis=0)
			scores_std = np.std(scores, axis=0)

			scores_mu_test = np.mean(scores_test, axis=0)
			scores_std_test = np.std(scores_test, axis=0)

			print('\nTrain and test metrics ==> tr_ite: {:0.3f}+-{:0.3f}, tr_ate: {:0.3f}+-{:0.3f}, tr_pehe: {:0.3f}+-{:0.3f}, te_ite: {:0.3f}+-{:0.3f}, te_ate: {:0.3f}+-{:0.3f}, te_pehe: {:0.3f}+-{:0.3f}'.format(scores_mu[0], scores_std[0], scores_mu[1], scores_std[1], scores_mu[2], scores_std[2], scores_mu_test[0], scores_std_test[0], scores_mu_test[1], scores_std_test[1], scores_mu_test[2], scores_std_test[2]))

	def test(self):

		## evaluation - final valid - Load best model over all the replications

		self.load_checkpoint(os.path.join(self.checkpoint_path, 'best_model_gen.pt'))

		(train, valid, test, contfeats, binfeats) = next(self.dataset.get_train_valid_test())
		(xdata, tdata, ydata), (y_cfte, mu0te, mu1te) = test
		xdata, tdata, ydata, y_cfte, mu0te, mu1te = torch.from_numpy(xdata).float(), torch.from_numpy(tdata).float(), torch.from_numpy(ydata).float(), torch.from_numpy(y_cfte).float(), torch.from_numpy(mu0te).float(), torch.from_numpy(mu1te).float()
		perm = binfeats + contfeats
		xdata = xdata[:, perm]

		if self.cuda_mode:
			xdata = xdata.cuda()

		xdata = Variable(xdata.float())

		y0t, y1t = self.get_y0_y1(xdata, L=100)
		y0t, y1t = y0t * self.ys + self.ym, y1t * self.ys + self.ym

		score_test = self.evaluator_test.calc_stats(y1t, y0t)

		print('\nOverall best model metrics ==> te_ite: {:0.3f}, te_ate: {:0.3f}, te_pehe: {:0.3f}'.format(score_test[0], score_test[1], score_test[2]))

	def get_y0_y1(self, x, shape=(), L=1, verbose=True):

		t0 = torch.zeros([x.size(0), 1])
		t1 = torch.ones([x.size(0), 1])

		if self.cuda_mode:
			t0 = t0.cuda()
			t1 = t1.cuda()

		t0 = Variable(t0.float())
		t1 = Variable(t1.float())

		muq_t0, sigmaq_t0, muq_t1, sigmaq_t1, qt = self.encoder.forward(x)

		for l in range(L):
			if L > 1 and verbose:
				sys.stdout.write('\rSample {}/{}'.format(l + 1, L))
				sys.stdout.flush()

			z_t0 = pyro.sample('latent_t0', dist.normal, muq_t0, sigmaq_t0)
			z_t1 = pyro.sample('latent_t1', dist.normal, muq_t1, sigmaq_t1)

			try:
				y0 +=  self.decoder.p_y_zt(z_t0, t0) / L
				y1 += self.decoder.p_y_zt(z_t1, t1) / L
			except UnboundLocalError:
				y0 = self.decoder.p_y_zt(z_t0, t0) / L
				y1 = self.decoder.p_y_zt(z_t1, t1) / L

		return y0.data, y1.data

	def model(self, data, prior = 'vamp', seperated = False):
		decoder = pyro.module('decoder', self.decoder)

		# Normal prior
		if prior == 'standard':
			z_mu, z_sigma = ng_zeros([data.size(0), self.z_dim]), ng_ones([data.size(0), self.z_dim])

			z = pyro.sample("latent", dist.normal, z_mu, z_sigma)

		elif prior == 'vamp':
			z_mu, z_sigma = self.vampprior()
		
			z_mu_avg = torch.mean(z_mu, 0)

			z_sigma_square = z_sigma * z_sigma
			z_sigma_square_avg = torch.mean(z_sigma_square, 0)
			z_sigma_avg = torch.sqrt(z_sigma_square_avg)

			z_mu_avg = z_mu_avg.expand(data.size(0), z_mu_avg.size(0))
			z_sigma_avg = z_sigma_avg.expand(data.size(0), z_sigma_avg.size(0))

			z = pyro.sample("latent", dist.normal, z_mu_avg, z_sigma_avg)

		x1, x2, t, y = decoder.forward(z)

		# pyro.sample('obs', torch.cat([x1, x2, t, y], 1))
		if seperated :
			return x1, x2, t, y , z
		return torch.cat([x1, x2, t, y], 1)

	def vampprior(self):
		# Minibatch of size n_pseudo_inputs of "one hot-encoded" embeddings for each pseudo-input
		idle_input = Variable(torch.eye(self.n_pseudo_inputs, self.n_pseudo_inputs), requires_grad = False)

		if self.cuda_mode:
			idle_input = idle_input.cuda()
 	
		# Generate pseudo-inputs from embeddings
		pseudo_inputs = self.encoder.forward_pseudo_inputs(idle_input)

		print(pseudo_inputs[3, :])
		print(pseudo_inputs[6, :])

        # Calculate mu and var for latent representation of all pseudo inputs  
		muq_t0, sigmaq_t0, muq_t1, sigmaq_t1, qt = self.encoder.forward(pseudo_inputs)
		z_mu = qt * muq_t1 + (1. - qt) * muq_t0
		z_sigma = qt * sigmaq_t1 + (1. - qt) * sigmaq_t0
		 
		return z_mu, z_sigma


	def guide(self, data):
		encoder = pyro.module('encoder', self.encoder)

		muq_t0, sigmaq_t0, muq_t1, sigmaq_t1, qt = encoder.forward(data[:, :(len(self.dataset.binfeats) + len(self.dataset.contfeats))])

		pyro.sample('latent', dist.normal, qt * muq_t1 + (1. - qt) * muq_t0, qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

	def checkpointing(self, rep=None):

		# Checkpointing
		print('Saving model...')
		ckpt = {'encoder_state': self.encoder.state_dict(),
		'decoder_state': self.decoder.state_dict(),
		'optimizer_state': self.svi.optim.get_state(),
		'history': self.history,
		'cur_epoch': self.cur_epoch}

		if rep:
			torch.save(ckpt, self.save_fmt.format(rep))
		else:
			torch.save(ckpt, os.path.join(self.checkpoint_path, 'best_model_gen.pt'))

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load model state
			self.encoder.load_state_dict(ckpt['encoder_state'])
			self.decoder.load_state_dict(ckpt['decoder_state'])
			# Load optimizer state
			self.svi.optim.set_state(ckpt['optimizer_state'])
			# Load history
			self.history = ckpt['history']
			self.cur_epoch = ckpt['cur_epoch']

		else:
			print('No checkpoint found at: {}'.format(ckpt))
