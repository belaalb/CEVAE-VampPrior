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
import evaluation

INFINITY = float('inf')

class TrainLoop(object):
	def __init__(self, encoder, decoder, optimizer, dataset, z_dim, n_pseudo_inputs, checkpoint_path = None, checkpoint_epoch = None, cuda = True):

		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.encoder = encoder
		self.decoder = decoder
		self.svi = SVI(self.model, self.guide, optimizer, loss="ELBO")
		self.dataset = dataset
		self.history = {'train_loss': [], 'valid_loss': []}
		self.z_dim = z_dim
		self.n_pseudo_inputs = n_pseudo_inputs

		if checkpoint_epoch is not None:

			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=10, n_reps=1, patience=100, save_every=1):

		last_best_val_loss = INFINITY

		train_iter = tqdm(enumerate(self.dataset.get_train_valid_test()))

		for t, (train, valid, test, contfeats, binfeats) in train_iter:
			train_loss = 0.0
			best_val_loss = np.inf
			print('Replication {}/{}'.format(t + 1, n_epochs))
			(xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
			(xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
			(xte, tte, yte), (y_cfte, mu0te, mu1te) = test
			xtr, ttr, ytr, y_cftr, mu0tr, mu1tr = torch.from_numpy(xtr), torch.from_numpy(ttr), torch.from_numpy(ytr), torch.from_numpy(y_cftr), torch.from_numpy(mu0tr), torch.from_numpy(mu1tr)
			xva, tva, yva, y_cfva, mu0va, mu1va = torch.from_numpy(xva), torch.from_numpy(tva), torch.from_numpy(yva), torch.from_numpy(y_cfva), torch.from_numpy(mu0va), torch.from_numpy(mu1va)
			xte, tte, yte, y_cfte, mu0te, mu1te = torch.from_numpy(xte), torch.from_numpy(tte), torch.from_numpy(yte), torch.from_numpy(y_cfte), torch.from_numpy(mu0te), torch.from_numpy(mu1te)

			perm = binfeats + contfeats
			xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]

			xalltr, talltr, yalltr = torch.cat((xtr, xva), 0), torch.cat((ttr, tva), 0), torch.cat((ytr, yva), 0)

			self.train_calc_stats = evaluation.calc_stats(yalltr, talltr, y_cf=torch.cat([y_cftr, y_cfva], 0), mu0=torch.cat([mu0tr, mu0va], 0), mu1=torch.cat([mu1tr, mu1va], 0))
			self.test_calc_stats = evaluation.calc_stats(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)

			# zero mean, unit variance for y during training
			ym, ys = torch.mean(ytr), torch.std(ytr)
			ytr, yva = (ytr - ym) / ys, (yva - ym) / ys

			train_data = Variable(torch.cat([xtr, ttr, ytr], 1).float())
			x_val, y_val, t_val = xva, yva, tva
			val_data = Variable(torch.cat([x_val, t_val, y_val], 1).float()) 
			## prepare data

			if self.cuda_mode:
				data = data.cuda()
			
			n_iter_per_epoch, idx = 10 * int(xtr.shape[0] / 100), np.arange(xtr.shape[0])
			for epoch in range(n_epochs):
				avg_loss = 0.0

				t0 = time.time()
				np.random.shuffle(idx)
				for j in range(n_iter_per_epoch):
					batch = torch.LongTensor(np.random.choice(idx, 100))
					x_train, y_train, t_train = xtr.index_select(0, batch), ytr.index_select(0, batch), ttr.index_select(0, batch)
					batch_train_data = Variable(torch.cat([x_train, t_train, y_train], 1).float())
					info_dict = self.svi.step(batch_train_data)
					avg_loss += info_dict
				
				avg_loss = avg_loss / n_iter_per_epoch
				avg_loss = avg_loss / 100				
				print("average train loss in epoch {}/{}: {} ".format(epoch + 1, n_epochs, avg_loss))
				
				train_loss = self.svi.evaluate_loss(train_data)
				self.history['train_loss'].append(train_loss)
				print("train loss in epoch {}/{}: {} ".format(epoch + 1, n_epochs, train_loss))

				x1, x2, t, y , z = self.model(batch_train_data,separated = True)
				print("YY : {}, {}, {}, {}".format(x1.size(), x2.size(), t.size(), y.size()))
				
				# val_loss = evaluation.val_loss(x1, x2, t,z, y,x_val, t_val, y_val)
				val_loss = self.svi.evaluate_loss(val_data)
				# to be completed YY :
				
				# print "temporary : " , val_loss
				self.history['valid_loss'].append(val_loss)
				print("validation loss in epoch {}/{}: {}".format(epoch + 1, n_epochs, val_loss))
				if val_loss <= best_val_loss:
					print('Improved validation bound, old: {:0.3f}, new: {:0.3f}'.format(best_val_loss, val_loss))
					best_val_loss = val_loss
					self.cur_epoch = epoch
					self.checkpointing()

	def test(self):

		y0, y1 = get_y0_y1(y_post, f0, f1, shape = yalltr.shape, L = 100)
		y0, y1 = y0 * ys + ym, y1 * ys + ym
		score = self.train_calc_stats(y1, y0)
		print("Final train score: {}".format(score))

		y0t, y1t = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=100)
		y0t, y1t = y0t * ys + ym, y1t * ys + ym
		score_test = self.test_calc_stats(y1t, y0t)
		print("Final test score: {}".format(score_test))

		print('Replication: {}/{}, tr_ite: {:0.3f}, tr_ate: {:0.3f}, tr_pehe: {:0.3f} \ te_ite: {:0.3f}, te_ate: {:0.3f}, te_pehe: {:0.3f}'.format(i + 1, reps, score[0], score[1], score[2], score_test[0], score_test[1], score_test[2]))

	def model(self, data, prior = 'vamp', separated = False):

		decoder = pyro.module('decoder', self.decoder)

		# Normal prior
	    if prior == 'standard':
			z_mu, z_sigma = ng_zeros([data.size(0), self.z_dim]), ng_ones([data.size(0), self.z_dim])

			z = pyro.sample("latent", dist.normal, z_mu, z_sigma)

	    elif self.args.prior == 'vamp':
			z_mu_minibatch, z_sigma_minibatch = self.vampprior()
			z = pyro.sample("latent", dist.normal, z_mu, z_sigma)

			z = z.expand(100, -1)
			
		x1, x2, t, y = decoder.forward(z) 

		# pyro.sample('obs', torch.cat([x1, x2, t, y], 1))
		if separated :
			return x1, x2, t, y , z
		return torch.cat([x1, x2, t, y], 1)

	def vampprior(self):
		# Minibatch of size n_pseudo_inputs of "one hot-encoded" embeddings for each pseudo-input
		idle_input = Variable(torch.eye(self.n_pseudo_inputs, self.n_pseudo_inputs), requires_grad = False)
 	
		# Generate pseudo-inputs from embeddings
        pseudo_inputs = self.encoder.forward_pseudo_inputs(idle_input)

        # Calculate mu and var for latent representation of all pseudo inputs  
		muq_t0, sigmaq_t0, muq_t1, sigmaq_t1, qt = self.encoder.forward(pseudo_inputs)
		z_mu = qt * muq_t1 + (1. - qt) * muq_t0
		z_sigma = qt * sigmaq_t1 + (1. - qt) * sigmaq_t0
		 
		return z_mu, z_sigma 
		

	def guide(self, data):
		encoder = pyro.module('encoder', self.encoder)

		muq_t0, sigmaq_t0, muq_t1, sigmaq_t1, qt = encoder.forward(data[:, :(len(self.dataset.binfeats) + len(self.dataset.contfeats))])

		pyro.sample('latent', dist.normal, qt * muq_t1 + (1. - qt) * muq_t0, qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'encoder_state': self.encoder.state_dict(),
		'decoder_state': self.decoder.state_dict(),
		'optimizer_state': self.svi.optim.get_state(),
		'history': self.history,
		'cur_epoch': self.cur_epoch}
		torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load model state
			self.encoder.load_state_dict(ckpt['encoder_state'])
			self.decoder.load_state_dict(ckpt['decoder_state'])
			# Load optimizer state
			self.svi.optim.load_state_dict(ckpt['optimizer_state'])
			# Load history
			self.history = ckpt['history']
			self.cur_epoch = ckpt['cur_epoch']

		else:
			print('No checkpoint found at: {}'.format(ckpt))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'models/m6-ihdp.pth.tar')

def load_model(filename='checkpoint.pth.tar'):
	if os.path.isfile(filename):
		print("=> loading checkpoint '{}'".format(resume))
		checkpoint = torch.load(args.resume)
		start_epoch = checkpoint['epoch']
		best_prec1 = checkpoint['best_prec1']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
