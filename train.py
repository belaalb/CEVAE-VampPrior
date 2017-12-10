from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
from pyro.optim import Adam, Adamax

import numpy as np

from datasets import IHDP

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

# Training settings
parser = argparse.ArgumentParser(description='CEVAE')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--reps', type=int, default=10, metavar='N', help='number of replications (default: 10)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--l2', type=float, default=0.0001, metavar='lambda', help='L2 wheight decay coefficient (default: 0.0001)')
parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
parser.add_argument('--checkpoint-path', type=str, default='./cp/', metavar='Path', help='Path for checkpointing')
parser.add_argument('--seed', type=int, default=10, metavar='S', help='random seed (default: 1)')
parser.add_argument('--opt', choices=['adam', 'adamax'], default='adam')
parser.add_argument('--d', type=int, default=20, metavar='S', help='latent dimension (default: 20)')
parser.add_argument('--nh', type=int, default=3, metavar='S', help='number of hidden layers (default: 3)')
parser.add_argument('--n-pseudo-inputs', type=int, default = 33, metavar='S', help='Number of pseudo-inputs')
parser.add_argument('--prior', type=int, default = 0, metavar='S', help='0: conventional prior; 1: VampPrior')
parser.add_argument('--h', type=int, default=200, metavar='S', help='number and size of hidden layers (default: 200)')
parser.add_argument('--save-every', type=int, default=10, metavar='N', help='how many epochs to wait before logging training status. Default is 10')
parser.add_argument('--no-plot', action='store_true', default=False, help='Disables plots')
parser.add_argument('--save-plots', action='store_true', default=False, help='Enables saving of plots')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

print('Cuda Mode is: {}'.format(args.cuda))

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

dataset = IHDP(replications=args.reps)

in_size = len(dataset.binfeats) + len(dataset.contfeats)

optimim_params = {'lr':args.lr, 'weight_decay':args.l2}

if args.opt == 'adam':
	optimizer = Adam(optimim_params)
elif args.opt == 'adamax':
	optimizer = Adamax(optimim_params)
else:
	print('Select correct optimizer')

trainer = TrainLoop(in_size, args.d, args.nh, args.h, args.n_pseudo_inputs, args.prior, torch.nn.functional.elu, optimizer, dataset, args.d, checkpoint_path=args.checkpoint_path, cuda=args.cuda)

trainer.train(n_epochs=args.epochs, n_reps=args.reps, batch_size=args.batch_size, save_every = args.save_every)
trainer.test()

if not args.no_plot:
	log = trainer.history
	for i, key in enumerate( log.keys() ):
		plt.figure(i+1)
		data = log[key]
		plt.errorbar(np.arange(data.shape[1])*args.save_every, data.mean(0), yerr=data.std(0))
		plt.title(key)
		if args.save_plots:
			plt.savefig(key+'.png')
		plt.show()
