from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
import model as model_
from pyro.optim import Adam

from datasets import IHDP

# Training settings
parser = argparse.ArgumentParser(description='Online transfer learning for emotion recognition tasks')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--reps', type=int, default=1, metavar='N', help='number of replications (default: 1)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs (default: 10)')
parser.add_argument('--patience', type=int, default=30, metavar='N', help='number of epochs without improvement to wait before stopping training (default: 30)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--l2', type=float, default=0.0001, metavar='lambda', help='L2 wheight decay coefficient (default: 0.0001)')
parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default='./cp/', metavar='Path', help='Path for checkpointing')
parser.add_argument('--seed', type=int, default=111, metavar='S', help='random seed (default: 1)')
parser.add_argument('--d', type=int, default=20, metavar='S', help='latent dimension (default: 20)')
parser.add_argument('--nh', type=int, default=3, metavar='S', help='number of hidden layers (default: 3)')
parser.add_argument('--h', type=int, default=200, metavar='S', help='number and size of hidden layers (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

dataset = IHDP(replications=args.reps)

in_size = len(dataset.binfeats) + len(dataset.contfeats)

encoder = model_.encoder(in_size, in_size + 1, args.d, args.nh, args.h, len(dataset.binfeats), len(dataset.contfeats), activation=torch.nn.functional.relu)
decoder = model_.decoder(args.d, args.nh, args.h, len(dataset.binfeats), len(dataset.contfeats), activation=torch.nn.functional.relu)

if args.cuda:
	encoder = encoder.cuda()
	decoder = decoder.cuda()

optimim_params = {'lr':args.lr, 'weight_decay':args.l2}
optimizer = Adam(optimim_params)

trainer = TrainLoop(encoder, decoder, optimizer, dataset, args.d, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

print('Cuda Mode is: {}'.format(args.cuda))

trainer.train(n_epochs=args.epochs,n_reps=args.reps, patience = args.patience)
trainer.test()
