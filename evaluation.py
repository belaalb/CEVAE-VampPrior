import numpy as np
import torch

class Evaluator(object):
	def __init__(self, y, t, y_cf=None, mu0=None, mu1=None):
		self.y = y
		self.t = t
		self.y_cf = y_cf
		self.mu0 = mu0
		self.mu1 = mu1
		if mu0 is not None and mu1 is not None:
		    self.true_ite = mu1 - mu0

	def rmse_ite(self, ypred1, ypred0):
		pred_ite = torch.zeros(self.true_ite.size())
		idx1, idx0 = np.where(self.t.cpu().numpy() == 1), np.where(self.t.cpu().numpy() == 0)
		idx1, idx0 = torch.LongTensor(idx1[0]), torch.LongTensor(idx0[0])

		ite1, ite0 = self.y[idx1[0]] - ypred0[idx1[0]], ypred1[idx0[0]] - self.y[idx0[0]]

		if self.true_ite.is_cuda:
			pred_ite = pred_ite.cuda()
			idx1, idx0 = idx1.cuda(), idx0.cuda()

		pred_ite[idx1] = ite1.expand_as(pred_ite[idx1])
		pred_ite[idx0] = ite0.expand_as(pred_ite[idx0])

		return torch.sqrt(torch.mean(torch.pow(self.true_ite - pred_ite, 2.), 0))

	def abs_ate(self, ypred1, ypred0):
		return torch.abs(torch.mean(ypred1 - ypred0, 0) - torch.mean(self.true_ite, 0))

	def pehe(self, ypred1, ypred0):
		return torch.sqrt(torch.mean(torch.pow((self.mu1 - self.mu0) - (ypred1 - ypred0), 2.), 0))

	def y_errors(self, y0, y1):
		ypred = (1 - self.t) * y0 + self.t * y1
		ypred_cf = self.t * y0 + (1 - self.t) * y1
		return self.y_errors_pcf(ypred, ypred_cf)

	def y_errors_pcf(self, ypred, ypred_cf):
		rmse_factual = torch.sqrt(torch.mean(torch.pow(ypred - self.y, 2.), 0))
		rmse_cfactual = torch.sqrt(torch.mean(torch.pow(ypred_cf - self.y_cf, 2.), 0))
		return rmse_factual, rmse_cfactual

	def calc_stats(self, ypred1, ypred0):

		ite = self.rmse_ite(ypred1, ypred0)
		ate = self.abs_ate(ypred1, ypred0)
		pehe = self.pehe(ypred1, ypred0)

		return ite[0], ate[0], pehe[0]
