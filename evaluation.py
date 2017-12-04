import numpy as np
import torch
from torch.autograd import Variable
import pyro.distributions as dist

class rmse_ite(torch.nn.Module):

	def __init__(self,y, t, mu0=None, mu1=None):
		self.y = y
		self.t = t
		self.mu0 = mu0
		self.mu1 = mu1
		if mu0 is not None and mu1 is not None:
			self.true_ite = mu1 - mu0
		super(rmse_ite,self).__init__()

	def forward(self, y1, y0):
		pred_ite = torch.zeros(self.true_ite)
		var1 = Variable(torch.from_numpy(np.ones(self.t.size())))
		var0 = Variable(torch.from_numpy(np.ones(self.t.size())))
		idx1 = (self.t ==  var1)
		idx0 = (self.t ==  var0)
		# idx1, idx0 = torch.where(self.t == 1), torch.where(self.t == 0)
		ite1, ite0 = self.y[idx1] - y0[idx1], y1[idx0] - self.y[idx0]
		pred_ite[idx1] = ite1
		pred_ite[idx0] = ite0
		return torch.sqrt(torch.mean(torch.square(self.true_ite - pred_ite)))
		
		
class abs_ate(torch.nn.Module):

	def __init__(self,mu0=None, mu1=None):
		self.mu0 = mu0
		self.mu1 = mu1
		if mu0 is not None and mu1 is not None:
			self.true_ite = mu1 - mu0
		super(abs_ate,self).__init__()

	def forward(self, y1, y0):
		return torch.abs(torch.mean(y1 - y0) - torch.mean(self.true_ite))

class pehe(torch.nn.Module):

	def __init__(self,mu0=None, mu1=None):
		self.mu0 = mu0
		self.mu1 = mu1
		super(pehe,self).__init__()

	def forward(self, y1, y0):
		return torch.sqrt(torch.mean(torch.square((self.mu1 - self.mu0) - (y1 - y0))))


class y_errors(torch.nn.Module):

	def __init__(self,y, t, y_cf=None,mu0=None, mu1=None):
		self.y = y
		self.y_cf = y_cf
		self.t = t
		self.mu0 = mu0
		self.mu1 = mu1
		super(y_errors,self).__init__()

	def forward(self, y0, y1):
		ypred = (1 - self.t) * y0 + self.t * y1
		ypred_cf = self.t * y0 + (1 - self.t) * y1
		return self.y_errors_pcf(ypred, ypred_cf)
	
	def y_errors_pcf(self, ypred, ypred_cf):
		rmse_factual = torch.sqrt(torch.mean(torch.square(ypred - self.y)))
		rmse_cfactual = torch.sqrt(torch.mean(torch.square(ypred_cf - self.y_cf)))
		return rmse_factual, rmse_cfactual
	
	
	
	
	
class y_errors_pcf(torch.nn.Module):
	
	def __init__(self,t,mu0=None, mu1=None):
		self.t = t
		self.mu0 = mu0
		self.mu1 = mu1
		super(y_errors,self).__init__()  

	def forward(self, ypred, ypred_cf):
		rmse_factual = torch.sqrt(torch.mean(torch.square(ypred - self.y)))
		rmse_cfactual = torch.sqrt(torch.mean(torch.square(ypred_cf - self.y_cf)))
		return rmse_factual, rmse_cfactual	

class calc_stats(torch.nn.Module):

	def __init__(self, y, t, y_cf=None, mu0=None, mu1=None):
		self.y = y
		self.t = t
		self.y_cf = y_cf
		self.mu0 = mu0
		self.mu1 = mu1
		if mu0 is not None and mu1 is not None:
			self.true_ite = mu1 - mu0
	
	def forward(self, ypred1, ypred0):
		self.rmse_ite = rmse_ite(y, t, mu0, mu1)
		self.abs_ate = abs_ate(mu0, mu1)
		self.pehe = pehe(mu0, mu1)
		ite = self.rmse_ite.forward(ypred1, ypred0)
		ate = self.abs_ate.forward(ypred1, ypred0)
		pehe = self.pehe.forward(ypred1, ypred0)
		return ite, ate, pehe

def val_loss(x1, x2, t, y , qz ,x_val, t_val, y_val):
#	print ("type2 : " , dist.normal.log_pdf())
#            # sample posterior predictive for p(y|z,t)
#            y_post = ed.copy(y, {z: qz, t: t_ph}, scope='y_post')
#            # crude approximation of the above
#            y_post_mean = ed.copy(y, {z: qz.mean(), t: t_ph}, scope='y_post_mean')
#            # construct a deterministic version (i.e. use the mean of the approximate posterior) of the lower bound
#            # for early stopping according to a validation set
#            y_post_eval = ed.copy(y, {z: qz.mean(), qt: t_ph, qy: y_ph, t: t_ph}, scope='y_post_eval')
#            x1_post_eval = ed.copy(x1, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='x1_post_eval')
#            x2_post_eval = ed.copy(x2, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='x2_post_eval')
#            t_post_eval = ed.copy(t, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='t_post_eval')
#            print x2
		logp_valid = torch.mean(torch.sum(y_val.log_prob(y) + t_val.log_prob(t), axis=1) + torch.sum(x1.log_prob(x_val[:, 0:len(binfeats)]), axis=1) + torch.sum(x2.log_prob(x_val[:, len(binfeats):]), axis=1) + torch.sum(z.log_prob(qz.mean()) - qz.log_prob(qz.mean()), axis=1))

if __name__ == '__main__' :
  X = np.asarray([[0.6946, 0.1328], [0.6563, 0.6873], [0.8184, 0.8047], [0.8177, 0.4517], 
				  [0.1673, 0.2775], [0.6919, 0.0439], [0.4659, 0.3032], [0.3481, 0.1996]], dtype=np.float32)
  X = torch.from_numpy(X)
  y = np.asarray((1,3,2,2,3,1,2,3), dtype=np.float32)
  y = torch.from_numpy(y)
  t = Variable(torch.from_numpy(np.ones(y.size())))
  # calc_stats = calc_stats(y,t)
  # calc_stats.forward(X,y)
	
