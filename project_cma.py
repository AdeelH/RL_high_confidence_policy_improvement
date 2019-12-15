import numpy as np
from hcpi import *
from time import perf_counter
from fourier_basis import FirstOrderFourierBasisFor1dState
# from data import parse_data
from matplotlib import pyplot as plt
import cma

class PolicyLinear():
	def __init__(self, nactions, phi, phi_nfeatures, theta=None):
		self.nactions = nactions
		self.phi = phi
		if theta is not None:
			self._theta = theta
		else:
			self._theta = np.random.normal(0, 1, size=(phi_nfeatures, nactions))

	def __call__(self, s, a=None):
		action_probs = self.getActionProbabilities(s)
		if a is not None:			
			return np.take_along_axis(action_probs, a[..., None], axis=-1).squeeze()
		else: 
			return np.random.choice(self.nactions, p=action_probs)

	@property
	def parameters(self):
		return self._theta.flatten()

	@parameters.setter
	def parameters(self, theta):
		self._theta = theta.reshape(self._theta.shape)

	def getActionProbabilities(self, s):
		out = np.dot(self.phi(s), self._theta)
		exps = np.exp(out - out.max(axis=-1, keepdims=True))
		return exps / exps.sum(axis=-1, keepdims=True)



# m, nA, k, theta_b, episodes, pi_vals_for_first_ep = parse_data('data.csv')
# L = max(len(ep['states']) for ep in episodes)
# S = np.zeros((len(episodes), L))
# A = np.zeros((len(episodes), L)).astype(int)
# R = np.zeros((len(episodes), L))
# ep_lens = np.zeros(len(episodes))
# for i, ep in enumerate(episodes):
# 	s, a, r = ep['states'], ep['actions'], ep['rewards']
# 	assert len(s) == len(a) == len(r)
# 	S[i, : len(s)] = s
# 	A[i, : len(a)] = a
# 	R[i, : len(r)] = r
# 	ep_lens[i] = ep['len']
# print(S.shape)


# pi_b = PolicyLinear(2, FirstOrderFourierBasisFor1dState(), 2)
# pi_b.parameters = theta_b
# print(theta_b)
# print(pi_b.parameters)
# for s, a, pi in zip(S[0], A[0], pi_vals_for_first_ep):
# 	print(s, a, pi_b(s, a), pi)

# np.savez('data', S=S, A=A, R=R, ep_lens=ep_lens, m=m, k=k, theta_b=theta_b)
data = np.load('data.npz')
S, A, R, ep_lens, m, k, theta_b = data['S'], data['A'], data['R'], data['ep_lens'], data['m'], data['k'], data['theta_b']

def train_val_split(D, ratio=.6, shuffle=True, dfilter=None):
	assert isinstance(D, tuple) or isinstance(D, list)
	N = len(D[0])
	if shuffle:
		idx = np.arange(N)
		np.random.shuffle(idx)
		D = tuple([d[idx] for d in D])
		if dfilter is not None:
			dfilter = dfilter[idx]

	split = int(ratio * N)
	D_train = tuple(d[: split] for d in D)
	D_val   = tuple(d[split :] for d in D)
	assert len(D_train[0]) + len(D_val[0]) == N

	if dfilter is not None:
		D_f = tuple(d[dfilter] for d in D)
		D_train_f = tuple(d[dfilter[: split]] for d in D_train)
		D_val_f = tuple(d[dfilter[split :]] for d in D_val)

		assert len(D_val_f[0]) <= len(D_val[0])
		print(len(D[0]), len(D_f[0]))
		return D, D_train, D_val, D_f, D_train_f, D_val_f

	print(len(D[0]))
	return D, D_train, D_val

returns = R.sum(axis=-1)
f = None
# f = (returns > -20) & (returns < 10)
# f = (returns < 0) #& (returns < 20)
D, D_c, D_s = train_val_split((S, A, R, ep_lens), ratio=.6, shuffle=True, dfilter=f)
D_nf, D_c_nf, D_s_nf = D, D_c, D_s

pi_b = PolicyLinear(2, FirstOrderFourierBasisFor1dState(), 2)
pi_b.parameters = theta_b

pi_e = PolicyLinear(2, FirstOrderFourierBasisFor1dState(), 2)

def theta_to_pi(theta):
	pi_e.parameters = theta
	return pi_e

delta = 0.001
c = D_s[2].sum(axis=-1).mean()
print(c)
eval_fn = hcpe_cma(D_c, D_s, pi_b, c, theta_to_pi, delta, batch_size=512)

theta_e = pi_e.parameters
theta_cma = cma.fmin(eval_fn, theta_e, 1.)

print(theta_cma)
pi_e.parameters = theta_cma[0]

print(pi_e.parameters)
print('D', pdis_batch(D, pi_e, pi_b)[0])
print('D_c', pdis_batch(D_c, pi_e, pi_b)[0])
print('D_s', pdis_batch(D_s, pi_e, pi_b)[0])
print('D_nf', pdis_batch(D_nf, pi_e, pi_b)[0])
print('D_c_nf', pdis_batch(D_c_nf, pi_e, pi_b)[0])
print('D_s_nf', pdis_batch(D_s_nf, pi_e, pi_b)[0])
# cma.plot()
# plt.show()
# cma.s.figshow()