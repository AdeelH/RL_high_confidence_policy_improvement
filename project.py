import numpy as np
from agents.cem import CEM
from agents.fchc import FCHC
from agents.ga import GA
from hcpi import *
from time import perf_counter
from fourier_basis import FirstOrderFourierBasisFor1dState
from data import parse_data


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
# for i, ep in enumerate(episodes):
# 	s, a, r = ep['states'], ep['actions'], ep['rewards']
# 	assert len(s) == len(a) == len(r)
# 	S[i, : len(s)] = s
# 	A[i, : len(a)] = a
# 	R[i, : len(r)] = r
# print(S.shape)


# pi_b = PolicyLinear(2, FirstOrderFourierBasisFor1dState(), 2)
# pi_b.parameters = theta_b
# print(theta_b)
# print(pi_b.parameters)
# for s, a, pi in zip(S[0], A[0], pi_vals_for_first_ep):
# 	print(s, a, pi_b(s, a), pi)

# np.savez('data', S=S, A=A, R=R, m=m, k=k, theta_b=theta_b)
data = np.load('data.npz')
S, A, R, m, k, theta_b = data['S'], data['A'], data['R'], data['m'], data['k'], data['theta_b']
print(S.shape)

np.random.seed(0)
N = len(S)
idx = np.arange(N)
np.random.shuffle(idx)
S, A, R = S[idx], A[idx], R[idx]

split = int(.6 * N)
D_c = S[: split], A[: split], R[: split]
D_s = S[split :], A[split :], R[split :]

assert len(D_c[0]) + len(D_s[0]) == N

pi_b = PolicyLinear(2, FirstOrderFourierBasisFor1dState(), 2)
pi_b.parameters = theta_b

pi_e = PolicyLinear(2, FirstOrderFourierBasisFor1dState(), 2)

def theta_to_pi(theta):
	pi_e.parameters = theta
	return pi_e

delta = 0.001
c = D_s[-1].sum(axis=-1).mean()
print(c)
# print(D_s[-1].sum(axis=-1))
eval_fn = hcpe(D_c, D_s, pi_b, c, theta_to_pi, delta)

bbo = CEM(
	theta = pi_e.parameters,
	sigma = 4, 
	popSize = 16, 
	numElite = 4, 
	epsilon = 4,
	evaluationFunction = eval_fn
)
# bbo = FCHC(
#     theta = pi_e.parameters,
#     sigma = 1.25, 
#     evaluationFunction = eval_fn
# )
# populationSize = 32
# bbo = GA(
#     populationSize = populationSize, 
#     numElite = 4, 
#     K_p = 16, 
#     alpha = 1., 
#     initPopulationFunction = lambda populationSize: np.random.randn(populationSize, pi_e.parameters.shape[0]), 
#     evaluationFunction = eval_fn
# )
t0 = perf_counter()
train_iters = 100
train_evals, val_evals = [0] * train_iters, [0] * train_iters
for i in range(train_iters):
	bbo.train()
	pi_e.parameters = bbo.parameters
	# train_evals[i] = bbo.eval
	# train_evals[i] = sum(bbo._evals) / populationSize
	train_evals[i], _ = pdis_batch(D_c, pi_e, pi_b)
	val_evals[i], _ = pdis_batch(D_s, pi_e, pi_b)
	print(train_evals[i], val_evals[i])
print(pi_e.parameters)
print(eval_fn(pi_e.parameters))
print()
print(f'elapsed: {perf_counter() - t0}')

from matplotlib import pyplot as plt
plt.plot(np.arange(len(train_evals)), train_evals)
plt.plot(np.arange(len(val_evals)), val_evals)
plt.show()

# # def print_history(ep):
# # 	for s, a, r in zip(*ep):
# # 		print('%.3f %d, %.3f' % (s, a, r))
# # 	print('-----------------------------')

# # print_history(run_episode(mdp, pi_b, max_steps=10))
# # print_history(run_episode(mdp, pi_e, max_steps=10))

# print(run_episodes(100, mdp, pi_b, max_steps=10)[2].mean(axis=0))
# print(eval_fn(pi_e.parameters))
# print(run_episodes(100, mdp, pi_e, max_steps=10)[2].mean(axis=0))
# for i in range(20):
# 	s = i / 10
# 	print(s, sum(pi_e(s) for _ in range(10))/10)