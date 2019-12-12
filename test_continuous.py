import numpy as np
from agents.cem import CEM
from hcpi import *
from time import perf_counter
from fourier_basis import FirstOrderFourierBasisFor1dState

def run_episode(mdp, policy, max_steps=1):
	# np.random.seed(0)
	mdp.reset()
	S = np.zeros(max_steps)
	A = np.zeros(max_steps).astype(int)
	R = np.zeros(max_steps)
	for t in range(max_steps):
		if mdp.isEnd:
			break
		S[t] = mdp.s
		A[t] = policy(S[t])
		mdp.step(A[t])
		R[t] = (mdp.gamma ** t) * mdp.r
		
	return S, A, R

def run_episodes(n, mdp, policy, max_steps=1, seed=0):
	np.random.seed(seed)
	S = np.zeros((n, max_steps))
	A = np.zeros((n, max_steps)).astype(int)
	R = np.zeros((n, max_steps))
	for i in range(n):
		mdp.reset()
		for t in range(max_steps):
			S[i, t] = mdp.s
			if mdp.isEnd:
				break
			A[i, t] = policy(S[i, t])
			mdp.step(A[i, t])
			R[i, t] = (mdp.gamma ** t) * mdp.r
			
	return S, A, R

class MdpContinuous():
	def __init__(self, hasEnded, transition_fn, R, gamma=1.):
		self.R = R
		self.hasEnded = hasEnded
		self.transition_fn = transition_fn
		self.gamma = gamma
		self.reset()

	def reset(self):
		# self.s = self.start
		self.s = np.random.uniform(0, 0.5)
		self.r = 0
		self.isEnd = False

	def step(self, a):
		self.r = self.R(self.s, a)
		self.s = self.transition_fn(self.s, a)
		self.isEnd = self.hasEnded(self.s)

class PolicyLinear():
	def __init__(self, nactions, phi, theta=None):
		self.nactions = nactions
		self.phi = phi
		if theta is not None:
			self._theta = theta
		else:
			self._theta = np.random.normal(0, 1, size=(2, nactions))

	# def phi(self, s):
	# 	out = np.empty((*np.array(s).shape, 2))
	# 	out[..., 0] = s
	# 	out[..., 1] = 1
	# 	return out

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


_transition_fn = lambda s, a: s + np.random.uniform(0, .5) if a == 0 else s + 1
_R = lambda s, a: -10 if 1. < s + a < 1.5 else s
mdp = MdpContinuous(lambda s: s > 1.5, _transition_fn, _R)
pi_b = PolicyLinear(2, FirstOrderFourierBasisFor1dState())
pi_e = PolicyLinear(2, FirstOrderFourierBasisFor1dState())

# print(run_episode(mdp, pi_b, max_steps=10))
# print(run_episodes(100, mdp, pi_b, max_steps=10)[2].mean(axis=0))

D_c = run_episodes(10000, mdp, pi_b, max_steps=10, seed=0)
D_s = run_episodes(10000, mdp, pi_b, max_steps=10, seed=1)

def theta_to_pi(theta):
	pi_e.parameters = theta
	return pi_e

delta = 0.0001
c = D_s[-1].sum(axis=-1).mean()
print(c)
eval_fn = hcpe(D_c, D_s, pi_b, c, theta_to_pi, delta)

bbo = CEM(
	theta = pi_e.parameters,
	sigma = 1, 
	popSize = 8, 
	numElite = 2, 
	epsilon = 2,
	evaluationFunction = eval_fn
)
t0 = perf_counter()
for _ in range(100):
	bbo.train()
print(pi_e.parameters)
print()
print(f'elapsed: {perf_counter() - t0}')

# def print_history(ep):
# 	for s, a, r in zip(*ep):
# 		print('%.3f %d, %.3f' % (s, a, r))
# 	print('-----------------------------')

# print_history(run_episode(mdp, pi_b, max_steps=10))
# print_history(run_episode(mdp, pi_e, max_steps=10))

print(run_episodes(100, mdp, pi_b, max_steps=10)[2].mean(axis=0))
print(eval_fn(pi_e.parameters))
print(run_episodes(100, mdp, pi_e, max_steps=10)[2].mean(axis=0))
for i in range(20):
	s = i / 10
	print(s, sum(pi_e(s) for _ in range(10))/10)
