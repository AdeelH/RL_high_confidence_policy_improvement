import numpy as np
from agents.cem import CEM
from hcpi import *
from time import perf_counter

def run_episode(mdp, policy, max_steps=1):
	# np.random.seed(0)
	mdp.reset()
	S = np.zeros(max_steps).astype(int)
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
	S = np.zeros((n, max_steps)).astype(int)
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

class MdpDiscrete():
	def __init__(self, start, terminals, transition_map, R, gamma=1.):
		self.start = start
		self.terminals = terminals
		self.R = R
		self.transition_map = transition_map
		self.gamma = gamma
		self.reset()

	def reset(self):
		self.s = self.start
		self.r = 0
		self.isEnd = False

	def step(self, a):
		self.r = self.R(self.s, a)
		self.s = self.transition_map[(self.s, a)]
		if self.s in self.terminals:
			self.isEnd = True

class PolicyTabular():
	def __init__(self, nstates, nactions, p=None):
		self.nactions = nactions
		self.nstates = nstates
		if p is not None:
			self.p = p
		else:
			self.p = np.ones((nstates, nactions)) / nactions
		self.precomputeActionProbabilities()

	def __call__(self, s, a=None):
		if a is not None:
			return self.p[s, a]
		return np.random.choice(self.nactions, p=self.p[s])

	@property
	def parameters(self):
		return self.p.flatten()
	
	@parameters.setter
	def parameters(self, p):
		self.p = p.reshape(self.p.shape)
		self.precomputeActionProbabilities()

	def precomputeActionProbabilities(self):
		exps = np.exp(self.p - self.p.max(axis=-1, keepdims=True))
		self.p = exps / exps.sum(axis=-1, keepdims=True)


mdp = MdpDiscrete(0, [1], {(0, 0): 1, (0, 1): 1}, lambda s, a: a + np.random.normal(0, 1))
pi_b = PolicyTabular(2, 2, p = np.array([[0, 0], [0, 0]]))
pi_e = PolicyTabular(2, 2)

# print(run_episodes(100, mdp, pi_e, max_steps=1)[2].mean(axis=0))

D_c = run_episodes(1000, mdp, pi_b, max_steps=1, seed=0)
D_s = run_episodes(1000, mdp, pi_b, max_steps=1, seed=1)

def theta_to_pi(theta):
	pi_e.parameters = theta
	return pi_e

delta = 0.01
c = D_s[-1].sum(axis=-1).mean()
print(c)
eval_fn = hcpe(D_c, D_s, pi_b, c, theta_to_pi, delta)

bbo = CEM(
	theta = pi_e.parameters,
	sigma = 1, 
	popSize = 4, 
	numElite = 1, 
	epsilon = 1,
	evaluationFunction = eval_fn
)
# t0 = perf_counter()
for _ in range(10):
	bbo.train()
print()
print(pi_e.p)
print()
# returns = train_agent_n_trials(env, policy, bbo, iters=iters, trials=trials)
# print(f'elapsed: {perf_counter() - t0}')

