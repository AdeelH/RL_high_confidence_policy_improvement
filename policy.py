import numpy as np

class FirstOrderFourierBasisFor1dState():

	def __call__(self, x):
		if np.isscalar(x):
			return np.array([1, np.cos(np.pi * x)])
		return np.dstack([np.ones_like(x), np.cos(np.pi * x)])


class PolicyLinear():
	def __init__(self, nactions, phi, phi_nfeatures, theta=None):
		self.nactions = nactions
		self.phi = phi
		self.phi_nfeatures = phi_nfeatures
		if theta is not None:
			self._theta = theta
		else:
			self.reset()

	def reset(self):
		self._theta = np.random.normal(0, 2, size=(self.phi_nfeatures, self.nactions))

	def __call__(self, s, a=None):
		action_probs = self.getActionProbabilities(s)
		if a is not None:
			# print(action_probs.sum(axis=-1).sum(), action_probs.size)
			return np.take_along_axis(action_probs, a[..., None], axis=-1).squeeze()
		else: 
			return np.random.choice(self.nactions, p=action_probs)
			# return np.argmax(action_probs)

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