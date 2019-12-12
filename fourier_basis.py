import numpy as np

class FirstOrderFourierBasisFor1dState():

	def __call__(self, x):
		if np.isscalar(x):
			return np.array([1, np.cos(np.pi * x)])
		return np.dstack([0 * x, np.cos(np.pi * x)])
