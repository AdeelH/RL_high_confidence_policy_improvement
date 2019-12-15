import numpy as np
from scipy.stats import t


def pdis_batch(D, pi_e, pi_b, batch_size=None):
	# R = discounted reward
	S, A, R, _ = D
	if batch_size is not None:
		idx = np.random.choice(len(S), size=batch_size)
		S, A, R = S[idx], A[idx], R[idx]
	pi_vals_e = pi_e(S, A)
	pi_vals_b = pi_b(S, A)
	pi_ratios = pi_vals_e / pi_vals_b
	# print('pi_vals_e', pi_vals_e)
	# print('pi_vals_b', pi_vals_b)
	# print('pi_ratios', pi_ratios)
	pi_ratios_cumprod = np.cumprod(pi_ratios, axis=-1)
	# print('pi_ratios_cumprod', pi_ratios_cumprod)
	pi_ratios_cumprod[np.isnan(pi_ratios_cumprod)] = 0
	# print('pi_ratios_cumprod', pi_ratios_cumprod)
	pdis_estimates = (pi_ratios_cumprod * R).sum(axis=-1)
	# print('pdis_estimates', pdis_estimates)

	return pdis_estimates.mean(), pdis_estimates

def hcpe(D_c, D_s, pi_b, c, theta_to_pi, delta, batch_size=None):
	n = D_s[0].shape[0]
	k = (2 / np.sqrt(n)) * t.ppf(1 - delta, n - 1)

	def _hcpe_closure(theta):
		pi_e = theta_to_pi(theta)
		J_estimate, pdis_estimates = pdis_batch(D_c, pi_e, pi_b)
		sigma_c = pdis_estimates.std()
		if J_estimate < c + sigma_c * k:
			return -np.inf
		return J_estimate
	
	return _hcpe_closure


def hcpe_cma(D_c, D_s, pi_b, c, theta_to_pi, delta, batch_size=None):
	n = D_s[0].shape[0]
	k = (2 / np.sqrt(n)) * t.ppf(1 - delta, n - 1)

	def _hcpe_closure(theta):
		pi_e = theta_to_pi(theta)
		J_estimate, pdis_estimates = pdis_batch(D_c, pi_e, pi_b, batch_size)
		sigma_c = pdis_estimates.std()
		return -J_estimate - np.log(J_estimate - (c + sigma_c * k))
	
	return _hcpe_closure
