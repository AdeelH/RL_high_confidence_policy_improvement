import numpy as np
from agents.fchc import FCHC
from hcpi import *
from policy import *
from data import *
from matplotlib import pyplot as plt


# m, nA, k, theta_b, episodes, pi_vals_for_first_ep = load_from_csv('data.csv')
# S, A, R, ep_lens = prepare_data(episodes, theta_b, save_to_file=False)

data = np.load('data.npz')
S, A, R, ep_lens, theta_b = data['S'], data['A'], data['R'], data['ep_lens'], data['theta_b']

pi_b = PolicyLinear(2, FirstOrderFourierBasisFor1dState(), 2)
pi_b.parameters = theta_b

# verify_pi_b(pi_b, pi_vals_for_first_ep, S, A)


def cand_safe_split(D, ratio=.6, shuffle=True, rfilter=None, cfilter=None):
	assert isinstance(D, tuple) or isinstance(D, list)
	N = len(D[0])
	if shuffle:
		idx = np.arange(N)
		np.random.shuffle(idx)
		D = tuple([d[idx] for d in D])
		if rfilter is not None:
			rfilter = rfilter[idx]

	split = int(ratio * N)
	D_cand = tuple(d[: split] for d in D)
	D_safe   = tuple(d[split :] for d in D)
	assert len(D_cand[0]) + len(D_safe[0]) == N

	if rfilter is not None or cfilter is not None:
		if rfilter is None:
			rfilter = slice(None, None)
			rfilter_cand = slice(None, None)
			rfilter_safe = slice(None, None)
		else:
			rfilter_cand = rfilter[: split]
			rfilter_safe   = rfilter[split: ]
		if cfilter is None:
			cfilter = slice(None, None)

		D_f = tuple(d[rfilter, cfilter] for d in D)
		D_cand_f = tuple(d[rfilter_cand, cfilter] for d in D_cand)
		D_safe_f = tuple(d[rfilter_safe, cfilter] for d in D_safe)

		assert len(D_safe_f[0]) <= len(D_safe[0])
		print(len(D[0]), len(D_f[0]))
		return D, D_cand, D_safe, D_f, D_cand_f, D_safe_f

	print(len(D[0]))
	return D, D_cand, D_safe

returns = R.sum(axis=-1)
f = None
# f = (returns > -10) & (returns < 10)
# f = (returns < 0)
# f = (ep_lens == 5) #& (returns > -20) & (returns < 10)
D_nf, D_c_nf, D_s_nf, D, D_c, D_s = cand_safe_split(
										D=(S, A, R, ep_lens.reshape(-1, 1)), 
										ratio=.6, shuffle=True, 
										rfilter=f, cfilter=slice(None, None)
									)

# bbo = CEM(
# 	theta = pi_e.parameters,
# 	sigma = 4, 
# 	popSize = 8, 
# 	numElite = 2, 
# 	epsilon = 4,
# 	evaluationFunction = eval_fn
# )


# populationSize = 16
# bbo = GA(
#     populationSize = populationSize, 
#     numElite = 2, 
#     K_p = 4, 
#     alpha = 2., 
#     initPopulationFunction = lambda populationSize: np.random.randn(populationSize, pi_e.parameters.shape[0]), 
#     evaluationFunction = eval_fn
# )
# pi_e = PolicyLinear(2, FirstOrderFourierBasisFor1dState(), 2)
# pi_e.parameters = np.zeros_like(pi_b.parameters)
# print(pi_e.parameters)
# print('D', pdis_batch(D, pi_e, pi_b, batch_size=1)[0])
# print('D_c', pdis_batch(D_c, pi_e, pi_b)[0])
# print('D_s', pdis_batch(D_s, pi_e, pi_b)[0])
# print('D_nf', pdis_batch(D_nf, pi_e, pi_b)[0])
# print('D_c_nf', pdis_batch(D_c_nf, pi_e, pi_b)[0])
# print('D_s_nf', pdis_batch(D_s_nf, pi_e, pi_b)[0])



def run():
	def train():
		bsz = 512
		eval_fn = hcpe(D_c, D_s, pi_b, c, theta_to_pi, delta, batch_size=bsz)

		bbo = FCHC(
		    theta = pi_e.parameters,
		    sigma = 1, 
		    evaluationFunction = eval_fn
		)
		train_iters = 20
		# train_evals, val_evals = [0] * train_iters, [0] * train_iters
		k = 5
		for i in range(train_iters):
			if (i + 1 ) % k == 0:
				bsz *= 2
				# k -= 1
				bbo.evaluationFunction = hcpe(D_c, D_s, pi_b, c, theta_to_pi, delta, batch_size=bsz)
			bbo.train()
			pi_e.parameters = bbo.parameters
			# train_evals[i] = bbo.eval
			# train_evals[i] = sum(bbo._evals) / populationSize
			# train_evals[i], _ = pdis_batch(D_c, pi_e, pi_b, batch_size=1024)
			# val_evals[i], _ = pdis_batch(D_s_nf, pi_e, pi_b)
			# print(train_evals[i], val_evals[i])

		print(pi_e.parameters)
		print('D', pdis_batch(D, pi_e, pi_b)[0])
		print('D_c', pdis_batch(D_c, pi_e, pi_b)[0])
		print('D_s', pdis_batch(D_s, pi_e, pi_b)[0])
		# print('D_nf', pdis_batch(D_nf, pi_e, pi_b)[0])
		# print('D_c_nf', pdis_batch(D_c_nf, pi_e, pi_b)[0])
		# print('D_s_nf', pdis_batch(D_s_nf, pi_e, pi_b)[0])
		print()
		return pdis_batch(D_s, pi_e, pi_b)[0] > 10


	delta = 0.0001
	c = D_s[2].sum(axis=-1).mean()

	for p in range(1, 30):
		np.random.seed(p)
		print(f'policy {p}')
		pi_e = PolicyLinear(2, FirstOrderFourierBasisFor1dState(), 2)
		# pi_e.parameters = pi_b.parameters

		def theta_to_pi(theta):
			pi_e.parameters = theta
			return pi_e

		for _ in range(10):
			pi_e.reset()
			if train(): 
				break

		# plt.plot(np.arange(len(train_evals)), train_evals)
		# plt.plot(np.arange(len(val_evals)), val_evals)
		# plt.show()

		with open(f'out/{p}.csv', 'w') as file:
			file.writelines([','.join('%.10f' % (x) for x in pi_e._theta.T.flatten()) + '\n'])

run()
