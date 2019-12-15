import numpy as np


def load_from_csv(filename):
	with open(filename, 'r') as f:
		read_int = lambda: int(f.readline())
		read_float_line = lambda: [float(x) for x in f.readline().split(',')]

		m = read_int()
		nA = read_int()
		k = read_int()

		theta_b = read_float_line()
		assert len(theta_b) == nA * (k + 1)**m
		theta_b = np.array(theta_b).reshape(nA, -1).T

		n = read_int()
		episodes = [None] * n
		for i in range(n):
			line = read_float_line()
			ep = np.array(list(zip(*[line[j::(m + 2)] for j in range(m + 2)])))
			assert len(line) == ep.size, f'{len(line), ep.size}'
			episodes[i] = {
				'states': ep[:, : m].reshape(-1),
				'actions': ep[:, -2].astype(int).reshape(-1),
				'rewards': ep[:, -1].reshape(-1)
			}
			episodes[i]['len'] = len(episodes[i]['states'])
		pi_vals_for_first_ep = read_float_line()

		return m, nA, k, theta_b, episodes, pi_vals_for_first_ep

def prepare_data(episodes, theta_b, save_to_file=True):
	L = max(len(ep['states']) for ep in episodes)
	S = np.zeros((len(episodes), L))
	A = np.zeros((len(episodes), L)).astype(int)
	R = np.zeros((len(episodes), L))
	ep_lens = np.zeros(len(episodes))
	for i, ep in enumerate(episodes):
		s, a, r = ep['states'], ep['actions'], ep['rewards']
		assert len(s) == len(a) == len(r)
		S[i, : len(s)] = s
		A[i, : len(a)] = a
		R[i, : len(r)] = r
		ep_lens[i] = ep['len']

	if save_to_file:
		np.savez('data', S=S, A=A, R=R, ep_lens=ep_lens, theta_b=theta_b)
	return S, A, R, ep_lens

def verify_pi_b(pi_b, pi_vals_for_first_ep, S, A):
	for s, a, pi in zip(S[0], A[0], pi_vals_for_first_ep):
		assert np.abs(pi_b(s, a) - pi) < 0.0001


# m, nA, k, theta_b, episodes, pi_vals_for_first_ep = parse_data('data.csv')
# print(m, nA, k)
# print(len(episodes))
# print(episodes[0])
# print(theta_b)
# print(pi_vals_for_first_ep)

# print(np.mean([ep['rewards'].sum() for ep in episodes]))
