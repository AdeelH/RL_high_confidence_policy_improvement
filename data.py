import numpy as np


def parse_data(filename):
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
		pi_vals_for_first_ep = read_float_line()

		return m, nA, k, theta_b, episodes, pi_vals_for_first_ep

# m, nA, k, theta_b, episodes, pi_vals_for_first_ep = parse_data('data.csv')
# print(m, nA, k)
# print(len(episodes))
# print(episodes[0])
# print(theta_b)
# print(pi_vals_for_first_ep)

# print(np.mean([ep['rewards'].sum() for ep in episodes]))
