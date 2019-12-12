import numpy as np

def top_k_inds(arr, k):
	'''Returns indices of the top k values in arr. Not necessarily sorted. Runs in linear time.'''
	return np.argpartition(arr, -k)[-k:]
