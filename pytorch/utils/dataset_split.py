import numpy as np
np.random.seed(1111)

def random_split(n_samples, m_bins, equal=True):
	all_indices = np.arange(n_samples)
	np.random.shuffle(all_indices)
	if equal:
		indices_list = np.split(all_indices, m_bins)
	else:
		split_points = np.random.choice(n_samples - 2, m_bins - 1, replace=False) + 1
		split_points.sort()
		indices_list = np.split(all_indices, split_points)
	
	return indices_list
