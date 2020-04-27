from read_convergence import parse, get_acc_lines, get_fairness, get_cffl_best
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dirname = 'logs/alpha_clip01'
acc_df = []
fairness = []

# for n in [5, 10]:
# for theta_ in [0.1, 1]:
for n, theta_ in [[5, 0.1], [5, 1], [10, 0.1], [10, 1]]:
	print(n, theta_)
	columns = ['accuracy','fairness']
	data_rows = []
	alpha_index = []
	for folder in os.listdir(dirname):
		if os.path.isfile(os.path.join(dirname, folder)): continue
		setup = parse(folder)
		alpha = setup['alpha']
		theta = setup['theta']
		n_workers = setup['P']
		if theta == theta_ and n_workers == n:

			alpha_index.append(alpha)
			best_worker_accs = get_cffl_best(dirname, folder)

			acc_lines = get_acc_lines(dirname, folder)
			dis_f, cffl_f = get_fairness(acc_lines)

			data_row = [best_worker_accs[2], cffl_f ]

			data_rows.append(data_row)

	df = pd.DataFrame(data=data_rows, columns=columns, index = alpha_index)
	df = df.sort_index()		
	_ = df.plot( kind= 'bar' , secondary_y= 'fairness' , rot= 0 )
	ax1, ax2 = plt.gcf().get_axes()

	ax1.set_ylim(df.accuracy.min()- 0.05, df.accuracy.max() + 0.05)
	ax2.set_ylim(df.fairness.min()- 0.05, df.fairness.max() + 0.05)

	ax1.set_ylabel('Accuracy')
	ax2.set_ylabel('Fairness')
	ax1.set_xlabel("alpha values")
	title = 'P{} without pretraining theta {}'.format(n, str(theta_))
	if n_workers == 10:
		print(title)
	plt.title(title)
	# plt.show()
	plt.savefig(os.path.join(dirname, title+'.png'))
	plt.clf()
	# plt.show()
	'''
	fig = plt.figure() # Create matplotlib figure

	ax = fig.add_subplot(111) # Create matplotlib axes
	ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

	width = 0.2
	df.accuracy.plot(kind='bar', ax=ax, width=width, position=1)
	df.fairness.plot(kind='bar', ax=ax2, width=width, position=0)

	ax.set_ylabel('Accuracy')
	ax2.set_ylabel('Fairness')
	'''
	# ax = df.plot.bar(rot=0)