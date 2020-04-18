import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fmt_styles = ['', '--','-.', 'o-', 'v-',
			 '^-', '<-','>-', '1-','2-',
			 '3-','4-','s-','p-','*-',
			 'h-', 'H-', '+-', 'X-','D-']

FONTSIZE = 17

Xlabel_NAMES = ['Epochs', 'Communication Rounds', 'Epochs (Communication Rounds)']
def plot(df, save_dir=None, name='Adult', plot_type=1, show=False):
	# Data
	index = np.arange(1, len(df)+1)

	for column, fmt in zip(df.columns, fmt_styles):
		plt.plot(index, column, fmt, data=df, label=column, )

	if len(df.columns) > 10:
		plt.legend(loc='lower right',fontsize=8.2)
	else:
		plt.legend(loc='lower right')

	plt.xlabel(Xlabel_NAMES[plot_type],  fontsize=FONTSIZE, fontweight='bold')
	plt.ylabel("Test Accuracy",  fontsize=FONTSIZE, fontweight='bold')
	

	# reformat the yticks
	if name =='Adult':
		locs, labels = plt.yticks()
		n_ticks = len(locs)
		y_ticks = np.linspace(locs[0], locs[-1], n_ticks)
		plt.yticks(np.round(np.array(locs), 2) ,fontsize=14)
		# plt.ylim(0.6, 1.0)
		# plt.yticks(np.arange(0.6, 0.90, step=0.05), size=14)
	else: # for MNIST
		locs, labels = plt.yticks()
		n_ticks = len(locs)
		y_ticks = np.linspace(locs[0], locs[-1], n_ticks)
		plt.yticks(np.round(np.array(locs), 2) ,fontsize=14)
		# plt.ylim(0.6, 1.02)
		# plt.yticks(np.arange(0.6, 1.02, step=0.05), size=14)

	plt.title(name, fontsize=FONTSIZE)
	plt.tight_layout()

	if save_dir:
		plt.savefig(save_dir)
		plt.clf()
	if show:
		plt.show()
	return



'''
def plot_one(df, save_dir=None, show=False, name='Adult'):
	# Data
	index = np.arange(1, len(df)+1)
	fmt_styles = ['ro-', 'c.-', 'm<-']
	for column, fmt in zip(df.columns, ['ro-', 'c.-', 'm<-']):
		plt.plot(index, column, fmt, data=df, label=column, )

	if len(df.columns) > 10:
		plt.legend(loc='lower right',fontsize=8.5)
	else:
		plt.legend(loc='lower right')	
	plt.xlabel("Epochs (Communication Rounds)",  fontsize=FONTSIZE, fontweight='bold')
	plt.ylabel("Test Accuracy",  fontsize=FONTSIZE, fontweight='bold')

	# reformat the yticks
	locs, labels = plt.yticks()
	n_ticks = len(locs)
	y_ticks = np.linspace(locs[0], locs[-1], n_ticks)
	plt.yticks(np.round(np.array(locs), 2) ,fontsize=14)
	
	# fix a constant size yticks
	plt.yticks(np.arange(0.6, 1.02, step=0.05),size=14)


	plt.title(name, fontsize=FONTSIZE)
	plt.tight_layout()

	
	if save_dir:
		plt.savefig(save_dir)
		plt.clf()
	if show:
		plt.show()
	return
'''