import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

fmt_styles = ['', '--','-.', 'o-', 'v-',
			 '^-', '<-','>-', '1-','2-',
			 '3-','4-','s-','p-','*-',
			 'h-', 'H-', '+-', 'X-','D-']

FONTSIZE = 16
def plot(df, save_dir=None,show=False, standalone=False):
	# Data
	index = np.arange(1, len(df)+1)

	for column, fmt in zip(df.columns, fmt_styles):
		plt.plot(index, column, fmt, data=df, label=column, )

	plt.legend(loc='lower right')
	plt.xlabel("Communication Rounds",  fontsize=FONTSIZE, fontweight='bold')
	if standalone:
		plt.xlabel("Epochs",  fontsize=FONTSIZE, fontweight='bold')

	plt.ylabel("Test Accuracy",  fontsize=FONTSIZE, fontweight='bold')
	locs, labels = plt.yticks()
	n_ticks = len(locs)
	y_ticks = np.linspace(locs[0], locs[-1], n_ticks)
	plt.yticks(np.round(np.array(locs), 2) ,fontsize=14)


	plt.title('Adult', fontsize=FONTSIZE)

	if save_dir:
		plt.savefig(save_dir)
		plt.clf()
	if show:
		plt.show()
	return


def plot_one(df, save_dir=None, show=False):
	# Data
	index = np.arange(1, len(df)+1)
	fmt_styles = ['ro-', 'c.-', 'm<-']
	for column, fmt in zip(df.columns, ['ro-', 'c.-', 'm<-']):
		plt.plot(index, column, fmt, data=df, label=column, )

	plt.legend(loc='lower right')
	plt.xlabel("Epochs (Communication Rounds)",  fontsize=FONTSIZE, fontweight='bold')
	plt.ylabel("Test Accuracy",  fontsize=FONTSIZE, fontweight='bold')
	
	# reformat the yticks
	locs, labels = plt.yticks()
	n_ticks = len(locs)
	y_ticks = np.linspace(locs[0], locs[-1], n_ticks)
	plt.yticks(np.round(np.array(locs), 2) ,fontsize=14)


	plt.title('Adult', fontsize=FONTSIZE)
	
	if save_dir:
		plt.savefig(save_dir)
		plt.clf()
	if show:
		plt.show()
	return