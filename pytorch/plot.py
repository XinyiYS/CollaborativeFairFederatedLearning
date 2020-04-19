import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


all_party_fmt_styles = ['', '--','-.', 'o-', 'v-',
			 '^-', '<-','>-', '1-','2-',
			 '3-','4-','s-','p-','*-',
			 'h-', 'H-', '+-', 'X-','D-']

best_worker_fmt_styles = ['ro-', 'c.-', 'm<-', 'b<-']

# Add 'b<-' for w/o pretrain
FONTSIZE = 17

Xlabel_NAMES = ['Communication Rounds', 'Epochs', 'Epochs (Communication Rounds)']
def plot(df, save_dir=None, name='adult', plot_type=1, show=False):
	# Data
	index = np.arange(1, len(df)+1)


	fmt_styles = best_worker_fmt_styles if plot_type == 2 else all_party_fmt_styles

	for column, fmt in zip(df.columns, fmt_styles):
		plt.plot(index, column, fmt, data=df, label=column, )

	if len(df.columns) > 10:
		plt.legend(loc='lower right',fontsize=8.2)
	else:
		plt.legend(loc='lower right')

	plt.xlabel(Xlabel_NAMES[plot_type],  fontsize=FONTSIZE, fontweight='bold')
	plt.ylabel("Test Accuracy",  fontsize=FONTSIZE, fontweight='bold')
	

	# reformat the yticks
	if name =='adult':
		bottom, top = 0.70, 0.85
	else: # for MNIST
		# for the classimbalance setting, lowest party has acc ~= 10%
		bottom, top = 0, 1.0


	plt.ylim(bottom, top)
	locs, labels = plt.yticks()
	n_ticks = len(locs)
	y_ticks = np.linspace(bottom, top, n_ticks)
	plt.yticks(np.round(np.array(locs), 2) ,fontsize=14)		

	plt.title(name.capitalize(), fontsize=FONTSIZE)
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