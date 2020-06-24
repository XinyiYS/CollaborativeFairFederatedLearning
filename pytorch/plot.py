import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


all_party_fmt_styles = ['', '--','-.', 'o-', 'v-',
			 '^-', '<-','>-', '1-','2-',
			 '3-','4-','s-','p-','*-',
			 'h-', 'H-', '+-', 'X-','D-',
			 '8-','P-','d-','|','_-']

best_worker_fmt_styles = ['ro-', 'c.-', 'gv-','m<-', 'b<-']

acc_top_bottom_limits = {'adult':[0.70, 0.85],
						'mr':  [0.5, 0.8],
						'sst': [0.2, 0.5],
						'mnist': [0.4, 1],
						'cifar10':[0, 0.7]}

c_th_fmt_style = 'k--'
# Add 'b<-' for w/o pretrain
FONTSIZE = 17

Xlabel_NAMES = ['Communication Rounds', 'Epochs', 'Epochs (Communication Rounds)']
def plot(df, save_dir=None, name='adult', plot_type=1, show=False, **kwargs):
	# pass in other use info using kwargs
	
	# Data
	index = np.arange(1, len(df)+1)

	fmt_styles = best_worker_fmt_styles if plot_type == 2 else all_party_fmt_styles

	for column, fmt in zip(df.columns, fmt_styles):

		if column == 'threshold':
			plt.plot(index, column, c_th_fmt_style, data=df, label=column, )
		else:
			plt.plot(index, column, fmt, data=df, label=column, )

	if len(df.columns) > 10:
		plt.legend(loc='lower left',fontsize=8)
	else:
		plt.legend(loc='lower right')

	plt.xlabel(Xlabel_NAMES[plot_type],  fontsize=FONTSIZE, fontweight='bold')
	
	ylabel = "Test Accuracy"
	if 'ylabel' in kwargs:
		ylabel = kwargs['ylabel']
	plt.ylabel(ylabel,  fontsize=FONTSIZE, fontweight='bold')
	

	# reformat the yticks
	if name in acc_top_bottom_limits:
		bottom, top = acc_top_bottom_limits[name]
	else:
		bottom, top = 0, 1.0

	if 'split' in kwargs and kwargs['split'] =='classimbalance':
		bottom, top = 0, 1.0


	if 'bottom' in kwargs and 'top' in kwargs:
		bottom = kwargs['bottom']
		top = kwargs['top']

	plt.ylim(bottom, top)
	locs, labels = plt.yticks()
	n_ticks = len(locs)
	y_ticks = np.linspace(bottom, top, n_ticks)
	plt.yticks(np.round(np.array(locs), 2) ,fontsize=14)		

	title = name.capitalize()
	if 'title' in kwargs:
		title = kwargs['title'].capitalize()

	plt.title(title, fontsize=FONTSIZE)
	plt.tight_layout()

	if save_dir:
		plt.savefig(save_dir)
	if show:
		plt.show()
	plt.clf()
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