import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fmt_styles = ['', '--','-.', 'o-', 'v-',
			 '^-', '<-','>-', '1-','2-',
			 '3-','4-','s-','p-','*-',
			 'h-', 'H-', '+-', 'X-','D-']
def plot(df, save_dir=None,show=False):
	# Data
	index = np.arange(1, len(df)+1)

	for column, fmt in zip(df.columns, fmt_styles):
		plt.plot(index, column, fmt, data=df, label=column, )

	plt.legend()
	plt.xlabel("Communication Rounds",  fontsize='large', fontweight='bold')
	plt.ylabel("Test Accuracy",  fontsize='large', fontweight='bold')
	plt.title('Adult')
	
	if save_dir:
		plt.savefig(save_dir)
		plt.clf()
	if show:
		plt.show()
	return



