import os
import json

# a = 'experiment_p10_e5-10-5_b16_size10000_lr0'
# print(a.replace('experiment', 'powerlaw'))
for folder in os.listdir():
	
	os.rename(folder, folder.replace('experiment', 'powerlaw'))