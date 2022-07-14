from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import json
def findFiles(path): return glob.glob(path)

# print(findFiles('data/names/*.txt'))

import unicodedata
import string
import random

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)

# print(unicodeToAscii('Ślusàrski'))

def readLines(filename):
	lines = open(filename, encoding='utf-8').read().strip().split('\n')
	return [unicodeToAscii(line) for line in lines]

import torch

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
	return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
	tensor = torch.zeros(1, n_letters)
	tensor[0][letterToIndex(letter)] = 1
	return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
	tensor = torch.zeros(len(line), 1, n_letters)
	for li, letter in enumerate(line):
		tensor[li][0][letterToIndex(letter)] = 1
	return tensor

def get_train_test(data_dir='datasets/names/data.pt', labels_dir='datasets/names/labels.pt', reference_dict_dir='datasets/names/reference_dict', train_test_ratio=0.8):

	try:
		data = torch.load(data_dir)
		labels = torch.load(labels_dir)
		with open(reference_dict_dir, 'r') as reference_dict_str:
			reference_dict = json.loads(reference_dict_str.read())
	except:
		category_lines = {}
		all_categories = []

		for filename in findFiles('datasets/names/names_txt/*.txt'):
			category = os.path.splitext(os.path.basename(filename))[0]
			lines = readLines(filename)

			# restrict the lengths of the names to this range
			lines = [line for line in lines  if 3 < len(line) < 11 ]
			if len(lines) > 300:
				random.shuffle(lines)
				category_lines[category] = lines[:600]
				all_categories.append(category)

		unpadded_features = []
		labels = []
		reference_dict = {}
		n_categories = len(all_categories)
		for label, cat  in enumerate(all_categories):
			reference_dict[label] = cat
			# one_hot_label = torch.zeros(n_categories)
			# one_hot_label[label] = 1
			for line in category_lines[cat]:
				unpadded_features.append(lineToTensor(line).view(-1, n_letters))
				labels.append(label)

		padded_features = torch.nn.utils.rnn.pad_sequence(unpadded_features, padding_value=-1)

		labels = torch.tensor(labels)
		padded_features = padded_features.permute(1,0,2)
		data = padded_features
		
		# save the tensors to data.pt and labels.pt
		torch.save(data, 'datasets/names/data.pt') # and torch.load('datasets/names_data.pt')
		torch.save(labels, 'datasets/names/labels.pt') # and torch.load('datasets/names_labels.pt')
		
		with open(reference_dict_dir, 'w') as file:
			file.write(json.dumps(reference_dict))



	split_index = int(train_test_ratio * len(data))

	from sklearn.utils import shuffle
	import numpy as np
	data , labels = shuffle(data, labels, random_state=1111)

	train_data, train_labels = data[:split_index ], labels[:split_index]
	test_data, test_labels =  data[split_index: ], labels[split_index:]	
	return train_data, train_labels, test_data, test_labels, reference_dict


if __name__ =='__main__':
	import os
	dirname = os.path.dirname(__file__)
	print(dirname)
	train_data, train_labels, test_data, test_labels = get_train_test(data_dir='../datasets/names/data.pt', labels_dir='../datasets/names/labels.pt', reference_dict_dir='../datasets/names/reference_dict', train_test_ratio=0.8)
	print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)



'''
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
'''