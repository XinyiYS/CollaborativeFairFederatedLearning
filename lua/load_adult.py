import pandas as pd
import numpy as np
import sklearn
import sklearn.preprocessing as preprocessing

# Source: https://www.valentinmihov.com/2015/04/17/adult-income-data-set/
def data_transform(df):
	"""Normalize features."""
	binary_data = pd.get_dummies(df)
	feature_cols = binary_data[binary_data.columns[:-2]]
	scaler = preprocessing.StandardScaler()
	data = pd.DataFrame(scaler.fit_transform(feature_cols), columns=feature_cols.columns)
	return data


def split_and_transform(original, labels, train_test_ratio):

	num_train = int(train_test_ratio * len(original))
	
	original = data_transform(original)

	train_data = original[:num_train]
	train_labels = labels[:num_train]

	test_data = original[num_train:]
	test_labels = labels[num_train:]

	return train_data, train_labels, test_data, test_labels

def get_train_test(dataset_dir='datasets/adult.csv', train_dir='datasets/adult.data', test_dir='datasets/adult.test', train_test_ratio=0.8):

	features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
			"Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
			"Hours per week", "Country", "Target"] 

	# Change these to local file if available

	# train_dir = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
	# test_dir= 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

	import os
	if os.path.isfile(dataset_dir):
		df = pd.read_csv(dataset_dir)

		positives = df[df['income']==1]
		negatives = df[df['income']==0][:len(positives)]

		df = pd.concat([positives, negatives])
		df = df.sample(frac=1).reset_index(drop=True)
		labels = df['income'].astype('float')
		del df["income"]

		return split_and_transform(df, labels, train_test_ratio)

	if not os.path.isfile(train_dir):
		# This will download 3.8M
		train_dir = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
	if not os.path.isfile(test_dir):
		test_dir= 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

	print('reading from', train_dir, test_dir)
	# This will download 3.8M
	original_train = pd.read_csv(train_dir, names=features, sep=r'\s*,\s*', 
								 engine='python', na_values="?")
	# This will download 1.9M
	original_test = pd.read_csv(test_dir, names=features, sep=r'\s*,\s*', 
								engine='python', na_values="?", skiprows=1)

	original = pd.concat([original_train, original_test])

	original = original.dropna()

	original['Target'] = original['Target'].replace('<=50K', 0).replace('>50K', 1)
	original['Target'] = original['Target'].replace('<=50K.', 0).replace('>50K.', 1)

	# create equal number of positive nad negative cases
	positives = original[original['Target']==1]
	negatives = original[original['Target']==0][:len(positives)]

	original  = pd.concat([positives, negatives])
	# shuffle the data set
	original = original.sample(frac=1).reset_index(drop=True)

	labels = original['Target'].astype('float')

	# Redundant column
	# there is an Education-Num column that captures the info in Education
	del original["Education"]

	# Remove target variable
	del original["Target"]

	train_data, train_labels, test_data, test_labels = split_and_transform(original, labels, train_test_ratio)
	return train_data, train_labels, test_data, test_labels


# if __name__ =='__main__':
# 	import os
# 	dirname = os.path.dirname(__file__)
# 	print(dirname)
train_data, train_labels, test_data, test_labels = get_train_test(dataset_dir='adult.csv', train_dir='/Users/lvlingjuan/MEGA/PPDL/Privacy_dl/FairAndPrivateFederatedLearning/FairAndPrivateFederatedLearning/pytorch/datasets/adult.data', test_dir='/Users/lvlingjuan/MEGA/PPDL/Privacy_dl/FairAndPrivateFederatedLearning/FairAndPrivateFederatedLearning/pytorch/datasets/adult.test')
print(train_data.shape, test_data.shape)
# print(type(train_data))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_data_normalized = sc.fit_transform(train_data)
test_data_normalized = sc.transform(test_data)
print(train_data_normalized.shape)
train_data.to_csv (r'train_data.csv', index = False, header=False)
train_labels.to_csv (r'train_labels.csv', index = False, header=False)
test_data.to_csv (r'test_data.csv', index = False, header=False)
test_labels.to_csv (r'test_labels.csv', index = False, header=False)

# import csv
# # writer.writerows(test_labels)
# with open('adult_test_labels.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=' ',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     writer.writerows(test_labels)

