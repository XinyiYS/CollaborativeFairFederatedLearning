from scipy.stats import powerlaw
# import matplotlib.pyplot as plt
import numpy as np
import math
# fig, ax = plt.subplots(1, 1)
# a = 1.65911332899
# x = np.linspace(powerlaw.ppf(0.01, a),powerlaw.ppf(0.99, a), 100)
# ax.plot(x, powerlaw.pdf(x, a),'r-', lw=5, alpha=0.6, label='powerlaw pdf')
# plt.show()

# mnist
trainSize=1000
val_shffl = np.random.permutation(trainSize)
val_size=int(0.1*trainSize)
val_shard = val_shffl[:val_size]

train_shard=list(set(val_shffl)-set(val_shard))
train_shard=np.random.permutation(train_shard)




netSize=9
a = 1.65911332899
party_size=60 # middle value
b=np.linspace(powerlaw.ppf(0.01, a),powerlaw.ppf(0.99, a), netSize)

shard_size=list(map(math.ceil, b/sum(b)*party_size*netSize))
shard_indices={}
accessed=0
for nid in range(netSize):
	shard_indices[nid]=train_shard[accessed:accessed+shard_size[nid]]
	accessed=accessed+shard_size[nid]
	print(accessed)
print(shard_size)