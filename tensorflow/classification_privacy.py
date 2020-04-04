from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import tensorflow as tf

import numpy as np

tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

"""Install TensorFlow Privacy."""

"""## Load and pre-process the dataset

Load the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and prepare the data for training.
"""

train, test = tf.keras.datasets.mnist.load_data()
train_data, train_labels = train
test_data, test_labels = test

train_data = np.array(train_data, dtype=np.float32) / 255
test_data = np.array(test_data, dtype=np.float32) / 255

train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

train_labels = np.array(train_labels, dtype=np.int32)
test_labels = np.array(test_labels, dtype=np.int32)

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

assert train_data.min() == 0.
assert train_data.max() == 1.
assert test_data.min() == 0.
assert test_data.max() == 1.

"""## Define and tune learning model hyperparameters
Set learning model hyperparamter values.
"""

epochs = 1
batch_size = 250

"""DP-SGD has three privacy-specific hyperparameters and one existing hyperamater that you must tune:

1. `l2_norm_clip` (float) - The maximum Euclidean (L2) norm of each gradient that is applied to update model parameters. This hyperparameter is used to bound the optimizer's sensitivity to individual training points. 
2. `noise_multiplier` (float) - The amount of noise sampled and added to gradients during training. Generally, more noise results in better privacy (often, but not necessarily, at the expense of lower utility).
3.   `microbatches` (int) - Each batch of data is split in smaller units called microbatches. By default, each microbatch should contain a single training example. This allows us to clip gradients on a per-example basis rather than after they have been averaged across the minibatch. This in turn decreases the (negative) effect of clipping on signal found in the gradient and typically maximizes utility. However, computational overhead can be reduced by increasing the size of microbatches to include more than one training examples. The average gradient across these multiple training examples is then clipped. The total number of examples consumed in a batch, i.e., one step of gradient descent, remains the same. The number of microbatches should evenly divide the batch size. 
4. `learning_rate` (float) - This hyperparameter already exists in vanilla SGD. The higher the learning rate, the more each update matters. If the updates are noisy (such as when the additive noise is large compared to the clipping threshold), a low learning rate may help the training procedure converge. 

Use the hyperparameter values below to obtain a reasonably accurate model (95% test accuracy):
"""

l2_norm_clip = 1.5
noise_multiplier = 1.3
num_microbatches = 250
learning_rate = 0.25

if batch_size % num_microbatches != 0:
    raise ValueError('Batch size should be an integer multiple of the number of microbatches')

"""## Build the learning model

Define a convolutional neural network as the learning model.
"""

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 8,
                           strides=2,
                           padding='same',
                           activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Conv2D(32, 4,
                           strides=2,
                           padding='valid',
                           activation='relu'),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

"""Define the optimizer and loss function for the learning model. Compute the loss as a vector of losses per-example rather than as the mean over a minibatch to support gradient manipulation over each training point."""

optimizer = DPGradientDescentGaussianOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=learning_rate)

loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, reduction=tf.losses.Reduction.NONE)

"""## Compile and train the learning model"""

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=epochs,
          validation_data=(test_data, test_labels),
          batch_size=batch_size)

"""## Measure the differential privacy guarantee

Perform a privacy analysis to measure the DP guarantee achieved by a training algorithm. Knowing the level of DP achieved enables the objective comparison of two training runs to determine which of the two is more privacy-preserving. At a high level, the privacy analysis measures how much a potential adversary can improve their guess about properties of any individual training point by observing the outcome of our training procedure (e.g., model updates and parameters). 

This guarantee is sometimes referred to as the **privacy budget**. A lower privacy budget bounds more tightly an adversary's ability to improve their guess. This ensures a stronger privacy guarantee. Intuitively, this is because it is harder for a single training point to affect the outcome of learning: for instance, the information contained in the training point cannot be memorized by the ML algorithm and the privacy of the individual who contributed this training point to the dataset is preserved.

In this tutorial, the privacy analysis is performed in the framework of Rényi Differential Privacy (RDP), which is a relaxation of pure DP based on [this paper](https://arxiv.org/abs/1702.07476) that is particularly well suited for DP-SGD.

Two metrics are used to express the DP guarantee of an ML algorithm:

1.   Delta ($\delta$) - Bounds the probability of the privacy guarantee not holding. A rule of thumb is to set it to be less than the inverse of the size of the training dataset. In this tutorial, it is set to **10^-5** as the MNIST dataset has 60,000 training points.
2.   Epsilon ($\epsilon$) - This is the privacy budget. It measures the strength of the privacy guarantee by bounding how much the probability of a particular model output can vary by including (or excluding) a single training point. A smaller value for $\epsilon$ implies a better privacy guarantee. However, the $\epsilon$ value is only an upper bound and a large value could still mean good privacy in practice.

Tensorflow Privacy provides a tool, `compute_dp_sgd_privacy.py`, to compute the value of $\epsilon$ given a fixed value of $\delta$ and the following hyperparameters from the training process:

1.   The total number of points in the training data, `n`.
2. The `batch_size`.
3.   The `noise_multiplier`.
4. The number of `epochs` of training.
"""

result = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
    n=60000, batch_size=250, noise_multiplier=1.3, epochs=epochs, delta=1e-5)

print(result)
"""The tool reports that for the hyperparameters chosen above, the trained model has an $\epsilon$ value of 1.18.

## Summary
In this tutorial, you learned about differential privacy (DP) and how you can implement DP principles in existing ML algorithms to provide privacy guarantees for training data. In particular, you learned how to:
*   Wrap existing optimizers (e.g., SGD, Adam) into their differentially private counterparts using TensorFlow Privacy
*   Tune hyperparameters introduced by differentially private machine learning
*   Measure the privacy guarantee provided using analysis tools included in TensorFlow Privacy
"""
