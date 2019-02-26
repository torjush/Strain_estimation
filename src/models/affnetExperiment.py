import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from affineNet import AffineNet
from losses import nncc2d
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-path', help='Path to dataset',
                    type=str, default='../../data/raw/Affine_test')
args = parser.parse_args()
tf.enable_eager_execution()


original = plt.imread(os.path.join(args.data_path, 'original.png'))
moving = plt.imread(os.path.join(args.data_path, 'warped.png'))
moving = moving[:, :, 0]

original = np.reshape(original, [1, 250, 250, 1])
# original = np.tile(original, [5, 1, 1, 1])
moving = np.reshape(moving, [1, 250, 250, 1])
# moving = np.tile(moving, [5, 1, 1, 1])


affnet = AffineNet(5)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

for i in range(5000):
    with tf.GradientTape() as tape:
        warped = affnet(tf.constant(original), tf.constant(moving))
        loss = nncc2d(tf.constant(moving), warped)
    print('{{"metric": "Loss", "value": {}, "step": {}}}'.format(loss, i + 1))
    grads = tape.gradient(loss, tf.trainable_variables())
    optimizer.apply_gradients(zip(grads, tf.trainable_variables()),
                              global_step=tf.train.get_or_create_global_step())

fig, ax = plt.subplots(ncols=3)
ax[0].imshow(original[0, :, :, 0])
ax[1].imshow(moving[0, :, :, 0])
ax[2].imshow(warped[0, :, :, 0])
plt.tight_layout()
plt.savefig('imgs.png')
plt.close()
